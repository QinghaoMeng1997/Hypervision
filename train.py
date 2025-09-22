import hdf5storage
import torch
import argparse
import os
import time
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from getdataset import TrainDataset_V1, ValidDataset_V1
from my_utils import AverageMeter, initialize_logger, save_checkpoint, Loss_RMSE, Loss_PSNR, Loss_TV, Loss_MRAE, Loss_SAM
from DataProcess import Data_Process
import torch.utils.data
from architecture import model_generator
import numpy as np


parser = argparse.ArgumentParser(description="Model training of Hypervision")
parser.add_argument("--method", type=str, default='srnet_small', help='Model')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--end_epoch", type=int, default=200, help="number of epochs")
parser.add_argument("--epoch_sam_num", type=int, default=1000, help="per_epoch_iteration")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--gpu_id", type=str, default='0', help='select gpu')
parser.add_argument("--pretrained_model_path", type=str, default=None, help='pre-trained model path')
parser.add_argument("--sigma", type=float, default=(0, 1 / 255, 2/255), help="Sigma of Gaussian Noise")
parser.add_argument("--mask_path", type=str, default='./', help='path of calibrated sensing matrix')
parser.add_argument("--output_folder", type=str, default='./exp/', help='output path')
parser.add_argument("--start_dir", type=int, default=(800, 800), help="size of test image coordinate")
parser.add_argument("--image_size", type=int, default=(512, 512), help="size of test image")
parser.add_argument("--train_patch_size", type=int, default=(512, 512), help="size of patch")
parser.add_argument("--valid_patch_size", type=int, default=(512, 512), help="size of patch")
parser.add_argument("--train_data_path", type=str, default="/", help='path datasets')
parser.add_argument("--valid_data_path", type=str, default="/", help='path datasets')

css_1 = hdf5storage.loadmat('./MASK/')['spectral']
css_2 = hdf5storage.loadmat('./MASK/')['spectral']
change_rate = css_2 / css_1
change_rate = np.expand_dims(np.expand_dims(change_rate, 1), 1)
change_rate = change_rate.astype(np.float32)

opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_mrae = Loss_MRAE()
criterion_sam = Loss_SAM()
criterion_tv = Loss_TV(TVLoss_weight=float(0.5))
data_processing = Data_Process()

mask_init = hdf5storage.loadmat(opt.mask_path)['mask']
print('mask_init:', mask_init.shape)
mask = mask_init[:, opt.start_dir[0]:opt.start_dir[0]+opt.image_size[0], opt.start_dir[1]:opt.start_dir[1] + opt.image_size[1]]
mask = np.maximum(mask, 0)
mask = mask / mask.max()
mask = torch.from_numpy(mask)
mask = mask.unsqueeze(0)
mask = mask.cuda()
mask = mask.repeat(opt.batch_size,1,1,1)

print('mask:', mask.dtype, mask.shape, mask.max(), mask.mean(), mask.min())

def main():
    cudnn.benchmark = True

    print("\nloading dataset ...")
    train_data = TrainDataset_V1(data_path=opt.train_data_path, patch_size=opt.train_patch_size, arg=True)
    print('len(train_data):', len(train_data))
    print(f"Iteration per epoch: {len(train_data)}")
    val_data = ValidDataset_V1(data_path=opt.valid_data_path, patch_size=opt.valid_patch_size, arg=True)
    print('len(valid_data):', len(val_data))
    output_path = opt.output_folder

    # iterations
    per_epoch_iteration = opt.epoch_sam_num // opt.batch_size
    total_iteration = per_epoch_iteration*opt.end_epoch

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = model_generator(opt.method, opt.pretrained_model_path)
    # model = model.cuda()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    if torch.cuda.is_available():
        criterion_rmse.cuda()
        criterion_psnr.cuda()
        criterion_tv.cuda()
        criterion_mrae.cuda()

    start_epoch = 0
    iteration = start_epoch * per_epoch_iteration

    #opt.init_lr
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.init_lr,
                                 betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration - iteration, eta_min=1e-6)

    log_dir = os.path.join(output_path, 'exper_dim16_lay_danshuru.log')
    logger = initialize_logger(log_dir)

    record_rmse_loss = 10000
    strat_time = time.time()

    while iteration < total_iteration:
        print('iteration:',iteration)
        model.train()
        losses = AverageMeter()

        train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2,
                    pin_memory=True, drop_last=True)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        for i, (HSIs) in enumerate(train_loader):

            HSIs = HSIs.cuda()
            inputs, targets = data_processing.get_mos_hsi(hsi=HSIs, mask=mask, sigma=opt.sigma, mos_size=512, hsi_input_size=512, hsi_target_size=512)
            inputs = Variable(inputs)
            targets = Variable(targets)

            lr = optimizer.param_groups[0]['lr']
            outputs = model(inputs)
            #calculate the hybrid loss
            loss_rmse = criterion_rmse(outputs, targets)
            loss_tv = criterion_tv(outputs, targets)
            loss_mrae = criterion_mrae(outputs, targets) * 0.2
            loss = loss_rmse + loss_tv + loss_mrae
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            losses.update(loss.data)
            iteration = iteration + 1

            if iteration % per_epoch_iteration == 0:
                epoch = iteration // per_epoch_iteration
                end_time = time.time()
                epoch_time = end_time - strat_time
                strat_time = time.time()
                rmse_loss, psnr_loss, mrae_loss, sam_loss = Validate(val_loader, model, mask)

                # Save model
                if torch.abs(
                        record_rmse_loss - rmse_loss) < 0.0001 or rmse_loss < record_rmse_loss or iteration % 10000 == 0:
                    print(f'Saving to {output_path}')
                    save_checkpoint(output_path, (epoch), iteration, model, optimizer)
                    if rmse_loss < record_rmse_loss:
                        record_rmse_loss = rmse_loss
                # print loss
                print(" Iter[%06d/%06d], Epoch[%06d], Time[%06d],  learning rate : %.9f, Train Loss: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f, Test MRAE: %.9f, Test SAM: %.9f "
                      % (iteration, total_iteration, epoch, epoch_time, lr, losses.avg, rmse_loss, psnr_loss, mrae_loss, sam_loss))

                logger.info(" Iter[%06d/%06d], Epoch[%06d], Time[%06d],  learning rate : %.9f, Train Loss: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f, Test MRAE: %.9f, Test SAM: %.9f "
                      % (iteration, total_iteration, epoch, epoch_time, lr, losses.avg, rmse_loss, psnr_loss, mrae_loss, sam_loss))

def Validate(val_loader, model, mask):
    model.eval()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_mrae = AverageMeter()
    losses_sam = AverageMeter()

    for i, (HSIs) in enumerate(val_loader):
        HSIs = HSIs.cuda()
        #Generate the measurements using traning HSIs and selected sub-pattern
        inputs, targets = data_processing.get_mos_hsi(hsi=HSIs, mask=mask, sigma=opt.sigma, mos_size=512, hsi_input_size=512, hsi_target_size=512)

        with torch.no_grad():
            outputs = model(inputs)

            loss_rmse = criterion_rmse(outputs, targets)
            loss_psnr = criterion_psnr(outputs, targets)
            loss_mrae = criterion_mrae(outputs, targets)
            loss_sam = criterion_sam(outputs, targets)
            losses_psnr.update(loss_psnr.data)
            losses_rmse.update(loss_rmse.data)
            losses_mrae.update(loss_mrae.data)
            losses_sam.update(loss_sam.data)
    return losses_rmse.avg, losses_psnr.avg, losses_mrae.avg, losses_sam.avg


if __name__ == '__main__':
    main()


