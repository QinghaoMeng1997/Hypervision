import torchvision.transforms as transforms
import torch
import os
import torch.utils.data
from architecture import model_generator
import numpy as np
from PIL import Image
import cv2
import time

patch_size = 512
def main():
    save_path = './Save_Img'
    img_path = 'Input_Img/img.png'
    img = Image.open(img_path)

    img_array = np.array(img)
    img_array = img_array[0:0 + 512, 0:0 + 512]
    img_array = img_array.astype(np.float32)
    img_array = img_array / img_array.max()
    img_array = np.ascontiguousarray(img_array)
    print('input_init:', img_array.shape)

    input_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).cuda()
    print('input_patch',input_tensor.shape)

    input = input_tensor.squeeze(0).squeeze(0)
    image = transforms.ToPILImage()(input)
    image.save('input.png')#存储输入

    model_srnet = model_generator('srnet_small', 'exp/net_1000epoch.pth')
    # model.freeze()
    model_srnet.eval()
    start_time = time.perf_counter()
    for i in range(1):
        outputs = model_srnet(input_tensor)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"操作执行时间: {execution_time:.6f} 秒")

    with torch.no_grad():
        my_tensor = outputs.squeeze().permute(1, 2, 0).cpu().numpy()

        # output_file_path = './output.h5'
        # with h5py.File(output_file_path, 'w') as hf:
        #     # 在文件中创建一个数据集来存储高光谱图像
        #     hf.create_dataset('hsi', data=my_tensor, compression="gzip", compression_opts=9)

        my_img = my_tensor[:, :, (7, 15, 20)]
        my_img = 255*my_img/my_img.max()
        cv2.imwrite(os.path.join(save_path, "{}".format('out') + ".png"), my_img)

        # my_tensor = 255 * my_tensor / my_tensor.max()
        # for i in range(61):
        #     ch_img = my_tensor[:, :, i]
        #     cv2.imwrite(os.path.join("./Channel_Img", "{}".format(i) + ".png"), ch_img)


if __name__=='__main__':
    main()