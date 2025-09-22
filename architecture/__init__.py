import torch
from .SRNet_Small import SRNetSmall
def model_generator(method, pretrained_model_path=None):


    if method == 'srnet_small':
        model = SRNetSmall(in_channels=1, mask_channels=61, out_channels=61, dim=32, deep_stage=1, num_blocks=[1], num_heads=[8]).cuda()

    else:
        print(f'Method {method} is not defined !!!!')

    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=True)


    return model
