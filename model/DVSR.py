from model.sr3_modules.diffusion import GaussianDiffusion
from model.sr3_modules.unet import UNet as UNet_dv
import torch.nn as nn


class DVSr(nn.Module):
    def __init__(self, img_size, device, loss_type):
        super().__init__()
        self.init_predictor = UNet_dv(in_channel=3, out_channel=3, inner_channel=64, with_noise_level_emb=False)
        self.diffusion = GaussianDiffusion(UNet_dv(inner_channel=32), img_size, conditional=True, loss_type=loss_type)
        self.device = device

    def forward(self, blur, sharp):
        # 将blur输入init_predictor
        init_predict = self.init_predictor(blur)
        # 获得残差
        residual = blur - init_predict
        schedule = {"schedule": "linear",
                    "n_timestep": 2000,
                    "linear_start": 1e-6,
                    "linear_end": 1e-2}
        self.diffusion.set_new_noise_schedule(schedule, self.device)
        self.diffusion.set_loss(self.device)
        # 以blur为条件，建模残差
        l_pix = self.diffusion(residual, blur)
        b, c, h, w = blur.shape
        l_pix = l_pix.sum() / int(b * c * h * w)
        return l_pix

