from model_dv.DVSR import DVSr
from dataset import GroPro_dataset, get_file_path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    center_crop_size = 256
    schedule = {"schedule": "linear",
                "n_timestep": 2000,
                "linear_start": 1e-6,
                "linear_end": 1e-2}
    s_img_path_test, b_img_path_test = get_file_path('images/test')
    test_dataset = GroPro_dataset(s_img_path_test, b_img_path_test, crop_size=center_crop_size, mode='test')
    test_dataloader = DataLoader(test_dataset, 1, pin_memory=True)
    model = DVSr(img_size=center_crop_size, device='cuda', loss_type='l1')
    checkpoint = torch.load('1000epoch_blur.pth', map_location='cuda:0')
    # model_dv.load_state_dict(checkpoint['model_dv'], strict=False)
    model.load_state_dict(checkpoint, strict=False)
    model.diffusion.set_new_noise_schedule(schedule, device='cuda')
    model = model.cuda()
    writer = SummaryWriter(log_dir='logs_inference')
    model.eval()
    num_img = 1
    count = 0
    for imgs in test_dataloader:
        b = imgs['blur']
        s = imgs['sharp']
        b = b.cuda()
        s = s.cuda()
        init_predict = model.init_predictor(b)
        res = (b - init_predict)
        writer.add_image('1_blur-initPre', res[0])
        residual = model.diffusion.p_sample_loop(b, continous=True)
        for i in range(0, residual.shape[0]):
            writer.add_image('2_residual', residual[i], i)
        writer.add_image('6_init_predict', init_predict.reshape(3, center_crop_size, center_crop_size), 1)
        writer.add_image('5_true_blur', b.reshape(3, center_crop_size, center_crop_size), 1)
        writer.add_image('3_sample', b[0]+residual[-1], 1)
        writer.add_image('4_sharp', s[0], 1)
        count += 1
        if count == num_img:
            break

