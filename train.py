import torch
from dataset import GroPro_dataset, get_file_path
from torch.utils.data import DataLoader
import torch.optim as optim
from model_dv.DVSR import DVSr
from model_dv.networks import init_weights
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel


class EMA(AveragedModel):
    def __init__(self, model, decay, device='cuda'):
        def ema_avg(avg_model_param, model_param):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg)


if __name__ == '__main__':
    torch.cuda.set_device('cuda:1')
    type_ = 'origin'
    # load是加载的预训练文件路径
    load = None
    # 训练图像大小，随机裁剪
    img_size = 128
    # 测试图片大小，中心裁剪
    center_crop_size = 720
    s_img_path_train, b_img_path_train = get_file_path('images/train')
    train_dataset = GroPro_dataset(s_img_path_train, b_img_path_train, img_size, mode='train')
    s_img_path_test, b_img_path_test = get_file_path('images/test')
    test_dataset = GroPro_dataset(s_img_path_test, b_img_path_test, crop_size=center_crop_size, mode='test')
    train_dataloader = DataLoader(train_dataset, 22, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, 1, pin_memory=True)
    # 定义模型
    model = DVSr(img_size=img_size, device='cuda', loss_type='l1')
    # 模型参数初始化
    init_weights(model, init_type='orthogonal')
    model = model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    epoch = 1000
    cur_epoch = 1
    if load:
        checkpoint = torch.load(load, 'cuda')
        model.load_state_dict(checkpoint['model_dv'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        cur_epoch = checkpoint['epoch'] + 1
    print(f'start epoch is {cur_epoch}')
    writer = SummaryWriter(log_dir='logs_{}'.format(type_))
    iterations = 0
    is_first = True
    count = 0
    # 多少epoch记录一次
    epoch_to_show = 50
    for i in tqdm(range(cur_epoch, epoch + 1), desc='train', total=epoch - cur_epoch + 1):
        model.train()
        l = 0
        for j, img in enumerate(train_dataloader):
            blur = img['blur']
            blur = blur.cuda()
            sharp = img['sharp']
            sharp = sharp.cuda()
            loss = model(blur, sharp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l += loss.item()
        writer.add_scalar('loss', l, i)
        if i % epoch_to_show == 0:
            count += 1
            model.eval()
            with torch.no_grad():
                img = test_dataset[0]
                img_blur = img['blur'].reshape(1, 3, center_crop_size, center_crop_size).cuda()
                img_sharp = img['sharp'].reshape(1, 3, center_crop_size, center_crop_size).cuda()
                init_predict = model.init_predictor(img_blur).reshape(3, center_crop_size, center_crop_size)
            if is_first:
                writer.add_image('1_input_blur', img_blur[0], 1)
                writer.add_image('6_sharp', img_sharp[0])
                is_first = False
            residual = model.diffusion.p_sample_loop(img_blur, continous=False)
            writer.add_image('2_init_predict', init_predict, count)
            writer.add_image('3_real_residual', img_sharp[0] - init_predict, count)
            writer.add_image('4_predict_residual', residual, count)
            sample = residual + init_predict
            writer.add_image('5_sample', sample, count)
        if i % 200 == 0:
            state = {'model_dv': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': i}
            torch.save(state, f'{i}epoch_{type_}.pth')
