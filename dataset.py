import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset
from PIL import Image
import os


def transform_img(img_list, crop_size, mode):
    to_tensor = v2.ToTensor()
    imgs = [to_tensor(img) for img in img_list]
    if mode == 'train':
        stack_img = torch.stack(imgs, 0)
        # 训练图片进行随机的左右和上下翻转
        transforms = v2.Compose([v2.RandomCrop(crop_size), v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()])
        img_t = transforms(stack_img)
        img_t = torch.unbind(img_t, 0)
    else:
        stack_img = torch.stack(imgs, 0)
        transforms = v2.CenterCrop(crop_size)
        img_t = transforms(stack_img)
        img_t = torch.unbind(img_t, 0)
    return img_t


def get_file_path(root):
    # root应该是train或test结尾
    ls = os.listdir(root)
    sharp_img_path = []
    blur_img_path = []
    for r in ls:
        path = os.path.join(root, r)
        sharp_path = os.path.join(path, 'sharp')
        blur_path = os.path.join(path, 'blur')
        sharp_img_list = os.listdir(sharp_path)
        blur_img_list = os.listdir(blur_path)
        for i in range(len(sharp_img_list)):
            sharp_img_path.append(os.path.join(sharp_path, sharp_img_list[i]))
            blur_img_path.append(os.path.join(blur_path, blur_img_list[i]))

    return sharp_img_path, blur_img_path


class GroPro_dataset(Dataset):
    def __init__(self, sharp_root, blur_root, crop_size=128, mode='train'):
        self.sharp_root = sharp_root
        self.blur_root = blur_root
        self.crop_size = crop_size
        self.mode = mode

    def __len__(self):
        return len(self.sharp_root)

    def __getitem__(self, index):
        sharp_image = Image.open(self.sharp_root[index])
        blur_img = Image.open(self.blur_root[index])
        img_t = transform_img([sharp_image, blur_img], crop_size=self.crop_size, mode=self.mode)
        # img_t = [img * 2. - 1. for img in img_t]
        return {'sharp': img_t[0], 'blur': img_t[1]}
