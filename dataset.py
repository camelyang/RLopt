import os
import cv2
import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

def make_path_list(root):
    path_list = []
    for filename in os.listdir(root):
        path_list.append(os.path.join(root, filename))

    return path_list

class SegDataset(Dataset):

    def __init__(self, img_path, mask_path, transform = None):

        self.img_paths = make_path_list(img_path)
        self.mask_paths = make_path_list(mask_path)
        self.transform = transform
    
    def __getitem__(self, index):
        
        # 读取image和mask
        img = Image.open(self.img_paths[index])
        # img = cv2.imread(self.img_paths[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)

        # 返回三通道张量(C, H, W)
        return img, mask

    def __len__(self):

        return len(self.img_paths)

# 测试
if __name__ == "__main__":
    imgpath = "D:/UNetPlus/CrackDataset/dataset-EdmCrack600/test_img_split"
    maskpath = "D:/UNetPlus/CrackDataset/dataset-EdmCrack600/test_lab_split"
    dataset = SegDataset(imgpath, maskpath, transforms.ToTensor())
    print(len(dataset))
    print(dataset[0][0])
    print(dataset[0][1].shape)
    img = transforms.ToPILImage()(dataset[0][1])
    img.show()
