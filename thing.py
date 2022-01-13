import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.imgs_path = "img_align_celeba/"
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])
        self.img_dim = (128, 128)
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        masking_image= cv2.imread('masks/14239.png',  cv2.IMREAD_UNCHANGED)
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        masking_image=cv2.resize(masking_image, self.img_dim)
        #ret, mask = cv2.threshold(masking_image, 0, 255, cv2.THRESH_BINARY)
        bg = cv2.bitwise_or(img,img,mask = masking_image)
        img = cv2.hconcat([bg, img])
        if self.transform:
            img = self.transform(img)
        #img = torch.from_numpy(img)
        return img

    
