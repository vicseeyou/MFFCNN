import os.path

import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision.transforms import transforms
import pandas as pd

class read_single_scale_img(Dataset):
    def __init__(self, img_dir, label_path):
        self.img_dir = img_dir
        self.img_name = sorted(os.listdir(img_dir))
        self.excel_data = pd.read_excel(label_path, header=None)

        self.transform = transform

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir, self.img_name[item])
        image = Image.open(img_path).convert("RGB")
        tensor_img = self.transform(image)
        #
        label_array = self.excel_data.values[item, :]
        label = torch.argmax(torch.tensor(label_array)).item()  #
        # name = self.excel_data.values[item,:]
        # name = self.img_name[item]
        return tensor_img, label

transform = transforms.Compose([
        transforms.ToTensor(),

    ])

class read_single_scale(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_name = sorted(os.listdir(img_dir))


        self.transform = transform

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir, self.img_name[item])
        image = Image.open(img_path).convert("RGB")
        tensor_img = self.transform(image)

        # name = self.excel_data.values[item,:]
        # name = self.img_name[item]
        return tensor_img

class read_multi_scale_img(Dataset):
    def __init__(self, img_dir1, img_dir2, label_path):
        self.img_dir1 = img_dir1
        self.img_dir2 = img_dir2
        self.img_name = sorted(os.listdir(img_dir1))
        self.excel_data = pd.read_excel(label_path, header=None)

        self.transform = transform

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_path1 = os.path.join(self.img_dir1, self.img_name[item])
        image1 = Image.open(img_path1).convert("RGB")
        tensor_img1 = self.transform(image1)
        img_path2 = os.path.join(self.img_dir2, self.img_name[item])
        image2 = Image.open(img_path2).convert("RGB")
        tensor_img2 = self.transform(image2)
        #
        label_array = self.excel_data.values[item, :]
        label = torch.argmax(torch.tensor(label_array)).item()
        # name = self.excel_data.values[item,:]
        # name = self.img_name[item]
        return tensor_img1, tensor_img2, label