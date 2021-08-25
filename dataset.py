from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as ttf
device = "cuda" if torch.cuda.is_available() else "cpu"


class GAN_Data(Dataset):
    def __init__(self, path_list, transforms= None):
        super().__init__()

        self.path_list = path_list
        self.transforms = transforms
        self.t = ttf.Resize((256, 256))
        self.blur = ttf.GaussianBlur(3, sigma=(0.1, 2.0))
    
    def __getitem__(self, idx):

        img_path = self.path_list[idx]

        img = np.array(Image.open(r"D:/Desktop/Medical Imaging/MRI_512/" + img_path).convert('RGB').resize((512, 512)))
        img = torch.tensor(img, dtype= torch.float).permute(2, 0, 1)

        if self.transforms:
            img = self.transforms(img)
        
        lr_img = self.blur(self.t(img))

        return lr_img.to(device) / 255., img.to(device) / 255.
    
    def __len__(self):
        return len(self.path_list)
