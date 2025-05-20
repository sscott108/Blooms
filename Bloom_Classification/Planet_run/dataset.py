import numpy as np
from os import listdir
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset
import segmentation_models_pytorch as smp
from torch.optim import Adam
import os
from matplotlib import pyplot as plt
import rasterio as rio
from rasterio.features import rasterize
from shapely.geometry import Polygon
import torch.nn.functional as F
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
warnings.filterwarnings('ignore')
from PIL import Image
import pickle as pkl

np.random.seed(420)
torch.manual_seed(420)
torch.cuda.manual_seed(420)  # If using CUDA

from sklearn.model_selection import train_test_split
import torchmetrics
import segmentation_models_pytorch as smp
from torch.utils.data import Subset
import random
from tqdm import tqdm
from pyproj import Transformer
from rasterio.windows import Window
from torchvision.models import resnet50
from sklearn.metrics import accuracy_score
torch.set_printoptions(sci_mode=False)
from torch.utils.data import WeightedRandomSampler
from matplotlib.colors import ListedColormap, BoundaryNorm

class FlipsAndTricks(object):
    def __call__(self, sample):

        planet = sample['planet']
        cicyano = sample['cicyano']
        sentinel = sample['sentinel']
        

        mirror = np.random.randint(0, 2)
        flip = np.random.randint(0, 2)
        
        if mirror:
#             print('horizontal flip')
            planet = np.fliplr(planet)
            cicyano = np.fliplr(cicyano)
            sentinel = np.fliplr(sentinel)

        if flip:
#             print('vertical flip')
            planet = np.flipud(planet)
            cicyano = np.flipud(cicyano)
            sentinel = np.flipud(sentinel)

        # rotate by [0,1,2,3]*90 deg
        rot = np.random.randint(0, 4)
#         print(f'rotated {rot* 90} degrees ')
        planet = np.rot90(planet, rot, axes=(0,1))
        cicyano = np.rot90(cicyano, rot, axes=(0,1))
        sentinel = np.rot90(sentinel, rot, axes=(0,1))
        
        sample['planet'] = planet.copy()
        sample['cicyano'] = cicyano.copy()
        sample['sentinel'] = sentinel.copy()
        return sample

class CropAndChop(object):
    def __call__(self,sample):
        
        imgdata = sample['planet']

        crop_height, crop_width = 512, 512
        
        height,width = imgdata.shape[0], imgdata.shape[1]
        x = np.random.randint(0, width - crop_width + 1)
        y = np.random.randint(0, height - crop_height + 1)
        
        imgdata = imgdata[y:y + crop_height, x:x + crop_width,: ] 
        
        sample['planet'] = imgdata

        return sample
        
class ToTensor(object):
    def __call__(self, sample):

        sample['planet'] =  torch.from_numpy(sample['planet'].copy())
        sample['sentinel'] = torch.from_numpy(sample['sentinel'].copy())
        sample['cicyano'] =  torch.from_numpy(sample['cicyano'].copy())
        return sample
    
# class FlipsAndTricks(object):
#     def __call__(self, sample):

#         imgdata = sample['img']
#         fptdata = sample['fpt']

#         mirror = np.random.randint(0, 2)
#         flip = np.random.randint(0, 2)
        
#         if mirror:
#             imgdata = np.fliplr(imgdata)

#         if flip:
#             imgdata = np.flipud(imgdata)

#         # rotate by [0,1,2,3]*90 deg
#         rot = np.random.randint(0, 4)
#         imgdata = np.rot90(imgdata, rot, axes=(1,2))
        
#         sample['img'] = imgdata.copy()
#         return sample
    
class Normalize(object):
    def __init__(self):
        #calculated from train loader images OF this dataset
        self.channel_means = np.array([50.5977, 46.5061, 50.8589])
        self.channel_stds = np.array([32.0780, 32.3217, 32.0690])

    def __call__(self, sample):
        #reshape below ensures the mean and std are broadcast across height/width dimensions
        sample['planet'] = (sample['planet']-self.channel_means.reshape(
            sample['planet'].shape[0], 1, 1))/self.channel_stds.reshape(
            sample['planet'].shape[0], 1, 1)

        return sample
def map_value(val):
    if val < 100:
        return 0
    # Compute the index of the interval based on increments of 15
    return (val - 100) // 15 + 1


class BloomDataset():
    def __init__(self, imgdir, lbldir, transforms=None):
        
        self.imgdir = imgdir
        self.lbldir = lbldir
        self.transforms = transforms
        self.imgpaths, self.lblpaths = [], []
        
        for file in sorted(listdir(self.lbldir)):
            if file.endswith('.jpg'):
                self.lblpaths.append(os.path.join(self.lbldir,file))

        for file in sorted(listdir(self.imgdir)):
            if file.endswith('.jpg'):
                self.imgpaths.append(os.path.join(self.imgdir,file))


    def __len__(self): return len(self.imgpaths)

    
    def __getitem__(self, idx):
        
        with rio.open(self.imgpaths[idx]) as rds:
            data = rds.read()
#             data = np.moveaxis(data, 0, 2) 
            img_arr = np.array(data)

            imgdata = torch.from_numpy(img_arr.copy())

        with rio.open(self.lblpaths[idx]) as rds:
            data = rds.read()
#             data = np.moveaxis(data, 0, 2) 
            fpt_arr = np.array(data)
           
            
            fptdata =  torch.from_numpy(fpt_arr.copy())
        

        sample = {'img': imgdata,
                  'fpt': fptdata,
                  'imgfile': self.imgpaths[idx]}

        if self.transforms:
            sample = self.transforms(sample)

        return sample
        
    def display(self, idx):
        sample = self[idx]
        imgdata = sample['img'] #.permute(1, 2, 0)
        fptdata = sample['fpt']
        imgdata = np.moveaxis(imgdata, 0, 2) 
        fptdata = np.moveaxis(fptdata, 0,2)
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(6,3))
        ax1.imshow(imgdata)
        ax1.axis('off')
        ax2.imshow(fptdata)
        ax2.axis('off')
        plt.show()
        return fig

    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet_pretrained = resnet50(pretrained=True)
        self.resnet_pretrained.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3),
                              stride=(2, 2),padding=(3, 3), bias=False)
        
        self.fc1 = nn.Linear(self.resnet_pretrained.fc.out_features, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # output size changed to 2
        self.dropout = nn.Dropout(p=0.35)
    def forward(self, image):
        img_features = self.resnet_pretrained(image)
        img_features = torch.flatten(img_features, 1)
        img_features = self.fc1(img_features)
        x = self.relu(img_features)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x





