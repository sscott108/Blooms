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
        planet = np.rot90(planet, rot, axes=(1,2))
        cicyano = np.rot90(cicyano, rot, axes=(1,2))
        sentinel = np.rot90(sentinel, rot, axes=(1,2))
        
        sample['planet'] = planet.copy()
        sample['cicyano'] = cicyano.copy()
        sample['sentinel'] = sentinel.copy()
        return sample

# class CropAndChop(object):
#     def __call__(self,sample):
        
#         imgdata = sample['planet']

#         crop_height, crop_width = 512, 512
        
#         height,width = imgdata.shape[0], imgdata.shape[1]
#         x = np.random.randint(0, width - crop_width + 1)
#         y = np.random.randint(0, height - crop_height + 1)
        
#         imgdata = imgdata[y:y + crop_height, x:x + crop_width,: ] 
        
#         sample['planet'] = imgdata

#         return sample
        
class CropAndChop(object):
    def __init__(self, crop_shape):
        self.crop_shape = crop_shape

    def __call__(self, sample):
      
        imgdata = sample['planet']

        crop_height, crop_width = self.crop_shape, self.crop_shape
        height, width = imgdata.shape[1], imgdata.shape[2]

        # Randomly select the top-left corner for cropping
        x = np.random.randint(0, width - crop_width + 1)
        y = np.random.randint(0, height - crop_height + 1)

        # Perform the cropping
        imgdata = imgdata[:, y:y + crop_height, x:x + crop_width]

        # Update the sample dictionary
        sample['planet'] = imgdata

        return sample

class ToTensor(object):
    def __call__(self, sample):

        sample['planet'] =  torch.from_numpy(sample['planet'].copy())
        sample['sentinel'] = torch.from_numpy(sample['sentinel'].copy())
        sample['cicyano'] =  torch.from_numpy(sample['cicyano'].copy())
        return sample
    
    
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

    

# class Bin(object):
#     def __call__(self, sample):
        
#         fptdata = sample['fpt']
        
#         min_class = 100
#         max_class = 249

#         mask = (fptdata >= min_class) & (fptdata <= max_class)

#         new_label = np.full_like(fptdata, 0)
#         new_label[fptdata < 31] = 1
#         new_label[(fptdata >= 31) & (fptdata < 100)] = 2
#         new_label[(fptdata >= 100) & (fptdata < 249)] = 3
# #         new_label[(fptdata >= 175) & (fptdata <= 249)] = 4

# #         mapped_labels = np.clip(np.array([map_value(val) for val in new_label.flatten()]), 0, 9) 
# #         mapped = mapped_labels.reshape(1, 320,320)

#         return {'img': sample['img'], 
#                 'fpt': new_label, 
#                 'imgfile': sample['imgfile']}

# def create_dataset(*args, apply_transforms=True, **kwargs):
#     if apply_transforms:
#         data_transforms = transforms.Compose([
#             Crop(),
#             FlipsAndTricks(),
#             Bin(),
#            ])
#     else: data_transforms = None

#     data = BloomDataset(*args, **kwargs, transforms=data_transforms)
#     return data
        



# flattened_labels = new_label.flatten()

# # Count the occurrences of each class
# unique_classes, counts = np.unique(flattened_labels, return_counts=True)

# # Plotting the counts
# plt.figure(figsize=(10, 6))
# plt.bar(unique_classes, counts, color='red')
# plt.xlabel('Class')
# plt.ylabel('Count')
# plt.title('Class Distribution in Segmentation Labels')
# plt.xticks(unique_classes)
# plt.show()

class encoding_block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(encoding_block,self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        model.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*model)
    def forward(self, x):
        return self.conv(x)    
    
class unet_model(nn.Module):
    def __init__(self,out_channels=4,features=[64, 128, 256, 512]):
        super(unet_model,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv1 = encoding_block(3,features[0])
        self.conv2 = encoding_block(features[0],features[1])
        self.conv3 = encoding_block(features[1],features[2])
        self.conv4 = encoding_block(features[2],features[3])
        self.conv5 = encoding_block(features[3]*2,features[3])
        self.conv6 = encoding_block(features[3],features[2])
        self.conv7 = encoding_block(features[2],features[1])
        self.conv8 = encoding_block(features[1],features[0])        
        self.tconv1 = nn.ConvTranspose2d(features[-1]*2, features[-1], kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)        
        self.bottleneck = encoding_block(features[3],features[3]*2)
        self.final_layer = nn.Conv2d(features[0],out_channels,kernel_size=1)
    def forward(self,x):
        skip_connections = []
        x = self.conv1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv2(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv3(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv4(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        x = self.tconv1(x)
        x = torch.cat((skip_connections[0], x), dim=1)
        x = self.conv5(x)
        x = self.tconv2(x)
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.conv6(x)
        x = self.tconv3(x)
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.conv7(x)        
        x = self.tconv4(x)
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.conv8(x)
        x = self.final_layer(x)
        return x
