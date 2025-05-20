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
        


import os
import torch
import numpy as np
import rasterio as rio
from rasterio.windows import Window
from pyproj import Transformer

class SentinelDataset:
    def __init__(self, lat, lon):
        """
        PyTorch dataset for extracting Sentinel data around specific lat/lon coordinates.

        Args:
            lat (float): Latitude of the crop center.
            lon (float): Longitude of the crop center.
        """
        self.lat = lat
        self.lon = lon

        # Directory containing the label files
        lbl_dir = '/work/srs108/all_data'
        self.cicyano_paths = []

        # Gather valid .tif file paths and filter invalid ones
        for folder in tqdm(sorted(os.listdir(lbl_dir))):
            print(folder)
            if folder not in [".DS_Store", ".ipynb_checkpoints"]:
                folder_path = os.path.join(lbl_dir, folder)
                for file in sorted(os.listdir(folder_path)):
                    if file.endswith('.tif'):
                        parts = file.split('.')
                        name = parts[-3]
                        if name == 'truecolor':
                            file_path = os.path.join(folder_path, file)
                            with rio.open(file_path) as rds:
                                data = rds.read([1, 2, 3])
                                crop_size = 3  
                                transformer = Transformer.from_crs("EPSG:4326", rds.crs, always_xy=True)
                                xx, yy = transformer.transform(self.lon, self.lat)
                                row, col = rds.index(xx, yy)

                                # Define the crop window
                                row_start = max(row - crop_size, 0)
                                row_end = min(row + crop_size, rds.height)
                                col_start = max(col - crop_size, 0)
                                col_end = min(col + crop_size, rds.width)

                                window = Window.from_slices((row_start, row_end), (col_start, col_end))
                                data_r = rds.read(window=window)
                                sentinel_data = np.array(data_r)

                                if np.sum(sentinel_data) == 0 or np.sum(sentinel_data) > 14000:
                                    pass
                                else: self.cicyano_paths.append(file_path)
  

    def __len__(self):
        return len(self.cicyano_paths)

    def __getitem__(self, idx):

        file = self.cicyano_paths[idx]

        with rio.open(file) as rds:
            # Read RGB channels (assuming [1, 2, 3] are R, G, B)
            data = rds.read([1, 2, 3])

            crop_size = 3  # Crop size in pixels

            # Convert lat/lon to image coordinates
            transformer = Transformer.from_crs("EPSG:4326", rds.crs, always_xy=True)
            xx, yy = transformer.transform(self.lon, self.lat)
            row, col = rds.index(xx, yy)

            # Define the crop window
            row_start = max(row - crop_size, 0)
            row_end = min(row + crop_size, rds.height)
            col_start = max(col - crop_size, 0)
            col_end = min(col + crop_size, rds.width)

            window = Window.from_slices((row_start, row_end), (col_start, col_end))
            data_r = rds.read(window=window)

        # Convert to tensor
        sentinel = torch.tensor(data_r, dtype=torch.float32)

        dy = file.split('/')[-1].split('.')[1][:4]
        dm = file.split('/')[-1].split('.')[2][:2]
        dd = file.split('/')[-1].split('.')[2][2:]
        d = str(dy)+'-' + str(dm)+'-' + str(dd)
        sample = {
            'sentinel': sentinel,
            'path': file,
            'date': d
        }
        return sample
