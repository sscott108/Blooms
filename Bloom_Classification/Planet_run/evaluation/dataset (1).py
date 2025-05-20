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
import datetime

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

class NCImageDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []  # List to store tuples of (file_path, class_label)
        
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file == ".DS_Store":
                    continue
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(root)
                if folder_name == "negative":
                    class_label = 0
                elif folder_name == "positive":
                    class_label = 1
                else:
                    continue  
                
                self.samples.append((file_path, class_label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, class_label = self.samples[idx]
        
        with rio.open(file_path) as src:
            image = src.read([1,2,3])  
            image = torch.tensor(image)  
        
        transform  = transforms.Compose([
                            transforms.ToPILImage(),
#                             transforms.RandomHorizontalFlip(),
#                             transforms.RandomRotation(degrees=30),
#                             transforms.CenterCrop(size=224),
                            transforms.ToTensor()
        ])
        sample = {'image': transform(image),
                 'class_lbl': class_label,
                 'file_path': file_path}
        return sample


def train(model, dataloader, criterion, opt, epoch, device):
    running_acc = 0
    running_loss = 0
    model.train()

    for i, batch in tqdm(enumerate(dataloader)):
        x = batch['image'].float().to(device)
        y = batch['class_lbl'].float().to(device)
        output = model(x)
        output_binary = np.zeros(output.shape)
        output_binary[output.cpu().detach().numpy() >= 0] = 1 #if logit is greater than 0, classify as bloom

        #loss
        loss = criterion(output.squeeze(dim=1),y) #BCEwL
        running_loss += loss.item()

        #accuracy
        label = y.cpu().detach().numpy()
        acc = accuracy_score(label,output_binary)
        running_acc += acc
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.empty_cache()
        
        
    return running_loss/len(dataloader), running_acc/len(dataloader)

def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    running_acc = 0

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            x = batch['image'].float().to(device)
            y = batch['class_lbl'].float().to(device)
            output = model(x)
            output_binary = np.zeros(output.shape)
            output_binary[output.cpu().detach().numpy() >= 0] = 1 #if logit is greater than 0, classify as bloom
#             print(output_binary)

            #loss
            loss = criterion(output.squeeze(dim=1),y) #BCEwL
            running_loss += loss.item()

            #accuracy
            label = y.cpu().detach().numpy()
            acc = accuracy_score(label,output_binary)
            running_acc += acc

            torch.cuda.empty_cache()
        
        return running_loss/len(dataloader), running_acc/len(dataloader)
            
def finding_sample_weights(dataset, weights_name, save=False):
    all_labels = [dataset[i]['class_lbl'] for i in tqdm(range(len(dataset)), position = 0, leave =True)]
    class_counts = np.bincount(all_labels)
    class_weights = 1.0 / class_counts 
    sample_weights = [class_weights[label] for label in all_labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    if save:
        torch.save((class_counts, class_weights, sample_weights), weights_name +'.pth')
    return class_counts, class_weights, sample_weights


class PseudoLabeledDataset(Dataset):
    def __init__(self, pseudo_labels):
        
        self.pseudo_labels = list()
        with open(pseudo_labels, 'rb') as fp:
            add_ons = pkl.load(fp)
            self.pseudo_labels, _ = add_ons
    
    def __len__(self):
        
        return len(self.pseudo_labels)
    
    def __getitem__(self, idx):
    
        file_path = self.pseudo_labels[idx]
        
        # Read image using rasterio
        with rio.open(file_path) as src:
            image = src.read([1,2,3])  # Reads the image as a NumPy array (C, H, W)
            image = torch.tensor(image)  # Convert to PyTorch tensor
        
            label =file_path.split('/')[-2]
            if label == "negative": class_label = 0
            elif label == "positive": class_label = 1
        
        transform  = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(degrees=30),
#                             transforms.CenterCrop(size=224),
                            transforms.ToTensor()
        ])
        
           
        sample = {'image': transform(image),
                 'class_lbl': class_label,
                 'file_path': file_path}
        return sample