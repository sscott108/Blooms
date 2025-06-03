import numpy as np
import torch
import pandas as pd
import torchvision
import os
import json
import requests
from requests.auth import HTTPBasicAuth
from IPython.display import display, HTML
from pprint import pprint
from pyproj import Transformer
import time
import sys
import getpass
import glob
from collections import defaultdict
from rasterio.merge import merge

from matplotlib import pyplot as plt
import rasterio
from torch import nn
import torch.nn.functional as F
seed= np.random.randint(0,10000)
torch.manual_seed(seed)
from tqdm import tqdm
torch.set_printoptions(sci_mode=False)
import matplotlib.dates as mdates
from os import listdir
from torch.utils.data import DataLoader, Dataset
from shapely.geometry import Polygon, mapping, shape
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings('ignore')
from PIL import Image
import datetime
from datetime import datetime, timedelta, date
from torchvision.models import resnet50
from torchvision import transforms

np.random.seed(420)
torch.manual_seed(420)
torch.cuda.manual_seed(420)  # If using CUDA

from rasterio.windows import Window
from sklearn.metrics import accuracy_score
torch.set_printoptions(sci_mode=False)

def plot_large_image(image_dir):
    dategroups = defaultdict(list)
    tif_files = glob.glob(os.path.join(image_dir, '*.tif'))
    for tif_file in tif_files:
        filename = os.path.basename(tif_file)
        date = filename.split('_')[0]
    #     print(date)
        dategroups[date].append(tif_file)
    
    for date, files in dategroups.items():
        src_files_to_mosaic = []
        for tif_file in files:
            src = rasterio.open(tif_file)
            src_files_to_mosaic.append(src)
            print(src_files_to_mosaic)

        mosaic, out_transform = merge(src_files_to_mosaic)

        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform
        })
        output_path = f'mosaic_{date}.png'
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        for src in src_files_to_mosaic:
            src.close()
            
    fig,ax = plt.subplots(1,1, figsize = (20,7))
    ax.imshow(mosaic.T)
    ax.axis('off')
    plt.show()


def crop_and_download(x, y, image_path, sites_total):
    for fi in tqdm(sorted(os.listdir(image_path))):
        if fi.endswith(".tif"):
            working_path = os.path.join(image_path, fi)
            with rasterio.open(working_path) as rds:
                transformer = Transformer.from_crs("EPSG:4326", rds.crs, always_xy=True)

                for site_name, lat, lon in sites_total:
                    xx, yy = transformer.transform(lon, lat)
                    row, col = rds.index(xx, yy)

                    if row > x and col > y:
                        try:
                            window = Window.from_slices(rows=(row - x, row + x), cols=(col - y, col + y))
                            data = rds.read(window=window)
                            data = np.moveaxis(data, 0, 2) 

                            if data.shape[0] != 2 * x or data.shape[1] != 2 * y: continue

                            black_space = np.mean(data / 255)

                            if black_space < 0.25 or black_space >= 0.9: continue

    #                         plt.imshow(data)
    #                         plt.axis('off')
    #                         plt.show()
                            out_dir = os.path.join(image_path, site_name)
                            os.makedirs(out_dir, exist_ok=True)

                            out_path = os.path.join(out_dir, fi)
                            im = Image.fromarray(data.astype(np.uint8))
                            im.save(out_path)

                        except Exception as e:
                            print(f"{site_name} in {fi} failed: {e}")
                            continue

                            
                            
def imagelinks_delayed(image_ids_set, item_type, API_KEY, max_retries=10):
    for image_id in list(image_ids_set)[:]:
        id_url = f'https://api.planet.com/data/v1/item-types/{item_type}/items/{image_id}/assets'
        try:
            result = requests.get(id_url, auth=HTTPBasicAuth(API_KEY, ''))
            links = result.json()["ortho_visual"]["_links"]
            self_link = links["_self"]
            activation_link = links["activate"]

            # Attempt activation
            requests.get(activation_link, auth=HTTPBasicAuth(API_KEY, ''))

            for attempt in range(max_retries):
                activation_status_result = requests.get(self_link, auth=HTTPBasicAuth(API_KEY, ''))
                status = activation_status_result.json().get("status", "")
                print(f'status: {status} for {image_id} (attempt {attempt + 1})')

                if status != 'activating':
                    break

                print(f"Waiting 2 minutes to retry for image_id: {image_id}")
                time.sleep(120)

        except Exception as e:
            print(f"Error processing image_id {image_id}: {e}")
            continue
                            


def image_download(folder_name, download_links):
    if not (os.path.isdir(folder_name)):
        print('Figure directory didn''t exist, creating now.')
        os.mkdir(folder_name)
    else:
        print('Figure directory exists.') 
        
    for image_name, download_link in download_links.items():
        try:
            # Use allow_redirects=False to prevent automatic redirects
            response = requests.get(download_link, stream=True, allow_redirects=True)

            if response.status_code == 302:  # 302 indicates a redirect
                # Get the final URL from the 'Location' header
                final_url = response.headers.get('Location')

                if final_url:
                    response = requests.get(final_url, stream=True)
                else:
                    print(f"Failed to retrieve {image_name}. Redirect URL not found.")
                    continue

            if response.status_code == 200:
                local_file_path = f'./{folder_name}/{image_name}.tif'

                with open(local_file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"Downloaded {image_name} as {local_file_path}")
            else:
                print(f"Failed to retrieve {image_name}. Status code: {response.status_code}")
        except Exception as e:
            print(f"An error occurred while downloading {image_name}: {e}")
            


class DailyMonitoringDataset(Dataset):
    def __init__(self, root_dir):
        self.current_day = []
        for root, dirs, files in os.walk(root_dir):
            if root == root_dir:  # skip the top-level folder
                continue
            for file in files:
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(root)

                self.current_day.append(file_path)
    
    def __len__(self):
        return len(self.current_day)
    
    def __getitem__(self, idx):
        file_path = self.current_day[idx]
        
        with rasterio.open(file_path) as src:
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
                 'file_path': file_path}
        return sample
    
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


def bloom_model_implementation(download_path, model_path):
    dataset = DailyMonitoringDataset(root_dir=download_path)
    date = DataLoader(dataset, batch_size=1, shuffle = False)
    model = CNN()
    model.load_state_dict(torch.load(model_path, 
    map_location=torch.device('cpu')))
    
    model.eval()
    tracking = {'date': [],'loc':[], 'pred': []}
    with torch.no_grad():
        for i, batch in tqdm(enumerate(date)):
            x = batch['image'].float()
            p = batch['file_path'] 

            output = model(x)
    #         logits = (output.cpu().detach().numpy()).astype(int)
            output_binary = (output >= 0)

            dates = [datetime.strptime(fp.split('/')[-1][:8], "%Y%m%d").strftime("%m-%d-%Y") for fp in p]
            locs = [fp.split('/')[-2] for fp in p]
            tracking['date'].extend(dates) 
            tracking['loc'].extend(locs)
            tracking['pred'].extend(output_binary.reshape(-1).tolist())  
            
    return tracking