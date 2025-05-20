import numpy as np
import torch
import pandas as pd
import torchvision
from torch import nn, optim
from torch.optim import Adam
import os
from matplotlib import pyplot as plt
import rasterio as rio
import torch.nn.functional as F
import seaborn as sns
from sklearn.metrics import jaccard_score
import pickle as pkl
seed= np.random.randint(0,10000)
torch.manual_seed(seed)
import torchmetrics
import segmentation_models_pytorch as smp
from torch.utils.data import Subset
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torchmetrics import JaccardIndex
torch.set_printoptions(sci_mode=False)
from sklearn.metrics import confusion_matrix
import seaborn as sns

def train_val_iou(loss_train, loss_val, epochs, save = True, fig_name=''):
    epoch = range(epochs)
    fig, ax = plt.subplots(1,1, figsize = (10,6))   
    ax.plot(epoch, loss_train, color='b', linewidth=2, label='Training')
    ax.plot(epoch, loss_val, color='r', linewidth=2, label='Validation')

    ax.set_xlabel('Iters')
    ax.set_ylabel('Iou')
    ax.set_title('Training and Validation Intersection-over-Union')
    ax.legend()
    plt.show()
    if save==True:
        fig.savefig(fig_name+'.png', transparent=False, facecolor='white', bbox_inches='tight')
        
def train_val_loss(loss_train, loss_val, epochs, save = True, fig_name=''):
    epoch = range(epochs)
    fig, ax = plt.subplots(1,1, figsize = (10,6))   
    ax.plot(epoch, loss_train, color='b', linewidth=2, label='Training')
    ax.plot(epoch, loss_val, color='r', linewidth=2, label='Validation')

    ax.set_xlabel('Iters')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    plt.show()
    if save==True:
        fig.savefig(fig_name +'.png', transparent=False, facecolor='white', bbox_inches='tight')
        
        
def train_val_acc(loss_train, loss_val, epochs, save = True, fig_name=''):
    epoch = range(epochs)
    fig, ax = plt.subplots(1,1, figsize = (10,6))   
    ax.plot(epoch, loss_train, color='b', linewidth=2, label='Training')
    ax.plot(epoch, loss_val, color='r', linewidth=2, label='Validation')

    ax.set_xlabel('Iters')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    plt.show()
    if save==True:
        fig.savefig(fig_name + '.png', transparent=False, facecolor='white', bbox_inches='tight')
        
def plot_roc_curve(fpr, tpr, auc, save=False):
    fig, ax = plt.subplots(1,1, figsize = (6,6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    plt.show()
    if save ==True:
        fig.savefig(fig_name + '.png', transparent=False, facecolor = 'white')
        
        
def bloom_image_plotting(batch, output_binary,y,c):

    num_images = len(batch['planet'])  # Number of images in the batch
    rows = num_images  # One row per image pair (planet and cicyano)
    cols = 2  # Two columns: one for the planet image, one for the cicyano label

    # Colors and normalization for the labels
    colors = ['black', 'seagreen', 'yellow', 'grey']
    cmap = ListedColormap(colors)
    boundaries = [0, 1, 2, 3, 4]
    norm = BoundaryNorm(boundaries, cmap.N)

    # Create the figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=(8, rows * 3))  # Smaller horizontal size
    axes = axes.reshape(rows, cols)  # Ensure consistent shape for indexing

    # Loop through the images and plot
    for i in range(num_images):
        # Plot the planet image
        axes[i, 0].imshow(batch['planet'][i])
        axes[i, 0].set_title(f'Output: {output_binary[i].item()}')
        axes[i, 0].axis('off')

        # Plot the cicyano label
        axes[i, 1].imshow(batch['cicyano'][i], cmap=cmap, norm=norm)
        axes[i, 1].set_title(f'Label: {y[i]}, FPT: {c[i]}')
        axes[i, 1].axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.3)  # Adjust spacing as needed
    plt.show()
    plt.close()