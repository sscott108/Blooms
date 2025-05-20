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

def mIOU(y, pred, num_classes=4):
    num_classes = 4
    pred = F.softmax(pred, dim=1)               # scale elements between [0,1] and sum to 1.    
    pred = torch.argmax(pred, dim=1).squeeze(1)   # index with highest probability 
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    y = y.view(-1)

    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (y == sem_class)

        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else: 
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)                   #iou's for each class calculated over the whole batch
    return np.mean(present_iou_list), iou_list

def pixel_accuracy(output, label):
    output = torch.argmax(output, dim=1)
    correct = torch.eq(output, label).float()
    accuracy = torch.mean(correct)
    return accuracy


def train_val_iou(loss_train, loss_val, epochs, save = True, fig_name=''):
    epoch = range(epochs)
    fig, ax = plt.subplots(1,1, figsize = (10,6))   
    ax.plot(epoch, loss_train, color='b', linewidth=0.5, label='Training')
    ax.plot(epoch, loss_val, color='r', linewidth=0.5, label='Validation')

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
        fig.savefig(fig_name+'.png', transparent=False, facecolor='white', bbox_inches='tight')
        
        
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
        fig.savefig(fig_name+'.png', transparent=False, facecolor='white', bbox_inches='tight')
        
        
def plot_roc_curve(fpr, tpr, auc, save=False):
    fig, ax = plt.subplots(1,1)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    plt.show()
    if save == True:
        fig.savefig('ROC.png', transparent=False, facecolor = 'white')
    
    
    