#Utils Section
from matplotlib import pyplot as plt
import random
import os
import numpy as np
import pandas as pd
from torchvision.utils import make_grid, save_image
import torch
import torchvision

def show_image(image, title="Single Image"):

    image = image.data.cpu().view(28, 28)

    figure = plt.figure()
    plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.axis('off')


def show_images(images, col=4, title="Multiple Images"):

    plt.figure(figsize=(20, 5))
    images = images.data.cpu().view(-1, 28, 28)

    for i in range(col):

        plt.subplot(1, col, i+1)
        plt.title(title)
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')


def disp_ori_recons(x, x_recons, col=2, epoch=0, is_random=True):
    
    batch_size = x_recons.shape[0]

    if is_random:
        batch_indices = [random.randint(0, batch_size-1) for _ in range(col)]
    else:
        batch_indices = [i for i in range(col)]

    if x is not None:
        
        x = x.data.cpu().view(-1, 28, 28)

        plt.figure(figsize=(10, 5))
        plt.suptitle(f'Epoch: {epoch} – Original vs Reconstructed')

        for i in range(col):
            
            plt.subplot(1, col, i+1)
            plt.imshow(x[batch_indices[i]], cmap='gray') #i+4*N
            plt.axis('off')

    x_recons = x_recons.data.cpu().view(-1, 28, 28)
    
    plt.figure(figsize=(10, 5))
    
    for i in range(col):

        plt.subplot(1, col, i+1)
        plt.imshow(x_recons[batch_indices[i]], cmap='gray')
        plt.axis('off')

######################################################################################################

def show_images_new(image, device='cpu', title="Images"):

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(make_grid(image.to(device)[:20], padding=2, normalize=True).cpu(), (1,2,0)))
    
def save_images_new(image, image_name, device='cpu'): 
    save_image(image, image_name, normalize=True)

def save_state(model, optimizer, epoch, model_name, path="./"):

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    # file_name = f'{model_name}_{epoch}.pth'
    file_name = f'{model_name}.pth'

    save_path = os.path.join(path, file_name)

    torch.save(state, save_path)

def load_state(model, optimizer, path, mode='train', device='cpu'):

    state = torch.load(path)

    model.load_state_dict(state['model_state_dict'])
    model.to(device)

    if mode == 'train':
        optimizer.load_state_dict(state['optimizer_state_dict'])
        model.train()
    else:
        model.train()
    
    epoch = state['epoch']