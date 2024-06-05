#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division

import warnings
warnings.filterwarnings('ignore')
import time
import os
import re
import glob
import copy
import json
import datetime
import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from model_initialization import set_parameter_requires_grad, initialize_model


SCALE_UP = 1000


class TimeRegressModule(nn.Module):
    def __init__(self, num_ftrs, num_classes):
        super(TimeRegressModule, self).__init__()
        self.linear = nn.Linear(num_ftrs, num_classes)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.linear(x))


class TimeClassifyModule(nn.Module):
    def __init__(self, num_ftrs, num_classes):
        super(TimeClassifyModule, self).__init__()
        self.linear = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.linear(x)


def time_loss_func(output, target, norm_time=True):
    if not norm_time:
        output = output/24.0
        target = target/24.0

    # Circular time loss for regression (this is failed idea)
    lower = torch.minimum(output, target)
    upper = torch.maximum(output, target)
    loss = torch.minimum(torch.abs(output-target), lower+(1-upper))
     
    # L1 Loss
    # loss = torch.abs(output-target)

    # Penalties for wrong samples
    # mask = loss > (4.0/24) # 4 hours
    # loss[mask] = 1.0

    # reduce mean
    loss = torch.mean(loss)

    # scale up loss to be easier to converge
    # loss *= SCALE_UP
    return loss


def train_model(model, device, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()*inputs.size(0)

            epoch_loss = running_loss/len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, 'best_model.tar')

            if phase == 'val':
                val_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time()-since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_loss_history


def test_model(model, device, test_loader, criterion):
    since = time.time()
    model.eval()   # Set model to evaluate mode

    running_loss = 0
    running_time_loss = 0
    running_preds = []
    # Iterate over data.
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            running_loss += criterion(outputs, labels.data)*inputs.size(0)
            running_time_loss += time_loss_func(outputs, labels.data)*inputs.size(0)
            running_preds += outputs.cpu().detach().numpy().tolist()[0]

    # statistics
    loss = running_loss.double()/len(test_loader.dataset)
    time_loss = running_time_loss.double()/len(test_loader.dataset)
    print('Loss: {:.4f}'.format(loss))
    print('Time loss: {:.4f}, equivalent {:.2f} hour(s)'.format(time_loss, time_loss*24))
    return running_preds


class ImageToTimeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.img_labels = df
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = vars(parser.parse_args())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data_dir = 'G:\\My Drive\\MIRFLICKR\\'

    df = pd.read_csv(os.path.join(data_dir, 'label_train.tsv'), sep='\t')
    print(df.head())
    df_train, df_test = train_test_split(df, test_size=0.4, random_state=42)
    df_test, df_val = train_test_split(df_test, test_size=0.25, random_state=42)

    label_column = 'norm_time'
    dfs = {
        'train': df_train[['img', label_column]],
        'val': df_val[['img', label_column]],
        'test': df_test[['img', label_column]]
    }
    print(len(df_train), len(df_val), len(df_test))

    # criterion = time_loss_func  # time_loss is circular loss, as 23h59 is close to 0h, instead of 0.99 and 0
    criterion = nn.L1Loss()

    if args['train']:
        model_name = 'efficientnet_b0'
        batch_size = 8
        num_epochs = 15
        layers_to_update = None # update all layers

        # Initialize the model for this run
        model_ft, input_size, params_to_update = initialize_model(
            model_name, 1, layers_to_update, device, use_pretrained=True, output_layer=TimeRegressModule)
        print(model_ft)

        optimizer_ft = optim.SGD(params_to_update, lr=0.01)# SGD(params_to_update, lr=0.001, momentum=0.9)

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        # Create training and validation datasets
        image_datasets = {x: ImageToTimeDataset(dfs[x], data_transforms[x]) for x in ['train', 'val']}
        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=1) for x in ['train', 'val']}

        # Train and evaluate
        model_ft, hist = train_model(model_ft, device, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

        torch.save(model_ft, os.path.join(data_dir, 'efficientnetb0_mirflickr25k.pth.tar'))

    if args['test']:
        input_size = 224

        model_ft = torch.load(os.path.join(data_dir, 'efficientnetb0_mirflickr25k.pth.tar'), map_location=device)

        test_dataset = ImageToTimeDataset(
            dfs['test'], 
            transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
        df_test['pred'] = test_model(model_ft, device, test_loader, criterion)
        df_test.to_csv('image_to_time_test_result.csv')
