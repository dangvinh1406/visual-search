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


def train_model(model, device, dataloaders, criterion, optimizer, num_epochs=25, threshold=0.5):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    preds = outputs > threshold
                    trues = labels > threshold

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == trues)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, 'best_model.tar')
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def test_model(model, device, test_loader, threshold=0.5):
    since = time.time()
    model.eval()   # Set model to evaluate mode

    running_corrects = 0
    # Iterate over data.
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

            # zero the parameter gradients
            # optimizer.zero_grad()

            # forward
            # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            preds = outputs > threshold
            trues = labels > threshold
            running_corrects += torch.sum(preds == trues)

    # statistics
    acc = running_corrects.double()/len(test_loader.dataset)
    print('Acc: {:.4f}'.format(acc))


class ImageTransientDataset(Dataset):
    def __init__(self, df, transform=None):
        self.img_labels = df
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        image = Image.open(img_path)
        label = torch.tensor([float(cell.split(',')[0]) for cell in self.img_labels.iloc[idx, 1:]])
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

    data_dir = 'G:\\My Drive\\transient\\'
    
    model_name = 'efficientnet_b0'
    num_classes = 40
    input_size = 224

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
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

    df = pd.read_csv(os.path.join(data_dir, 'annotations', 'annotations.tsv'), sep='\t', header=None, index_col=None)
    df[0] = df[0].apply(lambda x: os.path.join(data_dir, 'imageLD', x))
    # print(df.head())
    df_train, df_test = train_test_split(df, test_size=0.4, random_state=42)
    df_test, df_val = train_test_split(df_test, test_size=0.25, random_state=42)

    dfs = {
        'train': df_train,
        'val': df_val,
        'test': df_test
    }
    print(len(df_train), len(df_val), len(df_test))

    if args['train']:
        batch_size = 16
        num_epochs = 30
        layers_to_update = ['features.7', 'features.8']

        # Initialize the model for this run
        model_ft, input_size_, params_to_update = initialize_model(model_name, num_classes, layers_to_update, device, use_pretrained=True)
        # print(model_ft)
        assert input_size_ == input_size

        optimizer_ft = optim.Adam(params_to_update)# SGD(params_to_update, lr=0.001, momentum=0.9)
        criterion = nn.MSELoss()

        # Create training and validation datasets
        image_datasets = {x: ImageTransientDataset(dfs[x], data_transforms[x]) for x in ['train', 'val']}
        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

        # Train and evaluate
        model_ft, hist = train_model(model_ft, device, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

        torch.save(model_ft, os.path.join(data_dir, model_name+'_transient.pth.tar'))

    if args['test']:
        model_ft = torch.load(os.path.join(data_dir, model_name+'_transient.pth.tar'), map_location=device)

        test_dataset = ImageTransientDataset(dfs['test'], data_transforms['val'])

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
        test_model(model_ft, device, test_loader)
