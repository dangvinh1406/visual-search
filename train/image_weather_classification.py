# Commented out IPython magic to ensure Python compatibility.
from __future__ import print_function
from __future__ import division

import warnings
warnings.filterwarnings('ignore')
import time
import os
import glob
import copy
import json
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms

from model_initialization import set_parameter_requires_grad, initialize_model


def train_model(model, device, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
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
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

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


def test_model(model, device, test_loader):
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

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    # statistics
    acc = running_corrects.double()/len(test_loader.dataset)
    print('Acc: {:.4f}'.format(acc))


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = vars(parser.parse_args())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data_dir = 'G:\\My Drive\\RFS-dataset\\'
    
    model_name = 'efficientnet_b0'
    num_classes = 5
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

    # For the first time process RFS dataset
    RFS_process = False

    if RFS_process:
        ori_dir = os.path.join(data_dir, 'Dataset')
        ratios = [70, 10, 20]
        assert sum(ratios) == 100
        ratios = [r/100 for r in ratios]
        ratios[1] += ratios[0]
        ratios[2] += ratios[1]
        print(ratios)

        classes = glob.glob(os.path.join(ori_dir, '*'))
        for cls in classes:
            imgs = glob.glob(os.path.join(cls, '*.*'))
            n = len(imgs)
            data = {
                'train': imgs[:int(n*ratios[0])], 
                'val': imgs[int(n*ratios[0]):int(n*ratios[1])],
                'test': imgs[int(n*ratios[1]):]
            }
            for subset in ['train', 'val', 'test']:
                os.makedirs(cls.replace('Dataset', subset), exist_ok=True)
                for im in data[subset]:
                    try:
                        os.rename(im, im.replace('Dataset', subset))
                    except Exception as e:
                        print(e)

    for subset in ['train', 'val', 'test']:
        print(
            subset,
            len(glob.glob(os.path.join(data_dir, subset, '*', '*'))),
        )

    if args['train']:
        batch_size = 16
        num_epochs = 30
        layers_to_update = ['features.7', 'features.8']

        # Initialize the model for this run
        model_ft, input_size_, params_to_update = initialize_model(model_name, num_classes, layers_to_update, device, use_pretrained=True)
        # print(model_ft)
        assert input_size_ == input_size

        optimizer_ft = optim.Adam(params_to_update)# SGD(params_to_update, lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

        # Train and evaluate
        model_ft, hist = train_model(model_ft, device, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

        torch.save(model_ft, os.path.join(data_dir, model_name+'_rfs.pth.tar'))

    if args['test']:
        model_ft = torch.load(os.path.join(data_dir, model_name+'_rfs.pth.tar'), map_location=device)

        test_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'test'), 
            data_transforms['val']
        )

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
        test_model(model_ft, device, test_loader)