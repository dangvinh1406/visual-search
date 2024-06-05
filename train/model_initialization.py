#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division

import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torchvision
from torchvision import models


def set_parameter_requires_grad(model, layers_to_update):
    for name, param in model.named_parameters():
        param.requires_grad = False
        if layers_to_update is None:
            param.requires_grad = True
            continue
        for layer_name in layers_to_update:
            if layer_name in name:
                param.requires_grad = True


def initialize_model(model_name, num_classes, layers_to_update, device, use_pretrained=True, output_layer=nn.Linear):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, layers_to_update)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = output_layer(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'efficientnet_b0':
        model_ft = models.efficientnet_b0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, layers_to_update)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = output_layer(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, layers_to_update)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = output_layer(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg11_bn":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, layers_to_update)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = output_layer(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    model_ft = model_ft.to(device)
    print("Params to learn:")
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print('\t', name)
    return model_ft, input_size, params_to_update

