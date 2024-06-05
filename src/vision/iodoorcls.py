# https://github.com/AMANVerma28/Indoor-Outdoor-scene-classification
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import os

from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image

from src.vision.setting import SETTING, MODEL_PATH

RESOURCES = SETTING['iodoorcls']


class InOutdoorClassifyInstance:
    def __init__(self, backbone='resnet18', use_gpu=True):
        weight_url = RESOURCES['weights'][backbone]
        model_file = os.path.join(MODEL_PATH, weight_url.split('/')[-1])
        if not os.access(model_file, os.W_OK):
            raise RuntimeError('Missing model file. Please download '+weight_url)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.model = torch.load(model_file, map_location=self.device)
        
        # fix for loading old Pytorch version model
        if backbone == 'resnet18':
            self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.eval()

        # load the class label
        label_url = RESOURCES['class_labels']
        label_file = os.path.join(MODEL_PATH, label_url.split('/')[-1])
        if not os.access(label_file, os.W_OK):
            raise RuntimeError('Missing class labels file. Please download '+label_url)
        classes = list()
        iodoor_types = {
            '1': 'indoor',
            '2': 'outdoor'
        }
        self.ioclasses = {}
        with open(label_file) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
                self.ioclasses[classes[-1]] = iodoor_types[line.strip().split(' ')[1]]
        self.classes = tuple(classes)

        self.centre_crop = trn.Compose([
            trn.Resize((256, 256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.warmup()
        print('Loaded model {} to device {}'.format(type(self.model), self.device))


    def predict(self, image_path, rank=5):
        img_pil = Image.open(image_path).convert('RGB')
        input_img = self.centre_crop(img_pil).unsqueeze(0)
        input_img = input_img.to(self.device)

        # forward pass
        with torch.no_grad():
            logit = self.model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        # output the prediction
        predictions = []
        for i in range(0, rank):
            predictions.append({
                'class': self.classes[idx[i]],
                'ioclass': self.ioclasses[self.classes[idx[i]]],
                'probability':probs[i].cpu().detach().numpy().item()
            })

        return predictions


    def warmup(self):
        img_pil = Image.new('RGB', (256, 256))
        input_img = self.centre_crop(img_pil).unsqueeze(0)
        input_img = input_img.to(self.device)
        with torch.no_grad():
            self.model.forward(input_img)


class InOutdoorClassifyService:
    __instance = None

    @staticmethod
    def get_instance():
        if InOutdoorClassifyService.__instance is None:
            InOutdoorClassifyService.__instance = InOutdoorClassifyInstance()
        return InOutdoorClassifyService.__instance
