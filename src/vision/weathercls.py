# RFS Dataset: https://drive.google.com/file/d/1oyPhrwnG1rS9TRlcmDPIA4mTsfcB7sJu/view?usp=sharing

import torch
import torch.nn as nn
import os
import torchvision.models as models

from torch.autograd import Variable as V
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image

from src.vision.setting import SETTING, MODEL_PATH

RESOURCES = SETTING['weathercls']


class WeatherClassifyInstance:
    def __init__(self, backbone='resnet18', use_gpu=False):
        weight_url = RESOURCES['weights'][backbone]
        model_file = os.path.join(MODEL_PATH, weight_url.split('/')[-1])
        if not os.access(model_file, os.W_OK):
            raise RuntimeError('Missing model file. Please download '+weight_url)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.model = torch.load(model_file, map_location=self.device)
        
        # self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.eval()

        # load the class label
        label_url = RESOURCES['class_labels']
        label_file = os.path.join(MODEL_PATH, label_url.split('/')[-1])
        if not os.access(label_file, os.W_OK):
            raise RuntimeError('Missing class labels file. Please download '+label_url)
        classes = list()
        with open(label_file) as class_file:
            for line in class_file:
                classes.append(line.strip())
        self.classes = tuple(classes)

        self.centre_crop = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def predict(self, image_path):
        img_pil = Image.open(image_path).convert('RGB')
        input_img = V(self.centre_crop(img_pil).unsqueeze(0), volatile=True)
        input_img = input_img.to(self.device)

        # outputs = self.model(inputs)
        # _, preds = torch.max(outputs, 1)
        logit = self.model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        # output the prediction
        prediction = {
            'class': self.classes[idx[0]],
            'probability':probs[0].cpu().detach().numpy().item()
        }
        
        return prediction


class WeatherClassifyService:
    __instance = None

    @staticmethod
    def get_instance():
        if WeatherClassifyService.__instance is None:
            WeatherClassifyService.__instance = WeatherClassifyInstance()
        return WeatherClassifyService.__instance