import torch
import torch.nn as nn
import os
import math
import datetime
import torchvision.models as models

from torch.autograd import Variable as V
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image

from src.vision.setting import SETTING, MODEL_PATH

RESOURCES = SETTING['timergs']


class TimeRegressInstance:
    def __init__(self, backbone='efficientnet_b0', use_gpu=True):
        weight_url = RESOURCES['weights'][backbone]
        model_file = os.path.join(MODEL_PATH, weight_url.split('/')[-1])
        if not os.access(model_file, os.W_OK):
            raise RuntimeError('Missing model file. Please download '+weight_url)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.model = torch.load(model_file, map_location=self.device)
        self.model.eval()

        self.centre_crop = trn.Compose([
            trn.Resize((224, 224)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def predict(self, image_path):
        img_pil = Image.open(image_path).convert('RGB')
        input_img = V(self.centre_crop(img_pil).unsqueeze(0), volatile=True)
        input_img = input_img.to(self.device)

        norm_time = self.model.forward(input_img).cpu().detach().numpy().flatten()[0]

        prediction = {
            'norm_time': norm_time,
            'time': self.format_time(norm_time)
        }

        return prediction
    

    def format_time(self, t):
        hour, minute = math.modf(t*24)
        minute, second = math.modf(minute*60)
        second = int(second*60)
        return datetime.time(hour, minute, second).isoformat()


class TimeRegressService:
    __instance = None

    @staticmethod
    def get_instance():
        if TimeRegressService.__instance is None:
            TimeRegressService.__instance = TimeRegressInstance()
        return TimeRegressService.__instance