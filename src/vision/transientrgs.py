import warnings
warnings.filterwarnings("ignore")

import torch
import os

from torchvision import transforms as trn
from PIL import Image

from src.vision.setting import SETTING, MODEL_PATH

RESOURCES = SETTING['transientrgs']


class TransientRegressInstance:
    def __init__(self, backbone='efficientnet_b0', use_gpu=True):
        weight_url = RESOURCES['weights'][backbone]
        model_file = os.path.join(MODEL_PATH, weight_url.split('/')[-1])
        if not os.access(model_file, os.W_OK):
            raise RuntimeError('Missing model file. Please download '+weight_url)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.model = torch.load(model_file, map_location=self.device)
        self.model.eval()

        self.threshold = RESOURCES['threshold']

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
            trn.Resize((224, 224)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.warmup()
        print('Loaded model {} to device {}'.format(type(self.model), self.device))


    def predict(self, image_path):
        img_pil = Image.open(image_path).convert('RGB')
        input_img = self.centre_crop(img_pil).unsqueeze(0)
        input_img = input_img.to(self.device)

        predictions = []
        with torch.no_grad():
            logit = self.model.forward(input_img).cpu().detach().numpy().flatten()

        for i in range(len(logit)):
            if logit[i] > self.threshold:
                predictions.append({
                    'class': self.classes[i],
                    'score': float(logit[i])
                })

        predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
        return predictions


    def warmup(self):
        img_pil = Image.new('RGB', (224, 224))
        input_img = self.centre_crop(img_pil).unsqueeze(0)
        input_img = input_img.to(self.device)
        with torch.no_grad():
            self.model.forward(input_img).cpu().detach().numpy().flatten()


class TransientRegressService:
    __instance = None

    @staticmethod
    def get_instance():
        if TransientRegressService.__instance is None:
            TransientRegressService.__instance = TransientRegressInstance()
        return TransientRegressService.__instance