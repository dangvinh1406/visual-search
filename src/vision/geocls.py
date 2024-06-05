import warnings
warnings.filterwarnings("ignore")

import sys
import os
import torch
import torchvision
from PIL import Image

sys.path.append(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'GeoEstimation'))
from classification.train_base import MultiPartitioningClassifier

from src.vision.setting import SETTING, MODEL_PATH

RESOURCES = SETTING['geoest']


class CompatibleMPC(MultiPartitioningClassifier):
    def __init__(self, hparams):
        super(MultiPartitioningClassifier, self).__init__()
        self.save_hyperparameters(hparams)
        # set self.hparams = hparams doesn't work in newer version of pytorch-lightning

        self.partitionings, self.hierarchy = self.__init_partitionings()
        self.model, self.classifier = self.__build_model()

    def __init_partitionings(self):
        return self._MultiPartitioningClassifier__init_partitionings()
    
    def __build_model(self):
        return self._MultiPartitioningClassifier__build_model()


class GeolocationClassifyInstance:
    def __init__(self, use_gpu=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        root_path = os.path.join(MODEL_PATH, RESOURCES['folder'])
        self.model = CompatibleMPC.load_from_checkpoint(
            checkpoint_path=os.path.join(root_path, RESOURCES['checkpoint']),
            hparams_file=os.path.join(root_path, RESOURCES['param']),
            partitionings={ # override cell files paths
                'files': [
                    os.path.join(root_path, 's2-cells', 'cells_50_5000.csv'),
                    os.path.join(root_path, 's2-cells', 'cells_50_2000.csv'),
                    os.path.join(root_path, 's2-cells', 'cells_50_1000.csv')
                ],
                'ptype': 'default',
                'shortnames': ['coarse', 'middle', 'fine']
            }
        )
        self.model = self.model.eval().to(self.device)

        self.tfm = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        self.warmup()
        print('Loaded model {} to device {}'.format(type(self.model), self.device))


    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = torchvision.transforms.Resize(256)(image)
        crops = torchvision.transforms.FiveCrop(224)(image)
        crops_transformed = []
        for crop in crops:
            crops_transformed.append(self.tfm(crop))
        input_tensor = torch.stack(crops_transformed, dim=0).unsqueeze(0).to(self.device)
        meta = {'img_id': [''], 'img_path': ['']} # batch = 1
        _, pred_classes, pred_latitudes, pred_longitudes = self.model.inference(
            (input_tensor, meta)
        )
        output = {}
        for p_key in pred_classes.keys():
            output[p_key] = {
                "pred_class": pred_classes[p_key].cpu().detach().numpy().item(),
                "pred_lat": pred_latitudes[p_key].cpu().detach().numpy().item(),
                "pred_long": pred_longitudes[p_key].cpu().detach().numpy().item(),
            }
        return output
    

    def warmup(self):
        image = Image.new('RGB', (256, 256))
        crops = torchvision.transforms.FiveCrop(224)(image)
        crops_transformed = []
        for crop in crops:
            crops_transformed.append(self.tfm(crop))
        input_tensor = torch.stack(crops_transformed, dim=0).unsqueeze(0).to(self.device)
        meta = {'img_id': [''], 'img_path': ['']}
        self.model.inference((input_tensor, meta))


class GeolocationClassifyService:
    __instance = None

    @staticmethod
    def get_instance():
        if GeolocationClassifyService.__instance is None:
            GeolocationClassifyService.__instance = GeolocationClassifyInstance()
        return GeolocationClassifyService.__instance