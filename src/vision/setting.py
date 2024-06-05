import os

SETTING = {
    'iodoorcls': {
        'weights':{
            'resnet18': 'http://places2.csail.mit.edu/models_places365/whole_resnet18_places365_python36.pth.tar',
            'resnet50': 'http://places2.csail.mit.edu/models_places365/whole_resnet50_places365_python36.pth.tar',
            'densenet161': 'http://places2.csail.mit.edu/models_places365/whole_densenet161_places365_python36.pth.tar',
            'alexnet': 'http://places2.csail.mit.edu/models_places365/whole_alexnet_places365_python36.pth.tar'
        },
        'class_labels': 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    },
    'weathercls': {
        'weights':{
            'resnet18': 'resnet18_rfs_acc0.87.pth.tar'
        },
        'class_labels': 'categories_rfs.txt'
    },
    'transientrgs': {
        'weights':{
            'efficientnet_b0': 'efficientnet_b0_transient.pth.tar'
        },
        'class_labels': 'categories_transient.txt',
        'threshold': 0.5
    },
    'timergs':{
        'weights':{
            'efficientnet_b0': 'efficientnetb0_mirflickr25k.pth.tar'
        }
    },
    'geoest':{
        'folder': 'geoestimation',
        'checkpoint': 'epoch.014-val_loss.18.4833.ckpt',
        'param': 'hparams.yaml',
    },
}
    
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model'
)