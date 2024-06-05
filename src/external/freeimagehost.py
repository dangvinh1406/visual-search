import requests

from src.utils import CREDENTIAL


class ImageHostInstance:
    DEV_STUB = False

    def __init__(self, endpoint=None, key=None):
        if not key:
            info = CREDENTIAL
            endpoint = info['freeimagehost']['endpoint']
            key = info['freeimagehost']['key']
        self.key = key
        self.endpoint = endpoint

    
    def upload(self, image_path):
        if self.DEV_STUB:
            return 'stub/uploaded/image' 
        result = {}
        with open(image_path, 'rb') as image:
            file = {'source': image}
            data = {'key': self.key}
            try:
                response = requests.post(
                    self.endpoint, data=data, files=file)
                response.raise_for_status()
                result = response.json()
            except Exception as ex:
                raise ex
        return self.format(result)

    
    def format(self, data):
        return data['image']['url']


class ImageHostService:
    __instance = None

    @staticmethod
    def get_instance():
        if ImageHostService.__instance is None:
            ImageHostService.__instance = ImageHostInstance()
        return ImageHostService.__instance