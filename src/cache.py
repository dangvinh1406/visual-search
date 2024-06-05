import os
import hashlib
import glob
import json

from src.utils import DATASET_PATH

BUF_SIZE = 65536  # lets read stuff in 64kb chunks!


class CacheInstance:

    USE_CACHE = True

    def __init__(self, path=os.path.join(DATASET_PATH, 'images', '*.*')):
        img_files = glob.glob(path)
        json_files = {
            im_path: im_path.replace('images', 'jsons').replace(im_path.split('.')[-1], 'json')
            for im_path in img_files
        }
        self.cache = {
            self.get_md5(im_path): json.load(open(json_files[im_path]))
                for im_path in img_files
                if os.path.isfile(json_files[im_path])
        }
        print('Cache contains {} files'.format(len(self.cache)))

    def get_md5(self, image_path):
        md5 = hashlib.md5()
        with open(image_path, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                md5.update(data)
        return md5.hexdigest()
    
    def lookup(self, image_path):
        if not self.USE_CACHE:
            return None
        md5_str = self.get_md5(image_path)
        if md5_str in self.cache:
            return self.cache[md5_str]
        return None
    

class CacheService:
    __instance = None

    @staticmethod
    def get_instance():
        if CacheService.__instance is None:
            CacheService.__instance = CacheInstance()
        return CacheService.__instance