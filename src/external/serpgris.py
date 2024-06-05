import os
import json
from serpapi import GoogleSearch

from src.external.freeimagehost import ImageHostService
from src.utils import CREDENTIAL


class ReverseImageSearchInstance:
    DEV_STUB = False

    def __init__(self, key=None):
        if not key:
            info = CREDENTIAL
            key = info['serpapi']['key']
        self.key = key
        self.engine = 'google_reverse_image'
        self.pagination = 10

    
    def search(self, image_path):
        if self.DEV_STUB:
            stub_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', '..', 'data', os.path.basename(image_path)+'.stub')
            if os.path.exists(stub_file):
                print('Load stub data', stub_file)
                return self.format(json.load(open(stub_file)))

        uploaded_im_url = ImageHostService.get_instance().upload(image_path)
        start = 0
        pages = []
        while True:
            params = {
                'engine': self.engine,
                'image_url': uploaded_im_url,
                'api_key': self.key,
                'start': start
            }
            print('Retrieving search result from offset', start)
            search = GoogleSearch(params)
            result = search.get_dict()
            print('Done')
            if 'image_results' not in result or len(result['image_results']) == 0:
                break
            pages.append(result)
            start += self.pagination

        if self.DEV_STUB:
            stub_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', '..', 'data', os.path.basename(image_path)+'.stub')
            print('Dump stub data', stub_file)
            with open(stub_file, 'w') as stub:
                stub.write(json.dumps(pages, indent=4))
        
        return self.format(pages)


    def format(self, data):
        # remove first 2 records as these are relative search (not pages include matching images), 
        # follow the structure of Google's result which is crawled by SerpAPI 
        image_results = []
        offset = 0
        for i in range(len(data)):
            page = data[i]
            for record in page['image_results']:
                record['position'] += offset
                image_results.append(record)
            offset += len(page)
        return image_results[2:]


class ReverseImageSearchService:
    __instance = None

    @staticmethod
    def get_instance():
        if ReverseImageSearchService.__instance is None:
            ReverseImageSearchService.__instance = ReverseImageSearchInstance()
        return ReverseImageSearchService.__instance