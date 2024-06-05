import requests

from src.utils.url import get_domain
from src.utils import CREDENTIAL


class WebCategorizeInstance:
    def __init__(self, endpoint=None, key=None):
        if not key:
            info = CREDENTIAL
            endpoint = info['klazify_web_categorize']['endpoint']
            key = info['klazify_web_categorize']['key']
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': 'Bearer '+key,
            'cache-control': 'no-cache'
        }
        self.endpoint = endpoint


    def categorize(self, url):
        payload = '{\"url\":\"https://{}\"}\n'.format(get_domain(url))
        response = requests.request(
            'POST', self.endpoint, data=payload, headers=self.headers)
        print(response.text)


class WebCategorizeService:
    __instance = None

    @staticmethod
    def get_instance():
        if WebCategorizeService.__instance is None:
            WebCategorizeService.__instance = WebCategorizeInstance()
        return WebCategorizeService.__instance