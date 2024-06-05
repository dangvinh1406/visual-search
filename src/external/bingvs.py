import json
import requests

from src.utils import CREDENTIAL


class VisualSearchInstance:

    def __init__(self, endpoint=None, subcription_key=None):
        if not subcription_key:
            info = CREDENTIAL
            endpoint = info['bing_visual_search']['endpoint']
            subcription_key = info['bing_visual_search']['key']
        self.headers = {'Ocp-Apim-Subscription-Key': subcription_key}
        self.endpoint = endpoint


    def search(self, image_path):
        # TODO: need to resize image if too large to guarantee PagesIncluding
        result = {}
        with open(image_path, 'rb') as image:
            file = {'image': image}
            try:
                response = requests.post(self.endpoint, headers=self.headers, files=file)
                response.raise_for_status()
                result = response.json()
            except Exception as ex:
                raise ex
        return self.format(result)


    def __get_more_bing_images(self, next_offset, count):
        result = None
        try:
            response = requests.post(
                self.endpoint, headers=self.headers,
                json={
                    'knowledgeRequest' : {
                        'invokedSkills': ['SimilarImages'],
                        'offset': next_offset,
                        'count': count
                    }
                }
            )
            response.raise_for_status()
            result = response.json()       
        except Exception as ex:
            raise ex
            print(json.dumps(result, indent=4))
        return result


    def format(self, search_result):
        form_data = {}
        for tag in search_result['tags']:
            form_data[tag['displayName']] = tag

        default_actions = {}
        for action in form_data['']['actions']:
            default_actions[action['actionType']] = action
        form_data['']['actions'] = default_actions

        if 'PagesIncluding' in form_data['']['actions']:
            data = form_data['']['actions']['PagesIncluding']['data']['value']
            if 'currentOffset' in form_data['']['actions']['PagesIncluding']:
                # handle paging
                data += self.__get_more_bing_images(
                    next_offset=form_data['']['actions']['PagesIncluding']['nextOffset'],
                    count=form_data['']['actions']['PagesIncluding']['totalEstimatedMatches'])
            form_data['']['actions']['PagesIncluding']['data']['value'] = data
        else:
            raise ConnectionError('Bing Visual Search API not returns PagesIncluding')


        filter_data = form_data['']['actions']['PagesIncluding']['data']['value'],
        return filter_data


class VisualSearchService:
    __instance = None

    @staticmethod
    def get_instance():
        if VisualSearchService.__instance is None:
            VisualSearchService.__instance = VisualSearchInstance()
        return VisualSearchService.__instance