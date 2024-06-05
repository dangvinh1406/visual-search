import yaml
import json
import os
import requests

from src.utils import CREDENTIAL


class WeatherRetrieveInstance:
    def __init__(self, endpoint=None, key=None):
        if not key:
            info = CREDENTIAL
            endpoint = info['visualcrossing_weather_retrieve']['endpoint']
            key = info['visualcrossing_weather_retrieve']['key']
        self.endpoint = endpoint
        self.key = key

    
    def retrieve(self, location, start_date, end_date=None):
        params = '{location}/{start_date}'.format(
            location=location, start_date=start_date)
        if end_date:
            params += '/'+end_date
        response = requests.get(self.endpoint+params+'?key='+self.key)

        return response.json()


class WeatherRetrieveService:
    __instance = None

    @staticmethod
    def get_instance():
        if WeatherRetrieveService.__instance is None:
            WeatherRetrieveService.__instance = WeatherRetrieveInstance()
        return WeatherRetrieveService.__instance