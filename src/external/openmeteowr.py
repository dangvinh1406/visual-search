import requests
from datetime import datetime, timedelta

from src.utils import CREDENTIAL


class WeatherRetrieveInstance:
    DEV_STUB = False

    def __init__(self, endpoint=None):
        if not endpoint:
            info = CREDENTIAL
            endpoint = info['openmeteo_weather_retrieve']['endpoint']
        self.endpoint = endpoint
        self.query_params = 'hourly=apparent_temperature,precipitation,weathercode'


    def retrieve(self, location, start_date, delta_date=1):
        if self.DEV_STUB:
            location = (10.823, 106.6296)
            start_date = (datetime.today()-timedelta(days=30))
        if any(param is None for param in [location, start_date]):
            return {}
        end_date = start_date+timedelta(days=delta_date)
        start_date = start_date.date().isoformat()
        end_date = end_date.date().isoformat()
        lat, long = str(location[0]), str(location[1])
        url = '{}?latitude={}&longitude={}&start_date={}&end_date={}&{}'.format(
            self.endpoint, lat, long, start_date, end_date, self.query_params
        )
        print(url)
        response = requests.get(url)
        return response.json()


class WeatherRetrieveService:
    __instance = None

    @staticmethod
    def get_instance():
        if WeatherRetrieveService.__instance is None:
            WeatherRetrieveService.__instance = WeatherRetrieveInstance()
        return WeatherRetrieveService.__instance