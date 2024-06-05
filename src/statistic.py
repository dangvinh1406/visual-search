import os
import pandas
import langdetect
import exifread
import re
import datetime
from geopy.geocoders import Nominatim
from exif import Image as Exifim
# from PIL import Image, ExifTags

# from src.external.bingvs import VisualSearchService
# from src.external.serpgris import ReverseImageSearchService
from src.external.googlesbi import SearchByImageService
from src.external.freeimagehost import ImageHostService
# from src.external.visualcrossingwr import WeatherRetrieveService
from src.external.openmeteowr import WeatherRetrieveService
from src.vision.transientrgs import TransientRegressService
from src.vision.iodoorcls import InOutdoorClassifyService
from src.vision.geocls import GeolocationClassifyService
from src.chart import ChartManagerService
from src.rule import RuleCheckService
from src.cache import CacheService
from src.utils import TIME_FORMAT, CREDENTIAL, access
from src.utils.url import get_domain, DOMAIN


DEMO = os.environ.get('DEMO', 'true') == 'true'

CacheService.get_instance()
SearchByImageService.get_instance()
ImageHostService.get_instance()
WeatherRetrieveService.get_instance()
TransientRegressService.get_instance()
InOutdoorClassifyService.get_instance()
GeolocationClassifyService.get_instance()
ChartManagerService.get_instance()
RuleCheckService.get_instance()


class StatisticCollectInstance:
    DEV_STUB = DEMO
    unknown_lang = '-'

    geolocator = Nominatim(
        user_agent=CREDENTIAL['nominatim_openstreetmap']['agent'])

    def collect(self, image_path):
        data = CacheService.get_instance().lookup(image_path)
        if data is not None:
            data['inference'] = RuleCheckService.get_instance().infer(data)
            return data

        data = {
            'search': None,
            'vision': {
                'places': None,
                'transients': None,
                'locations': None,
            },
            'exif': {
                'captured_time': None,
                'captured_place': {
                    'latitude': None,
                    'longitude': None
                }
            },
            'weather': None,
            'inference': None
        }

        data['exif']['captured_time'] = self.extract_time_from_exif(image_path)
        data['exif']['captured_place'] = self.extract_place_from_exif(image_path)

        place_data = InOutdoorClassifyService.get_instance().predict(image_path)
        data['vision']['places'] = place_data
        location_data = GeolocationClassifyService.get_instance().predict(image_path)
        location_data['country'] = self.get_country_from_latlong(
            location_data['fine']['pred_lat'], location_data['fine']['pred_long']
        )
        data['vision']['locations'] = location_data
        transient_data = TransientRegressService.get_instance().predict(image_path)
        data['vision']['transients'] = transient_data

        if self.DEV_STUB:
            ImageHostService.get_instance().DEV_STUB = True
            SearchByImageService.get_instance().DEV_STUB = True
            RuleCheckService.get_instance().DEV_STUB = True
            WeatherRetrieveService.get_instance().DEV_STUB = True
        else:
            ImageHostService.get_instance().DEV_STUB = False
            SearchByImageService.get_instance().DEV_STUB = False
            RuleCheckService.get_instance().DEV_STUB = False
            WeatherRetrieveService.get_instance().DEV_STUB = False
        
        uploaded_im_url = ImageHostService.get_instance().upload(image_path)
        search_data = SearchByImageService.get_instance().search(uploaded_im_url)
        data['search'] = self.statistic(search_data)

        location, start_date = self.select_time_and_place(data)
        data['weather'] = WeatherRetrieveService.get_instance().retrieve(
            location, start_date)
        data['inference'] = RuleCheckService.get_instance().infer(data)
        return data


    def collec_pages_including_lang(self, x):
        try:
            return langdetect.detect(x)
        except:
            return self.unknown_lang

    
    def check_source(self, url):
        domain = get_domain(url)
        if domain in DOMAIN:
            return DOMAIN[domain]
        return ''
    

    def get_country_from_latlong(self, lat, long):
        coord = "{}, {}".format(lat, long)
        location = self.geolocator.reverse(coord, exactly_one=True, language='en')
        address = location.raw['address']
        city = address.get('city', '')
        state = address.get('state', '')
        country = address.get('country', '')
        return {
            'city': city,
            'state': state,
            'country': country
        }


    def statistic(self, data, 
        date_key='date', title_key='title', url_key='link'
    ):
        df = pandas.DataFrame(data)
        df = df.rename(columns={date_key: 'date', title_key: 'title', url_key:'link'})
        if len(df) <= 0:
            return data
        df['date'] = pandas.to_datetime(df['date'], errors='coerce')
        df = df.sort_values(by='date')
        if 'lang' not in df:
            df['lang'] = df['title'].apply(
                lambda x: self.collec_pages_including_lang(x))
        df['source_type'] = df['link'].apply(lambda x: self.check_source(x))
        data_dict = {}
        data_dict['charts'] = {
            'frequency_timeline': ChartManagerService.get_instance().analyze_freq_timeline_chart(
                df.dropna(subset=['date'])),
            'lang_percentage': ChartManagerService.get_instance().analyze_lang_percentage_chart(df)
        }
        df = df.fillna('')
        df['date'] = df['date'].apply(
            lambda x: x.strftime(TIME_FORMAT) if not pandas.isnull(x) else None)
        df_has_date = df.dropna(subset=['date'])
        data_dict['summary'] = {
            'newest': df_has_date.iloc[-1].to_dict() if len(df_has_date) else None,
            'oldest': df_has_date.iloc[0].to_dict() if len(df_has_date) else None,
            'unspecified_time': df['date'].isna().sum(),
            'highlight': {
                k: v.to_dict(orient='records') 
                for k, v in df.groupby('source_type') if k != ''
            }
        }
        data_dict['records'] = df.to_dict(orient='records')
        return data_dict


    def extract_time_from_exif(self, image_path):
        def collect_exif(image_path):
            exif = None
            with open(image_path, 'rb') as image:
                exif = exifread.process_file(image)
            for key in exif:
                exif[key] = exif[key].__str__()
            return exif

        exif = str(collect_exif(image_path))
        if not exif:
            return None
        moments = []
        for s in re.findall(r'\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}', exif):
            try:
                t = datetime.datetime.strptime(s, TIME_FORMAT)
                moments.append(t)
            except:
                continue

        if len(moments) == 0:
            return None

        oldest_moment = min(moments)
        return oldest_moment.strftime(TIME_FORMAT) # get oldest datetime in the file
    

    def extract_place_from_exif(self, image_path):

        def decimal_coords(coords, ref):
            decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
            if ref == "S" or ref == 'W' :
                decimal_degrees = -decimal_degrees
            return decimal_degrees

        lat, lon = None, None
        try:
            with open(image_path, 'rb') as src:
                img = Exifim(src)
        except:
            return {'latitude': lat, 'longitude': lon}
        if img.has_exif:
            try:
                (lat, lon) = (decimal_coords(img.gps_latitude,
                        img.gps_latitude_ref),
                        decimal_coords(img.gps_longitude,
                        img.gps_longitude_ref))
            except AttributeError:
                print('No Coordinates')
        else:
            print ('The Image has no EXIF information')
            
        return {'latitude': lat, 'longitude': lon}

    
    def select_time_and_place(self, data):
        selected_time = access(data, 'exif/captured_time')
        if not selected_time:
            selected_time = access(data, 'search/summary/oldest/date')
        selected_time = datetime.datetime.strptime(selected_time, TIME_FORMAT) if selected_time is not None else None

        location = (
            access(data, 'exif/captured_place/latitude'),
            access(data, 'exif/captured_place/longitude')
        )
        if not location[0]:
            location = (
                access(data, 'vision/locations/fine/pred_lat'),
                access(data, 'vision/locations/fine/pred_long')
            )
        location = location if location[0] is not None else None
        return location, selected_time


class StatisticCollectService:
    __instance = None

    @staticmethod
    def get_instance():
        if StatisticCollectService.__instance is None:
            StatisticCollectService.__instance = StatisticCollectInstance()
        return StatisticCollectService.__instance