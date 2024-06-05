from datetime import datetime
from geopy import distance as geodistance
from src.utils import access, TIME_FORMAT


PASS = 'pass'
TIME_CAP_AFTER_UPLOAD = 'Captured time from EXIF is after the first time the image appear on the internet'
TIME_CAP_NOT_MATCH_TIME_DAY = 'Captured time from EXIF does not match with time in the day'
TIME_CAP_NOT_MATCH_SEASON = 'Captured time from EXIF does not match with the season'
PLACE_CAP_FAR_ESTIMATED = 'Estimate geolocation is too far from the captured location in EXIF'
WEATHER_NOT_MATCH = 'Weather is not matched'

GEODISTANCE_THRESHOLD = 10  # degree ~ 110 km


class RuleCheckInstance:

    DEV_STUB = False

    def check_captured_time_before_uploaded_time(self, data):
        capture_time = access(data, 'exif/captured_time')
        upload_time = access(data, 'search/summary/oldest/date')
        if capture_time is None or upload_time is None:
            return True, PASS
        if datetime.strptime(capture_time, TIME_FORMAT) > datetime.strptime(upload_time, TIME_FORMAT):
            return False, TIME_CAP_AFTER_UPLOAD
        return True, PASS
    

    def check_captured_time_match_image_time(self, data):
        capture_time = access(data, 'exif/captured_time')
        if capture_time is None:
            return True, PASS
        capture_time = datetime.strptime(capture_time, TIME_FORMAT)

        image_time_tags = [
            tag['class'].split('/')[-1]
            for tag in access(data, 'vision/transients')
            if 'time/' in tag['class']
        ]
        if not image_time_tags:
            return True, PASS
        if 'daylight' in image_time_tags:
            if capture_time.hour > 16 or capture_time.hour < 5:
                return False, TIME_CAP_NOT_MATCH_TIME_DAY
        if 'night' in image_time_tags:
            if capture_time.hour < 18 and capture_time.hour > 4:
                return False, TIME_CAP_NOT_MATCH_TIME_DAY
        if 'midday' in image_time_tags:
            if capture_time.hour < 10 or capture_time.hour > 15:
                return False, TIME_CAP_NOT_MATCH_TIME_DAY

        image_moment_tags = [
            tag['class'].split('/')[-1]
            for tag in access(data, 'vision/transients')
            if 'moment/' in tag['class']
        ]
        if 'dawn-dusk' in image_moment_tags:
            if capture_time.hour not in range(16, 19) and capture_time.hour not in range(3, 6):
                return False, TIME_CAP_NOT_MATCH_TIME_DAY
        if 'sunrise-sunset' in image_moment_tags:
            if capture_time.hour not in range(16, 18) and capture_time.hour not in range(3, 5):
                return False, TIME_CAP_NOT_MATCH_TIME_DAY

        image_season_tags = [
            tag['class'].split('/')[-1]
            for tag in access(data, 'vision/transients')
            if 'season/' in tag['class']
        ]
        if 'spring' in image_season_tags:
            if capture_time.month not in [3, 4, 5, 6]:  # Northern Hemisphere
                lat = access(data, 'exif/captured_place/latitude')  # from exif
                if not lat:
                    lat = access(data, 'vision/locations/fine/pred_lat')  # from estimated gps
                if lat and lat < 0:
                    if capture_time.month not in range[9, 10, 11, 12]:  # Southern Hemisphere
                        return False, TIME_CAP_NOT_MATCH_SEASON
                else:
                    return False, TIME_CAP_NOT_MATCH_SEASON
        if 'summer' in image_season_tags:
            if capture_time.month not in [6, 7, 8, 9]:  # Northern Hemisphere
                lat = access(data, 'exif/captured_place/latitude')  # from exif
                if not lat:
                    lat = access(data, 'vision/locations/fine/pred_lat')  # from estimated gps
                if lat and lat < 0:
                    if capture_time.month not in [12, 1, 2, 3]:  # Southern Hemisphere
                        return False, TIME_CAP_NOT_MATCH_SEASON
                else:
                    return False, TIME_CAP_NOT_MATCH_SEASON
        if 'autumn' in image_season_tags:
            if capture_time.month not in [9, 10, 11, 12]:  # Northern Hemisphere
                lat = access(data, 'exif/captured_place/latitude')  # from exif
                if not lat:
                    lat = access(data, 'vision/locations/fine/pred_lat')  # from estimated gps
                if lat and lat < 0:
                    if capture_time.month not in [3, 4, 5, 6]:  # Southern Hemisphere
                        return False, TIME_CAP_NOT_MATCH_SEASON
                else:
                    return False, TIME_CAP_NOT_MATCH_SEASON
        if 'winter' in image_season_tags:
            if capture_time.month not in [12, 1, 2, 3]:  # Northern Hemisphere
                lat = access(data, 'exif/captured_place/latitude')  # from exif
                if not lat:
                    lat = access(data, 'vision/locations/fine/pred_lat')  # from estimated gps
                if lat and lat < 0:
                    if capture_time.month not in [6, 7, 8, 9]:  # Southern Hemisphere
                        return False, TIME_CAP_NOT_MATCH_SEASON
                else:
                    return False, TIME_CAP_NOT_MATCH_SEASON
        return True, PASS


    def check_capture_place_near_estimated_place(self, data):
        lat = access(data, 'exif/captured_place/latitude')
        lon = access(data, 'exif/captured_place/longitude')
        if not lat:
            return True, PASS
        estimated_lat = access(data, 'vision/locations/fine/pred_lat')
        estimated_lon = access(data, 'vision/locations/fine/pred_long')
        if geodistance.distance(
            (lat, lon), (estimated_lat, estimated_lon)
        ) > GEODISTANCE_THRESHOLD:
            return False, PLACE_CAP_FAR_ESTIMATED
        return True, PASS


    def check_weather_match_transients(self, data):
        image_weather_tags = [
            tag['class'].split('/')[-1]
            for tag in access(data, 'vision/transients')
            if 'weather/' in tag['class']
        ]
        return True, PASS
    

    def check_transients_match_uploaded_period_weather(self, data):
        image_weather_tags = [
            tag['class'].split('/')[-1]
            for tag in access(data, 'vision/transients')
            if 'weather/' in tag['class']
        ]
        return True, PASS


    def infer(self, data):

        if self.DEV_STUB:
            return {
                'conflicts':[
                    {'message': message} 
                    for message in [
                        TIME_CAP_AFTER_UPLOAD,
                        PLACE_CAP_FAR_ESTIMATED,
                        WEATHER_NOT_MATCH
                    ]
                ]
            }

        check_funcs = [
            func for func in dir(self) if func.startswith('check_')
        ]
        check_results = [
            getattr(self, func)(data) for func in check_funcs
        ]
        result = {
            'conflicts': [
                {'message': message} 
                for (is_pass, message) in check_results
                if not is_pass
            ]
        }
        return result


class RuleCheckService:
    __instance = None

    @staticmethod
    def get_instance():
        if RuleCheckService.__instance is None:
            RuleCheckService.__instance = RuleCheckInstance()
        return RuleCheckService.__instance
