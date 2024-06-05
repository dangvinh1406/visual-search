from src.external.bingvs import VisualSearchService
from src.vision.transientrgs import TransientRegressService
from src.vision.iodoorcls import InOutdoorClassifyService
from src.external.visualcrossingwr import WeatherRetrieveService


VisualSearchService.get_instance()
TransientRegressService.get_instance()
InOutdoorClassifyService.get_instance()
WeatherRetrieveService.get_instance()