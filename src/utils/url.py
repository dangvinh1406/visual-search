import os
import yaml
from urllib.parse import urlparse


DOMAIN = yaml.load(
    open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..' , 'data', 'domains.yaml'
    )),
    Loader=yaml.Loader
)

def get_domain(url, remove_www=True):
    domain = urlparse(url).netloc
    if remove_www:
        domain = domain.replace('www.', '')
    return domain