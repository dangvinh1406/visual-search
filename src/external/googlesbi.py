import datetime
import time
import os
import json
import selenium
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urlencode, unquote
from bs4 import BeautifulSoup

from src.utils import version


class SearchByImageInstance:
    url = 'https://www.google.com/searchbyimage?'
    client = 'firefox-b-d'
    pagination = 10
    lang = 'en'

    ERROR =  'This page appears when Google '\
            +'automatically detects requests '\
            +'coming from your computer network'
    DEV_STUB = False
    DEBUG = False
    WAITING_SEC = 20

    def __init__(self):
        pass


    def search(self, uploaded_im_url, limit_page=None):
        if self.DEV_STUB:
            stub_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', '..', 'data', 'example.stub')
            if os.path.exists(stub_file):
                print('Load stub data', stub_file)
                return self.format(json.load(open(stub_file)))

        start = 0
        pages = []
        p = 0
        options = webdriver.ChromeOptions()
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--headless")
        if version(selenium.__version__) >= version('4.10.0'):
            driver = webdriver.Chrome(options=options)
        else:
            executable_path = ChromeDriverManager().install()
            driver = webdriver.Chrome(executable_path, options=options)
        
        if self.DEBUG:
            for fname in os.listdir('./'):
                if fname.startswith("debug_"):
                    os.remove(fname)
        while True:
            params = {
                'hl': self.lang,
                'client': self.client,
                'image_url': uploaded_im_url,
                'start': str(start)
            }
            html = self.call_google_by_chrome(driver, params)
            if self.DEBUG:
                with open('debug_{}.html'.format(str(p)), 'w') as out:
                    out.write(html)
            try:
                records = self.parse_html(html)
            except:
                print('Error: Cannot parse HTML correctly')
                break
            if records is None:
                print('Error: Google limits queries, break at page', p)
                break
            if len(records) == 0:
                break
            pages += records

            start += self.pagination
            p += 1
            if limit_page and p >= limit_page:
                break
            time.sleep(self.WAITING_SEC)  # sleep to prevent Google block
        driver.close()
        if self.DEBUG:
            with open('debug.json', 'w') as out:
                json.dump(pages, out)
        return self.format(pages)


    def call_google_by_chrome(self, driver, params):
        url = self.url+urlencode(params)
        driver.get(url)
        html = driver.page_source
        return html 


    def format(self, data):
        return data


    def parse_html(self, html):
        if self.ERROR in html:
            return None
        soup = BeautifulSoup(html, features="html.parser")
        search_result = soup.find('div', id='rso')
        if not search_result:
            return []
        pages_including_tag = search_result.findChildren(recursive=False)[-1]
        pages_including_records = pages_including_tag.findChildren(
            recursive=False)[0].find_all('div', lang=True)

        pages_including = []
        for tag in pages_including_records:
            data = {
                'lang': tag['lang'],
                'link': None,
                'title': None,
                'content': None,
                'size': None,
                'date': None
            }
            a = tag.find_all('a', href=True)[0]
            data['link'] = a['href']
            data['title'] = a.text
            info_tags = a.parent.parent.parent.parent.parent.find_all(
                'span', recursive=True)
            i_ = 0
            for i in range(len(info_tags)):
                size = self.find_and_format_size(info_tags[i].text)
                if size:
                    data['size'] = size
                    i_ = i+1
                    continue

                parsed_date = self.find_and_format_datetime(info_tags[i].text)
                if parsed_date:
                    data['date'] = parsed_date
                    i_ = i+1
                    continue
            if i_ < len(info_tags):
                data['content'] = info_tags[i_].text
            pages_including.append(data)
        return pages_including

    
    def find_and_format_size(self, s):
        try:
            return [int(v) for v in s.split(u'\u00D7')]
        except:
            return None


    def find_and_format_datetime(self, dts):
        day_units = {
            0: ['sec', 'min', 'hour'],
            1: ['day'],
            7: ['week'],
            30: ['month'],
            365: ['year']
        }
        try:
            if 'ago' in dts:
                for num_day, day_unit in day_units.items():
                    if any(unit in dts for unit in day_unit):
                        num = int(dts.split(' ')[0])
                        dts = (datetime.today()-datetime.timedelta.days(num_day*num)).strftime.isoformat()
                        return dts
            else:
                dts = datetime.datetime.strptime(dts, '%b %d, %Y').isoformat()
        except:
            return None
        return dts


class SearchByImageService:
    __instance = None

    @staticmethod
    def get_instance():
        if SearchByImageService.__instance is None:
            SearchByImageService.__instance = SearchByImageInstance()
        return SearchByImageService.__instance