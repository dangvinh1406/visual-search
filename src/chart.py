import pandas

from src.abnormal import AbnormalDetectorService
from src.utils import LANGUAGE


AD = AbnormalDetectorService.get_instance()


class ChartManagerInstance:

    def analyze_lang_percentage_chart(self, df):
        chart = {
            'data': {
                LANGUAGE.get(t[0][:2], 'unknown'): {
                    'count': t[1]
                } 
                for t in df.groupby(by='lang').size().sort_values(ascending=False).items()
            },
            'total': len(df)
        }
        for l in chart['data'].keys():
            chart['data'][l]['percentage'] = chart['data'][l]['count']*100/len(df)

        # this part used for DeDigi Frontend
        limit_parts = 5
        chart['plot'] = {
            'labels': [cnt for cnt in chart['data'].keys()][:limit_parts],
            'data': [int(cnt['percentage']) for cnt in chart['data'].values()][:limit_parts]
        }
        if len(chart['data'].keys()) > limit_parts:
            chart['plot']['labels'][-1] = "others"
            chart['plot']['data'][-1] = 100-sum(chart['plot']['data'][:-1])
        chart['plot']['color'] = ["#41B883", "#E46651", "#00D8FF", "#DD1B16", "#808080"][
            :len(chart['plot']['labels'])]
        return chart


    def analyze_freq_timeline_chart(self, df):
        def ym2str(year_month):
            return str(year_month[0])+'/'+str(year_month[1]).zfill(2)

        def padding_zeros_missing_months_and_sort(series):
            years = [y[0] for y in series.index.tolist()]
            missings = {}
            for year in range(min(years), max(years)+1):
                for month in range(1, 13):
                    if (year, month) not in series:
                        missings[(year, month)] = 0
            series = pandas.concat([series, pandas.Series(data=missings)])
            return series.sort_index()
        
        def shorten_ym_labels(labels):
            '''
            Only keep yyyy/mm for appearing first time, the rest is mm
            '''
            years = []
            for i in range(len(labels)):
                if labels[i].split('/')[0] in years:
                    labels[i] = labels[i].split('/')[1]
                else:
                    years.append(labels[i].split('/')[0])
            return labels

        if len(df) == 0:
            return {
                'data': {},
                'seasonal': False,
                'max_count': 0,
                'abnormal': False,
                'plot': {
                    'labels': [],
                    'datasets': []
                }
            }
        df = df.set_index('date').sort_index()
        count_by_month = df.groupby(by=[df.index.year, df.index.month]).size()
        count_by_month = padding_zeros_missing_months_and_sort(count_by_month)
        abnormal_months, is_seasonal = AD.detect_abnormal_time_series(count_by_month)

        count_by_month = list(count_by_month.items())
        freq_timeline_chart = {
            'data': {
                ym2str(count_by_month[t][0]): {
                    'count': count_by_month[t][1],
                    'abnormal': abnormal_months[t]
                } 
                for t in range(len(count_by_month))
            },
            'seasonal': is_seasonal
        }
        freq_timeline_chart['max_count'] = max(
            x['count'] for x in freq_timeline_chart['data'].values())
        freq_timeline_chart['abnormal'] = any(
            x['abnormal'] for x in freq_timeline_chart['data'].values())
        
        # this part used for DeDigi Frontend
        plot = {
            'labels': shorten_ym_labels(list(freq_timeline_chart['data'].keys())),
            'datasets': [],
        }
        if freq_timeline_chart['abnormal']:
            plot['datasets'].append({
                'label': "Anomaly detected",
                'backgroundColor': "#DD1B16",
                'data': [
                    None if not d['abnormal'] else d['count']
                    for d in freq_timeline_chart['data'].values()
                ]
            })

        plot['datasets'].append({
            'label': "Number of pages includings",
            'backgroundColor': "#808080",
            'data': [d['count'] for d in freq_timeline_chart['data'].values()]
        })

        if freq_timeline_chart['seasonal']:
            plot['datasets'].append({
                'label': "Seasonal detected",
                'backgroundColor': "#00D8FF",
                'data': []
            })
        freq_timeline_chart['plot'] = plot
        return freq_timeline_chart


class ChartManagerService:
    __instance = None

    @staticmethod
    def get_instance():
        if ChartManagerService.__instance is None:
            ChartManagerService.__instance = ChartManagerInstance()
        return ChartManagerService.__instance