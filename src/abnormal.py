import numpy as np
# import seaborn as sb
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


class AbnormalDetectorInstance:

    seasonal_threshold = 3.0
    difference_seasonal_threshold = 0.2
    difference_backward_threshold = 0.3

    def detect_abnormal_time_series(self, series, visualize=False):
        epsilon = 1e-16

        def Kullback_Leibler(p, q):
            # assume p, q have sum = 1
            # epsilon is used to avoid zero issue
            P = p+epsilon
            Q = q+epsilon

            divergence = np.sum(P*np.log(P/Q))
            return divergence

        def calculate_KL_distance(chart1, chart2):
            # normalize charts to have sum = 1 as distributions
            norm_c1 = chart1/np.sum(chart1)
            norm_c2 = chart2/np.sum(chart2)
            # KL divergence is not symetric. We calculate back and forth
            return Kullback_Leibler(norm_c1, norm_c2)+Kullback_Leibler(norm_c2, norm_c1)

        def calculate_l1_distance(chart1, chart2):
            # normalize charts using max norm
            norm_c1 = normalize([chart1], norm='max')[0]
            norm_c2 = normalize([chart2], norm='max')[0]
            diff_norm = norm_c1-norm_c2
            return np.linalg.norm(diff_norm, ord=1), diff_norm

        years = sorted(list(set([y[0] for y in series.index.tolist()])))
        # if search result includes several years
        is_seasonal = False
        if len(years) > 1:
            # try to find seasonal features
            # 1. make distribution charts for each year
            charts = {}
            for year in years:
                charts[year] = {
                    'data': series[series.index.get_level_values(0)==year].sort_index()
                }
            # 2. calculate distribution distance between the first year and others
            for year in years:
                if year == years[0]:
                    charts[years[0]]['distance'] = 0
                    charts[years[0]]['is_normal'] = True
                else:
                    # charts[year]['distance'] = calculate_KL_distance(
                    charts[year]['distance'], charts[year]['diff_norm'] = calculate_l1_distance(
                        charts[years[0]]['data'].values, charts[year]['data'].values
                    )
                    if charts[year]['distance'] < self.seasonal_threshold:
                        is_seasonal = True
                        charts[year]['is_normal'] = True
                    else:
                        charts[year]['is_normal'] = False
            # 3. visualize charts if needed
            if visualize:
                fig, axes = plt.subplots(1, len(years), figsize=(5*len(years), 5))
                i = 0
                for year in years:
                    axes[i].set_title(
                        str(year)+':'+
                        '\n-distance:{}'.format(charts[year]['distance'])+
                        '\n-is_normal:{}'.format(charts[year]['is_normal'])
                    )
                    axes[i].set_ylim([0, np.max(series)])
                    axes[i].bar(np.arange(12), charts[year]['data'])
                    i += 1
        
        abnormal_months = [False]*len(series)
        if is_seasonal:
            # focus on years which are not similar to the first year
            pass
        else:
            # treat as one unique time series
            # 1. Normalize the whole series using min-max
            v = normalize([series.values], norm='max')[0]
            # 2. Calculate difference (backward derivative)
            diff = np.diff(v)
            # 3. Find months with large increasing
            abnormal_months = diff > self.difference_backward_threshold
            # 4. visualize chart if needed
            if visualize:
                series.index = series.index.map(
                    lambda t: datetime.datetime.strptime(str(t), '(%Y, %m)'))
                fig, axe = plt.subplots(1, 1, figsize=(20, 5))
                series = series.sort_index()
                axe.bar(
                    series.index.values[1:],
                    abnormal_months.astype(int)*series.max(),
                    color='red'
                )
                axe.plot(series)
            # append 1 first element as calculating backward
            abnormal_months = [False]+abnormal_months.tolist()

        return abnormal_months, is_seasonal


class AbnormalDetectorService:
    __instance = None

    @staticmethod
    def get_instance():
        if AbnormalDetectorService.__instance is None:
            AbnormalDetectorService.__instance = AbnormalDetectorInstance()
        return AbnormalDetectorService.__instance