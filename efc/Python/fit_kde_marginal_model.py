from datetime import datetime
from os import makedirs
from os.path import join
from typing import Union, Literal

import numpy as np
import pandas as pd
from config import start_evaluation, end_evaluation, base_path, model_results, kde_raw_data
from scipy import stats
from scipy.special import ndtr
from tqdm import tqdm

from utils import combine_json_results, read_price
from r_utils import crps_sample_single

start_import = datetime(2018, 12, 27)
window_length_days = (start_evaluation - start_import).days


def fit_kde_marginal_model(dist_str: Literal['normal', 'jsu'], dist_number, export_file_per_hour: bool = False):
    """Fit a KDE model to the errors of the forecast and compute the PIT values for the KDE model. Save the results
    to a pickle file."""
    df = read_price(start_date=start_import, end_date=end_evaluation)  # DataFrame with observation column 'obs'
    forecast_name = f'kde_{dist_str}_{dist_number}'
    makedirs(join(base_path, kde_raw_data, forecast_name), exist_ok=True)

    if dist_str == 'normal':
        normal_dist_loop = combine_json_results(join(base_path, f'distparams_probNN_normal{dist_number}'),
                                                start_date=start_import, end_date=end_evaluation)
        df['mean'] = normal_dist_loop['loc']  # Distributional neural network mean forecast
    elif dist_str == 'jsu':
        import tensorflow_probability as tfp
        tfd = tfp.distributions
        jsu_dist_loop = combine_json_results(join(base_path, f'distparams_probNN_jsu{dist_number}'),
                                             start_date=start_import, end_date=end_evaluation)
        df['mean'] = tfd.JohnsonSU(skewness=jsu_dist_loop['skewness'], tailweight=jsu_dist_loop['tailweight'],
                                   loc=jsu_dist_loop['loc'],
                                   scale=jsu_dist_loop['scale']).mean().numpy()
    else:
        ValueError(f'Unknown distribution string: {dist_str}')

    df['er'] = df['mean'] - df['obs']

    for date in tqdm(pd.date_range(start=start_evaluation, end=end_evaluation,
                                   freq='D')):  # tqdm is package for visualizing progress
        begin_train = date - pd.DateOffset(days=window_length_days)
        end_train = date - pd.DateOffset(hours=1)
        end_date = date + pd.DateOffset(hours=23)

        kde = stats.gaussian_kde(
            df.loc[begin_train:end_train, 'er']
        )  # one kde per day
        df.loc[date:end_date, 'pit'] = list(ndtr(np.ravel(- item + kde.dataset) / kde.factor).mean()
                                            for item in df.loc[date:end_date, 'er'])
        # kde.dataset: points the kde is generated with
        # kde.factor: kde bandwith

        # Create forecast sample for every hour
        for dt_loop in pd.date_range(date, end_date, freq='H'):
            sample = - df.loc[begin_train:end_train, 'er'].values + df.loc[
                dt_loop, 'mean']  # Forecast for obs = mean_forecast - errors
            if export_file_per_hour:
                np.savetxt(join(base_path, kde_raw_data, forecast_name, f'{dt_loop:%Y-%m-%d_%H}.txt'), sample)
            df.loc[dt_loop, 'crps'] = crps_sample_single(df.loc[dt_loop, 'obs'], sample)  # Compute crps based on sample

    df.to_pickle(join(base_path, model_results, f'{forecast_name}.pickle'))


def check_sample_computation():
    """ Function for checking the error computation and the pit computation in the kde approach.
    The error is subtracted from the mean forecast and the pit is computed by """
    n = 100
    true = stats.randint(low=0, high=100).rvs(size=n)
    errors = stats.norm(loc=1, scale=1).rvs(n)
    mean = true + errors

    # Check sampling computation
    forecasts = np.zeros((int(n / 2), int(n / 2)))
    pit_kde_er = np.zeros(int(n / 2))
    pit_approx_forecasts = np.zeros(int(n / 2))
    for i in range(int(n / 2)):
        forecasts[i, :] = mean[i + int(n / 2)] - errors[i:i + int(n / 2)]

        kde = stats.gaussian_kde(errors[i:i + int(n / 2)])
        pit_kde_er[i] = ndtr(
            np.ravel(- errors[i + int(n / 2)] + kde.dataset) / kde.factor).mean()  # Has to be multiplied by -1
        pit_approx_forecasts[i] = (forecasts[i, :] <= true[i + int(n / 2)]).mean()
    print(((true[int(n / 2):] - forecasts.mean(axis=1)) ** 2).mean() ** 1 / 2)

    # Check pit computation
    print(np.column_stack([pit_kde_er, pit_approx_forecasts])[:10])


if __name__ == '__main__':
    # Export Observation DataFrame
    df = read_price(start_date=start_import, end_date=end_evaluation)
    df.to_pickle(join(base_path, model_results, 'obs.pickle'))

    # Export per-file kde results
    export_file_per_hour = True
    for dist_number in range(1, 5):
        fit_kde_marginal_model('normal', dist_number, export_file_per_hour=export_file_per_hour)
        fit_kde_marginal_model('jsu', dist_number, export_file_per_hour=export_file_per_hour)
