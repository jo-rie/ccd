from datetime import datetime, timedelta
from os import makedirs
from os.path import join
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
import properscoring as ps
from config import nb_samples, base_path, model_results
from r_utils import crps_sample_single, crps_sample
from tqdm import tqdm
from utils import read_price, compute_naive_forecast_for_datetime, get_sample_forecast_day, combine_json_results


def compute_naive_forecast(begin_date: datetime, end_date: datetime, export_folder_files: str = None, d: int = 1456):
    """ Compute Naive Forecast for the period from begin_date to end_date using D samples from the past. If
    export_folder is given, a file with a sample per hour is created at the given folder."""
    df_naive_forecast = read_price()

    for datetime_loop in tqdm(pd.date_range(start=df_naive_forecast.index.min() + pd.DateOffset(days=7),
                                            end=end_date, freq='H'),
                              desc='Computing naive, first run'):
        df_naive_forecast.loc[datetime_loop, 'mean'] = compute_naive_forecast_for_datetime(df_naive_forecast,
                                                                                           datetime_loop)

    df_naive_forecast['resid'] = df_naive_forecast['obs'] - df_naive_forecast['mean']

    for datetime_loop in tqdm(pd.date_range(start=begin_date, end=end_date, freq='H'),
                              desc='Computing naive, second run'):
        # Sample from distribution: same hour, last 1456 days
        values_to_sample_from = df_naive_forecast[(df_naive_forecast.index.hour == datetime_loop.hour) &
                                                  (df_naive_forecast.index >= datetime_loop - pd.DateOffset(days=d)) &
                                                  (df_naive_forecast.index < datetime_loop)]['resid'].values
        sample = np.random.choice(values_to_sample_from, size=nb_samples, replace=True) + df_naive_forecast.loc[
            datetime_loop, 'mean']
        # Save sample to file
        if export_folder_files is not None:
            np.savetxt(join(export_folder_files, f'{datetime_loop:%Y-%m-%d_%H}.txt'), sample)
        # Compute CRPS and save to df_naive_forecast
        df_naive_forecast.loc[datetime_loop, 'crps'] = crps_sample_single(df_naive_forecast.loc[datetime_loop, 'obs'],
                                                                          sample)
        df_naive_forecast.loc[datetime_loop, 'pit'] = (sample <= df_naive_forecast.loc[datetime_loop, 'obs']).mean()
    return df_naive_forecast


def compute_lp_combination(begin_date: datetime, end_date: datetime, dist: Literal['jsu', 'normal'] = 'normal'):
    """ Compute a summary dataframe of the TLP of the DDNN-Normal forecast number dist_number for the period from
    begin_date to end_date."""
    df_combination = read_price(start_date=begin_date, end_date=end_date)

    for dt in tqdm(df_combination.index[::24], desc=f'Computing {dist.capitalize()} LP combination'):
        date = dt.date()
        data = np.concatenate([get_sample_forecast_day(join(base_path, f'forecasts_probNN_{dist}{x}'), date)
                               for x in range(1, 5)], axis=1)
        df_combination.loc[dt:dt + timedelta(hours=23), 'mean'] = data.mean(axis=1)
        df_combination.loc[dt:dt + timedelta(hours=23), 'median'] = np.median(data, axis=1)
        df_combination.loc[dt:dt + timedelta(hours=23), 'pit'] = np.mean(
            data <= df_combination.loc[dt:dt + timedelta(hours=23), 'obs'].values[:, np.newaxis], axis=1)
        df_combination.loc[dt:dt + timedelta(hours=23), 'crps'] = \
            crps_sample(obs=df_combination.loc[dt:dt + timedelta(hours=23), 'obs'], forecast_samples=data)[1]

    return df_combination


def compute_normal_dist(begin_date: datetime, end_date: datetime, dist_number: int):
    """ Compute a summary dataframe of the DDNN-Normal forecast number dist_number for the period from begin_date to
    end_date."""
    df_result = combine_json_results(join(base_path, f'distparams_probNN_normal{dist_number}'),
                                     start_date=begin_date, end_date=end_date)
    df_real = read_price()
    df_result = df_result.join(df_real, how='left')
    df_result['mean'] = stats.norm.mean(loc=df_result['loc'], scale=df_result['scale'])
    df_result['pit'] = stats.norm.cdf(df_result['obs'], loc=df_result['loc'],
                                      scale=df_result['scale'])
    df_result['crps'] = ps.crps_gaussian(x=df_result['obs'], mu=df_result['loc'],
                                         sig=df_result['scale'])
    return df_result


def compute_jsu_dist(begin_date: datetime, end_date: datetime, dist_number: int) -> pd.DataFrame:
    """ Compute a summary dataframe of the DDNN-JSU forecast number dist_number for the period from begin_date to
    end_date."""
    df_result = read_price(begin_date, end_date)

    # Compute sample-based as crps has no closed-form solution for JSU
    for day in tqdm(df_result.index[::24], desc=f'Computing DDNN-JSU forecasts {dist_number}'):
        end_day = day + pd.DateOffset(hours=23)

        obs = df_result.loc[day:end_day, 'obs'].values[:, np.newaxis]
        day_data = get_sample_forecast_day(join(base_path, f'forecasts_probNN_jsu{dist_number}'), day.date())
        df_result.loc[day:end_day, 'mean'] = day_data.mean(axis=1)
        df_result.loc[day:end_day, 'pit'] = np.mean(day_data <= obs, axis=1)
        df_result.loc[day:end_day, 'crps'] = crps_sample(obs=obs.flatten(), forecast_samples=day_data)[1]

    return df_result


if __name__ == '__main__':

    export_folder = join(base_path, model_results)
    makedirs(export_folder, exist_ok=True)

    begin_test = datetime(2020, 1, 1, 0)
    end_test = datetime(2020, 12, 31, 23)

    df_naive = compute_naive_forecast(begin_test, end_test)
    df_naive.to_pickle(join(export_folder, 'naive.pickle'))

    df_normal_lp = compute_lp_combination(begin_test, end_test, 'normal')
    df_normal_lp.to_pickle(join(export_folder, 'ddnn_normal_lp.pickle'))

    df_jsu_lp = compute_lp_combination(begin_test, end_test, 'jsu')
    df_jsu_lp.to_pickle(join(export_folder, 'ddnn_jsu_lp.pickle'))

    for dist_nb in tqdm(range(1, 5), desc='Computing DNN-Normal'):
        df_normal_dist = compute_normal_dist(begin_test, end_test, dist_nb)
        df_normal_dist.to_pickle(join(export_folder, f'ddnn_normal_{dist_nb}.pickle'))

    for dist_nb in tqdm(range(1, 5), desc='Computing DNN-JSU'):
        df_jsu_dist = compute_jsu_dist(begin_test, end_test, dist_nb)
        df_jsu_dist.to_pickle(join(export_folder, f'ddnn_jsu_{dist_nb}.pickle'))
