import json
from datetime import datetime, timedelta
from os import listdir
from os.path import join
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_pinball_loss
from scipy import stats
from statsmodels.stats.diagnostic import anderson_statistic

from config import base_path


# nb_bins = 11

def combine_json_results(folder_path: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """ Combine all json files in folder_path to one DataFrame. The data is assumed to be named after the date the distribution corresponds to.

    :param folder_path: folder with json files in it
    :param start_date: let dataframe start earliest with the specified date
    :param end_date: let dataframe end latest with the specified date
    :return: pd.DataFrame with a row per hour and a column per distribution parameter
    """
    # folder_path = '/home/ubuntu/Code/distributionalnn-raw/distparams_probNN_jsu1'
    json_file_list = [f for f in listdir(folder_path) if f.endswith('.json')]

    def read_json(file_name) -> pd.DataFrame:
        date = pd.to_datetime(Path(file_name).stem)
        with open(join(folder_path, file_name), 'r') as f:
            param_data = json.load(f)
        df_new = pd.DataFrame(
            index=pd.date_range(start=date, end=date + pd.DateOffset(hours=23), freq='H'),
            data=param_data
        )
        return df_new

    df = pd.concat(map(read_json, json_file_list))
    df = df.sort_index()

    df = restrict_df_to_dates(df, end_date, start_date)
    return df


def restrict_df_to_dates(df: pd.DataFrame, end_date: datetime = None, start_date: datetime = None) -> pd.DataFrame:
    """
    The restrict_df_to_dates function takes a dataframe and two dates as input.
    It returns the dataframe with all rows that are not between the start_date and end_date removed.
    If either date is None, then it will be ignored.

    :param df: Specify the dataframe that we want to filter
    :param end_date: Specify the end date of the dataframe
    :param start_date: Restrict the dataframe to only include rows with dates greater than or equal to start_date
    :return: A dataframe that is restricted to the dates specified by the user
    :doc-author: Trelent
    """
    if start_date is not None:
        if end_date is not None:
            df = df[(df.index >= start_date) & (df.index <= end_date)]
        else:
            df = df[(df.index >= start_date)]
    else:
        if end_date is not None:
            df = df[(df.index <= end_date)]
    return df


def get_sample_forecast(folder_path: str, dt: datetime) -> np.ndarray:
    """ Get a numpy vector of a forecast for a specific datetime.

    :param folder_path: The folder to be used (has to contain a text while named after the date of datetime
    :param dt: The date and hour the forecast should be for
    :return: a numpy array with a distribution sample
    """
    # folder_path = '/home/ubuntu/Code/distributionalnn-raw/forecasts_probNN_jsu1'
    # dt = datetime(2019, 4, 3, 15)
    filename = dt.strftime('%Y-%m-%d') + '.txt'

    data = np.loadtxt(join(folder_path, filename), delimiter=',')
    return data[:, dt.hour]


def get_sample_forecast_day(folder_path: str, dt: datetime) -> np.ndarray:
    """ Get a numpy vector of a forecast for a whole day.

    :param folder_path: The folder to be used (has to contain a text while named after the date of datetime
    :param dt: The date the forecast should be for
    :return: a numpy array with a distribution sample
    """
    # folder_path = '/home/ubuntu/Code/distributionalnn-raw/forecasts_probNN_jsu1'
    # dt = datetime(2019, 4, 3)
    filename = dt.strftime('%Y-%m-%d') + '.txt'

    data = np.loadtxt(join(folder_path, filename), delimiter=',')
    return data.T


def to_csv_and_pickle(df: pd.DataFrame, path: str):
    """ Save the dataframe df to the specified path as csv and pickle

    :param df: dataframe
    :param path: path without an extension to save
    :return:
    """
    df.to_csv(path + '.csv')
    df.to_pickle(path + '.pickle')


def read_price(start_date: datetime = None, end_date: datetime = None):
    """ Read csv file with price data and return it as dataframe with column 'obs' """
    df_real = pd.read_csv(join(base_path, 'Datasets/DE.csv'), index_col=0)
    df_real.index = [datetime.strptime(e, '%Y-%m-%d %H:%M:%S') for e in df_real.index]
    df_real = df_real.rename(columns={'Price': 'obs'})
    df_real = df_real[['obs']]
    df_real = restrict_df_to_dates(df_real, start_date=start_date, end_date=end_date)
    return df_real


def rmse(df: pd.DataFrame, col1: str, col2: str):
    """Compute rmse for the columns col1 and col2 of the dataframe and return it as float"""
    return np.sqrt(np.mean((df[col1] - df[col2]) ** 2))


def mae(df: pd.DataFrame, col1: str, col2: str):
    """Compute mae for the columns col1 and col2 of the dataframe and return it as float"""
    return np.mean(np.abs(df[col1] - df[col2]))


def compute_naive_forecast_for_datetime(df, dt: datetime, col='obs') -> float:
    """ Compute the naive forecast for the given datetime dt using the data in df column col"""
    if dt.weekday() in [0, 5, 6]:  # For Monday(0), Saturday(5), Sunday(6)
        forecast = df.loc[dt - timedelta(days=7), col]
    else:
        forecast = df.loc[dt - timedelta(days=1), col]
    return forecast


def compute_pinball(obs: np.ndarray, sample: np.ndarray, nb_quantiles: int = 99):
    quantile_array = np.linspace(1 / (nb_quantiles + 1), 1, nb_quantiles, endpoint=False)
    quantile_sample = np.zeros([len(obs), nb_quantiles])
    pinballs_per_point = np.zeros_like(quantile_sample)
    # Compute all quantiles due to memory restrictions separately per time step
    for i_window in range(len(obs)):
        quantile_sample[i_window, :] = np.quantile(sample[i_window, :], quantile_array)
    # Compute pinball los for all quantiles
    for (i_quan, quan) in enumerate(quantile_array):
        pinballs_per_point[i_window, i_quan] = (
            mean_pinball_loss(obs,
                              quantile_sample[:, i_quan],
                              alpha=quan, multioutput='raw_values'))
    return pinballs_per_point.mean(axis=1)


def ad_test(x: np.ndarray) -> float:
    """Return Anderson-Darling test statistic for uniformity for vector x."""
    return anderson_statistic(x, dist=stats.uniform, fit=False, params=(-1e-10, 1+1e-10))


def ks_test(x: np.ndarray) -> float:
    """Return Kolmogorv-Smirnoff test statistic for uniformity for vector x."""
    statistic, pvalue = stats.ks_1samp(x=x, cdf=stats.uniform(0,1).cdf)
    return statistic