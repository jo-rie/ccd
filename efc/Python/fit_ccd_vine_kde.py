import argparse
from datetime import datetime
# from multiprocessing_logging import install_mp_handler
import functools
import os
from glob import glob
from multiprocessing import Pool
from os import makedirs
from os.path import join, split
from typing import Union, Any, List
import logging

import numpy as np
import pandas as pd
from pyvinecopulib import Vinecop, FitControlsVinecop, Bicop, FitControlsBicop
import pyvinecopulib as pv
from scipy import integrate

from config import base_path, kde_raw_data, model_results
from kde_utils import KdeFit
from tqdm import tqdm

bicop_family_set = pv.all.copy()
bicop_family_set.remove(pv.bb7)
integration_n = 20000
vinecopcontrols = FitControlsVinecop(
    family_set=bicop_family_set  # Do not use bb7 as it is numerically unstable
)
bicopcontrols = FitControlsBicop(family_set=bicop_family_set)

start_kde_fit = datetime(2018, 12, 27)
start_ccd_fit = datetime(2019, 6, 27)

begin_test = datetime(2020, 1, 1)
end_test = datetime(2020, 12, 31)

dt_range = pd.date_range(begin_test, end_test, freq='D')

logging.basicConfig(filename=f'../logging/fit_ccd_vine_kde_{datetime.now()}.txt', encoding='utf-8',
                    level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

output_path_array = [join(base_path, f'data_{f}') for f in ['ccd_all_data_fit', 'ccd_last_year_fit',
                                                            'ccd_hourly_all_data',
                                                            'ccd_hourly_last_year']]
date_offset_array = [None, pd.DateOffset(years=1, hours=-1), None, pd.DateOffset(years=1, hours=-1)]
fit_hourly_array = [False, False, True, True]


def fit_single_day(dt: Any, output_path: str, fit_hourly: bool,
                   vine_window_length: Union[pd.DateOffset, None],
                   margins: List[int], fit_bicop: bool) -> None:
    """ Fit ccd model for the date dt

    :param fit_bicop: If true a pyvinecopulib.Bicop is fit, otherwise a Vinecop is used
    :param dt: argument accepted for start by pd.date_range
    :param output_path: path to save fitting results
    :param vine_window_length: length of fitting window for vine copula. If None, all past data is used
    :param fit_hourly: if True, a separate vine copula is fit for every hour
    :param margins: List of the margins to combine (e.g., [1, 2] for combining 1 and 2)
    :return: None
    """
    print(f'Starting {dt:%Y-%m-%d %H}:00')
    if vine_window_length is None:
        vine_window_length = pd.tseries.frequencies.to_offset(dt - pd.Timestamp('1970-01-01 00:00:00'))
        # Take as offset the duration to datetime base year
    makedirs(output_path, exist_ok=True)
    # Import data
    df_obs = pd.read_pickle(join(base_path, kde_raw_data, 'obs.pickle'))
    margin_data = []
    for i in margins:
        data_new = pd.read_pickle(join(base_path, model_results, f'kde_normal_{i}.pickle'))
        margin_data.append(data_new)

    df_ccd = pd.DataFrame(index=pd.date_range(dt, dt + pd.Timedelta(hours=23), freq='h'),
                          columns=['obs', 'pit', 'crps', 'mean', 'log_score'])

    end_train = dt - pd.Timedelta(hours=1)

    # Fit Vine for all hours
    if not fit_hourly:
        if not fit_bicop:
            vine_fit = Vinecop(
                np.column_stack(
                    [df['pit'].loc[(df.index > end_train - vine_window_length) & (df.index <= end_train)] for df in
                     margin_data]), controls=vinecopcontrols)  # Use only the last year of data
        else:
            vine_fit = Bicop(
                np.column_stack(
                    [df['pit'].loc[(df.index > end_train - vine_window_length) & (df.index <= end_train)] for df in
                     margin_data]), controls=bicopcontrols)
            vine_fit.to_json(join(output_path, f'bicop_{dt:%Y-%m-%d}.json'))
    # Evaluate for the following 23 hours
    for h in range(0, 24):
        if fit_hourly:
            if not fit_bicop:
                vine_fit = Vinecop(
                    np.column_stack(
                        [df['pit'].loc[(df.index > end_train - vine_window_length) &
                                       (df.index <= end_train) &
                                       (df.index.hour == h)] for df in
                         margin_data]), controls=vinecopcontrols)  # Use only the last year of data
            else:
                vine_fit = Bicop(
                    np.column_stack(
                        [df['pit'].loc[(df.index > end_train - vine_window_length) &
                                       (df.index <= end_train) &
                                       (df.index.hour == h)] for df in
                         margin_data]), controls=bicopcontrols)  # Use only the last year of data
                vine_fit.to_json(join(output_path, f'bicop_{dt:%Y-%m-%d %H}.json'))
        dt_hour = dt + pd.Timedelta(hours=h)
        dist_list = [KdeFit(
            np.loadtxt(join(base_path, kde_raw_data, f'kde_normal_{i}', f'{dt_hour:%Y-%m-%d_%H}.txt'))
        ) for i in margins]

        obs = df_obs.loc[dt_hour].values[0]
        integration_boundaries = (np.min([m.quantile(0.01) for m in dist_list]),
                                  np.max([m.quantile(0.99) for m in dist_list]))

        def vectorize_cdf(x: Union[float, np.ndarray]):
            return np.column_stack([dist.cdf(x) for dist in dist_list])

        def vectorize_pdf(x: Union[float, np.ndarray]):
            return np.column_stack([dist.pdf(x) for dist in dist_list])

        def ccd_likeli(x: Union[float, np.ndarray]):
            return vectorize_pdf(x).prod(axis=1) * vine_fit.pdf(vectorize_cdf(x))

        if integration_boundaries[0] > obs or obs > integration_boundaries[1]:
            logging.warning(
                f'Out of Bounds: obs is {obs}, bounds are {integration_boundaries[0]}, {integration_boundaries[1]}')
            if integration_boundaries[0] > obs:
                grid = np.linspace(obs, integration_boundaries[1], 2 * integration_n)
                pit = 0
            else:
                grid = np.linspace(integration_boundaries[0], obs, 2 * integration_n)
                pit = 1
            grid_vals = ccd_likeli(grid)
            normalization_constant = integrate.trapezoid(grid_vals, grid)

            grid_cdf_vals = np.zeros_like(grid)
            for i in range(1, len(grid_cdf_vals)):
                grid_cdf_vals[i] = integrate.trapezoid(grid_vals[:i], grid[:i]) / normalization_constant

            mean = integrate.trapezoid(grid_vals / normalization_constant * grid, grid)
            crps = integrate.trapezoid(((integration_boundaries[0] > obs) - grid_cdf_vals) ** 2, grid)
            log_score = np.nan
        else:
            left_grid = np.linspace(integration_boundaries[0], obs, integration_n)
            left_grid_vals = ccd_likeli(left_grid)
            left_integral = integrate.trapezoid(left_grid_vals, left_grid)
            right_grid = np.linspace(obs, integration_boundaries[1], integration_n)
            right_grid_vals = ccd_likeli(right_grid)
            right_integral = integrate.trapezoid(right_grid_vals, right_grid)
            normalization_constant = left_integral + right_integral

            def ccd_pdf(x: float):
                return ccd_likeli(x) / normalization_constant

            mean_grid = np.concatenate([left_grid, right_grid])
            mean_grid_vals = np.concatenate([left_grid_vals, right_grid_vals]) / normalization_constant
            mean_grid_vals = mean_grid_vals * mean_grid
            mean = integrate.trapezoid(mean_grid_vals, mean_grid)

            # Compute crps by reusing the computed values on the grid
            left_grid_cdf = np.zeros_like(left_grid)
            for i in range(1, len(left_grid_cdf)):
                left_grid_cdf[i] = integrate.trapezoid(left_grid_vals[:i], left_grid[:i]) / normalization_constant
            right_grid_cdf = np.zeros_like(right_grid)
            right_grid_cdf[0] = left_grid_cdf[-1]
            for i in range(1, len(left_grid_cdf)):
                right_grid_cdf[i] = integrate.trapezoid(right_grid_vals[:i], right_grid[:i]) / normalization_constant + \
                                    right_grid_cdf[0]
            crps = integrate.trapezoid(left_grid_cdf ** 2, left_grid) + integrate.trapezoid((1 - right_grid_cdf) ** 2,
                                                                                            right_grid)

            # Compute summary statistics
            pit = left_integral / (left_integral + right_integral)
            log_score = - np.log(ccd_pdf(obs))

        df_ccd.loc[dt_hour, 'obs'] = obs
        df_ccd.loc[dt_hour, 'pit'] = pit
        df_ccd.loc[dt_hour, 'crps'] = crps
        df_ccd.loc[dt_hour, 'mean'] = mean
        df_ccd.loc[dt_hour, 'log_score'] = log_score

        print(f'Finished {dt_hour:%Y-%m-%d %H:00} with obs={obs}, pit={pit:.4f}, mean={mean:.4f}')

    # save results
    df_ccd.to_pickle(join(output_path, f'ccd_results_{dt:%Y-%m-%d}.pickle'))
    df_ccd.to_csv(join(output_path, f'ccd_results_{dt:%Y-%m-%d}.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit CCD model with kde margins to data')
    parser.add_argument('--noparallel', action='store_true', help='Run the computation in parallel')
    parser.add_argument('--margins', nargs='*', type=int, default=[1, 2, 3, 4], help='List of margins to use')
    parser.add_argument('--models', nargs='*', type=int, default=[1, 2, 3, 4],
                        help='List of models to fit (1: all data, 2: last year of data, '
                             '3: all data (vine fit per hour), 4: last year of data (vine fit per hour)')
    parser.add_argument('--bistochastic', action='store_true',
                        help='Use a bivariate copula and store results per loop.')
    args = parser.parse_args()
    # makedirs(join(base_path, 'logs'), exist_ok=True)

    combination_output_path = join(base_path, f'results_ccd_{datetime.now():%Y%m%d}'
                                              f'_margins_{"".join([str(a) for a in args.margins])}'
                                              f'_models_{"".join([str(a) for a in args.models])}')
    makedirs(combination_output_path, exist_ok=True)

    logging.info(f'Starting Computation, export folder is {combination_output_path} ---')
    print(f'Starting Computation, export folder is {combination_output_path} ---')

    output_path_array = [output_path_array[i - 1] for i in args.models]
    date_offset_array = [date_offset_array[i - 1] for i in args.models]
    fit_hourly_array = [fit_hourly_array[i - 1] for i in args.models]

    for (output_path, date_offset, fit_hourly) in zip(output_path_array, date_offset_array, fit_hourly_array):
        logging.info(f'Starting loop {output_path} ---')
        makedirs(join(output_path), exist_ok=True)

        if not args.noparallel:
            pool = Pool()
            pool.map(
                functools.partial(fit_single_day, output_path=output_path, fit_hourly=fit_hourly,
                                  vine_window_length=date_offset, margins=args.margins, fit_bicop=args.bistochastic),
                dt_range)
        else:
            for dt_loop in tqdm(dt_range):
                # dt_loop = pd.to_datetime('2020-01-31 02:00:00')
                fit_single_day(dt_loop, output_path=output_path, fit_hourly=fit_hourly,
                               vine_window_length=date_offset, margins=args.margins, fit_bicop=args.bistochastic)

        ## Combine data
        file_list = glob(join(output_path, '*.pickle'))
        file_list.sort()
        pd_list = [pd.read_pickle(f) for f in file_list]
        pd_concat = pd.concat(pd_list)
        pd_concat.to_csv(join(combination_output_path, f'{os.path.split(output_path)[1]}.csv'))
        pd_concat.to_pickle(join(combination_output_path, f'{os.path.split(output_path)[1]}.pickle'))

        if args.bistochastic:
            file_list_json = glob(join(output_path, '*.json'))
            file_list_json.sort()
            pd_list = []
            for f in file_list_json:
                # Load Bicop from json
                bicop_loop = Bicop(f)
                date = pd.to_datetime(split(f)[1].split('.')[0].split('_')[1])
                # Extract name, rotation, tau
                df_loop = pd.DataFrame(index=[date], data={'family': bicop_loop.family, 'rotation': bicop_loop.rotation, 'tau': bicop_loop.tau})
                # Save to dataframe
                pd_list.append(df_loop)
            pd_concat = pd.concat(pd_list)
            pd_concat.to_csv(join(combination_output_path, f'{os.path.split(output_path)[1]}_bicops.csv'))
            pd_concat.to_pickle(join(combination_output_path, f'{os.path.split(output_path)[1]}_bicops.pickle'))

    logging.info('Finished Computation ---')
