from os import makedirs
from os.path import join
from typing import List, Union, Dict
from datetime import datetime
import json
import pickle

import numpy as np
import pandas as pd
from efc.Python.utils import ad_test, ks_test
from pyvinecopulib import Vinecop, Bicop, BicopFamily
from scipy import stats
from python_src.forecast_combination import fit_slp, fit_tlp, fit_blp, evaluate_tlp, evaluate_blp, evaluate_slp, \
    fit_emos_norm, evaluate_emos_norm
from python_src.ccd import estimate_ccd_copula, evaluate_ccd


def generate_data(copula: Union[Bicop, Vinecop], margins: np.ndarray[stats.rv_continuous], n: int):
    """ Generate n points following the specified copula and marginal distributions

    :param copula: the d-variate copula
    :param margins: list with the marginal distributions
    :param n: number of points to generate
    :return: nxd-matrix with observations in the xy-space
    """
    uv_data = copula.simulate(n=n)
    xy_data = np.zeros_like(uv_data)
    for i in range(uv_data.shape[1]):
        xy_data[:, i] = margins[i].ppf(uv_data[:, i])
    return xy_data


def run_ccd_simulation_wrong_margin(true_copula, true_margins: np.ndarray[stats.rv_continuous],
                                    test_margins: np.ndarray[stats.rv_continuous], n_train_margin: int,
                                    n_train_copula: int, n_test: int, export_path: str = None):
    """Run the ccd simulations where the margins used for fitting the copula are different from the true margins."""
    temp_name_run = f'sim_run_{datetime.now():%Y%m%d_%H%M%S}'
    # Generate data
    eps = generate_data(copula=true_copula, margins=true_margins, n=n_train_margin + n_train_copula + n_test)

    eps_train_margin = eps[:n_train_margin, :]
    eps_train_copula = eps[n_train_margin:n_train_margin + n_train_copula, :]
    eps_test = eps[n_train_margin + n_train_copula:, :]

    for (i_m, m) in test_margins:
        m.fit(eps_train_margin[:, i_m])

    # Fit CCD
    bicop_estimate = estimate_ccd_copula(eps_train_copula, test_margins)
    # bicop_estimate = true_copula

    ccd_results = evaluate_ccd(bicop_estimate, test_margins, eps_test)

    # Save results
    if export_path is not None:
        makedirs(export_path, exist_ok=True)
        bicop_estimate.to_json(join(export_path, f'{temp_name_run}-copula-estimate.json'))
        with open(join(export_path, f'{temp_name_run}-ccd-results.pickle'), 'wb') as f:
            pickle.dump(ccd_results, f)

    result = pd.DataFrame(summary_from_dict(ccd_results), index=[0])
    result.loc[0, 'copula_family_fit'] = bicop_estimate.family
    result.loc[0, 'copula_tau_fit'] = bicop_estimate.tau
    return result


def run_ccd_simulation(true_copula, margins: np.ndarray[stats.rv_continuous], n_train: int, n_test: int,
                       export_path: str = None):
    """Run simulation with only CCD fit."""
    temp_name_run = f'sim_run_{datetime.now():%Y%m%d_%H%M%S}'
    # Generate data
    eps = generate_data(copula=true_copula, margins=margins, n=n_train + n_test)

    eps_train = eps[:n_train, :]
    eps_test = eps[n_train:, :]

    # Fit CCD
    bicop_estimate = estimate_ccd_copula(eps_train, margins)

    ccd_results = evaluate_ccd(bicop_estimate, margins, eps_test)

    # Save results
    if export_path is not None:
        makedirs(export_path, exist_ok=True)
        bicop_estimate.to_json(join(export_path, f'{temp_name_run}-copula-estimate.json'))
        with open(join(export_path, f'{temp_name_run}-ccd-results.pickle'), 'wb') as f:
            pickle.dump(ccd_results, f)

    result = pd.DataFrame(summary_from_dict(ccd_results), index=[0])
    result.loc[0, 'copula_family_fit'] = bicop_estimate.family
    result.loc[0, 'copula_tau_fit'] = bicop_estimate.tau
    return result


def run_simulation(true_copula, margins: np.ndarray[stats.rv_continuous], n_train: int, n_test: int, export_path: str):
    """Run simulation with all methods."""
    makedirs(export_path, exist_ok=True)
    temp_name_run = f'sim_run_{datetime.now():%Y%m%d_%H%M%S}'
    # Generate data
    obs = stats.randint(1, 100).rvs((n_train + n_test, 1))
    eps = generate_data(copula=true_copula, margins=margins, n=n_train + n_test)
    forecasts = eps + obs

    obs_train = obs[:n_train, :]
    eps_train = eps[:n_train, :]
    forecasts_train = forecasts[:n_train, :]
    obs_test = obs[n_train:, :]
    eps_test = eps[n_train:, :]
    forecasts_test = forecasts[n_train:, :]

    # Fit CCD
    vinecop_estimate = estimate_ccd_copula(eps_train, margins)
    # vinecop_estimate = true_copula

    # Fit TLP, SLP, BLP
    tlp_fit = fit_tlp(eps_train, margins)
    slp_fit = fit_slp(eps_train, margins)
    blp_fit = fit_blp(eps_train, margins)
    emos_fit = fit_emos_norm(forecasts_train, obs_train)

    # Fit EMOS/BMA (using R(?))

    # Evaluate results
    ccd_results = evaluate_ccd(vinecop_estimate, margins, eps_test)
    tlp_results = evaluate_tlp(eps_test, margins, tlp_fit)
    slp_results = evaluate_tlp(eps_test, margins, slp_fit)
    blp_results = evaluate_tlp(eps_test, margins, blp_fit)
    emos_results = evaluate_emos_norm(forecasts_test, observations=obs_test, param_dict=emos_fit)

    # Save results
    vinecop_estimate.to_json(join(export_path, f'{temp_name_run}-copula-estimate.json'))
    with open(join(export_path, f'{temp_name_run}-ccd-results.pickle'), 'wb') as f:
        pickle.dump(ccd_results, f)
    with open(join(export_path, f'{temp_name_run}-tlp-results.pickle'), 'wb') as f:
        pickle.dump(tlp_results, f)
    with open(join(export_path, f'{temp_name_run}-slp-results.pickle'), 'wb') as f:
        pickle.dump(slp_results, f)
    with open(join(export_path, f'{temp_name_run}-blp-results.pickle'), 'wb') as f:
        pickle.dump(blp_results, f)
    with open(join(export_path, f'{temp_name_run}-emos-results.pickle'), 'wb') as f:
        pickle.dump(blp_results, f)

    return pd.DataFrame({name: summary_from_dict(eval_dict) for (name, eval_dict) in
                         zip(['tlp', 'slp', 'blp', 'emos', 'ccd'],
                             [tlp_results, slp_results, blp_results, emos_results, ccd_results])}).T


def summary_from_dict(eval_dict: Dict):
    """ Return a summary of evaluation measures for the simulation result in eval dict

    :param eval_dict: dict with keys 'rmse', 'mae', 'mean_log_score', and 'pit'
    :return: dict with 'rmse', 'mae', 'mean_log_score', 'ad_stat', and 'ks_stat'
    """
    return ({
        'rmse': eval_dict['rmse'],
        'mae': eval_dict['mae'],
        'mean_log_score': eval_dict['mean_log_score'],
        'ad_stat': ad_test(eval_dict['pit']),
        'ks_stat': ks_test(eval_dict['pit'])
    })


class SimRun:
    def __init__(self, copula_fam, copula_tau, margins, margin_name, n_train, n_test, n_runs):
        self.copula_fam = copula_fam
        self.copula_tau = copula_tau
        self.margins = margins
        self.margin_name = margin_name
        self.n_train = n_train
        self.n_test = n_test
        self.n_runs = n_runs
        self.bicop_fam = BicopFamily.__members__[copula_fam]
        self.cop_param = Bicop(self.bicop_fam).tau_to_parameters(copula_tau)

    def __str__(self):
        return f'{self.copula_fam}_tau_{self.copula_tau}_{self.margin_name}_n_train_{self.n_train}_n_test_{self.n_test}'
