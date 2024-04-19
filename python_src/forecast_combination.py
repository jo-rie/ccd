from typing import List, Union, Dict

import numpy as np
from scipy import optimize, stats, integrate


def fit_tlp(eps_data: np.ndarray, margins: Union[List, np.ndarray], score: str = 'log_score') -> Dict:
    """ Fit TLP weight to the error data in eps with the specified margins

    :param eps_data: Historical error data (nxd-matrix)
    :param margins: list of marginal distribution (mean 0)
    :param score: the score to be used, only implemented: 'log_score'
    :return: dict with parameter 'linear_factors'
    """
    n, d = eps_data.shape
    if score == 'log_score':
        score_from_k = lambda k: pdf2logscore(tlp_pdf(eps_data, margins, k)).mean()
    else:
        raise NotImplementedError(f'Scoring rule {score} not implemented')

    result = optimize.minimize(score_from_k, ((1 / d,) * d), bounds=(((0, 1),) * d),
                               constraints=({'type': 'eq', 'fun': lambda x: sum(x) - 1}))

    return {'linear_factors': result['x']}


def fit_blp(eps_data: np.ndarray, margins: Union[List, np.ndarray], score: str = 'log_score') -> Dict:
    """ Fit BLP parameters to the error data in eps with the specified margins

    :param eps_data: Historical error data (nxd-matrix)
    :param margins: list of marginal distribution (mean 0)
    :param score: the score to be used, only implemented: 'log_score'
    :return: dict with parameter 'linear_factors', 'beta_a' and 'beta_b'
    """
    n, d = eps_data.shape
    if score == 'log_score':
        score_from_k = lambda k: pdf2logscore(blp_pdf(eps_data, margins, k[:d], k[d], k[d + 1])).mean()
    else:
        raise NotImplementedError(f'Scoring rule {score} not implemented')

    result = optimize.minimize(score_from_k, (1 / d,) * d + (1, 1),
                               bounds=(((0, 1),) * d) + ((0, None), (0, None)),
                               constraints=({'type': 'eq', 'fun': lambda x: sum(x[:d]) - 1}))

    return {'linear_factors': result['x'][:d], 'beta_a': result['x'][d], 'beta_b': result['x'][d + 1]}


def fit_slp(eps_data: np.ndarray, margins: Union[List, np.ndarray], score: str = 'log_score') -> Dict:
    """ Fit SLP parameters to the error data in eps with the specified margins

    :param eps_data: Historical error data (nxd-matrix)
    :param margins: list of marginal distribution (mean 0)
    :param score: the score to be used, only implemented: 'log_score'
    :return: dict with 'linear_factors' and 'spread_adjustment'
    """
    n, d = eps_data.shape
    if score == 'log_score':
        score_from_k = lambda k: pdf2logscore(slp_pdf(eps_data, margins, k[:d], k[d])).mean()
    else:
        raise NotImplementedError(f'Scoring rule {score} not implemented')

    result = optimize.minimize(score_from_k, (1 / d,) * d + (1, ),
                               bounds=(((0, 1),) * d) + ((0, None),),
                               constraints=({'type': 'eq', 'fun': lambda x: sum(x[:d]) - 1}))

    return {'linear_factors': result['x'][:d], 'spread_adjustment': result['x'][d]}


def emos_pdf_norm(forecast_data: np.ndarray, observations: np.ndarray, intercept: float, linear_factors: np.ndarray, spread_intercept: float, spread_factor: float):
    """ Compute the EMOS pdf values for the given forecast data and parameters

    :param forecast_data: the forecasts
    :param observations: the observed values
    :param intercept: the intercept of the emos mean
    :param linear_factors: linear factors for the combination of forecasts
    :param spread_intercept: constant value of standard deviation
    :param spread_factor: ensemble-variance factor for standard deviation
    :return: array of pdf values
    """
    e_var = np.var(forecast_data, axis=1)
    return stats.norm.pdf(observations, intercept + np.dot(forecast_data, linear_factors), np.sqrt(spread_intercept + e_var * spread_factor))


def fit_emos_norm(forecast_data: np.ndarray, obs: np.ndarray, score: str = 'log_score') -> Dict:
    """

    :param obs: observation data
    :param forecast_data: Historical forecasts (nxd-matrix)
    :param score: the score to be used, only implemented: 'log_score'
    :return:
    """
    n, d = forecast_data.shape

    if score == 'log_score':
        score_from_k = lambda k: pdf2logscore(emos_pdf_norm(forecast_data, obs.flatten(), k[0], k[1:(d + 1)], k[d + 1], k[d+2])).mean()
    else:
        raise NotImplementedError(f'Scoring rule {score} not implemented')

    result = optimize.minimize(score_from_k, (0, ) + (1 / d,) * d + (1,) + (0, ),
                               bounds=((0, None),) + (((0, 1),) * d) + ((0 + 1e-6, None),) + ((0, None),),
                               constraints=({'type': 'eq', 'fun': lambda x: sum(x[1:d+1]) - 1}))

    return {'intercept': result['x'][0], 'linear_factors': result['x'][1:d+1], 'spread_intercept': result['x'][d+1], 'spread_factor': result['x'][d+2]}


def slp_pdf(eps_data: np.ndarray, margins: Union[np.ndarray, List], linear_factors: np.ndarray,
            spread_adjustment: float) -> np.ndarray:
    """ PDF for the SLP

    :param spread_adjustment: spread adjustment parameter in the SLP
    :param eps_data: forecast error data
    :param margins: marginal distributions
    :param linear_factors: linear factors of the SLP
    :return: array with the logarithmic scores
    """
    n, d = eps_data.shape
    return (1 / spread_adjustment *
            np.dot(
                np.column_stack([margins[i_d].pdf(eps_data[:, i_d] / spread_adjustment) for i_d in range(d)]),
                linear_factors
            ))


def slp_cdf(eps_data: np.ndarray, margins: Union[np.ndarray, List], linear_factors: np.ndarray,
            spread_adjustment: float) -> np.ndarray:
    """ CDF for the SLP

    :param spread_adjustment: spread adjustment parameter in the SLP
    :param eps_data: forecast error data
    :param margins: marginal distributions
    :param linear_factors: linear factors of the SLP
    :return: array with the logarithmic scores
    """
    n, d = eps_data.shape
    return (np.dot(
                np.column_stack([margins[i_d].cdf(eps_data[:, i_d] / spread_adjustment) for i_d in range(d)]),
                linear_factors
            ))


def pdf2logscore(pdf_vals: np.ndarray) -> np.ndarray:
    """ Compute logarthmic score from pdf values

    :param pdf_vals: array with pdf values of observations
    :return: array with logarithmic scores
    """
    return - np.log(pdf_vals)


def blp_pdf(eps_data: np.ndarray, margins: Union[np.ndarray, List], linear_factors: np.ndarray, beta_a: float,
            beta_b: float) -> np.ndarray:
    """ PDF values for the BLP

    :param beta_a: parameter a of the beta distribution
    :param beta_b: parameter b of the beta distribution
    :param eps_data: forecast error data
    :param margins: marginal distributions
    :param linear_factors: linear factors of the BLP
    :return: array with the logarithmic scores
    """
    n, d = eps_data.shape
    forecast_pdf = np.zeros_like(eps_data)
    forecast_cdf = np.zeros_like(eps_data)
    for i_d in range(d):
        forecast_pdf[:, i_d] = margins[i_d].pdf(eps_data[:, i_d])
        forecast_cdf[:, i_d] = margins[i_d].cdf(eps_data[:, i_d])
    return np.dot(forecast_pdf, linear_factors) * stats.beta(a=beta_a, b=beta_b).pdf(
        np.dot(forecast_cdf, linear_factors))


def blp_cdf(eps_data: np.ndarray, margins: Union[np.ndarray, List], linear_factors: np.ndarray, beta_a: float,
            beta_b: float) -> np.ndarray:
    """ CDF values for the BLP

    :param beta_a: parameter a of the beta distribution
    :param beta_b: parameter b of the beta distribution
    :param eps_data: forecast error data
    :param margins: marginal distributions
    :param linear_factors: linear factors of the BLP
    :return: array with the logarithmic scores
    """
    n, d = eps_data.shape
    forecast_cdf = np.zeros_like(eps_data)
    for i_d in range(d):
        forecast_cdf[:, i_d] = margins[i_d].cdf(eps_data[:, i_d])
    return stats.beta(a=beta_a, b=beta_b).cdf(
        np.dot(forecast_cdf, linear_factors))


def tlp_pdf(eps_data: np.ndarray, margins: Union[np.ndarray, List], linear_factors: np.ndarray) -> np.ndarray:
    n, d = eps_data.shape
    forecast_pdf_array = np.zeros_like(eps_data)
    for i_d in range(d):
        forecast_pdf_array[:, i_d] = margins[i_d].pdf(eps_data[:, i_d])

    return np.dot(forecast_pdf_array, linear_factors)


def evaluate_tlp(eps_data: np.ndarray, margins: Union[np.ndarray, List], param_dict: Dict) -> Dict:
    n, d = eps_data.shape
    forecast_pdf_array = np.zeros_like(eps_data)
    forecast_cdf_array = np.zeros_like(eps_data)
    for i_d in range(d):
        forecast_pdf_array[:, i_d] = margins[i_d].pdf(eps_data[:, i_d])
        forecast_cdf_array[:, i_d] = margins[i_d].cdf(eps_data[:, i_d])

    mean_array = np.dot(eps_data, param_dict['linear_factors'])
    pit_array = np.dot(forecast_cdf_array, param_dict['linear_factors'])
    log_score_array = pdf2logscore(tlp_pdf(eps_data, margins, param_dict['linear_factors']))

    return {
        'means': mean_array,
        'rmse': np.sqrt(np.mean(mean_array ** 2)),
        'mae': np.mean(np.abs(mean_array)),
        'pit': pit_array,
        'log_score': log_score_array,
        'mean_log_score': np.mean(log_score_array),
        'param': param_dict
    }


def evaluate_slp(eps_data: np.ndarray, margins: Union[np.ndarray, List], param_dict: Dict) -> Dict:
    mean_array = np.zeros(eps_data.shape[0])
    pit_array = slp_cdf(eps_data, margins, param_dict['linear_factors'], param_dict['spread_adjustment'])
    log_score_array = pdf2logscore(
        slp_pdf(eps_data, margins, param_dict['linear_factors'], param_dict['spread_adjustment']))

    for i in range(eps_data.shape[0]):
        mean_array[i] = integrate.quad(lambda x: x * slp_pdf(
            eps_data[i, :], margins,
            param_dict['linear_factors'], param_dict['spread_adjustment']), -np.inf, np.inf)

    return {
        'means': mean_array,
        'rmse': np.sqrt(np.mean(mean_array ** 2)),
        'mae': np.mean(np.abs(mean_array)),
        'pit': pit_array,
        'log_score': log_score_array,
        'mean_log_score': np.mean(log_score_array),
        'param': param_dict
    }


def evaluate_blp(eps_data: np.ndarray, margins: Union[np.ndarray, List], param_dict: Dict) -> Dict:
    mean_array = np.zeros(eps_data.shape[0])
    pit_array = blp_cdf(eps_data, margins, param_dict['linear_factors'], param_dict['beta_a'], param_dict['beta_b'])
    log_score_array = pdf2logscore(
        blp_pdf(eps_data, margins, param_dict['linear_factors'], param_dict['beta_a'], param_dict['beta_b']))

    for i in range(eps_data.shape[0]):
        mean_array[i] = integrate.quad(lambda x: x * blp_pdf(
            eps_data[i, :], margins,
            param_dict['linear_factors'], param_dict['beta_a'], param_dict['beta_b']), -np.inf, np.inf)

    return {
        'means': mean_array,
        'rmse': np.sqrt(np.mean(mean_array ** 2)),
        'mae': np.mean(np.abs(mean_array)),
        'pit': pit_array,
        'log_score': log_score_array,
        'mean_log_score': np.mean(log_score_array),
        'param': param_dict
    }


def evaluate_emos_norm(forecast_data: np.ndarray, observations: np.ndarray, param_dict: Dict) -> Dict:
    observations = observations.flatten()
    e_var = np.var(forecast_data, axis=1)
    mean_array = param_dict['intercept'] + np.dot(forecast_data, param_dict['linear_factors'])
    pit_array = stats.norm.cdf(observations, mean_array, np.sqrt(param_dict['spread_intercept'] + e_var * param_dict['spread_factor']))
    log_score_array = pdf2logscore(
        emos_pdf_norm(forecast_data, observations, param_dict['intercept'], param_dict['linear_factors'], param_dict['spread_intercept'], param_dict['spread_factor']))

    return {
        'means': mean_array,
        'rmse': np.sqrt(np.mean((mean_array - observations) ** 2)),
        'mae': np.mean(np.abs(mean_array - observations)),
        'pit': pit_array,
        'log_score': log_score_array,
        'mean_log_score': np.mean(log_score_array),
        'param': param_dict
    }
