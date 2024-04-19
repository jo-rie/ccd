from typing import Union, List

import numpy as np
from pyvinecopulib import Vinecop, Bicop, FitControlsBicop, BicopFamily
from scipy import integrate
from tqdm import tqdm, trange
import pyvinecopulib as pv

TRAPZ_N_DEFAULT = 10000
bicop_family_set = pv.parametric.copy()
bicop_family_set.remove(pv.bb7)


def estimate_ccd_copula(eps_data: np.ndarray,
                        margins: Union[np.ndarray, List, None] = None,
                        use_ranks: bool = False,
                        **vinecopcontrol: str) -> Bicop:
    """ Estimate ccd copula

    :param eps_data: the error data
    :param margins: List or np.ndarray of marginal distributions with cdf method
    :param vinecopcontrol: Named arguments to be passed on the Vinecop fitting
    :return:
    """
    if use_ranks:
        NotImplementedError('use_ranks is not yet implemented')
    eps_pit_data = np.zeros_like(eps_data)
    for i in range(len(margins)):
        eps_pit_data[:, i] = margins[i].cdf(eps_data[:, i])
    if eps_data.shape[1] == 2:
        vinecop_fit = Bicop()
        vinecop_fit.select(data=eps_pit_data,
                         controls=FitControlsBicop(family_set=bicop_family_set))  # for bb7 the pdf is unstable
    else:
        vinecop_fit = Vinecop()
        vinecop_fit.select_all(eps_pit_data, family_set=bicop_family_set, **vinecopcontrol)
    return vinecop_fit


def evaluate_single_ccd(copula: Vinecop, margins: Union[np.ndarray, List, None], eps_data: np.ndarray,
                        use_quad: bool = False):
    """ Evaluate a single ccd forecast based on the arguments

    :param eps_data: 1xd-matrix with the realized errors per forecast
    :param copula: Vinecop object modeling the joint error dependence of the d forecasts
    :param margins: list of length d with the marginal distributions
    :param use_quad: use scipy integrate's quad to compute integrals. If False, the trapezoidal method is used that is much faster

    :return: list with keys 'mean', 'pit', 'log_score',
    """
    if (eps_data.ndim > 1) & (eps_data.shape[0] > 1):
        ValueError('eps_data is supposed to have only one row')
    eps_data = eps_data.flatten()

    integration_boundaries = (
        np.min([eps_data[i_m] - 5 * np.sqrt(m.stats('v')) for (i_m, m) in enumerate(margins)] + [0]),
        np.max([eps_data[i_m] + 5 * np.sqrt(m.stats('v')) for (i_m, m) in enumerate(margins)] + [0]))

    def vectorize_cdf(x: Union[float, np.ndarray]):
        return np.column_stack([m.cdf(- x + eps_data[i_m]) for (i_m, m) in enumerate(margins)])

    def vectorize_pdf(x: Union[float, np.ndarray]):
        return np.column_stack([m.pdf(- x + eps_data[i_m]) for (i_m, m) in enumerate(margins)])

    def ccd_likeli(x: float):
        return vectorize_pdf(x).prod(axis=1) * copula.pdf(vectorize_cdf(x))

    if use_quad:
        left_integral = quad_with_increasing_grid(ccd_likeli, integration_boundaries[0], 0)
        right_integral = quad_with_increasing_grid(ccd_likeli, 0, integration_boundaries[1])
    else:
        left_integral = func_trapz(ccd_likeli, integration_boundaries[0], 0)
        right_integral = func_trapz(ccd_likeli, 0, integration_boundaries[1])
    normalization_constant = left_integral + right_integral
    pit = left_integral / normalization_constant
    if (pit < 0) | (pit > 1):
        print(f'PIT not in [0, 1]: left={left_integral}, right={right_integral}, pit={pit}')

    def ccd_pdf(x: float):
        return ccd_likeli(x) / normalization_constant

    def ccd_cdf(x: float):
        return quad_with_increasing_grid(ccd_likeli, integration_boundaries[0], x) / normalization_constant

    # Compute summary statistics
    # pit = ccd_cdf(0)
    if use_quad:
        mean = quad_with_increasing_grid(lambda x: ccd_pdf(x) * x, integration_boundaries[0],
                                         integration_boundaries[1])
    else:
        mean = func_trapz(lambda x: ccd_pdf(x) * x, integration_boundaries[0],
                          integration_boundaries[1])
    log_score = - np.log(ccd_pdf(0))

    return {
        'marginal_pdf': vectorize_pdf, 'marginal_cdf': vectorize_cdf, 'normalization_constant': normalization_constant,
        'ccd_likeli': ccd_likeli, 'ccd_pdf': ccd_pdf, 'ccd_cdf': ccd_cdf,
        'pit': pit, 'mean': mean, 'log_score': log_score
    }


def evaluate_ccd(copula: Vinecop, margins: Union[np.ndarray, List, None], eps_data: np.ndarray,
                 disable_progress: bool = False):
    """ Evaluate ccd forecast based on the arguments

    :param eps_data: nxd-matrix with the realized errors per forecast
    :param copula: Vinecop object modeling the joint error dependence of the d forecasts
    :param margins: list of length d with the marginal distributions

    :return: list with keys 'means', 'log_scores', 'pits', 'rmse', 'mae', and 'mean_log_score'
    """
    n = eps_data.shape[0]

    mean_array = np.zeros(n)
    log_score_array = np.zeros(n)
    pit_array = np.zeros(n)

    for i in trange(n, desc='Computing CCD: ', disable=disable_progress):
        single_result = evaluate_single_ccd(copula, margins, eps_data[i, :])

        # Compute summary statistics
        pit_array[i] = single_result['pit']
        mean_array[i] = single_result['mean']
        log_score_array[i] = single_result['log_score']

    rmse = np.sqrt(np.mean(mean_array ** 2))
    mae = np.mean(np.abs(mean_array))

    return {
        'means': mean_array, 'log_scores': log_score_array, 'pit': pit_array,
        'rmse': rmse, 'mae': mae, 'mean_log_score': log_score_array.mean()
    }


def func_trapz(func, a: float, b: float, n_steps: int = TRAPZ_N_DEFAULT):
    """ Compute the integral of func from a to b using n_steps and the trapezoidal rule

    :param func: function to integrate
    :param a: left bound
    :param b: right bound
    :param n_steps: number of steps
    :return: integral estimate
    """
    grid = np.linspace(a, b, n_steps)
    # grid_func_vals = np.zeros_like(grid)
    # for i in range(len(grid_func_vals)):
    grid_func_vals = func(grid)
    return integrate.trapz(grid_func_vals, grid)


def quad_with_increasing_grid(func, a: float, b: float) -> float:
    """ Integrate function with increasing number of subdivisions if the integration did not succeed in the standard setting

    :param func: function to integrate
    :param a: left boundary
    :param b: right boundary
    :return: value of numerical integral
    """
    int_result = integrate.quad(func, a, b, full_output=1)
    if len(int_result) == 4:
        int_result = integrate.quad(func, a, b, full_output=1, limit=100)
        if len(int_result) == 4:
            int_result = integrate.quad(func, a, b, full_output=1, limit=1000)
            if len(int_result) == 4:
                int_result_left = integrate.quad(func, a, np.min([b, 0]), full_output=1, limit=1000)
                int_result_right = integrate.quad(func, np.min([b, 0]), b, full_output=1, limit=1000)
                if len(int_result_left) == 4 | len(int_result_right) == 4:
                    return np.nan
                else:
                    return int_result_left[0] + int_result_right[0]
    return int_result[0]
