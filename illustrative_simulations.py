"""
Script to create PIT histograms for missspecifications

- wrong margin: whats happens if one uses a normal instead of the (true) t distribution for marginal modelling?
- wrong copula: which copula types behave well for misspecified copula?
"""
import pickle
from multiprocessing import Pool
from os import makedirs
from os.path import join
from datetime import datetime

# Imports
import numpy as np
import scipy.stats
from pyvinecopulib import Bicop, BicopFamily
from scipy import stats
from tqdm import tqdm

from python_src.ccd import evaluate_ccd
from python_src.simulation_utils import generate_data

n_test = int(1e4)
cop_families = ['gaussian', 'gumbel', 'clayton', 'frank']
bicop_families = [getattr(BicopFamily, f) for f in cop_families]

# %% Wrong margin


t_dist = stats.t(loc=0, scale=np.sqrt(1 / 2), df=4)
norm_dist = stats.norm(loc=0, scale=1)
true_margin = np.array([t_dist, t_dist])
fit_margin = np.array([norm_dist, norm_dist])
data_export_root_wrong_tau = f'{datetime.now():%Y%m%d-%H%M}-results_wrong_margin'


def run_for_tau(true_tau):
    overall_result = {}
    for (true_family, tf_name) in zip(bicop_families, cop_families):
        # create data
        true_copula = Bicop(
            family=true_family,
            parameters=Bicop(true_family).tau_to_parameters(true_tau)
        )
        eps = generate_data(copula=true_copula, margins=true_margin, n=n_test)
        # fit_array = [stats.norm.fit(eps[:, i]) for i in range(2)]
        # stats_norm_array = [stats.norm(loc=fit_array[i][0], scale=fit_array[i][1]) for i in range(2)]
        res_dict = evaluate_ccd(copula=true_copula, margins=fit_margin, eps_data=eps)
        overall_result[f'true_{tf_name}'] = res_dict
    with open(join(data_export_root_wrong_tau, f'results_tau_{true_tau}.pickle'), 'wb') as f:
        pickle.dump(overall_result, f, protocol=pickle.HIGHEST_PROTOCOL)


# %% Wrong copula

data_export_root_wrong_copula = f'{datetime.now():%Y%m%d-%H%M}-results_wrong_copula'
u_var1 = stats.uniform(-np.sqrt(12) / 2, np.sqrt(12))
t_var1 = stats.t(loc=0, scale=np.sqrt(1 / 2), df=4)
# Grid plot of PIT histograms; columns: true copula; rows: estimated copula
config_array_wrong_copula = []
for (margins, margin_name) in zip([np.array([stats.norm(), stats.norm()]),
                                   np.array([u_var1, u_var1]),
                                   np.array([t_var1, t_var1])], ['N', 'U', 't']):
    for true_tau in [.2, .4, .6, .7, .8, .9, .95]:
        config_array_wrong_copula.append((margins, margin_name, true_tau))


def run_for_margin_and_name(margin_name_tau_tuple):
    margins = margin_name_tau_tuple[0]
    margin_name = margin_name_tau_tuple[1]
    true_tau = margin_name_tau_tuple[2]
    temp_name = f'm_{margin_name}_tau_{true_tau}'
    overall_result = {}
    for (true_family, tf_name) in zip(bicop_families, cop_families):
        # create data
        true_copula = Bicop(
            family=true_family,
            parameters=Bicop(true_family).tau_to_parameters(true_tau)
        )
        # print(f'true {true_copula.str()}')
        eps = generate_data(copula=true_copula, margins=margins, n=n_test)
        # Evaluate data with all other copulas
        for (fit_family, fit_name) in zip(bicop_families, cop_families):
            fit_copula = Bicop(
                family=fit_family,
                parameters=Bicop(fit_family).tau_to_parameters(true_tau)
            )
            res_dict = evaluate_ccd(copula=fit_copula, margins=margins, eps_data=eps, disable_progress=True)
            # print(f'fit {fit_copula.str()}; KS: {ks_test(res_dict["pit"]):.4f}')
            overall_result[f'true_{tf_name}_fit_{fit_name}'] = res_dict
    with open(join(data_export_root_wrong_copula, f'{temp_name}.pickle'), 'wb') as f:
        pickle.dump(overall_result, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    pool = Pool(6)
    print('Starting wrong margin...')
    makedirs(data_export_root_wrong_tau, exist_ok=True)
    for _ in tqdm(pool.map(
            run_for_tau, [0.4, 0.8]
    ), total=2):
        pass
    print('Finished')
    print('Starting wrong copula...')
    makedirs(data_export_root_wrong_copula, exist_ok=True)
    print('\n'.join([f'({name} - {tau})' for (_, name, tau) in config_array_wrong_copula]))
    for _ in tqdm(pool.map(
            run_for_margin_and_name, config_array_wrong_copula
    ), total=len(config_array_wrong_copula)):
        pass

    # run_for_margin_and_name(config_array[1])

    print('Finished')
