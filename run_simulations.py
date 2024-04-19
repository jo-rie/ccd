"""
Script for running 2D simulations comparing the ccd method to tlp, slp and blp (using the true margins) and with emos and bma.
"""
import pickle
from multiprocessing import Pool
from os import makedirs
# from pathos.multiprocessing import ProcessingPool as Pool
from os.path import join

import numpy as np
import pandas as pd
from pyvinecopulib import Bicop
from scipy import stats
from tqdm import tqdm

from python_src.simulation_utils import run_simulation, SimRun

export_postfix = ''

u_var1 = stats.uniform(-np.sqrt(12) / 2, np.sqrt(12))
t_var1 = stats.t(loc=0, scale=np.sqrt(1 / 2), df=4)
n_train = 5000
n_test = 1000
n_runs = 100


# Create Simulation runs
sim_runs = []
for (margins, margin_name) in zip([np.array([stats.norm(), stats.norm()]),
                                   np.array([u_var1, u_var1]),
                                   np.array([t_var1, t_var1])], ['N', 'U', 't']):
    sim_runs.append(SimRun(copula_fam='indep', copula_tau=0, margins=margins, margin_name=margin_name, n_train=n_train,
                           n_test=n_test, n_runs=n_runs))
    for tau in [0.4, 0.8]:
        for cop_fam in ['gaussian', 'gumbel', 'clayton', 'frank']:
        # for cop_fam in ['gaussian', 'gumbel']:
            sim_runs.append(SimRun(copula_fam=cop_fam, copula_tau=tau, margins=margins, margin_name=margin_name,
                                   n_train=n_train, n_test=n_test, n_runs=n_runs))


def run_simRun(simrun):
    """Function for a single simulation run."""
    export_path = join('results' + export_postfix, str(simrun))
    results_list = []
    copula = Bicop(family=simrun.bicop_fam, parameters=simrun.cop_param)
    for i in range(simrun.n_runs):
        result_new = run_simulation(copula, simrun.margins, simrun.n_train, simrun.n_test,
                                    export_path=export_path)
        result_new['run'] = i
        results_list.append(result_new)
    result_combined = pd.concat(results_list)
    result_combined.to_pickle(join(export_path, f'{simrun}.pickle'))
    result_combined.to_csv(join(export_path, f'{simrun}.csv'))


if __name__ == '__main__':
    # Run the simulations in parallel.
    exp_path = join('results' + export_postfix)
    makedirs(exp_path, exist_ok=True)

    pool = Pool(4)

    for _ in tqdm(pool.map(
            run_simRun, sim_runs
            ), total=len(sim_runs)):
        pass
    # for simrun in tqdm(sim_runs):
    #     run_simRun(simrun)

    with open(join(exp_path, 'simulation_overview.pickle'), 'wb') as f:
        pickle.dump(sim_runs, f)