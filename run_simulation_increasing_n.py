"""
Run CCD simulation for increasing number of samples in (copula) train set for various copulas and
"""
import pickle
from multiprocessing import Pool
from os.path import join

from python_src.simulation_utils import run_ccd_simulation, SimRun

import numpy as np
import pandas as pd
from pyvinecopulib import Bicop, BicopFamily, Vinecop
from scipy import stats
from tqdm import tqdm

export_root = 'results_increasing_n_subset_simruns'

u_var1 = stats.uniform(-np.sqrt(12) / 2, np.sqrt(12))
t_var1 = stats.t(loc=0, scale=np.sqrt(1 / 2), df=4)
# n_train_array = [50, 100, 200, 400, 800, 1000, 2000, 4000, 8000]
# n_train_array = [50, 100, 500, 1000, 5000]
n_train_array = [10, 25, 75, 200, 500]
n_test = int(1e6)
n_runs = 50

# cop_fam_array = ['gaussian', 'gumbel', 'clayton', 'frank']
cop_fam_array = ['gaussian', 'gumbel']

# Generate the simulation runs
sim_runs = []
for (margins, margin_name) in zip([np.array([stats.norm(), stats.norm()]),
                                   np.array([u_var1, u_var1]),
                                   np.array([t_var1, t_var1])], ['N', 'U', 't']):
    for n_train in n_train_array:

        sim_runs.append(
            SimRun(copula_fam='indep', copula_tau=0, margins=margins, margin_name=margin_name, n_train=n_train,
                   n_test=n_test, n_runs=n_runs))
        for tau in [0.4, 0.8]:
            for cop_fam in cop_fam_array:
                sim_runs.append(SimRun(copula_fam=cop_fam, copula_tau=tau, margins=margins, margin_name=margin_name,
                                       n_train=n_train, n_test=n_test, n_runs=n_runs))


def run_simRun(simrun):
    """Function to run a single simulation run."""
    print(f'Starting {simrun}')
    export_path = join(export_root, str(simrun))
    results_list = []
    copula = Bicop(family=simrun.bicop_fam, parameters=simrun.cop_param)
    for i in range(simrun.n_runs):
        result_new = run_ccd_simulation(copula, simrun.margins, simrun.n_train, simrun.n_test,
                                        export_path=export_path)
        result_new['run'] = i
        results_list.append(result_new)
    result_combined = pd.concat(results_list, ignore_index=True)
    result_combined.to_pickle(join(export_root, f'{simrun}.pickle'))
    result_combined.to_csv(join(export_root, f'{simrun}.csv'))


if __name__ == '__main__':
    # Run the simulations in parallel
    pool = Pool(4)

    for _ in tqdm(pool.map(
            run_simRun, sim_runs
    ), total=len(sim_runs)):
        pass

    with open(join(export_root, f'simulation_overview.pickle'), 'wb') as f:
        pickle.dump(sim_runs, f)
