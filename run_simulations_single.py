"""
Script for running 2D simulations comparing the ccd method to tlp, slp and blp (using the true margins) and with emos and bma.
"""
from datetime import datetime
from os.path import join

import numpy as np
from pyvinecopulib import Bicop, BicopFamily, Vinecop
from scipy import stats

from python_src.simulation_utils import run_simulation, run_ccd_simulation

## Set up parameters

true_copula_family = 'gumbel'
true_copula_bicop_family = BicopFamily.__members__[true_copula_family]
true_copula_tau = 0.7
n_train = 5000
n_test = 100

true_copula = Bicop(
    family=true_copula_bicop_family,
    parameters=Bicop(true_copula_bicop_family).tau_to_parameters(true_copula_tau)
)
margins = np.array([stats.norm(), stats.norm()])

# result = run_simulation(true_copula, margins, n_train, n_test, export_path=join(f'results', f'{datetime.now():%Y%m%d_%H%M%S}'))

result = run_ccd_simulation(true_copula, margins, n_train, n_test)
print(result)