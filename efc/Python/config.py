from os.path import join
from os import makedirs

import pandas as pd

import sys
sys.path.append('Python')

base_path = '..'
nn_evaluation_path = 'data_nn_evaluation'
makedirs(nn_evaluation_path, exist_ok=True)

naive_path = 'prob_naive'
makedirs(join(base_path, naive_path), exist_ok=True)

model_results = 'data_paper'
makedirs(join(base_path, model_results), exist_ok=True)

kde_raw_data = 'data_margins_kde'

end_evaluation = pd.to_datetime('2020-12-31 23:00:00')
start_evaluation = end_evaluation - pd.Timedelta(days=553, hours=23)

start_eval_all = pd.to_datetime('2020-12-01 00:00:00')
end_eval_all = end_evaluation


nb_samples = 10000 # M in paper
D_naive_default = 1456
