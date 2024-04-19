from datetime import datetime
from os import makedirs
from os.path import join

import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator
from tqdm import tqdm
import seaborn as sns

from plot_utils import fig_with_size, initialize_mpl, plot_pit_normal, plot_pit_jsu, save_fig, pair_error_plot
from efc.Python.plot_utils import pit_plot
from config import start_evaluation, end_evaluation, base_path

from utils import ks_test, ad_test, restrict_df_to_dates

plot_output_path = '../paper_plots'
makedirs(plot_output_path, exist_ok=True)

begin_test = datetime(2020, 1, 1)
end_test = datetime(2020, 12, 31)

initialize_mpl()


def rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())


def mae(y_pred, y_true):
    return np.abs(y_pred - y_true).mean()


# %% Plot pair plots of (raw) forecast errors

pair_error_plot(join(plot_output_path, 'joint_error_plot_normal.pdf'), dist_type='Normal')
pair_error_plot(join(plot_output_path, 'joint_error_plot_jsu.pdf'), dist_type='JSU')

# %% Structure of analysis for all models:
# Add tuple (path-to-pickle, plot-name, name-in-table)
# Iterate over tuples: plot pit, add analysis to results

model_list = []
df_list = {}

for dist_number in range(1, 5): model_list.append(
    (f'../data_paper/ddnn_jsu_{dist_number}.pickle', f'ddnn_jsu_{dist_number}', f'DDNN-JSU-{dist_number}'))
for dist_number in range(1, 5): model_list.append(
    (f'../data_paper/ddnn_normal_{dist_number}.pickle', f'ddnn_normal_{dist_number}', f'DDNN-Normal-{dist_number}'))
for dist_number in range(1, 5): model_list.append(
    (f'../data_paper/kde_jsu_{dist_number}.pickle', f'kde_jsu_{dist_number}', f'KDE-JSU-{dist_number}'))
for dist_number in range(1, 5): model_list.append(
    (f'../data_paper/kde_normal_{dist_number}.pickle', f'kde_normal_{dist_number}', f'KDE-Normal-{dist_number}'))

model_list += [
    ('../data_paper/ddnn_jsu_lp.pickle', 'jsu_lp', 'JSU-TLP'),
    ('../data_paper/ddnn_normal_lp.pickle', 'normal_lp', 'Normal-TLP'),
    ('../results_ccd_20230104/data_ccd_all_data_fit.pickle', 'ccd_all', 'CCD-All'),
    ('../results_ccd_20230104/data_ccd_hourly_all_data.pickle', 'ccd_hourly', 'CCD-Hourly'),
    ('../results_ccd_20230104/data_ccd_hourly_last_year.pickle', 'ccd_hourly_1y', 'CCD-1Y-Hourly'),
    ('../results_ccd_20230104/data_ccd_last_year_fit.pickle', 'ccd_1y', 'CCD-1Y'),
    ('../results_ccd_20240123_margins_13_models_12/data_ccd_all_data_fit.pickle', 'ccd_13_all', 'CCD-13-All'),
    ('../results_ccd_20240123_margins_13_models_12/data_ccd_last_year_fit.pickle', 'ccd_13_1y', 'CCD-13-1Y'),
    ('../results_ccd_20240124_margins_14_models_12/data_ccd_all_data_fit.pickle', 'ccd_14_all', 'CCD-14-All'),
    ('../results_ccd_20240124_margins_14_models_12/data_ccd_last_year_fit.pickle', 'ccd_14_1y', 'CCD-14-1Y'),
]


def evaluate_single_model(path: str, plot_name: str, table_name: str):
    # Plot PIT
    df = pd.read_pickle(path)
    df = restrict_df_to_dates(df, start_date=begin_test, end_date=end_test)
    fig, ax = fig_with_size(4, nb_vertically=4)
    pit_plot(df['pit'], filepath=join(plot_output_path, f'{plot_name}_pit.pdf'), ax=ax)
    df_list[table_name] = df


def evaluate_bicop(path, plot_name, table_name):
    print(plot_name)
    path_bicop = path[:-7] + '_bicops' + path[-7:]
    df = pd.read_pickle(path_bicop)
    df['family'] = df['family'].apply(lambda x: x.name)
    # print('Rotation:')
    # print(df['rotation'].value_counts())
    # print('Family:')
    # print(df['family'].value_counts())
    # print(f"Rotation != 0: {df.index[df['rotation'] != 0]} ({df['rotation'][df['rotation'] != 0]}), "
    #       f"Family != student: {df.index[df['family'] != 'student']} ({df['family'][df['family'] != 'student']})")
    # Rotation 180 with bb1 copula for 2020-01-25 and (1, 4)
    # df.reset_index(inplace=True)
    fig, ax = fig_with_size(2)
    df['tau'].plot(ax=ax)
    # sns.scatterplot(df, x='index', y='tau', hue='family', size='rotation', sizes={0: 0.5, 180: 2})
    ax.set(xlabel='Date', ylabel=r'$\hat{\tau}$', ylim=(0, 1))
    # Turn off minor ticks
    ax.xaxis.set_minor_locator(NullLocator())
    save_fig(fig, join(plot_output_path, f'{plot_name}_bicop.pdf'))


for (path, plot_name, table_name) in model_list:
    evaluate_single_model(path, plot_name, table_name)

for (path, plot_name, table_name) in model_list[-4:]:
    evaluate_bicop(path, plot_name, table_name)

# %% Compute Scores for all measures and store them

df_scores = pd.DataFrame(columns=['MAE', 'RMSE', 'CRPS', 'KS', 'AD'], index=list(df_list.keys()))

for model_name in df_list.keys():
    pits = np.array(df_list[model_name]['pit'].values, dtype=np.float32)
    df_scores.loc[model_name, ['MAE', 'RMSE', 'CRPS', 'KS', 'AD']] = [
        mae(df_list[model_name]['obs'], df_list[model_name]['mean']),
        rmse(df_list[model_name]['obs'], df_list[model_name]['mean']),
        df_list[model_name]['crps'].mean(),
        ks_test(pits),
        ad_test(pits)
    ]

print(df_scores.to_string(float_format=lambda x: f"{x:.4f}"))

with open(join(plot_output_path, 'scores.tex'), 'wt') as f:
    df_scores.drop(columns=['AD']).to_latex(f, float_format="%.4f")
