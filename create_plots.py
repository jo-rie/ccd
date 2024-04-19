import pickle
import random
from os import makedirs
from os.path import join
from timeit import Timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pyvinecopulib import BicopFamily, Bicop
from scipy import stats

from python_src.ccd import evaluate_single_ccd
from python_src.plot_utils import initialize_mpl, fig_with_size, save_fig, boxplot_settings, create_and_save_boxplot
from efc.Python.plot_utils import pit_plot
from efc.Python.utils import ks_test
from setup import base_path_code_plots

initialize_mpl()


# %% Test plots
def plots_test():
    def get_pit_sample(n=100):
        import random
        return np.array([random.uniform(0, 1) for i in range(n)])

    pit_vals = get_pit_sample()
    fig, ax = fig_with_size()
    pit_plot(u=pit_vals, ax=ax)
    save_fig(fig, join(base_path_code_plots, 'test_pit_plot.pdf'))

    fig, ax = fig_with_size()
    ax.boxplot(stats.norm().rvs(1000), **boxplot_settings)
    save_fig(fig, join(base_path_code_plots, 'test_boxplot.pdf'))


# %% Simulation Box Plots for comparison
def plots_comparison():
    data_root = 'results'
    base_path_code_plots_tmp = join(base_path_code_plots, 'results_comparison')
    makedirs(base_path_code_plots_tmp, exist_ok=True)
    with open(join(data_root, f'simulation_overview.pickle'), 'rb') as f:
        sim_run_array = pickle.load(f)

    for sim_run in sim_run_array:
        print(f'Starting {sim_run}')
        df_simrun = pd.read_pickle(join(data_root, f'{sim_run}.pickle'))
        df_simrun['method'] = df_simrun.index.str.upper()
        if sim_run.margin_name != 'N': # Exclude EMOS for non-Gaussian margins
            df_simrun = df_simrun[df_simrun['method'] != 'EMOS']
            positions = [3, 4, 2, 1]
        else:
            positions = [3, 5, 4, 2, 1]
        for col in ['rmse', 'mean_log_score', 'ad_stat', 'ks_stat']:
            fig, ax = fig_with_size(4, nb_vertically=5)
            create_and_save_boxplot(df=df_simrun, col=col, by='method', fig=fig, ax=ax,
                                    fname=join(base_path_code_plots_tmp, f'{sim_run}_{col}.pdf'), positions=positions,
                                    xlabel='')


# %% Simulation Box Plots for increasing-n-simulations
def plots_increasing_n():
    data_root = 'results_increasing_n_subset_simruns'
    base_path_code_plots_tmp = join(base_path_code_plots, 'results_increasing_n')
    makedirs(base_path_code_plots_tmp, exist_ok=True)
    with open(join(data_root, f'simulation_overview.pickle'), 'rb') as f:
        sim_run_array = pickle.load(f)

    cop_tau_pair_array = set([(s.bicop_fam, s.copula_tau) for s in sim_run_array])

    for cop_tau in cop_tau_pair_array:
        print(f'\n\n{cop_tau}')
        sim_run_array_loop = [s for s in sim_run_array if (s.bicop_fam == cop_tau[0]) & (s.copula_tau == cop_tau[1])]
        # Read all pandas arrays
        pd_df_array = []
        for s in sim_run_array_loop:
            new_df = pd.read_pickle(join(data_root, f'{s}.pickle'))
            new_df['n'] = s.n_train
            pd_df_array.append(new_df)
            print(f'{s.n_train}:')
            print(new_df.value_counts('copula_family_fit'))
        df_sim_runs = pd.concat(pd_df_array)
        for col in ['rmse', 'mean_log_score', 'ad_stat', 'ks_stat']:
            fig, ax = fig_with_size(4, factor_height=0.8)
            create_and_save_boxplot(df=df_sim_runs, col=col, by='n', fig=fig, ax=ax,
                                    fname=join(base_path_code_plots_tmp,
                                               f'{str(cop_tau[0]).split(".")[-1]}_{cop_tau[1]}_{col}.pdf'))


# %% Plots for wrong margins
def plots_wrong_margin():
    data_export_root = '20230824-1303-results_wrong_margin'
    cop_family_array = ['gaussian', 'gumbel', 'clayton', 'frank']
    base_path_code_plots_tmp = join(base_path_code_plots, 'results_wrong_margin')
    makedirs(base_path_code_plots_tmp, exist_ok=True)
    for true_tau in [.4, .8]:
        with open(join(data_export_root, f'results_tau_{true_tau}.pickle'), 'rb') as f:
            overall_result = pickle.load(f)
        for cop_family in cop_family_array:
            result_dict = overall_result[f'true_{cop_family}']
            fig, ax = fig_with_size(4, factor_height=0.8)
            pit_plot(result_dict['pit'], ax=ax, bins=20)
            ax.set_ylim(0, 2)
            ax.tick_params(axis='x', labelrotation = 45)
            save_fig(fig, join(base_path_code_plots_tmp, f'true_tau_{true_tau}_cop_{cop_family}.pdf'))


# %% Plots for wrong copula
def plots_wrong_copula():
    # data_export_root = '20230824-1303-results_wrong_copula'
    # data_export_root = '20240131-0619-results_wrong_copula'
    # data_export_root = '20240131-2110-results_wrong_copula'
    data_export_root = '20240131-2217-results_wrong_copula'
    cop_family_array = ['frank', 'gaussian', 'clayton', 'gumbel']
    cop_family_array_reverse = cop_family_array.copy()
    cop_family_array_reverse.reverse()
    base_path_code_plots_tmp = join(base_path_code_plots, 'results_wrong_copula')
    makedirs(base_path_code_plots_tmp, exist_ok=True)
    for margin_name in ['N', 'U', 't']:
        for true_tau in [.2, .4, .6, .7, .8, .9, .95]:
            temp_name = f'm_{margin_name}_tau_{true_tau}'
            with open(join(data_export_root, f'{temp_name}.pickle'), 'rb') as f:
                overall_result = pickle.load(f)
            # Create plots with column and row headings
            # ks_array = np.zeros((len(cop_family_array), len(cop_family_array)))
            # ls_array = np.zeros((len(cop_family_array), len(cop_family_array)))
            # rmse_array = np.zeros((len(cop_family_array), len(cop_family_array)))
            fig, axes = fig_with_size(nb_vertically=2, nrows=4, ncols=4, sharey='all')
            for i in range(len(cop_family_array)):
                for j in range(len(cop_family_array)):
                    ax: plt.Axes = axes[i, j]
                    element = f'true_{cop_family_array_reverse[j]}_fit_{cop_family_array[i]}'
                    result_dict = overall_result[element]
                    pit_plot(result_dict['pit'], ax=ax, bins=16)
                    # ks_array[i, j] = ks_test(result_dict['pit'])
                    # ls_array[i, j] = result_dict['mean_log_score']
                    # rmse_array[i, j] = result_dict['rmse']
                    ax.set_ylim(0, 2)
                    # axes[i, j].set_title(element)
            for ax, col in zip(axes[0, :], [f'True: {cop_fam.capitalize()}' for cop_fam in cop_family_array_reverse]):
                ax.set_title(col)
            for ax, row in zip(axes[:, 0], [f'Fit:\n{cop_fam.capitalize()}' for cop_fam in cop_family_array]):
                ax.set_ylabel(row)
            save_fig(fig, join(base_path_code_plots_tmp, f'{temp_name}.pdf'))
            # pd.DataFrame(index=[f'Fit: {c}' for c in cop_family_array],
            #              columns=[f'True: {c}' for c in cop_family_array_reverse],
            #              data=ks_array).to_csv(
            #     join(base_path_code_plots_tmp, f'{temp_name}_ks_stats.csv'), float_format='%.3f')
            # pd.DataFrame(index=[f'Fit: {c}' for c in cop_family_array],
            #              columns=[f'True: {c}' for c in cop_family_array_reverse],
            #              data=ls_array).to_csv(
            #     join(base_path_code_plots_tmp, f'{temp_name}_ls_stats.csv'), float_format='%.3f')
            # pd.DataFrame(index=[f'Fit: {c}' for c in cop_family_array],
            #          columns=[f'True: {c}' for c in cop_family_array_reverse],
            #          data=rmse_array).to_csv(
            # join(base_path_code_plots_tmp, f'{temp_name}_rmse_stats.csv'), float_format='%.3f')


# %% Forecast sample visualization

def plot_forecast_samples():
    cop_family_array = ['frank', 'gaussian', 'clayton', 'gumbel']
    margins = np.array([stats.norm(), stats.norm()])
    bicop_families = [getattr(BicopFamily, f) for f in cop_family_array]
    base_path_code_plots_tmp = join(base_path_code_plots, 'sample_forecasts')
    makedirs(base_path_code_plots_tmp, exist_ok=True)
    DEF_LINEWIDTH = .5
    COP_TAU = .8
    for (i_eps, eps) in enumerate([np.array([-.5, .7]), np.array([1, 1.2])]):
        fig, ax = fig_with_size(nb_horizontally=2)
        xgrid = np.linspace(-3, 3, 1000)
        # plot forecasts
        ax.axvline(eps[0], label=r'$\hat{x}_1$', color='grey', linestyle='dashed', linewidth=DEF_LINEWIDTH)
        ax.axvline(eps[1], label=r'$\hat{x}_2$', color='grey', linestyle='dotted', linewidth=DEF_LINEWIDTH)
        for i in range(2):
            ax.plot(xgrid, stats.norm(loc=eps[i]).pdf(xgrid), label=r'$f^' + f'{i + 1}' + r'$',
                    linewidth=DEF_LINEWIDTH / 2)
        for (cop_name, bicop_fam) in zip(cop_family_array, bicop_families):
            res_dict = evaluate_single_ccd(
                Bicop(family=bicop_fam, parameters=Bicop(bicop_fam).tau_to_parameters(COP_TAU)), margins=margins,
                eps_data=eps)
            ax.plot(xgrid, res_dict['ccd_pdf'](xgrid), label=f'CCD, {cop_name[:2].capitalize()}',
                    linewidth=DEF_LINEWIDTH)
        ax.legend()
        save_fig(fig, join(base_path_code_plots_tmp, f'sample_{i_eps}.pdf'))


if __name__ == '__main__':
    print('Starting plots comparison')
    plots_comparison()
    print('Starting plots increasing n')
    plots_increasing_n()
    print('Starting plots wrong margins')
    plots_wrong_margin()
    print('Starting plots wrong copula')
    plots_wrong_copula()
    print('Starting plots sample forecasts')
    plot_forecast_samples()
