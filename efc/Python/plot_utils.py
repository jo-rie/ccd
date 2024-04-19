from os.path import join
from typing import Tuple, Union, Literal, Optional

from config import base_path
from nptyping import (NDArray, Shape, Int32)
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
import pandas as pd
from scipy.stats import norm, kendalltau
import seaborn as sns

from utils import combine_json_results, read_price, restrict_df_to_dates
from config import start_eval_all, end_eval_all

# Widths and heights
text_width_pts = 345 # Elsevier
pts_to_inch = 1 / 72.27
text_width = text_width_pts * pts_to_inch
default_fig_height = text_width / 3.1
default_fig_width = text_width
fig_factor_horizontally = 1.05  # Additional space for room between figures


MPL_ALPHA = .5
MPL_S = .5

cycler_sym = mpl.cm.get_cmap('PRGn')  # plt.cm.PRGn  # Symmetric plot colors
cycler_01 = mpl.cm.get_cmap('YlGn')  # 0-1 plot colors
plot_style_ca = {'s': 2, 'alpha': .45, 'vmin': 0, 'vmax': 1}
list_of_cmaps = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

boxplot_settings = {
    'patch_artist': True,
    'boxprops': {"facecolor": mpl.colors.to_rgba("C0", 0.5), "edgecolor": "C0", "linewidth": 1.5},
    'whiskerprops': {'linewidth': .75},
    'medianprops': {'color': 'C0', 'linewidth': 1.0},
    'flierprops': {'markeredgecolor': 'C0', 'linewidth': 0.75},
    'showmeans': True
}


def initialize_mpl() -> None:
    plt.rcParams.update({
        'figure.dpi': 600,
        "text.usetex": True,
        'font.size': 4,
        "font.family": "serif",
        "font.serif": ["STIXTwoText"],
        'pgf.preamble': r'\usepackage{unicode-math,fontspec}\setmathfont[Extension={.otf},Path=/System/Library/Fonts/Supplemental/]{STIXTwoMath}\setsansfont[Path=/System/Library/Fonts/Supplemental/]{STIXTwoText}',
        'pgf.rcfonts': False,
        "figure.figsize": (default_fig_width, default_fig_height),
        'axes.labelsize': 5,
        'legend.fontsize': 5,
        # "pgf.texsystem": "pdflatex",
        # "pgf.preamble": "\n".join([
        #     r"\usepackage[utf8x]{inputenc}",
        #     r"\usepackage[T1]{fontenc}",
        #     r"\usepackage{cmbright}",
        # ]),
    })
    mpl.use('pgf')


def fig_with_size(nb_horizontally=1, nb_vertically=1, fig_height=None, fig_width=None, factor_height=None,
                  **subplots_kwargs) -> Tuple[
    plt.Figure, Union[plt.Axes, np.ndarray]]:
    """Return a figure so that nb_horizontally fit next to each other and nb_horizontally fit below each other"""
    if fig_height is None:
        if factor_height is None:
            if nb_vertically == 1:
                fig_height = default_fig_height
            else:
                fig_height = text_height / (nb_vertically * fig_factor_vertically)
        else:
            fig_height = default_fig_height * factor_height
    if fig_width is None:
        fig_width = default_fig_width / (nb_horizontally * fig_factor_horizontally)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), **subplots_kwargs)
    return fig, ax


def save_fig(fig: plt.Figure, path: str):
    """Save the figure to the path"""
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pit_normal(dist_number: int, save_path: str, filename: str = 'pit_histogram_raw.pdf'):
    """Plot pit of ddnn normal forecast with number dist_number. The histogram is saved in the folder save_path."""
    # Load data
    df = combine_json_results(join(base_path, f'distparams_probNN_normal{dist_number}'),
                              start_date=start_eval_all, end_date=end_eval_all)
    df_real = read_price()
    df = df.join(df_real, how='left')
    df['pit_dist'] = norm.cdf(df['obs'], loc=df['loc'], scale=df['scale'])

    fig, ax = fig_with_size(3)
    pit_plot(df['pit_dist'], filepath=join(save_path, filename), ax=ax)


def plot_pit_jsu(dist_number: int, save_path: str, filename: str = 'pit_histogram_raw.pdf'):
    """Plot pit of ddnn jsu forecast with number dist_number. The histogram is saved in the folder save_path."""
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    # Load data
    df = combine_json_results(join(base_path, f'distparams_probNN_jsu{dist_number}'),
                              start_date=start_eval_all, end_date=end_eval_all)
    df_real = read_price()
    df = df.join(df_real, how='left')
    df['pit_dist'] = df.apply(lambda x: tfd.JohnsonSU(
        skewness=x['skewness'], tailweight=x['tailweight'],
        loc=x['loc'], scale=x['scale']).cdf(x['obs']), axis=1)

    fig, ax = fig_with_size(3)
    pit_plot(df['pit_dist'], filepath=join(save_path, filename), ax=ax)


def pair_error_plot(filename: str, dist_type: Literal['Normal', 'JSU']):
    """Plot pair plots of the errors for all raw distributional ddnn outputs"""
    mpl.use('pdf')
    df_list = []
    # Load data for all normal distributions
    for dist_number in range(1, 5):
        if dist_type.lower() == 'normal':
            df = pd.read_pickle(f'../data_paper/ddnn_normal_{dist_number}.pickle')
        elif dist_type.lower() == 'jsu':
            df = pd.read_pickle(f'../data_paper/ddnn_jsu_{dist_number}.pickle')
        else:
            ValueError(f'dist_type {dist_type} is not known.')
        df = restrict_df_to_dates(df, start_date=start_eval_all, end_date=end_eval_all)
        # df_real = read_price()
        # df = df.join(df_real, how='left')
        df[f'DDNN-{dist_type}-{dist_number}'] = df['mean'] - df['obs']
        df_list.append(df[f'DDNN-{dist_type}-{dist_number}'])
    df_all = pd.concat(df_list, axis=1)
    # sns_plot = sns.pairplot(df_all, height=default_fig_width / (4 * fig_factor_horizontally))
    # save_fig(sns_plot.figure, filename)
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(default_fig_width, default_fig_width),
                             sharex=False, sharey=False)
    for i in range(4):
        for j in range(4):
            if i > j:  # scatter plot
                axes[i, j].scatter(df_all[f'DDNN-{dist_type}-{i + 1}'], df_all[f'DDNN-{dist_type}-{j + 1}'],
                                   alpha=MPL_ALPHA, s=MPL_S)
            if i == j:  # histogram
                axes[i, j].hist(df_all[f'DDNN-{dist_type}-{i + 1}'], bins=20, density=True)
            if i < j:  # contour plot
                # axes[i, j].hist2d(df_all[f'DDNN-{dist_type}-{i+1}'], df_all[f'DDNN-{dist_type}-{j+1}'])
                sns.kdeplot(data=df_all, x=f'DDNN-{dist_type}-{i + 1}', y=f'DDNN-{dist_type}-{j + 1}',
                            ax=axes[i, j], linewidths=0.5)
                axes[i, j].set(xlabel=None, ylabel=None)
                tau_estimate = kendalltau(df_all[f'DDNN-{dist_type}-{i + 1}'],
                                          df_all[f'DDNN-{dist_type}-{j + 1}']).statistic
                axes[i, j].annotate(xy=(.95, .05), text=r'$\tau = ' + f'{tau_estimate:.3f}' + r'$',
                                    va='bottom', ha='right', xycoords='axes fraction')
        axes[i, 0].set(ylabel=f'DDNN-{dist_type}-{i + 1}')
        axes[-1, i].set(xlabel=f'DDNN-{dist_type}-{i + 1}')
    save_fig(fig, filename)
    mpl.use('pgf')


def pit_plot(u: np.ndarray, ax: Optional[plt.Axes] = None, title: Optional[str] = None,
             x_label: Optional[str] = None, y_label: Optional[str] = None,
             filepath: Optional[str] = None) -> plt.Axes:
    """
    The pit_plot function plots the probability integral transform (PIT) of a given array of pit values.

    :param filepath: Path to save pit in, if None, the figure is not saved
    :param u: np.ndarray: Specify the input array
    :param ax: Optional[plt.Axes]: Pass in an existing axes object to plot on
    :param title: Optional[str]: Set the title of the plot
    :param x_label: Optional[str]: Set the x-axis label of the plot
    :param y_label: Optional[str]: Set the y-axis label
    :return: A tuple of a figure and an axis
    """
    fig_created = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig_created = True
    if any(u < 0.):
        print("Warning: u contains values < 0: {}".format(u[u < 0]))
    if any(u > 1.):
        print("Warning: u contains values > 1: {}".format(u[u > 1]))
    ax.hist(u, range=(0, 1), density=True)
    ax.set(title=title, xlabel=x_label, ylabel=y_label, xlim=(0, 1))
    ax.axhline(1, color='grey', linewidth=0.2, linestyle='--')
    if filepath is not None:
        save_fig(ax.get_figure(), filepath)
    return ax
