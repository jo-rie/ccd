from typing import Tuple, Union
from nptyping import (NDArray, Shape, Int32)
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# Widths and heights
# text_width_pts = 345 # Elsevier
text_width_pts = 418  # dissertation
pts_to_inch = 1 / 72.27
text_width = text_width_pts * pts_to_inch
text_height = 555 * pts_to_inch
default_fig_height = text_width / 3.1
default_fig_width = text_width
fig_factor_horizontally = 1.05  # Additional space for room between figures
fig_factor_vertically = 1.25  # Additional space for room between figures

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
    'medianprops':{'color': 'C0', 'linewidth': 1.0},
    'flierprops': {'markeredgecolor': 'C0', 'linewidth': 0.75},
    'showmeans': True
}


def initialize_mpl() -> None:
    plt.rcParams.update({
        'figure.dpi': 600,
        "text.usetex": True,
        'font.size': 4,
        # "font.family": "serif",
        # "font.serif": ["Palatino"],
        'pgf.preamble': r'\usepackage{unicode-math,fontspec}\setmathfont[Extension={.otf},Path=/System/Library/Fonts/Supplemental/]{STIXTwoMath}\setsansfont[Path=/System/Library/Fonts/Supplemental/]{STIXTwoText}',
        'pgf.rcfonts': False,
        "figure.figsize": (default_fig_width, default_fig_height),
        'axes.labelsize': 5,
        'legend.fontsize': 5,
    })
    mpl.use('pgf')


def fig_with_size(nb_horizontally=1, nb_vertically=1, fig_height=None, fig_width=None, factor_height=None, **subplots_kwargs) -> Tuple[
    plt.Figure, Union[plt.Axes, np.ndarray]]:
    """Return a figure so that nb_horizontally fit next to each other and nb_horizontally fit below each other"""
    if fig_height is None:
        if factor_height is None:
            if nb_vertically > 1:
                fig_height = text_height / (nb_vertically * fig_factor_vertically)
            else:
                fig_height = default_fig_height
        else:
            fig_height = default_fig_height * factor_height
    if fig_width is None:
        fig_width = default_fig_width / (nb_horizontally * fig_factor_horizontally)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), **subplots_kwargs)
    return fig, ax


def save_fig(fig: plt.Figure, path: str):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def create_and_save_boxplot(df: pd.DataFrame, col: str, by: str, fig: plt.Figure, ax: plt.Axes, fname: str, xlabel: str = None, **boxplot_kwargs):
    df.boxplot(column=col, by=by, ax=ax, **(boxplot_settings | boxplot_kwargs))
    fig.suptitle('')
    ax.set(title='')
    ax.set_xticklabels(ax.get_xticklabels(), va='top', rotation=45, ha='center')
    # ax.axis["bottom"].major_ticklabels.set_va("baseline")
    if xlabel is not None:
        ax.set(xlabel=xlabel)
    save_fig(fig, fname)
