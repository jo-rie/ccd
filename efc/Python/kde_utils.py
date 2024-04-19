import numpy as np
from scipy import interpolate, stats
from scipy.special import ndtr

n_points_kde = int(1e4)


class KdeFit:
    def __init__(self, data: np.ndarray):
        self.kde = stats.gaussian_kde(data)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self.kde.pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.array(x, copy=False, ndmin=1)
        if len(x) > 1:
            return np.array(list(ndtr(np.ravel(item - self.kde.dataset) / self.kde.factor).mean()
                                 for item in x))
        else:
            return ndtr(np.ravel(x - self.kde.dataset) / self.kde.factor).mean()

    def quantile(self, q: float) -> float:
        return np.quantile(self.kde.dataset, q=q)

    def __str__(self):
        return f'KDE with {self.kde.dataset.shape[1]} data points'
