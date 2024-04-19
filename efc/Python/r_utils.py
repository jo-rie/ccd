from typing import Tuple

import numpy as np
import rpy2
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
scoringRules = importr('scoringRules')


def crps_sample(obs: np.ndarray, forecast_samples: np.ndarray) -> Tuple[float, np.ndarray]:
    """ Compute the crps of obs given the n forecasts specified as m forecast samples per observation (forecast_samples is n x m). Returns the mean crps and crps per observation"""
    assert len(obs) == forecast_samples.shape[0]
    crps_values = np.zeros_like(obs)
    for i in range(len(obs)):
        crps_values[i] = crps_sample_single(obs[i], forecast_samples[i, :])
    # crps_values = scoringRules.crps_sample(obs[:], forecast_samples.T) # does not work because of conversion issues from numpy to r
    return crps_values.mean(), crps_values


def crps_sample_single(obs: float, forecast_samples: np.ndarray) -> float:
    """ Compute the crps of a single observation obs given the m forecasts in forecast_samples (forecast_samples is n x m)."""
    forecast_samples = forecast_samples.flatten()
    crps_value = scoringRules.crps_sample(obs, forecast_samples[np.newaxis, :])
    return crps_value[0]
