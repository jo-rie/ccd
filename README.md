# Code for the paper "Combining point forecasts to calibrated probabilistic forecasts using copulas"

This repository contains code to reproduce the results of the paper "Combining point forecasts to calibrated probabilistic forecasts using copulas" and apply the CCD method to further combination problems. 
The main script is `python_src/ccd.py`. 

The simulation studies can be run through `run_simulations.py` for the comparison to other methods and `run_simulation_increasing_n.py`for the analysis with respect to the CCD performance for increasing sample size. 
The plots are generated in `create_plots.py` (note that the plot folder has to be specified in the `setup.py` file).
The plots illustrating the method itself are in `illustrative_simulations.py`. 

The code for the example on the electricity price data is in the folder `efc`.
Note that the [code](https://github.com/gmarcjasz/distributionalnn) from the Paper [Distributional neural networks for electricity price forecasting](https://www.sciencedirect.com/science/article/abs/pii/S0140988323003419) is not included here but necessary to compute the forecasts to be combined.
As in the original data, the scripts are assumed to be run from the `efc` directory. 
The marginal kde forecasts are computed in the script `fit_kde_marginal_model.py`. 
The combination is computed in `fit_ccd_vine_kde.py`.
The other models are computed in `fit_non_ccd_marginal_models.py`. 
`plots_paper.py` generates the plots.
Note that the computations use the `rpy2` package to call functions from the `scoringRules` package in R.