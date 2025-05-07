###############################################################################
# Overview
###############################################################################

This repository contains code implementing the Additive Multi-Index GP model in Li et al. (2025): "Additive Multi-Index Gaussian process modeling, with application to multi-physics surrogate modeling of the quark-gluon plasma".

Please cite this paper if you use the code in this repository as part of a published research project.

###############################################################################
# Software and Package Requirements
###############################################################################

The following code should run as intended with R (version 4.2.3) and Python (version 3.8). 

Required Python libraries include:
- torch
- GPy
- gpytorch
- pandas
- linear_operator
- tqdm
- matplotlib numpy (1.16.4)
- scipy
- properscoring
- seaborn

Note that, due to a recent update with numpy, the latest numpy version will not work with GPy, thus our use of an earlier version above. We include a **environment.yml** file for the Python libraries.

Required R libraries include:
- tgp (2.4-23)
- scoringRules (1.1.3)
- BART (2.9.9)
- lhs (1.2.0)

###############################################################################
# Files and Folders Description
###############################################################################

- **au_data**: Folder containing data files for our Quark Gluon Plasma application.
  - **au_design.txt** contains the 17-dimensional design that is used for all observables.
  - **au_y.txt** contains the response variables for each of the four considered observables.
- **R_Results**: Folder containing scripts for running the Single Index Model/BART in R. These fit the R methods and save the performance metrics to this folder to be plotted by the Python methods. 
- **paper_plots**: Folder storing the .png plots in the paper.
- **reviewer_plots**: Folder storing the .png plots in reviewer response.
- **sim_results**: Folder storing pickle files that contain performance results of the Python-implemented methods.
- **helpers.py**: Wrappers to fit Python-implemented models and collect performance metrics.
- **sim_functions.py**: Python code for simulation experiments.
- **rp.py**: Files required for implementing the Diverse Projected Additive Pursuit (DPA) method.
- **Variational_Functions.py**: Files required for implementing variational inference for DPA.

The remaining relevant files are Jupyter notebooks:
- The name of each Jupyter notebook begins with a number that represents the section of the paper that it reproduces.
- Notebooks with "Plots" in their name generate the plots, while the other notebooks run the simulation/emulation tasks.
- Each section can be run independently of the others. Within each section, the notebooks with "Plots" in their name should be run last. The Plot notebooks are divided and labeled by which figure they reproduce. 

For convenience, we provide below the corresponding figures that each plotting notebook reproduces:
- **Simulation_Plots.ipynb**: Figures 4, 5
- **Variable_Selection_Plots.ipynb**: Figures 6, 11 (Appendix C)
- **Emulation_Plots.ipynb**: Figures 2, 8, 9, 10
- **Appendix_Plots.ipynb**: Figure 12 (Appendix D)
- **FitCheck.ipynb**: Figure 13 (Appendix E)
