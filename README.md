# bayes_covid
The repository for the class project in Foundations of Predictive Computational Science, Spring 2020.

The folder includes three main files: 

    model_covid.py: functions for three models for infectious disease growth and a noise model.

    misfit_covid.py: the misfit class for evaluating the log-likelihood function in the MCMC algorithm.

    prior.py: the prior class (uniform, gaussian, kde) for sampling from prior and evaluating the log-likelihood function

    mg.py: the implementation of the MCMC algorithm with Metrapolis-Hasting proposals.

A jupyer notebook is included for running an example of the Bayesian calibration/validation/prediction process that apperas in the Oden Institute report.
