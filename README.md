# jupyter-lmfit
A jupyter notebook interface for interactive non-linear model fitting based on lmfit (https://lmfit.github.io/lmfit-py/).

With support for a number of models that are built into lmfit, easily fit data using composite models of common model functions for peaks (Voigt, Lorentzian, Gaussian, Pearson7, ...) and more (Linear, Quadratic, Exponential, ...).

Current features:
- automatic plot updating after change of parameters
- easily add multiple models as necessary
- ability to constrain parameters to vary or stay constant in the fits (using the checkbox)
- set max/min values for parameters

Hopeful future upgrades:
- [lmfit](https://lmfit.github.io/lmfit-py/) makes it easy to set parameter contraints relative to one another (e.g. peak1_height = 2 * peak2_height) using the 'expr' keyword in the parameters.  This feature has not yet been implemented.

# Interface
![Demo Screenshot](../assets/ipylmfit_interface.png?raw=true)

# Demo Animation
![Demo Animation](../assets/ipylmfit_demo.gif?raw=true)
