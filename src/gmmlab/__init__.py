"""gmmlab: from-scratch Gaussian mixtures, EM, and active-learning experiments.

Nothing is exported yet. Modules are added only once their mathematics has been
worked through in the research log:

    gaussian.py   multivariate normal density (log-space, numerically safe)
    gmm.py        GaussianMixture: fit (EM), predict_proba (responsibilities)
    datasets.py   synthetic mixtures with a controllable separation parameter
    active.py     acquisition rules (random, max-entropy, ...) + the AL loop
"""

__version__ = "0.0.1"
