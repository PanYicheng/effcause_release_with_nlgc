# EffCause

This repository is the implementation of an accelerated version of DyCause as in [^1]. The main experiment records are in `effcause-exps.ipynb`.

Other important files:
* `main_effcause.py` EffCause algorithm.
* `main_dycause_mp_new.py` DyCause algorithm.
* `effcause_rca.py` EffCause for RCA.
* `***_test.py` are the test scripts on different datasets.
* `environment.yaml` the environment configuration.
* `r/*` nonlinear Granger causality test. This has not been integrated into EffCause for efficiency problems. Please install a R environment and the R package `generalCorr` before using.

> Note: Some data loading functions in the folder `utility_funcs` may have a little path bugs. Please modify them if data loading errors occur.
