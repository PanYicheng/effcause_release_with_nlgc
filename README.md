# EffCause

Official implementation of TKDD 2024 paper "EffCause: Discover Dynamic Causal Relationships Efficiently from Time-Series" (https://dl.acm.org/doi/10.1145/3640818). 

This repository is the implementation of an accelerated version of DyCause as in [1]. The main experiment records are in `effcause-exps.ipynb`.

Other important files:
* `main_effcause.py` EffCause algorithm.
* `main_dycause_mp_new.py` DyCause algorithm.
* `effcause_rca.py` EffCause for RCA.
* `***_test.py` are the test scripts on different datasets.
* `environment.yaml` the environment configuration.
* `r/*` nonlinear Granger causality test. This has not been integrated into EffCause for efficiency problems. Please install an R environment and the R package `generalCorr` before using.

> Note: Some data loading functions in the folder `utility_funcs` may have a little path bugs. Please modify them if data loading errors occur.

## References

[1] DyCause: Crowdsourcing to Diagnose Microservice Kernel Failure [[link]](https://ieeexplore.ieee.org/abstract/document/10005849).

## Citations
Please cite the paper and star this repo if you use EffCause and find it interesting/useful, thanks! Feel free to open an issue if you have any questions.

```bibtex
@article{pan2024effcause,
  title={EffCause: Discover Dynamic Causal Relationships Efficiently from Time-Series},
  author={Pan, Yicheng and Zhang, Yifan and Jiang, Xinrui and Ma, Meng and Wang, Ping},
  journal={ACM Transactions on Knowledge Discovery from Data},
  year={2024},
  publisher={ACM New York, NY}
}

@article{pan2023dycause,
  title={DyCause: Crowdsourcing to Diagnose Microservice Kernel Failure},
  author={Pan, Yicheng and Ma, Meng and Jiang, Xinrui and Wang, Ping},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2023},
  publisher={IEEE}
}
```
