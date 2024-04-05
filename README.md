# Column Subset Selection for Optimal Experiment Design
This repository contains MATLAB and Python code for optimal sensor placement using the column subset selection approach. It accompanies the paper
> Eswar, S., Rao, V., & Saibaba, A. K. (2024). Bayesian D-Optimal Experimental Designs via Column Subset Selection. Submitted. [arXiv preprint](https://arxiv.org/abs/2402.16000).

## Requirements
The [MATLAB](matlab/) code relies on the following packages to setup the test problems.
1. [IR Tools](https://github.com/jnagy1/IRtools)
2. [AIR Tools II](https://github.com/jakobsj/AIRToolsII)

The [Python](python/) code requires the following packages for test problems.
1. [PyOED](https://gitlab.com/ahmedattia/pyoed)
2. [GPy](https://github.com/SheffieldML/GPy)
 
## License
To use these codes in your research, please see the [License](LICENSE). If you find our code useful, please consider citing our paper.
```bibtex
@article{eswar2024bayesian,
  title={Bayesian D-Optimal Experimental Designs via Column Subset Selection},
  author={Eswar, Srinivas and Rao, Vishwas and Saibaba, Arvind K},
  journal={arXiv preprint arXiv:2402.16000},
  year={2024}
}
```
All figures and data from this paper can be generated via the scripts in [paper\_figure](paper_figures/).

## Funding
This work was supported was supported, in part, by the National Science Foundation through the award DMS-1845406 and
the Department of Energy through the awards DE-SC0023188 and DE-AC02-06CH11357.
