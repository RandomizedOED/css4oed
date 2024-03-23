# How to use
Simply run `prdiffusion_paper_figs.m` or `seismic_paper_figs.m` to reproduce the data in
> Eswar, S., Rao, V., & Saibaba, A. K. (2024). Bayesian D-Optimal Experimental Designs via Column Subset Selection: The Power of Reweighted Sensors. arXiv preprint arXiv:2402.16000.

The following Matlab packages/functions are needed to run.
1. [IR Tools](https://github.com/jnagy1/IRtools) - For setting up the test problems.
2. [AIR Tools II](https://github.com/jakobsj/AIRToolsII) - For setting up the test problems.
3. [randsample](https://www.mathworks.com/help/stats/randsample.html) - Beware that this function is not bundled in a default Matlab install.
4. [franke](https://www.mathworks.com/help/curvefit/franke.html) - Same problems as `randsample.m`.
5. [breakxaxis](https://www.mathworks.com/matlabcentral/fileexchange/42905-break-x-axis?s_tid=FX_rc1_behav) - Needed for plotting some histograms.
