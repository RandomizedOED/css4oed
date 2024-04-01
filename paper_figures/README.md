# Reproducibility scripts
Scripts to generate the data in
> Eswar, S., Rao, V., & Saibaba, A. K. (2024). Bayesian D-Optimal Experimental Designs via Column Subset Selection. arXiv preprint arXiv:2402.16000.

# Dependencies
The following Matlab packages/functions are needed to run these scripts.
1. [IR Tools](https://github.com/jnagy1/IRtools) - For setting up the test problems.
2. [AIR Tools II](https://github.com/jakobsj/AIRToolsII) - For setting up the test problems.
3. [randsample](https://www.mathworks.com/help/stats/randsample.html) - Beware that this function is not bundled in a default Matlab install.
4. [franke](https://www.mathworks.com/help/curvefit/franke.html) - Same problems as `randsample.m`.
5. [breakxaxis](https://www.mathworks.com/matlabcentral/fileexchange/42905-break-x-axis?s_tid=FX_rc1_behav) - Needed for plotting some histograms.

# Contents
All the data was generated on a laptop computer with a 12th Gen Intel(R) Core(TM) i7-1280P processor and 32 GB of memory using Matlab (version R2023b).  
1. `[prdiffusion|seismic]_paper_figs.m` - Generates problem setup and all data related to the quality of the solutions (D-optimality and reconstruction) for the test problems.
2. `[prdiffusion|seismic]_timing.m` - Benchmarks the running times of the different algorithms. Warning: This is particularly slow for the matrix-free version of the Heat problem (`prdiffusion_timing.m`).
