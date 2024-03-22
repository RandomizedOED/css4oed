function [obj, data_m, pr_m] = compute_obj_cached(F, Gn_inv, Gp_inv, y, mu, alpha, x)
% COMPUTE_OBJ_CACHED Cached version of compute_obj where the row sampling is 
% assumed to applied beforehand and not included in the function.
%
% Calculates the different terms in the linear Bayesian inverse
% problem setting. The following optimization is typically solved
% min_{x} \|Fx - y\|_{\Gamma_n^{-1}}^2 + \alpha*\|x-\mu\|_{\Gamma_p^{-1}}^2.
% The first term in the objective corresponds to data misfit and the 
% second term is the prior misfit.
%
% In case we sample rows of F via S the following objective is computed.
% min_{x} \|S*(Fx - y)\|_{\Gamma_n^{-1}}^2 + \alpha*\|x-\mu\|_{\Gamma_p^{-1}}^2.
%
% Input:
%  F      - Forward operator for the inverse problem
%  Gn     - Noise covariance matrix (Noise is assumed to be Gaussian)
%  Gp_inv - Prior precision matrix (prior is assumed to be Gaussian)
%  y      - Observations/data collected
%  mu     - Prior mean (prior is assumed to be Gaussian)
%  alpha  - Regularization parameter
%  x      - Candidate solution
% Output:
%  obj    - Objective evaluated at x
%  data_m - Data misfit term
%  pr_m   - Prior misfit term
  y_data = (F * x) - y;
  y_pr   = x - mu;

  data_m = 0.5 * (y_data' * (Gn_inv * y_data));
  pr_m   = 0.5 * (y_pr' * (Gp_inv * y_pr));
  obj    = data_m + (alpha * pr_m);
end
