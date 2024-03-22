function [obj, data_m, pr_m] = compute_obj(F, Gn, Gp_inv, y, mu, alpha, x, S)
% COMPUTE_OBJ Calculates the different terms in the linear Bayesian inverse
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
%  S      - Row sampler matrix (optional)
% Output:
%  obj    - Objective evaluated at x
%  data_m - Data misfit term
%  pr_m   - Prior misfit term
  arguments
    F
    Gn
    Gp_inv
    y
    mu
    alpha (1,1) = 1.0
    S = []
  end

  % Form the sampled matrices
  sel_F, sel_Gn_inv, sel_y = form_selops(F, Gn, y, S);

  % Compute the objective
  y_data = (sel_F * x) - sel_y;
  y_pr   = x - mu;

  data_m = 0.5 * (y_data' * (sel_Gn_inv * y_data));
  pr_m   = 0.5 * (y_pr' * (Gp_inv * y_pr));
  obj    = data_m + (alpha * pr_m);
end
