function [x, r, obj, data_m, pr_m] = solve_invprob(F, Gn, Gp_inv, y, mu,...
                                      alpha, S, solve_cg, maxiters, precon)
% SOLVE_INVPROB Solves the linear Bayesian inverse problem given by
% min_{x} \|Fx - y\|_{\Gamma_n^{-1}}^2 + \alpha*\|x-\mu\|_{\Gamma_p^{-1}}^2.
% The least squares problem can be solved via a direct solver or iteratively
% via the Conjugate-Gradient method.
% The method also optionally accepts a row sampling matrix S to solve the 
% problem after sensor placement via OED.
%
% Input:
%  F        - Forward operator for the inverse problem
%  Gn       - Noise covariance matrix (Noise is assumed to be Gaussian)
%  Gp_inv   - Prior precision matrix (prior is assumed to be Gaussian)
%  y        - Observations/data collected
%  mu       - Prior mean (prior is assumed to be Gaussian)
%  alpha    - Regularization parameter
%  S        - Row sampler matrix (optional)
%  solve_cg - Flag for using an iterative method CG (default true)
%  maxiters - Maximum number of CG iterations (default 300)
%  precon   - Preconditioner for CG
% Output:
%  x      - Solution vector
%  r      - Residual of the least-squares solution
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
    solve_cg (1,1) = true
    maxiters (1,1) = 300
    precon = []
  end

  % Form the sampled matrices
  [sel_F, sel_Gn_inv, sel_y] = form_selops(F, Gn, y, S);

  % Form the RHS for the solve
  b = (sel_F' * (sel_Gn_inv * sel_y)) + (alpha * (Gp_inv * mu));

  if (solve_cg) % Solve via CG
    % Form the operator for the solve
    A = @(x) (sel_F'*(sel_Gn_inv*(sel_F*x))) + (alpha*(Gp_inv*x));
    [x, exit_code] = pcg(A, b, 1e-6, maxiters, precon);
    if (exit_code > 0)
      fprintf('CG exited with code %d.\n', exit_code);
    end
    r = norm(A(x) - b, 2);
  else          % Solve via Gaussian Elimination
    % Form the dense operator for the solve
    A = (sel_F' * (sel_Gn_inv * sel_F)) + (alpha * Gp_inv);
    x = A \ b;
    r = norm(A*x - b, 2);
  end

  % Use the cached version of objective calculations
  [obj, data_m, pr_m] = compute_obj_cached(sel_F, sel_Gn_inv, Gp_inv,...
                                    sel_y, mu, alpha, x);
end
