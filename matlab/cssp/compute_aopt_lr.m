function a = compute_aopt_lr(Wk, Gp, Gpk)
% COMPUTE_AOPT_PROX Computes the Bayesian A-optimality criterion
%   for an operator A = Uk*Sk*Vk^T with Ck = Ak(:, idx).
%              \phi_A (C) = \trace{Gp^{1/2}( I + CC^T)^{-1}Gp^{1/2}}
% Input:
%  Wk  - Sk*Vk(idx, :)^T the scaled and sampled right singular vectors.
%  Gp  - Prior covariance matrix (Gp = G^T G).
%  Gpk - Projected covariance matrix (Gpk = Uk^T*G^T*G*Uk).
% Output:
%  a - A-optimality
  [~, k] = size(Wk);

  tGp = trace(Gp);
  tWk = trace(Wk*((eye(k) + Wk'*Wk)\(Wk'*Gpk)));
  a   = tGp - tWk;
end
