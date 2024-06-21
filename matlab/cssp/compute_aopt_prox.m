function a = compute_aopt_prox(F)
% COMPUTE_AOPT_PROX Computes the Bayesian A-optimality criterion
%   for an operator F assuming no prior.
%              \phi_A (F) = \trace( I + FF^T)^{-1}
% Input:
%  F - Input operator (will be densified).
% Output:
%  d - D-optimality
  [m, n] = size(F);
  if (~isa(F, "double"))
    F = F * eye(n);
  elseif(issparse(F))
    F = full(F);
  end
  s = svd(F);
  a = sum(1 ./ (1+s.^2)) + max(m-n, 0);
end
