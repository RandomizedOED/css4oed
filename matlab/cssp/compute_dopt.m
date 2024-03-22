function d = compute_dopt(F)
% COMPUTE_DOPT Computes the Bayesian D-optimality criterion for an operator F.
%              \phi_D (F) = \logdet( I + FF^T)
% Input:
%  F - Input operator (will be densified).
% Output:
%  d - D-optimality
  if (~isa(F, "double"))
    F = F * eye(size(F,2));
  elseif(issparse(F))
    F = full(F);
  end
  s = svd(F);
  d = sum(log(1+s.^2));
end
