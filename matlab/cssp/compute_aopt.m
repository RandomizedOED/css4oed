function a = compute_aopt(F, Gp)
% COMPUTE_AOPT_PROX Computes the Bayesian A-optimality criterion
%   for an operator F.
%              \phi_A (F) = \trace{Gp^{1/2}( I + FF^T)^{-1}Gp^{1/2}}
% Input:
%  F - Input operator (will be densified).
% Output:
%  d - D-optimality
  [~, n] = size(F);
  if (~isa(F, "double"))
    F = F * eye(n);
  elseif(issparse(F))
    F = full(F);
  end
  % Attempt 1: chol + qr + solve
  %G = chol(Gp);
  %R = qr([eye(size(F, 1)); F']);
  %L = R'\G';
  %a = norm(L, 'fro')^2;
  tGp = trace(Gp);
  tF  = trace(F*((eye(n) + F'*F)\(F'*Gp)));
  a   = tGp - tF;
end
