function [P, S, smpl, pr] = colsample(Vkt, l, beta)
% COLSAMPLE Sample the columns of a matrix with orthogonal rows.
% Sampling probabilities is calculated as a mixture of a uniform
% distribution and one proportional to the norms of the columns.
% Please refer to the paper for details. TODO: Add paper citation.
%
% Input:
%  Vkt  - Matrix whose columns are to be sampled
%  l    - Number of columns to be sampled
%  beta - Mixing parameter
% Output:
%  P    - Unweighted column sampling matrix
%  S    - Weighted column sampling matrix
%  smpl - Column indices selected
%  pr   - Sampling probabilities
  arguments
    Vkt
    l (1,1) {mustBeInteger, mustBePositive}
    beta (1,1) {mustBeInRange(beta, 0.0, 1.0)} = 0.9
  end
  n = size(Vkt, 2);

  % Form the leverage scores
  tau = sum(Vkt.^2, 1);
  tau = tau / sum(tau);

  pr = (beta * tau) + ((1-beta) * (1/n));

  % Sample with replacement
  smpl = randsample(n, l, true, pr);

  % Form the weighted sampling matrix
  rows = smpl;
  cols = 1:l;
  vals = sqrt(1 ./ (l * pr(smpl)));
  S    = sparse(rows, cols, vals, n, l);

  % Form the selection matrix
  P = sparse(rows, cols, ones(size(smpl)), n, l);
end
