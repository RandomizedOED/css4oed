function [Uk, Sk, Vk] = randsvd(A, k, p, q)
% RANDSVD A randomized SVD algorithm based on Algorithm 5.1 in [HMT11].
% Computes approximate rank-k SVD of an input matrix A.
% Input:
%  A - Input matrix or linear operator (matvecs must be defined)
%  k - Rank of the approximate SVD to be computed
%  p - Oversampling parameter
%  q - Number of subspace iterations to be performed
% Output:
%  Uk - Top-k left singular vectors
%  Sk - Top-k diagonal singular values matrix
%  Vk - Top-k right singular vectors
% References:
% [HMT11] Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp. 
%         "Finding structure with randomness: Probabilistic algorithms 
%         for constructing approximate matrix decompositions." 
%         SIAM review 53.2 (2011): 217-288. 
  arguments
    A
    k
    p (1,1) {mustBeInteger, mustBeNonnegative}
    q (1,1) {mustBeInteger, mustBeNonnegative} = 2
  end

  n     = size(A, 2);
  Omega = randn(n, k+p);

  Y      = A * Omega;
  [Q, ~] = qr(Y, 'econ');

  for i = 1:q
    Y1     = A' * Q;
    [Q, ~] = qr(Y1, 'econ');

    Y      = A * Q;
    [Q, ~] = qr(Y, 'econ');
  end

  B = Q' * A;
  [Ub, Sb, Vb] = svd(B, 'econ');

  Uk = Q * Ub(:, 1:k);
  Sk = Sb(1:k, 1:k);
  Vk = Vb(:, 1:k);
end
