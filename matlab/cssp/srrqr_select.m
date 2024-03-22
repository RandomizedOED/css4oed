function [p, num_swaps] = srrqr_select(M, k, f, verbose)
% SRRQR_SELECT A naive version of the strong rank-revealing QR algorithm
% given k. This is Algorithm 4 from [GE96]. Since we are only interested
% in the columns selected we only return the pivots and not the factors.
% Regular QR with column pivoting is used for initializing the pivots.
%
% Input:
%  M - The matrix on which to perform sRRQR
%  k - Number of columns to be selected
%  f - Improvement in determinant which causes a swap (f >= 1.0)
% Output:
%  p         - Indices/columns selected
%  num_swaps - Extra swaps performed over the QRCP selection
% References:
% [GE96] Gu, Ming, and Stanley C. Eisenstat. 
%        "Efficient algorithms for computing a strong rank-revealing QR factorization." 
%        SIAM Journal on Scientific Computing 17, no. 4 (1996): 848-869.
  arguments
    M
    k (1,1) {mustBeInteger, mustBeNonnegative}
    f (1,1) {mustBeGreaterThanOrEqual(f, 1.0)} = 1.0
    verbose (1,1) {mustBeNumericOrLogical} = false
  end

  % Run QRCP for a good initialisation
  [m, n]    = size(M);
  [Q, R, p] = qr(M, 'econ', 'vector');

  % Swap till determinant increases
  inc_found = true;
  num_swaps = 0;

  while (inc_found)
    A     = R(1:k, 1:k);
    AinvB = A \ R(1:k, k+1:n); % A^{-1}B 

    if (verbose)
      fprintf("Number of swaps performed: %d\n", num_swaps);
      s = svd(A);
      d = sum(log(s));
      fprintf("logdet(A): %.4f\n", d);
    end

    if (m <= n)
      C = R(k+1:m, k+1:n);
    else
      C = R(k+1:n, k+1:n);
    end

    % Compute the column norms of C
    gamma = zeros(n-k, 1);
    for ccol = 1:n-k
      gamma(ccol) = norm(C(:, ccol), 2);
    end

    % Compute row norms of A^{-1}
    Ainv  = pinv(A);
    omega = zeros(k, 1);
    
    for arow = 1:k
      omega(arow) = norm(Ainv(arow, :), 2);
    end

    % Find indices i and j that maximizes AinvB(i,j)^2 + (omega(i)*gamma(j))^2
    tmp    = omega * gamma';
    F      = AinvB.^2 + tmp.^2;
    [i, j] = find(F > f, 1);

    if (isempty(i)) % No swap found
      inc_found = false;
    else
      num_swaps = num_swaps + 1;
      
      % Swap and retriangularize
      R(:, [i j+k]) = R(:, [j+k i]);
      p([i j+k])    = p([j+k i]);
      [Q, R]        = qr(R, 'econ');
    end
  end

  fprintf("Number of swaps performed: %d\n", num_swaps);
end
