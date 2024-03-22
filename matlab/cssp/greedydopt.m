function [idx, dopt, S, detA] = greedydopt(A, k)
% GREEDYDOPT Perform greedy column subset selection by maximizing the
% D-optimal criterion for each column sequentially.
%
% Input:
%  A - The matrix on which to perform CSSP (will be densified)
%  k - Number of columns to be selected
% Output:
%  idx  - Indices/columns selected
%  dopt - D-optimality of the selected columns
%  S    - Column selection matrix
%  detA - Determinant of the selected columns
  arguments
    A
    k (1,1) {mustBeInteger, mustBePositive}
  end

  [m, n] = size(A);

  % Densify A if needed
  if (~isa(A, "double"))
    A = A * eye(n);
  end

  % Setup matrices to be updated
  Ainv = eye(m);
  idx  = zeros(k, 1);
  cols = 1:n;
  detA = 1.0;

  for j = 1:k
    B = Ainv * A(:, cols);

    % Compute the Ainv norms
    colnorms = zeros(length(cols), 1);
    for bcol = 1:length(cols)
      colnorms(bcol) = A(:, cols(bcol))' * B(:, bcol);
    end

    % Pick the maximum column
    [maxval, maxidx] = max(colnorms);
    
    % Update stuff
    idx(j) = cols(maxidx); % Add the selected column to greedy set
    cols(cols == cols(maxidx)) = []; % Remove the current column from selection
   
    % Update determinant
    % det(A + v v.T) = det(A)(1 + v.T A^{-1} v)
    detA = detA * (1 + maxval);

    % Sherman-Morrison
    % (A + v v.T)^{-1} = A^{-1} - (A^{-1}v v.T A^{-1})/(1 + v.T A^{-1} v)
    bmax = B(:, maxidx); 
    Ainv = Ainv - ((bmax * bmax')/(1 + maxval));
  end

  St   = form_selmat(idx, n);
  S    = St';
  dopt = compute_dopt(A*S);
end
