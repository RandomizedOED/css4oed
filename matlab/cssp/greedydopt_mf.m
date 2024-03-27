function [idx, dopt, S, detA] = greedydopt_mf(A, k)
% GREEDYDOPT_MF Perform greedy column subset selection by maximizing the
% D-optimal criterion for each column sequentially. It performs these
% calculations in a matrix-free manner.
%
% Input:
%  A - The matrix on which to perform CSSP
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
  
  % Setup matrices to be updated
  % Ainv will be used implicitly
  idx  = zeros(k, 1);
  cols = 1:n;
  detA = 1.0;
  C    = zeros(m, k);

  for j = 1:k
    % Need to compute all norms c^T A^{-1} c
    % A^{-1} = (I_m + C C.T)^{-1} =  I_m - C (I_k + C.T C)^{-1} C.T = I_m - C B^{-1} C.T
    B = eye(j-1) + (C(:, 1:j-1)' * C(:, 1:j-1)); 

    maxcol  = zeros(m, 1);
    maxval  = -inf;
    maxidx  = 0;

    % Loop through all the remaining columns
    e_bcol = zeros(n, 1);
    for bcol = 1:length(cols)
      % Extract the column bcol
      e_bcol(cols(bcol)) = 1.0;
      a_bcol             = A*e_bcol;
      
      % Compute the norm
      v_bcol    = C(:, 1:j-1)'*a_bcol;
      bcol_norm = norm(a_bcol, 2)^2 - (v_bcol'*(B\v_bcol));      

      % Reset e_bcol
      e_bcol(cols(bcol)) = 0.0;

      % Check if the column is max norm
      if (bcol_norm > maxval)
        maxval = bcol_norm;
        maxidx = bcol;
        maxcol = a_bcol;
      end
    end 
  
    % Update stuff
    idx(j) = cols(maxidx); % Add the selected column to greedy set
    cols(cols == cols(maxidx)) = []; % Remove the current column from selection
   
    % Update determinant
    % det(A + v v.T) = det(A)(1 + v.T A^{-1} v)
    detA = detA * (1 + maxval);
    
    % Update C
    C(:, j) = maxcol;
  end

  St   = form_selmat(idx, n);
  S    = St';
  dopt = compute_dopt(A*S);
end
