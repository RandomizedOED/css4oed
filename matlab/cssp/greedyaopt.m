function [idx, aopt, S] = greedyaopt(A, Gp, k)
  arguments
    A
    Gp
    k (1,1) {mustBeInteger, mustBePositive}
  end

  [m, n] = size(A);

  % Densify A if needed
  if (~isa(A, "double"))
    A = A * eye(n);
  end
  
  % Cache the cholesky of the prior
  Gp_chol = chol(Gp);

  % Setup variables to be updated
  aopt = trace(Gp);
  idx  = zeros(k, 1);
  cols = 1:n;
  C    = zeros(m, k);

  for j = 1:k
    % Find the column with maximum decrease in A-opt
    V  = solve_with_swm(C(:,1:j-1), A(:, cols));
    GV = Gp_chol*V;

    Gvnormsq = sum(GV.^2, 1);
    udotv    = arrayfun(@(i) A(:, cols(i))'*V(:, i), 1:length(cols));
    trdiff   = Gvnormsq ./ (1 + udotv);

    % Pick the largest difference in trace
    [trdec, sel_col_idx] = max(trdiff);

    % Update stuff
    aopt    = aopt - trdec;
    C(:, j) = A(:, cols(sel_col_idx));

    idx(j) = cols(sel_col_idx);           % Add the selected column to greedy set
    cols(cols == cols(sel_col_idx)) = []; % Remove the current column from selection
  end

  St = form_selmat(idx, n);
  S  = St';
end
