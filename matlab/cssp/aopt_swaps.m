function [p, num_swaps] = aopt_swaps(A, Gp, idx, verbose)
  arguments
    A
    Gp
    idx
    verbose (1,1) {mustBeNumericOrLogical} = false
  end 
  [n, m] = size(A);

  % Get the initial column subset
  p = idx;
  k = length(idx);

  % Swap till A-opt increases
  inc_found = true;
  num_swaps = 0;

  % Cache the cholesky of the prior
  Gp_chol = chol(Gp);

  % Compute current A-opt prox
  cur_aopt = compute_aopt(A(:, p), Gp);
  if (verbose)
    fprintf("Current A-opt: %.4f\n", cur_aopt);
  end

  while(inc_found)
    % Find the column with minimum increase in A-opt
    V  = solve_with_smw(A(:, p), A(:, p));
    GV = Gp_chol*V;

    Gvnormsq = sum(GV.^2, 1);
    udotv    = arrayfun(@(i) A(:, p(i))'*V(:, i), 1:length(p));
    trdiff   = Gvnormsq ./ (1 - udotv);

    % Pick the smallest increase in trace
    [trinc, rem_col_idx] = min(trdiff);

    % Remove the column
    rem_col = p(rem_col_idx);
    min_inc = cur_aopt + trinc;
    p_rem_j = setdiff(p, rem_col);

    % Find the column with maximum decrease in A-opt
    choices = setdiff(1:m, p);

    V  = solve_with_smw(A(:, p_rem_j), A(:, choices));
    GV = Gp_chol*V;

    Gvnormsq = sum(GV.^2, 1);
    udotv    = arrayfun(@(i) A(:, choices(i))'*V(:, i), 1:length(choices));
    trdiff   = Gvnormsq ./ (1 + udotv);

    % Pick the largest difference in trace
    [trdec, sel_col_idx] = max(trdiff);

    % Check if a swap is performed
    swap_aopt = min_inc - trdec;
    sel_col   = choices(sel_col_idx);

    if (swap_aopt < cur_aopt)
      if (verbose)
        fprintf("Swap Found!\n");
        fprintf("Current A-opt: %.4f\n", cur_aopt);
        fprintf("Swapped A-opt: %.4f\n", swap_aopt);
      end
      % Swap found
      p         = [setdiff(p, rem_col) sel_col];
      cur_aopt  = swap_aopt;
      num_swaps = num_swaps + 1;
    else
      % No swap found
      if (verbose)
        fprintf("No swap found.\n");
      end
      inc_found = false;
    end
  end
end

function [V] = solve_with_smw(C, U)
% Helper function to solve the following with Sherman-Morrison-Woodbury
%                (I + C C^T) V = U
% where C is a n x k low-rank matrix.
  [~, k] = size(C);
  
  B = eye(k) + C'*C;
  V = U - (C*(B\(C'*U)));
end
