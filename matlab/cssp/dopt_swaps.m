function [p, num_swaps] = dopt_swaps(A, idx, verbose)
  arguments
    A
    idx
    verbose (1,1) {mustBeNumericOrLogical} = false
  end 
  [n, m] = size(A);

  % Get the initial column subset
  p = idx;
  k = length(idx);

  % Swap till D-opt increases
  inc_found = true;
  num_swaps = 0;

  % Compute current D-opt
  cur_dopt = compute_dopt(A(:, p));
  if (verbose)
    fprintf("Current D-opt: %.4f\n", cur_dopt);
  end

  while(inc_found)
    % Find the column with minimum decrease in D-opt
    V       = solve_with_smw(A(:, p), A(:, p));
    udotv   = arrayfun(@(i) A(:, p(i))'*V(:, i), 1:length(p));
    detdiff = (1 - udotv);

    % Pick the smallest decrease in determinant
    [detdec, rem_col_idx] = max(detdiff);

    % Remove the column
    rem_col = p(rem_col_idx);
    min_dec = cur_dopt * detdec;
    p_rem_j = setdiff(p, rem_col);

    % Find the column with maximum increase in D-opt
    choices = setdiff(1:m, p);

    V       = solve_with_smw(A(:, p_rem_j), A(:, choices));
    udotv   = arrayfun(@(i) A(:, choices(i))'*V(:, i), 1:length(choices));
    detdiff = (1 + udotv);

    % Pick the largest increase in determinant
    [detinc, sel_col_idx] = max(detdiff);

    % Check if a swap is performed
    swap_dopt = min_dec * detinc;
    sel_col   = choices(sel_col_idx);

    if (swap_dopt > cur_dopt)
      if (verbose)
        fprintf("Swap Found!\n");
        fprintf("Current D-opt: %.4f\n", cur_dopt);
        fprintf("Swapped D-opt: %.4f\n", swap_dopt);
      end
      % Swap found
      p         = [setdiff(p, rem_col) sel_col];
      cur_dopt  = swap_dopt;
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

