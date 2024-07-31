function [p, num_swaps] = dopt_swaps2(A, idx, f, verbose)
  arguments
    A
    idx
    f (1,1) {mustBeGreaterThanOrEqual(f, 1.0)} = 1.0
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
    % Tracking the swap
    rem_col   = 0;
    sel_col   = 0;
    swap_dopt = cur_dopt;

    % Compute all the downdate determinants at once
    V       = solve_with_smw(A(:, p), A(:, p));
    udotv   = arrayfun(@(i) A(:, p(i))'*V(:, i), 1:length(p));
    detdiff = log(1 - udotv);

    % Downdate determinants
    detdecs   = cur_dopt + detdiff;

    % Find the columns to swap
    for jj = 1:k
      % Remove the jj column
      p_rem_jj = [p(1:jj-1) p(jj+1:end)];
      jj_dopt  = detdecs(jj);

      % Swap in a column
      choices = setdiff(1:m, p);

      V       = solve_with_smw(A(:, p_rem_jj), A(:, choices));
      udotv   = arrayfun(@(i) A(:, choices(i))'*V(:, i), 1:length(choices));
      detdiff = log(1 + udotv);
  
      % Pick the largest difference in determinant
      [detinc, sel_col_idx] = max(detdiff);
  
      % Compute the D-opt difference
      jj_dopt    = jj_dopt + detinc;
      jj_sel_col = choices(sel_col_idx);

      if (jj_dopt > swap_dopt)
        if (verbose)
          fprintf("Intermediate swap found.\n");
          fprintf("Current D-opt: %.4f\n", swap_dopt);
          fprintf("Swapped D-opt: %.4f\n", jj_dopt);
        end
        rem_col   = p(jj);
        sel_col   = jj_sel_col;
        swap_dopt = jj_dopt;
      end
    end    
    % Swap if needed
    if (swap_dopt > f*cur_dopt)
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
