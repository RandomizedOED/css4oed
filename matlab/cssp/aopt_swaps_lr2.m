function [p, num_swaps] = aopt_swaps_lr2(Uk, Sk, Vk, Gp, idx, verbose)
  arguments
    Uk
    Sk
    Vk
    Gp
    idx
    verbose (1,1) {mustBeNumericOrLogical} = false
  end 
  %[n, k] = size(Uk);
  [m, ~] = size(Vk);
  %Ak = Uk*Sk*Vk';

  % Compute the scaled right singular vectors
  Wk = Sk*Vk';

  % Get the initial column subset
  p = idx;
  k = length(idx);

  % Swap till A-opt increases
  inc_found = true;
  num_swaps = 0;

  % Cache the cholesky of the prior
  Gp_chol  = chol(Gp);
  Gpk_chol = Gp_chol*Uk;
  Gpk      = Gpk_chol'*Gpk_chol;

  % Compute current A-opt prox
  cur_aopt = compute_aopt_lr(Wk(:, p), Gp, Gpk);
  if (verbose)
    fprintf("Current A-opt : %.4f\n", cur_aopt);
    %fprintf("Current A-opt (chk) : %.4f\n", compute_aopt(Ak(:,p), Gp));
  end

  while(inc_found)
    % Tracking the swap
    rem_col   = 0;
    sel_col   = 0;
    swap_aopt = cur_aopt;

    % Compute all the downdate traces at once
    V  = (eye(k) + Wk(:, p)*Wk(:, p)') \ Wk(:, p);
    GV = Gpk_chol*V;

    Gvnormsq = sum(GV.^2, 1);
    udotv    = arrayfun(@(i) Wk(:, p(i))'*V(:, i), 1:length(p));
    trdiff   = Gvnormsq ./ (1 - udotv);

    % Downdate traces
    trincs   = cur_aopt + trdiff;

    % Find the columns to swap
    for jj = 1:k
      % Remove the jj column
      p_rem_jj = [p(1:jj-1) p(jj+1:end)];
      jj_aopt  = trincs(jj);

      % Swap in a column
      choices = setdiff(1:m, p);

      V  = (eye(k) + Wk(:, p_rem_jj)*Wk(:, p_rem_jj)') \ Wk(:, choices);
      GV = Gpk_chol*V;

      Gvnormsq = sum(GV.^2, 1);
      udotv    = arrayfun(@(i) Wk(:, choices(i))'*V(:, i), 1:length(choices));
      trdiff   = Gvnormsq ./ (1 + udotv);
  
      % Pick the largest difference in trace
      [trdec, sel_col_idx] = max(trdiff);
  
      % Compute the A-opt difference
      jj_aopt    = jj_aopt - trdec;
      jj_sel_col = choices(sel_col_idx);

      if (jj_aopt < swap_aopt)
        if (verbose)
          fprintf("Intermediate swap found.\n");
          fprintf("Current A-opt: %.4f\n", swap_aopt);
          fprintf("Swapped A-opt: %.4f\n", jj_aopt);
        end
        rem_col   = p(jj);
        sel_col   = jj_sel_col;
        swap_aopt = jj_aopt;
      end
    end
    % Swap if needed
    if (swap_aopt < cur_aopt)
      % Swap found
      p         = [setdiff(p, rem_col) sel_col];
      cur_aopt  = swap_aopt;
      num_swaps = num_swaps + 1;
      if (verbose)
        fprintf("Swap Found!\n");
        fprintf("Current A-opt: %.4f\n", cur_aopt);
        fprintf("Swapped A-opt: %.4f\n", swap_aopt);
        %fprintf("Swapped A-opt (chk) : %.4f\n", compute_aopt(Ak(:,p), Gp));
      end
    else
      % No swap found
      if (verbose)
        fprintf("No swap found.\n");
      end
      inc_found = false;
    end
  end
end
