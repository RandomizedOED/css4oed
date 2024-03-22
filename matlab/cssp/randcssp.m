function [pidx, dopt, S, Vk] = randcssp(A, k, typ, p, q, l, beta, seed)
% RANDCSSP Randomized column subset selection algorithms. Three styles of 
% methods are present.
% 1. GKS      - QR performed on right singular vectors
% 2. RAF      - QR performed directly on the sketch (Adjoint-free method)
% 3. Sampling - Sample columns randomly
% Please refer to the paper for exact details. TODO: Add citations
%
% Input:
%  A    - Input matrix or linear operator
%  k    - Number of columns to be selected
%  typ  - CSSP method to employ (qr/srrqr/colsample/hybrid/adjfree)
%  p    - Oversampling for the sketch
%  q    - Number of subspace iterations for randsvd
%  l    - Number of samples for sampling based methods
%  beta - Mixing parameter for sampling based methods
%  seed - Random seed for reproducibility
% Output:
%  pidx - Columns selected
%  dopt - D-optimality of selected columns
%  S    - Column sampling matrix
%  Vk   - Right singular vectors of A (for GKS and sampling methods) or
%         sketch of A for adjoint-free method
  arguments
    A
    k (1,1) {mustBeInteger, mustBeNonnegative}
    typ (1,1) {mustBeMember(typ, ["qr", "srrqr", "colsample", "hybrid", "adjfree"])} = "qr"
    p (1,1) {mustBeInteger, mustBePositive} = k
    q (1,1) {mustBeInteger, mustBeNonnegative} = 2
    l (1,1) {mustBeInteger, mustBeNonnegative} = ceil(k * log(k))
    beta (1,1) {mustBeNonnegative} = 0.9
    seed (1,1) {mustBeInteger} = 0
  end

  if (seed > 0)
    rng(seed, 'twister');
  end

  [m, n] = size(A);
  w      = min(m, n);

  % Fix bounds on p and l
  if (k + p > w)
    p = w - k;
  end
  if (l < k)
    l = 2*k;
  end
  if (l > w)
    l = w;
  end
  
  % Sketch Vk
  if (typ == "adjfree")
    d     = k + p;
    Omega = (1/sqrt(d)).*randn(d, m);
    Vk    = Omega*A;
  else
    [~, ~, Vk] = randsvd(A, k, p, q);
  end

  % Get the columns out
  if (typ == "qr")
    [~, ~, p] = qr(Vk', 'econ', 'vector');
    pidx      = p(1:k);
  elseif (typ == "srrqr")
    [pidx, ~] = srrqr_select(Vk', k);
  elseif (typ == "colsample")
    [~, ~, pidx, ~] = colsample(Vk', l, beta);
  elseif (typ == "hybrid")
    % Note this method is the same as weightedcssp when we 
    % use the unweighted sampler P
    [~, S, smpl, ~] = colsample(Vk', l, beta);
    [~, ~, p]       = qr(Vk'*S, 'econ', 'vector');
    pidx            = smpl(p(1:k));
  elseif (typ == "adjfree")
    [~, ~, p] = qr(Vk, 'econ', 'vector');
    pidx      = p(1:k);
  else
    error("Algo not supported.");
  end

  St   = form_selmat(pidx, n);
  S    = St';
  dopt = compute_dopt(A * S);
end
