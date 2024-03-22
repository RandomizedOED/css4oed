function [kidx, dopt, S, Vk] = detcssp(A, k, typ, qrtyp)
% DETCSSP Deterministic column subset selection algorithms based on
% the GKS approach.
% Please refer to the paper for exact details. TODO: Add citations
%
% Input:
%  A     - Input matrix or linear operator
%  k     - Number of columns to be selected
%  typ   - SVD algorithm to employ (dense/svds)
%  qrtyp - QR algorithm to employ (qr/srrqr)
% Output:
%  pidx - Columns selected
%  dopt - D-optimality of selected columns
%  S    - Column sampling matrix
%  Vk   - Right singular vectors of A
  arguments
    A
    k (1,1) {mustBeInteger, mustBePositive}
    typ (1,1) {mustBeMember(typ, ["svds", "dense"])} = "svds"
    qrtyp (1,1) {mustBeMember(qrtyp, ["qr", "srrqr"])} = "qr"
  end

  % Create a function handle for A (needed for svds)
  function z = Afun(x, y)
    if (y == "notransp") 
      z = A*x;
    elseif (y == "transp") 
      z = A'*x;
    end
  end

  % Compute the SVD
  if (typ == "svds")
    [~, ~, Vk] = svds(@Afun, size(A), k);
  elseif (typ == "dense")
    if (~isa(A, "double"))
      A = A * eye(size(A, 2));
    end
    [~, ~, V] = svd(A);
    Vk = V(:, 1:k);
  else
    error("SVD solver not supported.");
  end

  % Compute QR with pivoting
  if (qrtyp == "qr")
    [~, ~, p] = qr(Vk', 'econ', 'vector');
    kidx      = p(1:k);
  elseif (qrtyp == "srrqr")
    [kidx, ~] = srrqr_select(Vk', k);
  else
    error("QR algorithm not supported.");
  end
 
  % Compute D-optimality
  St        = form_selmat(kidx, size(A, 2));
  S         = St';
  dopt      = compute_dopt(A*S);
end
