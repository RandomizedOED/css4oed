function S = form_selmat(idx, m)
% FORM_SELMAT Generates a row-selection matrix S. 
%       F_s = S*F
% Here F_s is the sampled matrix where rows of F are given in the
% vector idx.
% Input:
%  idx - vector containing the row indices to be selected
%  m   - Total number of rows in F
% Output:
%  S   - COO matrix which performs the selection.
  k = length(idx);
  S = sparse(1:k, idx, ones(size(idx)), k, m);
end
