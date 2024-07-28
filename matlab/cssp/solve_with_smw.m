function [V] = solve_with_smw(C, U)
% Helper function to solve the following with Sherman-Morrison-Woodbury
%                (I + C C^T) V = U
% where C is a n x k low-rank matrix.
  [~, k] = size(C);
  
  B = eye(k) + C'*C;
  V = U - (C*(B\(C'*U)));
end
