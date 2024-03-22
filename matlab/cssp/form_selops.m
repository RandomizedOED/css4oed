function [sel_F, sel_Gn_inv, sel_y] = form_selops(F, Gn, y, S)
% FORM_SELOPS Form the sampled forward operator, sampled noise precision matrix, 
% and the sampled data. The sampling is performed via the row selection matrix S.
%
% Input:
%  F  - Forward operator
%  Gn - Noise covariance matrix
%  y  - Data/observations
%  S  - Sensor/row selection matrix (optional, default is to select all sensors)
% Output:
%  sel_F      - Sampled forward operator
%  sel_Gn_inv - Precision matrix of the selected sensors
%  sel_y      - Sampled observations
  % No selection matrix passed
  if ((nargin == 3) || (isempty(S)))
    if (~isa(Gn, "double"))
      Gn = Gn * eye(size(Gn, 2));
    end

    sel_F      = F;
    sel_Gn_inv = pinv(Gn);
    sel_y      = y;
  else
    sel_F      = S * F;
    sel_Gn_inv = form_selnsinv(Gn, S);
    sel_y      = S * y; 
  end
end
