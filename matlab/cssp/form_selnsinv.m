function [sel_Gn_inv] = form_selnsinv(Gn, S)
% FORM_SELNSINV Compute the inverse of the sampled noise covariance
% matrix (precision matrix). The sampling is performed via the 
% row selection matrix S.
%
% Input:
%  Gn - Noise covariance matrix
%  S  - Sensor/row selection matrix
% Output:
%  sel_Gn_inv - Precision matrix of the selected sensors
  if (~isa(Gn, "double"))
    Gn = Gn * eye(size(Gn, 2));
  end

  sel_Gn     = (S * Gn) * S';
  sel_Gn_inv = pinv(sel_Gn);
end
