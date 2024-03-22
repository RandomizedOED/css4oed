function [Q] = LinInt(x,y, xr, yr)
% [Q] = LinInt(x,y,z,xr,yr,zr)
%
% This function does a local linear interpolation
% computed for each receiver point in turn
%

nx = length(x);
ny = length(y);

np = length(xr);

Q = sparse(np,nx*ny);

for i = 1:np

    % fprintf('Point %d\n',i);
    [dx, ind_x] = findclosest(x,xr(i));
    [dy, ind_y] = findclosest(y',yr(i));


    Dx =  (x(ind_x(2)) - x(ind_x(1)));
    Dy =  (y(ind_y(2)) - y(ind_y(1)));
    
    % Get the row in the matrix
    v = zeros(nx, ny);
    v( ind_x(1),  ind_y(1)) = dx(2)*dy(2);
    v( ind_x(2),  ind_y(1)) = dx(1)*dy(2);
    v( ind_x(1),  ind_y(2)) = dx(2)*dy(1);
    v( ind_x(2),  ind_y(2)) = dx(1)*dy(1);

    vt = v';
    Q(i,:) = vt(:)'/(Dx*Dy);

end
end

function [dx, ind_x] = findclosest(x, xr)

    [~,im] = min(abs(xr-x));
    if  xr - x(im) >= 0    % Point on the left
        ind_x(1) = im;
        ind_x(2) = im+1;
    elseif  xr - x(im) < 0 % Point on the right
        ind_x(1) = im-1;
        ind_x(2) = im;
    end
    dx(1) = xr - x(ind_x(1));
    dx(2) = x(ind_x(2)) - xr;
end