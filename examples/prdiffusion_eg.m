%% Add cssp stuff to path and seed
addpath(genpath("../matlab/"));
seed = 4177;
rng(seed, "twister");

%% User parameters
nx      = 2^6;   % grid points
nobs    = 10;    % observation points (in 1D)
kappa2  = 80;    % prior nugget
alpha   = 0.1;   % regularization parameter
densify = true;  % form the forward operator
npct    = 0.02;  % noise percentage
use_cg  = false; % Solver options

%% Setup the 2D diffusion problem

% Setup the mesh, stiffness, and mass matrices
mesh = rect_grid2(0, 1, 0, 1, nx, nx);
[K,M,FreeNodes] = Stiff_Mass(mesh);
ndof = size(mesh.p,1);

% Problem setup
n = nx+1;
ProblemOptions = PRset('phantomImage', 'smooth');
[A,~,~, ProblemInfo] = PRdiffusion(n,ProblemOptions);

x = linspace(0,1,n);
y = linspace(0,1,n);
[X,Y] = meshgrid(x,y);

% True solution
f = franke(X,Y);
xt = f(:);

% Observation operator
xr = linspace(0.2,0.8,nobs);
yr = linspace(0.2,0.8,nobs);
[Xr,Yr] = meshgrid(xr,yr);
H = LinInt(x,y,Xr(:),Yr(:));

% Setup the forward operator
F = funMat( @(x) H*A(x, 'notransp'), @(x) A(H'*x, 'transp'),...
              [size(H,1), length(xt)]);
bt = F*xt;
[N,sigma] = WhiteNoise(bt, npct);
sigma2 = sigma^2;
bn = bt + N;

% Form the prior
K = K + kappa2*M;
R = chol(M);

% Densify the operators if needed
if (densify)
  Ft = F'*eye(size(F,1));
  F = Ft';
end

%% Show the problem
figure;
set(gcf, 'units', 'inches');
set(gcf, 'position', [0, 0, 12, 3]);

% Get same color limits
cmin  = inf;
cmax  = -inf;
ttraj = A(xt, 'notransp'); % true trajectory 
for x = {xt, ttraj, bn}
  cmin = min(cmin, min(x{1}));
  cmax = max(cmax, max(x{1}));
end

% Plot the true initial conditions
subplot(1,3,1);
pcolor(X, Y, reshape(xt,n,n)); shading interp; 
clim([cmin cmax]); colorbar;
axis square;
title('Initial Conditions');

% Plot the true trajectory and sensor locations
subplot(1,3,2);
pcolor(X, Y, reshape(ttraj,n,n)); shading interp; 
clim([cmin cmax]); colorbar;
hold on;
plot(Xr, Yr, 'ks');
axis square;
title('Trajectory and Sensors');

% Plot the observations
subplot(1,3,3);
pcolor(Xr, Yr, reshape(bn,sqrt(size(H,1)),sqrt(size(H,1))));
shading interp; 
clim([cmin cmax]); colorbar;
axis square;
title('Observations');

%% Form the preconditioned operator
% Prior stuff
Gp_inv  = K*(M\K);
mu      = zeros(size(xt));

% Noise stuff
Gn      = sigma2 * eye(size(F,1));

% Form the preconditioned operator (special case)
% In general you want: Apr = (Gp_sqrt)*(F')*(Gn_inv_sqrt)
Apr = sqrt(alpha)*R*(K\F')/sigma;

%% Solve the full problem
[x_full, ~, ~, ~, ~] = solve_invprob(F, Gn, Gp_inv, bn, mu, alpha,...
                                [], use_cg);

full_dopt = compute_dopt(Apr);
full_rerr = norm(x_full-xt)/norm(xt);
fprintf("Full operator D-optimality  : %.4f\n", full_dopt);
fprintf("Full operator relative error: %.4f\n", full_rerr);

%% Look at full operator solution
figure;
set(gcf, 'units', 'inches');
set(gcf, 'position', [0, 0, 12, 3]);

% Get same color limits
cmin  = inf;
cmax  = -inf;
for x = {xt, ttraj, x_full}
  cmin = min(cmin, min(x{1}));
  cmax = max(cmax, max(x{1}));
end

% Plot the true initial conditions
subplot(1,3,1);
pcolor(X, Y, reshape(xt,n,n)); shading interp; 
clim([cmin cmax]); colorbar;
axis square;
title('Initial Conditions');

% Plot the true trajectory and sensor locations
subplot(1,3,2);
pcolor(X, Y, reshape(ttraj,n,n)); shading interp; 
clim([cmin cmax]); colorbar;
hold on;
plot(Xr, Yr, 'ks');
axis square;
title('Trajectory and Sensors');

% Plot the observations
subplot(1,3,3);
pcolor(X, Y, reshape(x_full,n,n)); shading interp; 
clim([cmin cmax]); colorbar;
axis square;
title('Full operator reconstuction');

%% CSSP arguments
k      = 30;
kalpha = 0.1;

%% Run CSSP
% SVD + QR
[detqr_kidx, detqr_dopt, S_detqr, ~] = detcssp(Apr, k);

[x_detqr, ~, ~, ~, ~] = solve_invprob(F, Gn, Gp_inv, bn, mu, kalpha,...
                                S_detqr', use_cg);

detqr_rerr = norm(x_detqr-xt)/norm(xt);

% RandSVD + QR
[qr_kidx, qr_dopt, S_qr, ~] = randcssp(Apr, k);

[x_qr, ~, ~, ~, ~] = solve_invprob(F, Gn, Gp_inv, bn, mu, kalpha,...
                                S_qr', use_cg);

qr_rerr = norm(x_qr-xt)/norm(xt);

fprintf("D-optimality.\n")
fprintf("Full operator  : %.4f\n", full_dopt);
fprintf("DetGKS (QRCP)  : %.4f\n", detqr_dopt);
fprintf("RandGKS (QRCP) : %.4f\n\n", qr_dopt);

fprintf("Relative Error.\n");
fprintf("Full operator  : %.4f\n", full_rerr);
fprintf("DetGKS (QRCP)  : %.4f\n", detqr_rerr);
fprintf("RandGKS (QRCP) : %.4f\n", qr_rerr);

%% Look at the solutions returned
figure;
set(gcf, 'units', 'inches');
set(gcf, 'position', [0, 0, 8, 6]);

% Get same color limits
cmin  = inf;
cmax  = -inf;
for x = {xt, ttraj, x_detqr, x_qr}
  cmin = min(cmin, min(x{1}));
  cmax = max(cmax, max(x{1}));
end

% Plot the true initial conditions
subplot(2,2,1);
pcolor(X, Y, reshape(xt,n,n)); shading interp; 
clim([cmin cmax]); colorbar;
axis square;
title('Initial Conditions');

% Plot the true trajectory and sensor locations
subplot(2,2,2);
pcolor(X, Y, reshape(A(xt, 'notransp'),n,n)); shading interp; 
clim([cmin cmax]); colorbar;
hold on;
pts = [Xr(:), Yr(:)];
plot(Xr, Yr, 'ks');
plot(pts(detqr_kidx,1),pts(detqr_kidx,2), 'rx','DisplayName','Det (QR)');
plot(pts(qr_kidx,1),pts(qr_kidx,2), 'bo','DisplayName','Rand (QR)');
axis square;
title('Trajectory and Sensors');

% Plot the detcssp results
subplot(2,2,3);
pcolor(X, Y, reshape(x_detqr,n,n)); shading interp; 
clim([cmin cmax]); colorbar;
axis square;
title('DetGKS (QR)');

% Plot the randcssp results
subplot(2,2,4);
pcolor(X, Y, reshape(x_qr,n,n)); shading interp; 
clim([cmin cmax]); colorbar;
axis square;
title('RandGKS (QR)');
