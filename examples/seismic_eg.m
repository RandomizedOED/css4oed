%% Add cssp stuff to path and seed
addpath(genpath("../matlab/"));
seed = 4177;
rng(seed, "twister");

%% User parameters
% Model parameters
n        = 2^6;       % Image size
wmodel   = 'fresnel'; % wavemodel
nsrcs    = 1;         % no. of sources
nrecs    = 256;       % no. of receivers
retspmat = true;      % matrix return type

kappa2  = 80;    % prior nugget
alpha   = 0.1;   % regularization parameter
densify = true;  % form the forward operator
npct    = 0.02;  % noise percentage
use_cg  = false; % Solver options

%% Setup the 2D seismic problem

ProblemOptions = PRset(...
    'phantomImage', 'smooth',... % phantomImage
    'wavemodel', wmodel,...      % wavemodel - string that defines the type of problem
    's', nsrcs,...               % s - number of sources in the right side of the domain.
    'p', nrecs,...               % p - number of receivers (seismographs)
    'sm',retspmat);              % sm - logical; if true (default) then A is a sparse matrix, otherwise
                                 %      it is a function handle.
[F, d, xt, ProblemInfo] = PRseismic(n, ProblemOptions);

bt = F*xt;
[N,sigma] = WhiteNoise(d, npct);
sigma2 = sigma^2;
bn = d + N;

% Setup the mesh, stiffness, and mass matrices
nx = n-1;
mesh = rect_grid2(0, 1, 0, 1, nx, nx);
[K,M,FreeNodes] = Stiff_Mass(mesh);
ndof = size(mesh.p,1);
R = chol(M);
K = K + kappa2*M;

% Grid for plotting stuff
x = linspace(0,1,n);
y = linspace(0,1,n);
[X,Y] = meshgrid(x,y);

% Densify the operators if needed
if (densify)
  F = full(F);
end

%% Show the problem
figure;
set(gcf, 'units', 'inches');
set(gcf, 'position', [0, 0, 8, 8]);

% Plot the true initial conditions
pcolor(X, Y, flipud(reshape(xt,n,n))); shading interp; colorbar;
hold on;

% Place the receivers along the top and left axes
plot(linspace(0, 1, nrecs/2), ones(nrecs/2, 1), 'ks');
hold on;
plot(zeros(nrecs/2, 1), linspace(0, 1, nrecs/2), 'ks');

% Place the source on the right axis
plot(1, 0.5, 'bo', 'MarkerFaceColor','b');

axis square;
title('Initial Conditions');

%% Form the preconditioned operator
% Prior stuff
Gp_inv = K*(M\K);
mu     = zeros(size(xt));

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
set(gcf, 'position', [0, 0, 8, 3]);

% Get same color limits
cmin  = inf;
cmax  = -inf;
for x = {xt, x_full}
  cmin = min(cmin, min(x{1}));
  cmax = max(cmax, max(x{1}));
end

% Plot the true initial conditions
subplot(1,2,1);
pcolor(X, Y, flipud(reshape(xt,n,n))); shading interp; 
clim([cmin cmax]); colorbar;
axis square;
title('Initial Conditions');

% Plot the observations
subplot(1,2,2);
pcolor(X, Y, flipud(reshape(x_full,n,n)));
shading interp; 
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
for x = {xt, x_full, x_detqr, x_qr}
  cmin = min(cmin, min(x{1}));
  cmax = max(cmax, max(x{1}));
end

% Plot the true initial conditions
subplot(2,2,1);
pcolor(X, Y, flipud(reshape(xt,n,n))); shading interp; 
clim([cmin cmax]); colorbar;
axis square;
title('Initial Conditions');

% Plot the full operator results
subplot(2,2,2);
pcolor(X, Y, flipud(reshape(x_full,n,n))); shading interp; 
clim([cmin cmax]); colorbar;
axis square;
title('Full operator');

% Generate sensor locations - (from bottom left to top right);
Xr = [zeros(1, nrecs/2), linspace(0,1,nrecs/2)];
Yr = [linspace(0,1,nrecs/2), ones(1,nrecs/2)];

% Plot the detcssp results
subplot(2,2,3);
pcolor(X, Y, flipud(reshape(x_detqr,n,n))); shading interp; 
clim([cmin cmax]); colorbar;
hold on;
plot(Xr(detqr_kidx), Yr(detqr_kidx), 'rx'); hold on;
plot(1,0.5,'bo', 'MarkerFaceColor','b');
axis square;
title('DetGKS (QRCP)');

% Plot the randcssp results
subplot(2,2,4);
pcolor(X, Y, flipud(reshape(x_qr,n,n))); shading interp; 
clim([cmin cmax]); colorbar;
hold on;
plot(Xr(qr_kidx), Yr(qr_kidx), 'rx'); hold on;
plot(1,0.5,'bo', 'MarkerFaceColor','b');
axis square;
title('RandGKS (QRCP)');
