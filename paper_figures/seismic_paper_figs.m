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
rsvd_p  = 20;    % Oversampling for randsvd
rsvd_q  = 1;     % Subspace iterations for randsvd

%% Output params
save_file = true;
save_figs = true;

out_dir  = "./results";
expname  = "prseismic";
tstamp   = string(datetime("now", "Format", "y.MM.d'T'HH:mm"));

if (save_file)
  out_file = fullfile(out_dir,strcat(expname,"-",tstamp,'.mat'));
  fprintf("Saving data at : %s\n", out_file);
end

if (save_figs)
  fig_dir          = fullfile(out_dir, strcat(expname,"-figs-",tstamp));
  [st, msg, msgID] = mkdir(fig_dir);
  assert(st, "Error creating figures folder.");
  fprintf("Saving figures at : %s\n", fig_dir);
end

%% Setup the 2D seismic problem

ProblemOptions = PRset(...
    'phantomImage', 'smooth',... % phantomImage
    'wavemodel', wmodel,...      % wavemodel - string that defines the type of problem
    's', nsrcs,...               % s - number of sources in the right side of the domain.
    'p', nrecs,...               % p - number of receivers (seismographs)
    'sm',retspmat);              % sm - logical; if true (default) then A is a sparse matrix, otherwise
                                 %      it is a function handle.
[F, d, xt, ProblemInfo] = PRseismic(n, ProblemOptions);

% Rescale them to [0, 1]^2
F = F/n;
d = d/n;

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

%% Form the preconditioned operator
% Prior stuff
Gp_inv = K*(M\K);
mu     = zeros(size(xt));

% Noise stuff
Gn      = sigma2 * eye(size(F,1));

% Form the preconditioned operator (special case)
% In general you want: Apr = (Gp_sqrt)*(F')*(Gn_inv_sqrt)
Apr = sqrt(alpha)*R*(K\F')/sigma;

%% Check out the spectra of A
[Upr, Spr, Vpr] = svd(Apr);
s_Apr   = diag(Spr);
pow_Apr = cumsum(s_Apr.^2) / sum(s_Apr.^2);

%% Solve the full problem
[x_full, ~, ~, ~, ~] = solve_invprob(F, Gn, Gp_inv, bn, mu, alpha,...
                                [], use_cg);

full_dopt = compute_dopt(Apr);
full_rerr = norm(x_full-xt)/norm(xt);
fprintf("Full operator D-optimality  : %.4f\n", full_dopt);
fprintf("Full operator relative error: %.4f\n", full_rerr);

%% CSSP arguments
m           = size(Apr,2); % No. of candidate columns
num_sensors = [1:20 30:10:100];
kalpha      = alpha;

%% Run CSSP
rng(seed, "twister");

% Regular CSSP
% Store the solutions
qr_sols  = cell(length(num_sensors),1);
hy_sols  = cell(length(num_sensors),1);
gd_sols  = cell(length(num_sensors),1);
af_sols  = cell(length(num_sensors),1);

% Store the sampling matrix
qr_Ss  = cell(length(num_sensors),1);
hy_Ss  = cell(length(num_sensors),1);
gd_Ss  = cell(length(num_sensors),1);
af_Ss  = cell(length(num_sensors),1);
af_Ys  = cell(length(num_sensors),1);

% Metric results
qr_rerrs  = zeros(length(num_sensors),1);
hy_rerrs  = zeros(length(num_sensors),1);
gd_rerrs  = zeros(length(num_sensors),1);
af_rerrs  = zeros(length(num_sensors),1);

qr_dopts  = zeros(length(num_sensors),1);
hy_dopts  = zeros(length(num_sensors),1);
gd_dopts  = zeros(length(num_sensors),1);
af_dopts  = zeros(length(num_sensors),1);

qr_norms_VktSinv  = zeros(length(num_sensors),1);
hy_norms_VktSinv  = zeros(length(num_sensors),1);
gd_norms_VktSinv  = zeros(length(num_sensors),1);
af_norms_VktSinv  = zeros(length(num_sensors),1);

idx = 1;
for kidx=1:length(num_sensors)
  k  = num_sensors(kidx);
  Vk = Vpr(:,1:k);

  % Run the rand QR algo
  [~, qr_dopt, S_qr, ~] = randcssp(Apr, k, "qr", rsvd_p, rsvd_q);
  [x_qr, ~, ~, ~, ~]    = solve_invprob(F, Gn, Gp_inv, bn, mu, kalpha,...
                              S_qr', use_cg);
  qr_rerr               = norm(x_qr - xt)/norm(xt);

  % Compute the conditioning
  VktS                    = Vk'*S_qr;
  if (rank(VktS) == k)
    qr_norms_VktSinv(kidx) = norm(inv(VktS));
  else
    qr_norms_VktSinv(kidx) = inf;
  end


  % Run the hybrid algo
  [~, hy_dopt, S_hy, ~] = randcssp(Apr, k, "hybrid", rsvd_p, rsvd_q);
  [x_hy, ~, ~, ~, ~]    = solve_invprob(F, Gn, Gp_inv, bn, mu, kalpha,...
                              S_hy', use_cg);
  hy_rerr               = norm(x_hy - xt)/norm(xt);
  
  % Compute the conditioning
  VktS                    = Vk'*S_hy;
  if (rank(VktS) == k)
    hy_norms_VktSinv(kidx) = norm(inv(VktS));
  else
    hy_norms_VktSinv(kidx) = inf;
  end


  % Greedy
  [~, gd_dopt, S_gd, ~] = greedydopt_mf(Apr, k);
  [x_gd, ~, ~, ~, ~]    = solve_invprob(F, Gn, Gp_inv, bn, mu, kalpha,...
                                S_gd', use_cg);
  gd_rerr               = norm(x_gd - xt)/norm(xt);
  % Compute the conditioning
  VktS                     = Vk'*S_gd;
  if (rank(VktS) == k)
    gd_norms_VktSinv(kidx) = norm(inv(VktS));
  else
    gd_norms_VktSinv(kidx) = inf;
  end


  % Adjoint-free
  [~, af_dopt, S_af, Y1] = randcssp(Apr, k, "adjfree", rsvd_p);
  [x_af, ~, ~, ~, ~]     = solve_invprob(F, Gn, Gp_inv, bn, mu, kalpha,...
                                S_af', use_cg);
  af_rerr                = norm(x_af - xt)/norm(xt);
  % Compute the conditioning
  VktS                     = Vk'*S_af;
  if (rank(VktS) == k)
    af_norms_VktSinv(kidx) = norm(inv(VktS));
  else
    af_norms_VktSinv(kidx) = inf;
  end

  % Store the results
  % QR
  qr_sols{idx}  = x_qr;
  qr_Ss{idx}    = S_qr;
  qr_dopts(idx) = qr_dopt;
  qr_rerrs(idx) = qr_rerr;

  % Hybrid
  hy_sols{idx}  = x_hy;
  hy_Ss{idx}    = S_hy;
  hy_dopts(idx) = hy_dopt;
  hy_rerrs(idx) = hy_rerr;

  % Greedy
  gd_sols{idx}  = x_gd;
  gd_Ss{idx}    = S_gd;
  gd_dopts(idx) = gd_dopt;
  gd_rerrs(idx) = gd_rerr;

  % Adjoint free
  af_sols{idx}  = x_af;
  af_Ss{idx}    = S_af;
  af_Ys{idx}    = Y1;
  af_dopts(idx) = af_dopt;
  af_rerrs(idx) = af_rerr;

  % Print out the results
  fprintf("D-optimality for k = %d.\n", k)
  fprintf("Full operator : %.4f\n", full_dopt);
  fprintf("QR            : %.4f\n", qr_dopt);
  fprintf("Hy            : %.4f\n", hy_dopt);
  fprintf("Greedy        : %.4f\n", gd_dopt);
  fprintf("Adjfree       : %.4f\n\n", af_dopt);

  fprintf("Relative Error for k = %d.\n", k);
  fprintf("Full operator: %.4f\n", full_rerr);
  fprintf("QR           : %.4f\n", qr_rerr);
  fprintf("Hy           : %.4f\n", hy_rerr);
  fprintf("Greedy       : %.4f\n", gd_rerr);
  fprintf("Adjfree      : %.4f\n\n", af_rerr);

  idx = idx + 1;
end

% Store the results
algo_dopts  = [qr_dopts hy_dopts gd_dopts af_dopts];

algo_rerrs  = [qr_rerrs hy_rerrs gd_rerrs af_rerrs];

algo_conds  = [qr_norms_VktSinv hy_norms_VktSinv...
               gd_norms_VktSinv af_norms_VktSinv];

if (save_file)
  fprintf("Saving data.\n");
  save(out_file);
end

%% Random trials
nruns = 100;
rk_num_sensors = [5 10 20 30 50];
rng(seed, 'twister');

% Store the solutions
rand_sols  = cell(length(rk_num_sensors),1);

% Store the sampling matrix
rand_Ss  = cell(length(rk_num_sensors),1);

% Metric results
rand_rerrs = zeros(length(rk_num_sensors),nruns);
rand_dopts = zeros(length(num_sensors),nruns);

kidx = 1;
for k = rk_num_sensors
  fprintf("Working on k = %d\n", k);

  % Create the sol and sample arrays
  rand_sols{kidx} = cell(nruns,1);
  rand_Ss{kidx}   = cell(nruns,1);

  for ii = 1:nruns
    if (mod(ii,10) == 0)
      fprintf("Working on random design: %d\n", ii);
      if (save_file)
        save(out_file);
      end
    end

    % Get a random selection
    ridx = randsample(m, k, false);
    S_rt = form_selmat(ridx, m);
    S_r  = S_rt';

    % Solve with the random selection
    r_dopt            = compute_dopt(Apr*S_r);
    [x_r, ~, ~, ~, ~] = solve_invprob(F, Gn, Gp_inv, bn, mu, kalpha,...
                              S_r', use_cg);
    r_rerr            = norm(x_r-xt)/norm(xt);

    % Store answers
    rand_sols{kidx}{ii} = x_r;
    rand_Ss{kidx}{ii}   = S_r;

    rand_rerrs(kidx,ii) = r_rerr;
    rand_dopts(kidx,ii) = r_dopt;
  end
  kidx = kidx + 1;
end

if (save_file)
  fprintf("Saving data.\n");
  save(out_file);
end

%% Run a data completion experiment
rng(seed, "twister");
dc_kvals = 1:20;
dc_sols  = cell(length(dc_kvals),1);
dc_pdata = cell(length(dc_kvals),1);
dc_Ss    = cell(length(dc_kvals),1);
dc_Ps    = cell(length(dc_kvals),1);
dc_Vks   = cell(length(dc_kvals),1);

% Metrics captures
dc_derrs = zeros(length(dc_kvals),1);
dc_rerrs = zeros(length(dc_kvals),1);

for kidx=1:length(dc_kvals)
  [~, ~, S_dc, Vk_dc] = randcssp(Apr, dc_kvals(kidx));
  P_dc                = Vk_dc*((S_dc'*Vk_dc)\S_dc');

  % Complete data
  Pbn                 = P_dc*bn;
  dc_derr             = norm(Pbn-bn)/norm(bn);

  % Solve with completed data
  [x_dc, ~, ~, ~, ~] = solve_invprob(F, Gn, Gp_inv, Pbn, mu, ...
                            kalpha, [], use_cg);
  dc_rerr            = norm(x_dc-xt)/norm(xt);

  % Store results
  dc_sols{kidx}  = x_dc;
  dc_pdata{kidx} = Pbn;
  dc_Ss{kidx}    = S_dc;
  dc_Ps{kidx}    = P_dc;
  dc_Vks{kidx}   = Vk_dc;

  dc_derrs(kidx) = dc_derr;
  dc_rerrs(kidx) = dc_rerr;
end

if (save_file)
  fprintf("Saving data.\n");
  save(out_file);
end

%% Start plotting stuff
% Problem introduction and one reconstruction
kidx  = 10;
kval  = num_sensors(kidx);
x_plt = qr_sols{kidx};
S_plt = qr_Ss{kidx};

% Find the selected indices 
[plt_ind, ~, ~] = find(S_plt);
fprintf("Showing QR solution for k = %d.\n", num_sensors(kidx));

figure;
set(gcf, 'units', 'inches');
set(gcf, 'position', [0, 0, 10, 4]);

% Generate sensor locations - (from bottom left to top right);
Xr = [zeros(1, nrecs/2), linspace(0,1,nrecs/2)];
Yr = [linspace(0,1,nrecs/2), ones(1,nrecs/2)];

% Get same color limits
cmin  = inf;
cmax  = -inf;
for x = {xt, x_plt}
  cmin = min(cmin, min(x{1}));
  cmax = max(cmax, max(x{1}));
end

% Plot the true initial conditions
subplot(1,2,1);
pcolor(X, Y, flipud(reshape(xt,n,n))); 
shading interp; 
clim([cmin cmax]); colorbar; hold on;
%plot(Xr, Yr, 'ks'); hold on;
plot(Xr(plt_ind), Yr(plt_ind), 'rx'); hold on;
plot(1,0.5,'bo', 'MarkerFaceColor','b');
axis square;
title('Initial Conditions');

subplot(1,2,2);
pcolor(X, Y, flipud(reshape(x_plt,n,n))); 
shading interp; 
clim([cmin cmax]); colorbar;
axis square;
title('Reconstruction');

if (save_figs)
  figname = sprintf("problem-setup1-%d",kval);
  export_fig(fullfile(fig_dir,figname), "-pdf", "-transparent");
  export_fig(fullfile(fig_dir,figname), "-png", "-transparent");
  close all;
end

%% Problem with a source-receiver Fresnel zone
% Problem introduction and one reconstruction
kidx  = 20;
kval  = num_sensors(kidx);
x_plt = qr_sols{kidx};
S_plt = qr_Ss{kidx};

% Find the selected indices 
[plt_ind, ~, ~] = find(S_plt);
fprintf("Showing QR solution for k = %d.\n", num_sensors(kidx));

figure;
rect    = [0, 0, 15, 4];
fsize   = 10;
ttlsize = 12;

set(gcf, 'units', 'inches');
set(gcf, 'Position', rect);
set(gcf, 'OuterPosition',rect);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'defaultaxesfontsize', fsize);
set(gcf, 'defaulttextfontsize', fsize);

% Generate sensor locations - (from bottom left to top right);
Xr = [zeros(1, nrecs/2), linspace(0,1,nrecs/2)];
Yr = [linspace(0,1,nrecs/2), ones(1,nrecs/2)];

% Grab a Fresnel zone
sel_idx = 1;
x_fz    = F(plt_ind(sel_idx), :);

% Get same color limits
cmin  = inf;
cmax  = -inf;
for x = {xt, x_fz, x_plt}
  cmin = min(cmin, min(x{1}));
  cmax = max(cmax, max(x{1}));
end

% Plot the true initial conditions
subplot(1,3,1);
pcolor(X, Y, flipud(reshape(xt,n,n))); 
shading interp; 
clim([cmin cmax]); colorbar; hold on;
%plot(Xr, Yr, 'ks'); hold on;
plot(Xr(plt_ind), Yr(plt_ind), 'rx'); hold on;
plot(1,0.5,'bo', 'MarkerFaceColor','b');
axis square;
title('Initial Conditions', 'FontSize', ttlsize);

% Plot a Fresnel zone
subplot(1,3,2);
pcolor(X, Y, flipud(reshape(x_fz,n,n))); 
shading interp; 
colorbar; hold on;
%clim([cmin cmax]); colorbar; hold on; % Skip reweithing for F/n.
%plot(Xr, Yr, 'ks'); hold on;
plot(Xr(plt_ind), Yr(plt_ind), 'rx'); hold on;
plot(1,0.5,'bo', 'MarkerFaceColor','b');
axis square;
title('Fresnel Zone', 'FontSize', ttlsize);

% Plot the reconstruction
subplot(1,3,3);
pcolor(X, Y, flipud(reshape(x_plt,n,n))); 
shading interp; 
clim([cmin cmax]); colorbar;
axis square;
title(sprintf('Reconstruction (k=%d)', kval), 'FontSize', ttlsize);


if (save_figs)
  figname = sprintf("problem-setup2-%d",kval);
  export_fig(fullfile(fig_dir,figname), "-pdf", "-transparent");
  export_fig(fullfile(fig_dir,figname), "-png", "-transparent");
  close all;
end

%% Plots against random trials
fprintf("Plotting results against random trials.\n");
rerr_hobjs = cell(length(rk_num_sensors),1);
dopt_hobjs = cell(length(rk_num_sensors),1);
nbins      = 20;

for kidx = 1:length(rk_num_sensors)
  figure;
  rect    = [0, 0, 15, 6];
  fsize   = 16;
  lwidth  = 2;
  labsize = 16;

  set(gcf, 'units', 'inches');
  set(gcf, 'Position', rect);
  set(gcf, 'OuterPosition',rect);
  set(gcf, 'PaperPositionMode', 'auto');
  set(gcf, 'defaultaxesfontsize', fsize);
  set(gcf, 'defaulttextfontsize', fsize);

  % Grab the data
  kval   = rk_num_sensors(kidx);
  rdopts = rand_dopts(kidx, :);
  rrerrs = rand_rerrs(kidx, :);
  
  % Grab the CSSP solutions
  algo_kidx = find(num_sensors == kval);
  qr_dopt   = qr_dopts(algo_kidx);
  qr_rerr   = qr_rerrs(algo_kidx);
  dispname  = sprintf("Top-%d RandGKS", kval);

  % Form the D-optimality histogram
  subplot(1,2,1);
  dhist = histogram(rdopts, nbins); hold on;
  maxyd = max(dhist.Values);
  plot([full_dopt, full_dopt], [0, maxyd], 'k-', 'LineWidth', lwidth);
  hold on;
  plot([qr_dopt, qr_dopt], [0, maxyd], 'r--', 'LineWidth', lwidth); 
  xlabel('D-Optimality Criterion', 'FontSize', labsize);
  ylabel('Counts', 'FontSize', labsize);
  dopt_hobjs{kidx} = dhist;

  % Form the relerr histogram
  subplot(1,2,2);
  rhist = histogram(rrerrs, nbins); hold on;
  maxyr = max(rhist.Values);
  plot([full_rerr, full_rerr], [0, maxyr], 'k-', 'LineWidth', lwidth); 
  hold on;
  plot([qr_rerr, qr_rerr], [0, maxyr], 'r--', 'LineWidth',lwidth); 
  xlabel('Relative Error', 'FontSize', labsize);
  ylabel('Counts', 'FontSize',labsize);
  rerr_hobjs{kidx} = rhist;
  legend({'','Full',dispname}, 'FontSize',labsize);

  if (save_figs)
    figname = sprintf("histogram-%d", kval);
    export_fig(fullfile(fig_dir,figname), "-png", "-transparent");
    print(fullfile(fig_dir,figname), "-depsc");
    close all;
  end
end

%% Plots against random trials (break x-axis)
fprintf("Plotting results against random trials (with break).\n");

for kidx = 1:length(rk_num_sensors)
  figure;
  rect    = [0, 0, 15, 6];
  fsize   = 12;
  lwidth  = 2;
  labsize = 15;

  set(gcf, 'units', 'inches');
  set(gcf, 'Position', rect);
  set(gcf, 'OuterPosition',rect);
  set(gcf, 'PaperPositionMode', 'auto');
  set(gcf, 'defaultaxesfontsize', fsize);
  set(gcf, 'defaulttextfontsize', fsize);

  % Grab the data
  kval   = rk_num_sensors(kidx);
  rdopts = rand_dopts(kidx, :);
  rrerrs = rand_rerrs(kidx, :);
  
  % Grab the CSSP solutions
  algo_kidx = find(num_sensors == kval);
  qr_dopt   = qr_dopts(algo_kidx);
  qr_rerr   = qr_rerrs(algo_kidx);
  dispname  = sprintf("Top-%d RandGKS", kval);

  % Form the D-optimality histogram
  subplot(1,2,1);
  dhist = histogram(rdopts, nbins); hold on;
  maxyd = max(dhist.Values);
  plot([full_dopt, full_dopt], [0, maxyd], 'k-', 'LineWidth', lwidth); 
  hold on;
  plot([qr_dopt, qr_dopt], [0, maxyd], 'r--', 'LineWidth', lwidth);
  xlabel('D-Optimality Criterion', 'FontSize', labsize);
  ylabel('Counts', 'FontSize', labsize);
  legend({'','Full',dispname}, 'FontSize', labsize);

  xlim([min(rdopts)-0.2 full_dopt+0.5]);
  hb = breakxaxis([(qr_dopt + 0.5) (full_dopt - 0.5)]);

  % Form the relerr histogram
  subplot(1,2,2);
  rhist = histogram(rrerrs, nbins); hold on;
  maxyr = max(rhist.Values);
  plot([full_rerr, full_rerr], [0, maxyr], 'k-', 'LineWidth', lwidth);
  hold on;
  plot([qr_rerr, qr_rerr], [0, maxyr], 'r--', 'LineWidth', lwidth);
  xlabel('Relative Error', 'FontSize', labsize);
  ylabel('Counts', 'FontSize', labsize);
  legend({'','Full',dispname}, 'FontSize', labsize);

  if (save_figs)
    figname = sprintf("bhistogram-%d", kval);
    export_fig(fullfile(fig_dir,figname), "-png", "-transparent");
    print(fullfile(fig_dir,figname), "-depsc");
    close all;
  end
end

%% Plot the results versus k
figure;
%rect    = [0, 0, 12, 4];
fsize   = 12;
lwidth  = 1.5;
labsize = 12;

set(gcf, 'units', 'inches');
%set(gcf, 'Position', rect);
%set(gcf, 'OuterPosition',rect);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'defaultaxesfontsize', fsize);
set(gcf, 'defaulttextfontsize', fsize);

plot(num_sensors, qr_rerrs, 'x-', 'DisplayName', 'RandGKS Relerr', ...
  'LineWidth', lwidth);
hold on;
plot(num_sensors, full_rerr*ones(length(num_sensors),1), 'k--', ...
          'DisplayName', 'Full Relerr', 'LineWidth', lwidth);
xlabel('Number of sensors selected', 'FontSize', labsize);
ylabel('Relative Error', 'FontSize', labsize);
ylim([min(qr_rerrs)-0.05, max(qr_rerrs)+0.05]);
yyaxis right;
plot(num_sensors, qr_dopts, 'o-', 'DisplayName', 'RandGKS D-opt', ...
  'LineWidth', lwidth);
hold on;
plot(num_sensors, full_dopt*ones(length(num_sensors),1), 'k:', ...
          'DisplayName', 'Full D-Opt', 'LineWidth', lwidth);
ylabel("D-Optimality", 'FontSize', labsize);
legend('FontSize', labsize);

if (save_figs)
  figname = sprintf("rdvaryk-%d", kval);
  export_fig(fullfile(fig_dir,figname), "-png", "-transparent");
  print(fullfile(fig_dir,figname), "-depsc");
  close all;
end

%% Plot the data completion experiments

rect    = [0, 0, 8, 6];
fsize   = 16;
lwidth  = 1.5;
labsize = 16;
ttlsize = 20;

figure;

set(gcf, 'units', 'inches');
set(gcf, 'Position', rect);
set(gcf, 'OuterPosition',rect);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'defaultaxesfontsize', fsize);
set(gcf, 'defaulttextfontsize', fsize);

plot(dc_kvals, dc_derrs, 'ro-', 'LineWidth', lwidth); 
hold on;
plot(dc_kvals, ones(length(dc_kvals),1)*npct, 'k--', ...
  'LineWidth', lwidth);
ylim([npct-0.01, max(dc_derrs)+0.01]);
xlabel('Number of sensors', 'FontSize', labsize);
ylabel("Relative Error", 'FontSize', labsize);
title('Data Completion Error', 'FontSize', ttlsize);
legend({'Completed Data', 'Noise Level'}, 'FontSize', labsize);
if (save_figs)
  figname = sprintf("dcderr-%d", max(dc_kvals));
  export_fig(fullfile(fig_dir,figname), "-png", "-transparent");
  export_fig(fullfile(fig_dir,figname), "-pdf", "-transparent");
  print(fullfile(fig_dir,figname), "-depsc");
  close all;
end

figure;

set(gcf, 'units', 'inches');
set(gcf, 'Position', rect);
set(gcf, 'OuterPosition',rect);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'defaultaxesfontsize', fsize);
set(gcf, 'defaulttextfontsize', fsize);

plot(dc_kvals, dc_rerrs, 'ro-', 'LineWidth', lwidth);hold on;
plot(dc_kvals, ones(length(dc_kvals),1)*full_rerr, 'k--', ...
  'LineWidth', lwidth);
ylim([full_rerr-0.02, max(dc_rerrs)+0.05]);
xlabel('Number of sensors', 'FontSize', labsize);
ylabel("Relative Error", 'FontSize', labsize);
title('Reconstruction Error', 'FontSize', ttlsize);
legend({'Completed Data', 'Full Data'}, 'FontSize', labsize);
if (save_figs)
  figname = sprintf("dcrerr-%d", max(dc_kvals));
  export_fig(fullfile(fig_dir,figname), "-png", "-transparent");
  export_fig(fullfile(fig_dir,figname), "-pdf", "-transparent");
  print(fullfile(fig_dir,figname), "-depsc");
  close all;
end

