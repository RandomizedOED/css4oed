%% Add cssp stuff to path and seed
addpath(genpath("../matlab/"));
seed = 4177;
rng(seed, "twister");

%% User parameters
nx      = 2^6;   % grid points
nobs    = 10;    % observation points (in 1D)
kappa2  = 80;    % prior nugget
alpha   = 0.1;   % regularization parameter
npct    = 0.02;  % noise percentage
use_cg  = false; % Solver options

%% Output params
save_file = true;
save_figs = true;

out_dir  = "./results";
expname  = "prdiffusion-timing";
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

%% Form the preconditioned operator
% Prior stuff
Gp_inv  = K*(M\K);
mu      = zeros(size(xt));

% Noise stuff
Gn      = sigma2 * eye(size(F,1));

% Form the preconditioned operator (special case)
% In general you want: Apr = (Gp_sqrt)*(F')*(Gn_inv_sqrt)
% Apr = sqrt(alpha)*R*(K\F')/sigma;
Apr = funMat(@(x) (sqrt(alpha)/sigma)*(R*(K\(F'*x))),...
              @(x) (sqrt(alpha)/sigma)*(F*(K'\(R'*x))),...
              [size(F,2), size(F,1)]);

%% Timing parameters
m           = size(Apr, 2); % Number of candidates
num_sensors = [10 30];      % Number of sensors to select
rsvd_p      = 20;           % Oversampling for randsvd
rsvd_qs     = 1;            % Subspace iterations for randsvd
densify     = true;         % Make the operator dense
nruns       = 10;           % Number of runs to average

if (densify)
  Apr = Apr*eye(m);
end

%% Time the operators
fwop_times = zeros(nruns, 1);
adop_times = zeros(nruns, 1);

% Time forward operator
for ii = 1:nruns
  x = rand(size(Apr, 1), 1);
  tic;
  y = Apr'*x;
  fwop_times(ii) = toc;
end

% Time adjoint operator
for ii = 1:nruns
  x = rand(size(Apr, 2), 1);
  tic;
  y = Apr*x;
  adop_times(ii) = toc;
end

if (save_file)
  fprintf("Saving data.\n");
  save(out_file);
end

%% Measure all times
qr_times = zeros(length(num_sensors), length(rsvd_qs), nruns);
hy_times = zeros(length(num_sensors), length(rsvd_qs), nruns);
gd_times = zeros(length(num_sensors), length(rsvd_qs), nruns);
gf_times = zeros(length(num_sensors), length(rsvd_qs), nruns);
af_times = zeros(length(num_sensors), length(rsvd_qs), nruns);

for kk = 1:length(num_sensors)
  for qq = 1:length(rsvd_qs)
    k      = num_sensors(kk);
    rsvd_q = rsvd_qs(qq);
    fprintf("Running with k = %d and q = %d.\n", k, rsvd_q);
    
    % Time RandGKS
    fprintf("Working on RandGKS.\n");
    
    for ii=1:nruns
      tic;
      [qr_idx, ~, ~, ~] = randcssp(Apr, k, "qr", rsvd_p, rsvd_q);
      qr_times(kk, qq, ii) = toc;
    end

    % Time Hybrid
    fprintf("Working on Hybrid.\n");
    
    for ii=1:nruns
      tic;
      [hy_idx, ~, ~, ~] = randcssp(Apr, k, "hybrid", rsvd_p, rsvd_q);
      hy_times(kk, qq, ii) = toc;
    end
    
    % Time Greedy
    fprintf("Working on Greedy.\n");

    for ii=1:nruns
      tic;
      [gd_idx, ~, ~, ~] = greedydopt(Apr, k);
      gd_times(kk, qq, ii) = toc;
    end

    % Time Greedy (matrix-free)
    fprintf("Working on Greedy (matrix-free).\n");

    for ii=1:nruns
      tic;
      [gf_idx, ~, ~, ~] = greedydopt_mf(Apr, k);
      gf_times(kk, qq, ii) = toc;
    end

    % Time Adjoint-free
    fprintf("Working on Adjoint-free.\n");

    for ii=1:nruns
      tic;
      [af_idx, ~, ~, ~] = randcssp(Apr, k, "adjfree", rsvd_p);
      af_times(kk, qq, ii) = toc;
    end

    if (save_file)
      fprintf("Saving data.\n");
      save(out_file);
    end
    fprintf("\n\n");
  end
end

%% Aggregate times
med_times = cell(length(num_sensors), length(rsvd_qs));
min_times = cell(length(num_sensors), length(rsvd_qs));
max_times = cell(length(num_sensors), length(rsvd_qs));
avg_times = cell(length(num_sensors), length(rsvd_qs));
std_times = cell(length(num_sensors), length(rsvd_qs));
num_mvs   = cell(length(num_sensors), length(rsvd_qs));

for kk = 1:length(num_sensors)
  for qq = 1:length(rsvd_qs)
    k      = num_sensors(kk);
    rsvd_q = rsvd_qs(qq);

    % Aggregate the times
    med_times{kk, qq} = [median(qr_times(kk, qq, :)), ...
                         median(hy_times(kk, qq, :)), ...
                         median(gd_times(kk, qq, :)), ...
                         median(gf_times(kk, qq, :)), ...
                         median(af_times(kk, qq, :))];

    min_times{kk, qq} = [min(qr_times(kk, qq, :)), ...
                         min(hy_times(kk, qq, :)), ...
                         min(gd_times(kk, qq, :)), ...
                         min(gf_times(kk, qq, :)), ...
                         min(af_times(kk, qq, :))];

    max_times{kk, qq} = [max(qr_times(kk, qq, :)), ...
                         max(hy_times(kk, qq, :)), ...
                         max(gd_times(kk, qq, :)), ...
                         max(gf_times(kk, qq, :)), ...
                         max(af_times(kk, qq, :))];
  
    avg_times{kk, qq} = [mean(qr_times(kk, qq, :)), ...
                         mean(hy_times(kk, qq, :)), ...
                         mean(gd_times(kk, qq, :)), ...
                         mean(gf_times(kk, qq, :)), ...
                         mean(af_times(kk, qq, :))];

    std_times{kk, qq} = [std(qr_times(kk, qq, :)), ...
                         std(hy_times(kk, qq, :)), ...
                         std(gd_times(kk, qq, :)), ...
                         std(gf_times(kk, qq, :)), ...
                         std(af_times(kk, qq, :))];
  
    num_mvs{kk, qq}   = [(2*rsvd_q+2)*(k + rsvd_p), ...
                         (2*rsvd_q+2)*(k + rsvd_p), ...
                         m, ...
                         0.5*(k*(2*m - k + 1)), ...
                         (k + rsvd_p)];

  end
end

if (save_file)
  fprintf("Saving data.\n");
  save(out_file);
end

%% Print operator results
fprintf("Oper  Med    Min    Max    Avg    Std\n");
fprintf("Fw    %5.4f %5.4f %5.4f %5.4f %5.4f\n", ...
               median(fwop_times), min(fwop_times), max(fwop_times), ...
               mean(fwop_times), std(fwop_times));
fprintf("Adj   %5.4f %5.4f %5.4f %5.4f %5.4f\n", ...
               median(adop_times), min(adop_times), max(adop_times), ...
               mean(adop_times), std(adop_times));
fprintf("\n\n");

%% Print algorithm results

for kk = 1:length(num_sensors)
  for qq = 1:length(rsvd_qs)
    k      = num_sensors(kk);
    rsvd_q = rsvd_qs(qq);
    fprintf("k = %d q = %d\n", k, rsvd_q);
    fprintf("Algorithm  Med      Min      Max      Avg      Std    MVs\n");
    fprintf("QR         %8.4f %8.4f %8.4f %8.4f %5.4f %d\n", ...
               med_times{kk, qq}(1), min_times{kk, qq}(1), ...
               max_times{kk, qq}(1), avg_times{kk, qq}(1), ...
               std_times{kk, qq}(1), num_mvs{kk, qq}(1));
    fprintf("Hybrid     %8.4f %8.4f %8.4f %8.4f %5.4f %d\n", ...
               med_times{kk, qq}(2), min_times{kk, qq}(2), ...
               max_times{kk, qq}(2), avg_times{kk, qq}(2), ...
               std_times{kk, qq}(2), num_mvs{kk, qq}(2));
    fprintf("Greedy     %8.4f %8.4f %8.4f %8.4f %5.4f %d\n", ...
               med_times{kk, qq}(3), min_times{kk, qq}(3), ...
               max_times{kk, qq}(3), avg_times{kk, qq}(3), ...
               std_times{kk, qq}(3), num_mvs{kk, qq}(3));
    fprintf("Greedy (F) %8.4f %8.4f %8.4f %8.4f %5.4f %d\n", ...
               med_times{kk, qq}(4), min_times{kk, qq}(4), ...
               max_times{kk, qq}(4), avg_times{kk, qq}(4), ...
               std_times{kk, qq}(4), num_mvs{kk, qq}(4));
    fprintf("Adjfree    %8.4f %8.4f %8.4f %8.4f %5.4f %d\n", ...
               med_times{kk, qq}(5), min_times{kk, qq}(5), ...
               max_times{kk, qq}(5), avg_times{kk, qq}(5), ...
               std_times{kk, qq}(5), num_mvs{kk, qq}(5));
    fprintf("\n\n");
  end
end

%% Plot a matvecs vs k plot
mv_rsvd = @(k, p, q) (2*q + 2)*(k + p); 
mv_af   = @(k, p) k+p;
mv_gd   = @(k, m) (m*k) + (k/2) - (k.^2/2);

% Figure parameters
plot_cols = 1:50;
plot_p    = 20;
plot_qs   = [0 1 2];

% Figure properties
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

% GKS methods
for plot_q = plot_qs
  semilogy(plot_cols, mv_rsvd(plot_cols, plot_p, plot_q), 'o-' ,...
      'DisplayName', sprintf("GKS (p = %d, q = %d)", plot_p, plot_q), ...
      'LineWidth', lwidth);
  hold on;
end

% Adjoint-free method
semilogy(plot_cols, mv_af(plot_cols, plot_p), 'x-', ...
     'LineWidth', lwidth, ...
     'DisplayName', sprintf("RAF (p = %d)", plot_p));
hold on;

% Greedy method
semilogy(plot_cols, mv_gd(plot_cols, m), 's-', ...
  'LineWidth', lwidth, 'DisplayName', "Greedy");
hold on;

% Densifying the matrix
semilogy(plot_cols, m*ones(length(plot_cols), 1), 'k--', ...
  'LineWidth', lwidth, 'DisplayName', 'Dense');

title("Heat", 'FontSize', ttlsize);
xlabel("Number of sensors", 'FontSize', labsize);
ylabel("Number of PDE solves", 'FontSize', labsize);
legend('NumColumns', 2, 'Location', 'northwest');

if (save_figs)
  figname = sprintf("mvscl-%d", max(plot_cols));
  export_fig(fullfile(fig_dir,figname), "-png", "-transparent");
  export_fig(fullfile(fig_dir,figname), "-pdf", "-transparent");
  print(fullfile(fig_dir,figname), "-depsc");
  close all;
end
