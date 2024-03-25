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
densify = false; % form the forward operator
npct    = 0.02;  % noise percentage
use_cg  = false; % Solver options

%% Output params
save_file = true;

out_dir  = "./results";
expname  = "prseismic-timing";
tstamp   = string(datetime("now", "Format", "y.MM.d'T'HH:MM"));

if (save_file)
  out_file = fullfile(out_dir,strcat(expname,"-",tstamp,'.mat'));
  fprintf("Saving data at : %s\n", out_file);
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
num_sensors = [10 50]; % Number of sensors to select
rsvd_p      = 20;      % Oversampling for randsvd
rsvd_qs     = [0 1 2]; % Subspace iterations for randsvd

nruns = 10;

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

%% Display times
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
                         median(af_times(kk, qq, :))];

    min_times{kk, qq} = [min(qr_times(kk, qq, :)), ...
                         min(hy_times(kk, qq, :)), ...
                         min(gd_times(kk, qq, :)), ...
                         min(af_times(kk, qq, :))];

    max_times{kk, qq} = [max(qr_times(kk, qq, :)), ...
                         max(hy_times(kk, qq, :)), ...
                         max(gd_times(kk, qq, :)), ...
                         max(af_times(kk, qq, :))];
  
    avg_times{kk, qq} = [mean(qr_times(kk, qq, :)), ...
                         mean(hy_times(kk, qq, :)), ...
                         mean(gd_times(kk, qq, :)), ...
                         mean(af_times(kk, qq, :))];

    std_times{kk, qq} = [std(qr_times(kk, qq, :)), ...
                         std(hy_times(kk, qq, :)), ...
                         std(gd_times(kk, qq, :)), ...
                         std(af_times(kk, qq, :))];
  
    num_mvs{kk, qq}   = [(2*rsvd_q+1)*(k + rsvd_p), ...
                         (2*rsvd_q+1)*(k + rsvd_p), ...
                         size(Apr, 2), ...
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
    fprintf("k = %d q=%d\n", k, rsvd_q);
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
    fprintf("Adjfree    %8.4f %8.4f %8.4f %8.4f %5.4f %d\n", ...
               med_times{kk, qq}(4), min_times{kk, qq}(4), ...
               max_times{kk, qq}(4), avg_times{kk, qq}(4), ...
               std_times{kk, qq}(4), num_mvs{kk, qq}(4));
    fprintf("\n\n");
  end
end