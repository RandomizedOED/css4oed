% Assemble stiffness and mass matrices 

function [A,M,FreeNodes] = Stiff_Mass(mesh)

% Generate boundary data; reset all boundary markers to Dirichlet / Neumann
mesh.e(:,3) = 2;

% number of nodes
N = max( max(mesh.t(:,1:3)) );

%% Initialization of free nodes.
FreeNodes = 1:N; 

%% Compute geometric quantities and gradient of local basis
[Dphi,area] = gradbasis(mesh.p,mesh.t);  % using the code from iFEM by L. Chen


%% Generate stiffness and mass matrices
NT = size(mesh.t,1);
Mt = zeros(NT,3,3);
At = zeros(NT,3,3);
for i = 1:3
    for j = 1:3        
        At(:,i,j) = (Dphi(:,1,i).*Dphi(:,1,j) + Dphi(:,2,i).*Dphi(:,2,j)).*area;
        Mt(:,i,j) = area*((i==j)+1)/12;
    end    
end

%% Assemble the mass matrix in Omega
M = sparse(N,N);
A = sparse(N,N);
for i = 1:3
    krow = mesh.t(:,i);
    for j = 1:3
        kcol = mesh.t(:,j);        
        M = M + sparse(krow,kcol,Mt(:,i,j),N,N);    
        A = A + sparse(krow,kcol,At(:,i,j),N,N);
    end 
end    
clear At Mt
