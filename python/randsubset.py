import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, svds, aslinearoperator
from scipy.linalg import qr as qrcp
from scipy.sparse import coo_array, csr_array

from pyoed.assimilation.smoothing.fourDVar import VanillaFourDVar
from pyoed.models.error_models.Gaussian import GaussianErrorModel

###############################################################################
#                                                                             #
#                        Operator creation utilities                          #
#                                                                             #
###############################################################################

def create_pyoed_oper(model, obsop, prior, obserr, tspan, chkpts):
  '''
  Create a PyOED operator from a given model.

  :param model  : PyOED time-dependent PDE model
  :param obsop  : PyOED observation operator
  :param prior  : Prior Operator (assumed to Gaussian)
  :prior obserr : PyOED observation error operator
  :param tspan  : (t0, tf) iterable with two entries specifying of
                  the time integration window
  :param chkpts : times at which to store the computed solution,
                  must be sorted and lie within tspan

  :returns F_oper : A PyOED operator which can integrate a given state
                    through tspan and store states at the checkpoints
  '''
  # Create a 4D Var object
  F_oper = VanillaFourDVar()
  
  # Register the problem
  F_oper.register_model(model)
  F_oper.register_assimilation_window(tspan)
  F_oper.register_observation_operator(obsop)

  # Prior and noise can be optional
  if prior is not None:
    F_oper.register_prior_model(prior)

  if obserr is not None:
    F_oper.register_observation_error_model(obserr)
  
  # Register some "fake" observations at checkpoints
  fake_obs = np.random.rand(obsop.shape[0])

  for t in chkpts:
    F_oper.register_observation(t, fake_obs)

  return F_oper

def create_noise_oper(pyoed_noise_model):
  '''
  Converts a PyOED noise model to a SciPy LinearOperator

  :param pyoed_noise_model : PyOED noise model

  :returns A : A SciPy LinearOperator which matvec defined
               as by the noise covariance square root
  '''
  n = pyoed_noise_model.size
  f = lambda x: pyoed_noise_model.covariance_sqrt_matvec(x)
  A = LinearOperator((n, n), matvec=f, rmatvec=f)
  
  return A 

def A_forward_op(F_oper, prior=None, noise=None):
  '''
  Creates a function which can be used as the matvec function
  to SciPy's LinearOperator class.

  :params F_oper : PyOED model forward operator
  :params prior  : PyOED prior operator (should be able to apply
                   the covariance square root)
  :params noise  : PyOED noise operator (should be able to apply
                   the covariance inverse square root)

  :returns apply_forward : A function which takes in a vector and 
                           applies the prior, model, and noise 
                           in order.
  '''
  def apply_forward(x):
    # Flatten stuff
    if (x.ndim > 1):
      x = np.squeeze(x)
        
    chkpts = F_oper.observation_times

    # Apply prior is present
    if prior is not None:
      x = prior.covariance_sqrt_matvec(x)

    # Apply the model
    F_obs, _ = F_oper.apply_forward_operator(x, checkpoints=chkpts, 
                              scale_by_noise=False, save_states=False)

    y = F_obs[chkpts[-1]]

    # Scale by noise covariance if present
    if noise is not None:
      y = noise.covariance_sqrt_inv_matvec(y)

    return y

  return apply_forward

def A_adjoint_op(F_oper, prior=None, noise=None):
  '''
  Creates a function which can be used as the rmatvec function
  to SciPy's LinearOperator class.

  :params F_oper : PyOED model adjoint operator
  :params prior  : PyOED prior operator (should be able to apply
                   the covariance square root)
  :params noise  : PyOED noise operator (should be able to apply
                   the covariance inverse square root)

  :returns apply_adjoint : A function which takes in a vector and 
                           applies the noise, model adjoint, and 
                           prior in order.
  '''
  def apply_adjoint(y):
    # Flatten stuff
    if (y.ndim > 1):
      y = np.squeeze(y)
        
    chkpts = F_oper.observation_times

    # Scale by noise covariance if present
    if noise is not None:
      y = noise.covariance_sqrt_inv_matvec(y)

    # Apply the adjoint
    x, _ = F_oper.apply_forward_operator_adjoint(y, chkpts[-1], 
                    scale_by_noise=False)

    # Apply prior if present
    if prior is not None:
      x = prior.covariance_sqrt_matvec(x)

    return x

  return apply_adjoint

###############################################################################
#                                                                             #
#                               CSSP utilities                                #
#                                                                             #
###############################################################################

def form_selmat(idx, m):
  '''
  Generates a row-selection matrix S. 
      F_s = S * F
  Here F_s is the sampled matrix where rows of F are given in the
  vector idx.

  :param idx : Vector containing the row indices to be selected
  :param m   : Total number of rows in F

  :returns sel_mat : A COO matrix which performs the selection
  '''
  k       = len(idx)
  sel_mat = coo_array((np.ones(k), [np.arange(k), idx]), shape=(k, m))
  
  return sel_mat

# Simple SRRQR algorithm
def srrqr_select(M, k, f=1.0, verbose=False):
  '''
  SRRQR_SELECT A naive version of the strong rank-revealing QR algorithm
  given k. This is Algorithm 4 from [GE96]. Since we are only interested
  in the columns selected we only return the pivots and not the factors.
  Regular QR with column pivoting is used for initializing the pivots.

  :param M : The matrix on which to perform sRRQR
  :param k : Number of columns to be selected
  :param f : Improvement in determinant which causes a swap (f >= 1.0)
  
  :returns p         : Indices/columns selected
  :returns num_swaps : Extra swaps performed over the QRCP selection
  References:
  [GE96] Gu, Ming, and Stanley C. Eisenstat. 
         "Efficient algorithms for computing a strong rank-revealing QR factorization." 
         SIAM Journal on Scientific Computing 17, no. 4 (1996): 848-869.
  '''
  m, n = M.shape

  # QRCP for a good initialisation
  Q, R, P = qrcp(M, mode='economic', pivoting=True)

  # Swap till determinant increases
  inc_found = True
  num_swaps = 0

  while (inc_found):
    A     = R[0:k, 0:k]
    AinvB = np.linalg.solve(A, R[0:k, k:n]) # A^{-1}B

    if verbose:
      print(f"Number of swaps performed: {num_swaps}")
      sdet, logabsdet = np.linalg.slogdet(A)
      print(f"det(A): {sdet * np.exp(logabsdet):e}")

    if (m <= n):
      C = R[k:m, k:n]
    else:
      C = R[k:n, k:n]

    # Compute the column norms of C
    gamma = np.zeros(C.shape[1])
    
    for ccol in range(C.shape[1]):
      gamma[ccol] = np.linalg.norm(C[:, ccol], 2)

    # Find row norms of A^{-1}
    Ainv  = np.linalg.pinv(A)
    omega = np.zeros(Ainv.shape[0])

    for arow in range(A.shape[0]):
      omega[arow] = np.linalg.norm(Ainv[arow, :], 2)

    # Find indices i and j that maximizes AinvB(i, j)^2 + (omega[i]*gamma[j])^2
    tmp  = np.outer(omega, gamma)
    F    = AinvB**2 + tmp**2
    I, J = np.nonzero(F > f)

    if (len(I) == 0): # No swap found
      inc_found = False
    else:
      num_swaps += 1
      i, j       = I[0], J[0]
      
      # Swap and retriangularize
      R[:, [i, j+k]] = R[:, [j+k, i]]
      P[[i, j+k]]    = P[[j+k, i]]
      Q, R           = qrcp(R, mode='economic')

  print(f"Number of swaps performed: {num_swaps}")

  # Return the selected indices
  return P[:k], num_swaps

def randsvd(A, k, p, q=2):
  '''
  RANDSVD A randomized SVD algorithm based on Algorithm 5.1 in [HMT11].
  Computes approximate rank-k SVD of an input matrix A.
  
  :param A : Input matrix or linear operator (matvecs must be defined)
  :param k : Rank of the approximate SVD to be computed
  :param p : Oversampling parameter
  :param q : Number of subspace iterations to be performed

  :returns Uk  : Top-k left singular vectors
  :returns Sk  : Top-k diagonal singular values matrix
  :returns Vkt : Top-k right singular vectors (transposed)
  References:
  [HMT11] Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp. 
          "Finding structure with randomness: Probabilistic algorithms 
          for constructing approximate matrix decompositions." 
          SIAM review 53.2 (2011): 217-288.
  '''
  n     = A.shape[1]
  Omega = np.random.randn(n, k+p)

  Y = A @ Omega
  Q, _ = qrcp(Y, mode='economic')

  for _ in range(q):
    Y1   = A.T @ Q
    Q, _ = qrcp(Y1, mode='economic')

    Y    = A @ Q
    Q, _ = qrcp(Y, mode='economic')

  B    = Q.T @ A
  Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
  U         = Q @ Ub[:, :k]

  return U, s[:k], Vt[:k, :]

def colsample(Vkt, l, beta=0.9):
  '''
  COLSAMPLE Sample the columns of a matrix with orthogonal rows.
  Sampling probabilities is calculated as a mixture of a uniform
  distribution and one proportional to the norms of the columns.
  Please refer to the paper for details. TODO: Add paper citation.

  :param Vkt  : Matrix whose columns are to be sampled
  :param l    : Number of columns to be sampled
  :param beta : Mixing parameter

  :returns P    : Unweighted column sampling matrix
  :returns S    : Weighted column sampling matrix
  :returns smpl : Column indices selected
  :returns pr   : Sampling probabilities
  '''
  # Form the leverage scores
  n   = Vkt.shape[1]
  tau = np.sum(Vkt**2, axis=0)
  tau = tau / np.sum(tau)

  pr  = (beta * tau) + ((1 - beta) * (1/n))

  # Sample with replacement
  smpl = np.random.choice(n, size=l, replace=True, p=pr)

  # Form the mixing matrix
  cols = np.arange(l)
  vals = np.sqrt(1 / (l * pr[smpl]))
  S    = coo_array((vals, [smpl, cols]), shape=(n, l))

  # Form the selection matrix
  P    = coo_array((np.ones_like(smpl), [smpl, cols]), shape=(n, l))

  return P, S, smpl, pr

def compute_dopt(A):
  '''
  Computes the Bayesian D-optimality criterion for an operator F.
               \phi_D (F) = \logdet( I + FF^T)

  :param A: Input operator (will be densified).
  
  :returns dopt : D-optimality
  :returns A    : Densified input operator
  '''
  if (isinstance(A, LinearOperator)):
    # Form the full matrix
    A = A @ np.eye(A.shape[1])

  # Compute the D-optimality criteria
  sgn, lgdet = np.linalg.slogdet(np.eye(A.shape[0]) + A @ A.T)
  dopt       = sgn * lgdet
  
  return dopt, A

###############################################################################
#                                                                             #
#                               CSSP methods                                  #
#                                                                             #
###############################################################################

def greedydopt(A, k):
  '''
  Perform greedy column subset selection by maximizing the
  D-optimal criterion for each column sequentially.
  
  :param A: The matrix on which to perform CSSP (will be densified)
  :param k: Number of columns to be selected

  :returns idx  : Indices/columns selected
  :returns dopt : D-optimality of the selected columns
  :returns S    : Column selection matrix
  :returns det  : Determinant of the selected columns
  '''
  if (isinstance(A, LinearOperator)):
    # Form the full matrix
    A = A @ np.eye(A.shape[1])

  Ainv = np.eye(A.shape[0])
  idx  = []
  cols = list(range(A.shape[1]))
  det  = 1.0

  for _ in range(k):
    B = Ainv @ A[:, cols]

    # Compute the Ainv norms
    colnorms = np.zeros(len(cols))

    for j in range(len(cols)):
      colnorms[j] = A[:, cols[j]].T @ B[:, j]

    # Pick the maximum column
    maxidx = np.argmax(colnorms)
    maxval = colnorms[maxidx]

    # Update stuff
    idx.append(cols[maxidx])
    cols = list(set(cols) - set(idx))

    # det(A + v v.T) = det(A)(1 + v.T A^{-1} v)
    det  = det * (1 + maxval)

    # Sherman-Morrison: 
    # (A + v v.T)^{-1} = A^{-1} - (A^{-1}v v.T A^{-1})/(1 + v.T A^{-1} v)  
    Ainv = Ainv - ((np.outer(B[:, maxidx], B[:, maxidx]))/ (1 + maxval)) 

  # Compute D-Optimality
  S       = form_selmat(idx, A.shape[1])
  dopt, _ = compute_dopt(A @ aslinearoperator(S.T))

  return idx, dopt, S.T, det

def detcssp(A, k, typ='svds', qrtyp='qrcp'):
  '''
  Deterministic column subset selection algorithms based on
  the GKS approach.
  Please refer to the paper for exact details. TODO: Add citations

  :param A     : Input matrix or linear operator
  :param k     : Number of columns to be selected
  :param typ   : SVD algorithm to employ (dense/svds)
  :param qrtyp : QR algorithm to employ (qr/srrqr)

  :returns kidx : Columns selected
  :returns dopt : D-optimality of selected columns
  :returns S    : Column sampling matrix
  :returns Vkt  : Right singular vectors of A (transposed)
  '''
  # Compute the SVD
  if (typ == 'svds'):
    _, _, Vkt  = svds(A, k)
  elif (typ == 'dense'):
    if (isinstance(A, LinearOperator)):
      A = A @ np.eye(A.shape[1])
    
    _, _, Vt = np.linalg.svd(A)
    Vkt      = Vt[:k, :]
  else:
    print("SVD solver not supported.")
    return

  if (qrtyp == 'qrcp'):
    _, _, pidx = qrcp(Vkt, mode='economic', pivoting=True)
    kidx       = pidx[:k]
  elif (qrtyp == 'srrqr'):
    kidx, _ = srrqr_select(Vkt, k)
  else:
    print("QR algorithm not supported.")
    return

  # Compute D-Optimality
  S       = form_selmat(kidx, A.shape[1])
  dopt, _ = compute_dopt(A @ aslinearoperator(S.T))

  return kidx, dopt, S.T, Vkt

def randcssp(A, k, typ='qr', p=None, q=None, l=None, beta=None, rseed=None):
  '''
  Randomized column subset selection algorithms. Three styles of 
  methods are present.
  1. GKS      - QR performed on right singular vectors
  2. RAF      - QR performed directly on the sketch (Adjoint-free method)
  3. Sampling - Sample columns randomly
  Please refer to the paper for exact details. TODO: Add citations

  :param A    : Input matrix or linear operator
  :param k    : Number of columns to be selected
  :param typ  : CSSP method to employ (qr/srrqr/colsample/hybrid/adjfree)
  :param p    : Oversampling for the sketch
  :param q    : Number of subspace iterations for randsvd
  :param l    : Number of samples for sampling based methods
  :param beta : Mixing parameter for sampling based methods
  :param seed : Random seed for reproducibility
 
  :returns pidx : Columns selected
  :returns dopt : D-optimality of selected columns
  :returns S    : Column sampling matrix
  :returns Vkt  : Right singular vectors of A (transposed, for GKS and sampling methods) 
                  or sketch of A for adjoint-free method
  '''
  # Seed if necessary
  if rseed is not None:
    np.random.seed(rseed)
  
  # Get the shapes
  m, n = A.shape
  w = min(m, n)
  
  # Fill all the defaults
  if (p is None):
    p = k
    if (k + p > w):
      p = w - k

  if (q is None):
    q = 2

  if (l is None):
    l = int(np.ceil(k * np.log(k)))
    if (l < k):
      l = 2*k
    if (l > w): # w in min(m, n)
      l = w

  if (beta is None):
    beta = 0.9      

  # Sketch Vk
  if (typ == "adjfree"):
    d     = k + p
    Omega = (1/np.sqrt(d)) * np.random.randn(d, m)
    Vkt   = Omega @ A
  else
    _, _, Vkt = randsvd(A, k, p, q)

  # Get the columns out
  # QR on Vkt
  if (typ == 'qr'):
    _, _, pidx = qrcp(Vkt, mode='economic', pivoting=True)
    pidx = pidx[:k]
  # SRRQR on Vkt
  elif (typ == 'srrqr'):
    pidx, _ = srrqr_select(Vkt, k)
  # Sampling from Vkt
  elif (typ == 'colsample'):
    _, _, pidx, _ = colsample(Vkt, l, beta)
  # Sample and cut
  elif (typ == 'hybrid'):
    # Note this method is the same as weightedcssp when we 
    # use the unweighted sampler P
    _, Sc, smpl, _ = colsample(Vkt, l, beta)
    _, _, cols    = qrcp(Vkt @ Sc, mode='economic', pivoting=True)
    pidx = smpl[cols[:k]]
  elif (typ == 'adjfree'):
    _, _, pidx = qrcp(Vkt, mode='economic', pivoting=True)
    pidx = pidx[:k]
  else:
    print("Algo not supported")
    return

  # Compute D-optimality
  S       = form_selmat(pidx, A.shape[1])
  dopt, _ = compute_dopt(A @ aslinearoperator(S.T))

  return pidx, dopt, S.T, Vkt

###############################################################################
#                                                                             #
#                    Inverse problem solve functions                          #
#                                                                             #
###############################################################################

def form_selnsinv(Gn, S):
  '''
  Compute the inverse of the sampled noise covariance
  matrix (precision matrix). The sampling is performed via the 
  row selection matrix S.
  
  :param Gn : Noise covariance matrix
  :param S  : Sensor/row selection matrix
  
  :returns sel_Gn_inv : Precision matrix of the selected sensors
  '''
  # Assume S is a row-selection operator (COO matrix)
  if (not isinstance(S, LinearOperator)):
    S = aslinearoperator(S)

  if (isinstance(Gn, LinearOperator)):
    Gn = Gn @ np.eye(Gn.shape[1])

  sel_Gn     = (S @ Gn) @ S.T
  sel_Gn_inv = np.linalg.pinv(sel_Gn)

  return aslinearoperator(sel_Gn_inv)

def form_selops(F, Gn, y, S=None):
  '''
  Form the sampled forward operator, sampled noise precision matrix, 
  and the sampled data. The sampling is performed via the row selection matrix S.
  
  :param F  : Forward operator
  :param Gn : Noise covariance matrix
  :param y  : Data/observations
  :param S  : Sensor/row selection matrix (optional, default is to select all sensors)
  
  :returns sel_F      : Sampled forward operator
  :returns sel_Gn_inv : Precision matrix of the selected sensors
  :returns sel_y      : Sampled observations
  '''
  if (S is not None):
    # Assume S is a row-selection operator (COO matrix)
    if (not isinstance(S, LinearOperator)):
      S = aslinearoperator(S)

    # Form the sampled operators
    sel_Gn_inv = form_selnsinv(Gn, S)
    sel_F      = S @ F
    sel_y      = S @ y
  else:
    if (isinstance(Gn, LinearOperator)):
      Gn = Gn @ np.eye(Gn.shape[1])

    # Full solution
    sel_F      = F
    sel_Gn_inv = np.linalg.pinv(Gn)
    sel_y      = y

  # Return everything as operators
  if (not isinstance(sel_Gn_inv, LinearOperator)):
    sel_Gn_inv = aslinearoperator(sel_Gn_inv)
    
  if (not isinstance(sel_F, LinearOperator)):
    sel_F = aslinearoperator(sel_F)

  return sel_F, sel_Gn_inv, sel_y

def compute_obj(F, Gn, Gp_inv, y, mu, alpha, x, S=None):
  '''
  Calculates the different terms in the linear Bayesian inverse
  problem setting. The following optimization is typically solved
  min_{x} \|Fx - y\|_{\Gamma_n^{-1}}^2 + \alpha*\|x-\mu\|_{\Gamma_p^{-1}}^2.
  The first term in the objective corresponds to data misfit and the 
  second term is the prior misfit.
  
  In case we sample rows of F via S the following objective is computed.
  min_{x} \|S*(Fx - y)\|_{\Gamma_n^{-1}}^2 + \alpha*\|x-\mu\|_{\Gamma_p^{-1}}^2.
  
  :param F      : Forward operator for the inverse problem
  :param Gn     : Noise covariance matrix (Noise is assumed to be Gaussian)
  :param Gp_inv : Prior precision matrix (prior is assumed to be Gaussian)
  :param y      : Observations/data collected
  :param mu     : Prior mean (prior is assumed to be Gaussian)
  :param alpha  : Regularization parameter
  :param x      : Candidate solution
  :param S      : Row sampler matrix (optional)

  :returns obj    : Objective evaluated at x
  :returns data_m : Data misfit term
  :returns pr_m   : Prior misfit term
  '''
  # Assume S is a row-selection operator (COO matrix)
  if (not isinstance(S, LinearOperator)):
    S = aslinearoperator(S)

  # Form the sampled matrices
  sel_F, sel_Gn_inv, sel_y = form_selops(F, Gn, y, S)

  # Compute the objective
  y_data = (sel_F @ x) - sel_y
  y_pr   = x - mu
  
  data_m = 0.5 * (y_data.T @ (sel_Gn_inv @ y_data))
  pr_m   = 0.5 * (y_pr.T @ (Gp_inv @ y_pr))
  obj    = data_m + (alpha * pr_m)

  return obj, data_m, pr_m

def compute_obj_cached(F, Gn_inv, Gp_inv, y, mu, alpha, x):
  '''
  Cached version of compute_obj where the row sampling is 
  assumed to applied beforehand and not included in the function.
  
  Calculates the different terms in the linear Bayesian inverse
  problem setting. The following optimization is typically solved
  min_{x} \|Fx - y\|_{\Gamma_n^{-1}}^2 + \alpha*\|x-\mu\|_{\Gamma_p^{-1}}^2.
  The first term in the objective corresponds to data misfit and the 
  second term is the prior misfit.
  
  In case we sample rows of F via S the following objective is computed.
  min_{x} \|S*(Fx - y)\|_{\Gamma_n^{-1}}^2 + \alpha*\|x-\mu\|_{\Gamma_p^{-1}}^2.
  
  :param F      : Forward operator for the inverse problem
  :param Gn     : Noise covariance matrix (Noise is assumed to be Gaussian)
  :param Gp_inv : Prior precision matrix (prior is assumed to be Gaussian)
  :param y      : Observations/data collected
  :param mu     : Prior mean (prior is assumed to be Gaussian)
  :param alpha  : Regularization parameter
  :param x      : Candidate solution

  :returns obj    : Objective evaluated at x
  :returns data_m : Data misfit term
  :returns pr_m   : Prior misfit term
  '''
  # Utility function when matrices are pre-formed.

  # Compute the objective
  y_data = (F @ x - y)
  y_pr   = x - mu
  
  data_m = 0.5 * (y_data.T @ (Gn_inv @ y_data))
  pr_m   = 0.5 * (y_pr.T @ (Gp_inv @ y_pr))
  obj    = data_m + (alpha * pr_m)

  return obj, data_m, pr_m

def solve_invprob(F, Gn, Gp_inv, y, mu, alpha, 
                    S=None, solve_cg=True, maxiters=300, precon=None):
  '''
  Solves the linear Bayesian inverse problem given by
  min_{x} \|Fx - y\|_{\Gamma_n^{-1}}^2 + \alpha*\|x-\mu\|_{\Gamma_p^{-1}}^2.
  The least squares problem can be solved via a direct solver or iteratively
  via the Conjugate-Gradient method.
  The method also optionally accepts a row sampling matrix S to solve the 
  problem after sensor placement via OED.
  
  :param F        : Forward operator for the inverse problem
  :param Gn       : Noise covariance matrix (Noise is assumed to be Gaussian)
  :param Gp_inv   : Prior precision matrix (prior is assumed to be Gaussian)
  :param y        : Observations/data collected
  :param mu       : Prior mean (prior is assumed to be Gaussian)
  :param alpha    : Regularization parameter
  :param S        : Row sampler matrix (optional)
  :param solve_cg : Flag for using an iterative method CG (default true)
  :param maxiters : Maximum number of CG iterations (default 300)
  :param precon   : Preconditioner for CG
  
  :returns x      : Solution vector
  :returns r      : Residual of the least-squares solution
  :returns obj    : Objective evaluated at x
  :returns data_m : Data misfit term
  :returns pr_m   : Prior misfit term
  '''
  # Assume S is a row-selection operator
  # Form the sampled matrices
  sel_F, sel_Gn_inv, sel_y = form_selops(F, Gn, y, S)
 
  # Form the operators for the solve
  A = (sel_F.T @ (sel_Gn_inv @ sel_F)) + (alpha * Gp_inv)
  b = (sel_F.T @ (sel_Gn_inv @ sel_y)) + (alpha * (Gp_inv @ mu))

  if (solve_cg): # Solve via CG
    x, exit_code = cg(A, b, maxiter=maxiters, M=precon)

    if (exit_code > 0):
      print(f"CG exited with code {exit_code}") 
  else:           # Solve via Gaussian Elimination
    # Form the full matrix if needed
    if (isinstance(A, LinearOperator)):
      A = A @ np.eye(A.shape[1])

    x = np.linalg.solve(A, b)

  # Compute the residual and objective
  r = np.linalg.norm(A @ x - b)

  # Use the cached version of objective calculations
  obj, data_m, pr_m = compute_obj_cached(
                        sel_F, sel_Gn_inv, Gp_inv, sel_y, mu, alpha, x)

  return x, r, obj, data_m, pr_m

def solve_invprob_cached(F, Gn_inv, Gp_inv, y, mu, alpha, 
                    solve_cg=True, maxiters=300, precon=None):
  '''
  Cached version of solve_invprob. Assume all matrices are formed.

  Solves the linear Bayesian inverse problem given by
  min_{x} \|Fx - y\|_{\Gamma_n^{-1}}^2 + \alpha*\|x-\mu\|_{\Gamma_p^{-1}}^2.
  The least squares problem can be solved via a direct solver or iteratively
  via the Conjugate-Gradient method.
  The method also optionally accepts a row sampling matrix S to solve the 
  problem after sensor placement via OED.
  
  :param F        : Forward operator for the inverse problem
  :param Gn       : Noise covariance matrix (Noise is assumed to be Gaussian)
  :param Gp_inv   : Prior precision matrix (prior is assumed to be Gaussian)
  :param y        : Observations/data collected
  :param mu       : Prior mean (prior is assumed to be Gaussian)
  :param alpha    : Regularization parameter
  :param S        : Row sampler matrix (optional)
  :param solve_cg : Flag for using an iterative method CG (default true)
  :param maxiters : Maximum number of CG iterations (default 300)
  :param precon   : Preconditioner for CG
  
  :returns x      : Solution vector
  :returns r      : Residual of the least-squares solution
  :returns obj    : Objective evaluated at x
  :returns data_m : Data misfit term
  :returns pr_m   : Prior misfit term
  '''
  # Utility function when sampled matrices are pre-formed
  
  # Form the operators for the solve
  A = (F.T @ (Gn_inv @ F)) + (alpha * Gp_inv)
  b = (F.T @ (Gn_inv @ y)) + (alpha * (Gp_inv @ mu))

  if (solve_cg): # Solve via CG
    x, exit_code = cg(A, b, maxiter=maxiters, M=precon)

    if (exit_code > 0):
      print(f"CG exited with code {exit_code}") 
  else:           # Solve via Gaussian Elimination
    # Form the full matrix if needed
    if (isinstance(A, LinearOperator)):
      A = A @ np.eye(A.shape[1])

    x = np.linalg.solve(A, b)

  # Compute the residual and objective
  r = np.linalg.norm(A @ x - b)

  # Use the cached version of objective calculations
  obj, data_m, pr_m = compute_obj_cached(F, Gn_inv, Gp_inv, y, mu, alpha, x)

  return x, r, obj, data_m, pr_m

# Hacky functions to work on dense data
def form_selnsinv_den(Gn, S):
  '''
  Assuming all inputs matrices are dense.
  Compute the inverse of the sampled noise covariance
  matrix (precision matrix). The sampling is performed via the 
  row selection matrix S.
  
  :param Gn : Noise covariance matrix
  :param S  : Sensor/row selection matrix
  
  :returns sel_Gn_inv : Precision matrix of the selected sensors
  '''
  # Assume S is a row-selection operator (COO/CSR matrix)
  # Gn is the dense noise covariance matrix
  sel_Gn     = (S @ Gn) @ S.T
  sel_Gn_inv = np.linalg.pinv(sel_Gn)

  return sel_Gn_inv

def form_selops_den(F, Gn, y, S=None):
  '''
  Assuming all input matrices are dense.
  Form the sampled forward operator, sampled noise precision matrix, 
  and the sampled data. The sampling is performed via the row selection matrix S.
  
  :param F  : Forward operator
  :param Gn : Noise covariance matrix
  :param y  : Data/observations
  :param S  : Sensor/row selection matrix (optional, default is to select all sensors)
  
  :returns sel_F      : Sampled forward operator
  :returns sel_Gn_inv : Precision matrix of the selected sensors
  :returns sel_y      : Sampled observations
  '''
  # Gn is the dense noise covariance matrix
  # F is the dense forward operator
  if (S is not None):
    # Assume S is a row-selection operator (COO matrix)
    # Form the sampled operators
    sel_Gn_inv = form_selnsinv_den(Gn, S)
    sel_F      = S @ F
    sel_y      = S @ y
  else:
    # Full solution
    sel_F      = F
    sel_Gn_inv = np.linalg.pinv(Gn)
    sel_y      = y

  return sel_F, sel_Gn_inv, sel_y

def solve_invprob_den(F, Gn, Gp_inv, y, mu, alpha, 
                    S=None, solve_cg=True, maxiters=300, precon=None):
  '''
  Assuming all input matrices are dense.
  Solves the linear Bayesian inverse problem given by
  min_{x} \|Fx - y\|_{\Gamma_n^{-1}}^2 + \alpha*\|x-\mu\|_{\Gamma_p^{-1}}^2.
  The least squares problem can be solved via a direct solver or iteratively
  via the Conjugate-Gradient method.
  The method also optionally accepts a row sampling matrix S to solve the 
  problem after sensor placement via OED.
  
  :param F        : Forward operator for the inverse problem
  :param Gn       : Noise covariance matrix (Noise is assumed to be Gaussian)
  :param Gp_inv   : Prior precision matrix (prior is assumed to be Gaussian)
  :param y        : Observations/data collected
  :param mu       : Prior mean (prior is assumed to be Gaussian)
  :param alpha    : Regularization parameter
  :param S        : Row sampler matrix (optional)
  :param solve_cg : Flag for using an iterative method CG (default true)
  :param maxiters : Maximum number of CG iterations (default 300)
  :param precon   : Preconditioner for CG
  
  :returns x      : Solution vector
  :returns r      : Residual of the least-squares solution
  :returns obj    : Objective evaluated at x
  :returns data_m : Data misfit term
  :returns pr_m   : Prior misfit term
  '''
  # Assume S is a row-selection operator
  # Form the sampled matrices
  sel_F, sel_Gn_inv, sel_y = form_selops_den(F, Gn, y, S)
 
  # Form the operators for the solve
  A = (sel_F.T @ (sel_Gn_inv @ sel_F)) + (alpha * Gp_inv)
  b = (sel_F.T @ (sel_Gn_inv @ sel_y)) + (alpha * (Gp_inv @ mu))

  x = np.linalg.solve(A, b)

  # Compute the residual and objective
  r = np.linalg.norm(A @ x - b)

  # Use the cached version of objective calculations
  obj, data_m, pr_m = compute_obj_cached(
                        sel_F, sel_Gn_inv, Gp_inv, sel_y, mu, alpha, x)

  return x, r, obj, data_m, pr_m

