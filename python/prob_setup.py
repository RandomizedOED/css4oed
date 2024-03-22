import numpy as np
import scipy

from pyoed.models.simulation_models.advection_diffusion import AdvectionDiffusion1D
from pyoed.models.observation_operators.interpolation import SelectionOperator, CartesianInterpolator
from pyoed.models.error_models.Gaussian import GaussianErrorModel
from pyoed.assimilation.smoothing.fourDVar import VanillaFourDVar

def setup_advdiff1D(nx=41, dt=0.1, obspct=0.6, c=0.1, nu=0.1, time_span=(0,2),
                     chk_pts=np.arange(0.2,1.0,0.2), npct=0.01,
                     obstype="select", rseed=2027):
    '''
    Setup an example Advection-Diffusion 1D problem.
    '''
    
    # Get the model setup
    model = AdvectionDiffusion1D(configs={'nx':nx, 'dt':dt, 'c':c, 'nu':nu})
    
    # Setup the selection operator
    if (obstype == "select"):
        n_obs = int(obspct * (model.state_size-2))
        s_pts = np.sort(np.random.choice(np.arange(1, model.state_size-1), 
                                                 n_obs, replace=False))
        obsop = SelectionOperator(configs={'model':model, 
                                'observation_indexes':s_pts})
    elif (obstype == "cartesian"):
        n_obs = int(obspct * (model.state_size))
        s_pts = np.linspace(-0.8, 0.8, num=n_obs)
        obsop = CartesianInterpolator(configs=
                        {'model_grid': model.get_model_grid(), 
                         'observation_grid': s_pts})
    else:
        assert("Observation operator not supported.")
    
    # Setup the initial conditions and get some observations
    tIC = model.create_initial_condition()

    # Get some true trajectories
    _, ttraj = model.integrate_state(tIC, tspan=time_span, checkpoints=chk_pts)
    
    # Setup the noise
    noise_std = npct * np.linalg.norm(obsop(ttraj[-1])) / np.sqrt(obsop.shape[0])
    noise_var = noise_std**2
    obsns = GaussianErrorModel(configs={'size':obsop.shape[0], 
                'variance':noise_var, 'random_seed': rseed})

    # Get some observations
    obs = []
    for t in ttraj:
        obs.append(obsns.add_noise(obsop(t)))

    return model, n_obs, s_pts, obsop, obsns, tIC, ttraj, obs, noise_var

