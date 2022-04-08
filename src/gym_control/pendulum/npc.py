import logging
from typing import Tuple, List

from tqdm import tqdm

import numpy as np  # otherwise torch segfaults on my mac!?!

import gym

import torch
from torch import nn
import torch.nn.functional as F

from gym_control import GenericController


logger = logging.getLogger(__file__)

    
class NeuroPredictiveController(GenericController, nn.Module):
    
    def __init__(
        self,
        hidden_dim: int = 128,
        learn_dynamics: bool = False,
        **kwargs
    ):
        """Neural predictive controller that takes an input state and returns 
        an action.

        Parameters
        ----------
        hidden_dim, optional
            dimension of hidden layer, by default 128
        learn_dynamics, optional
            whether to also learn the parameters c1, c2, and l in the dynamics, 
            by default False
        """        
        
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.learn_dynamics = learn_dynamics
        
        # Parameters from gym env that we don't plan to "learn"
        _env = gym.make("Pendulum-v1")
        self.max_torque = _env.max_torque
        self.dt = _env.dt
        
        # Here we can either use exogenously defined parameter values for the
        # physics, or we can make them learnable parameters.  See the forward
        # method for how these are used in the dynamics.
        if learn_dynamics:
            self.c1 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
            self.c2 = torch.nn.Parameter(torch.ones(1), requires_grad=True) 
            self.l = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        
        # Policy neural network, weights initialized to zero.
        self.policy = nn.Sequential( 
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        for layer in self.policy:
            if isinstance(layer, nn.Linear):
                layer.weight.data.fill_(0.)


    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        thdot: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[any, any]]:
        """Forward pass of neural network that returns an action and, 
        optionally, predicted x,y values needed to learn the dynamics model.

        Parameters
        ----------
        x
            x-coordinate of pendulum head, size = (batch_size, 1)
        y
            y-coordinate of pendulum head, size = (batch_size, 1)
        thdot
            angular velocity of pendulum, size = (batch_size, 1)

        Returns
        -------
        u
            control input
        (x, y) 
            predicted xy-coordinates. The method returns x,y=None,None if 
            learn_dynamics is False.
        """        
        
        # Compute the action and squash it into valid range.
        u = torch.cat((x, y, thdot), axis=-1)
        u = self.policy(u)
        u = self.max_torque * (2 * torch.sigmoid(u) - 1)
        
        if self.learn_dynamics:
            # Advance the physics using computed action and _model_ of dynamics.
            # 1. Compute the angle with correct quadrant.
            th = torch.atan2(y, x)
            
            # Integrate angular velocity using _modeled_ values of c1, c2
            thdot = thdot + (self.c1 * torch.sin(th) + self.c2 * u) * self.dt 
            
            # Integrate angle given latest velocity
            th = th + thdot * self.dt
            
            # Compute x,y using _modeled_ value of the pendulum length
            x = self.l * torch.cos(th) 
            y = self.l * torch.sin(th)
        else:
            x, y = None, None
        
        return u, (x, y)


    def solve(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        """Compute the control input for a given starting state.

        Parameters
        ----------
        obs
            gym observation array

        Returns
        -------
            control input
        """    
        
        # Parse the obs array into required state variables
        x = torch.from_numpy(np.array([obs[0]])).reshape(-1, 1)        
        y = torch.from_numpy(np.array([obs[1]])).reshape(-1, 1)    
        thdot = torch.from_numpy(np.array([obs[2]])).reshape(-1, 1)

        # Call the forward method to get the action from policy.  We aren't 
        # learning when we call this method, just running the policy.
        u, _ = self(x, y, thdot)
        
        return np.array([u.detach().numpy().squeeze()])


def train(learn_dynamics: bool = False) -> Tuple[GenericController, List[float]]:
    """Training loop for the neural controller.

    Parameters
    ----------
    learn_dynamics, optional
        Whether or not to learn the dynamics, by default False.  If True, the
        training uses 1000 epochs, otherwise 300.

    Returns
    -------
    controller
        trained controller
        
    losses
        list of training losses
    """

    # The env gives us the "true" parameter values
    _env = gym.make("Pendulum-v1")
    g = _env.g
    l = _env.l
    m = _env.m
    max_speed = _env.max_speed
    dt = _env.dt
    c1 = 3 * g / (2 * l)
    c2 = 3 / (m * l**2)

    # Learning hyperparameters
    bsz = 128
    max_steps = 40
    num_epochs = 1000 if learn_dynamics else 300

    torch.manual_seed(42)
    
    # When initializing the controller, you can give it the exact physical 
    # parameters and only focus on learning a policy.  _OR_, you can set a 
    # parameter to None and try to learn it in addition to the policy.
    model = NeuroPredictiveController(learn_dynamics=learn_dynamics)
    
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    losses = []
    pb = tqdm(range(num_epochs))
    try:
        for _ in pb:

            # Track the various losses
            control_loss = 0.
            dynamics_loss = 0.

            # Initialize simulation
            th = np.pi * (2 * torch.rand(bsz, 1) - 1)
            thdot = 2 * torch.rand(bsz, 1) - 1
            x = l * torch.cos(th)
            y = l * torch.sin(th)
            
            us = []  # keep a history of actions for each episode to monitor norm

            for _ in range(max_steps):

                # Call the policy model which can also return predictions for x,y
                u, (x_pred, y_pred) = model(x, y, thdot)
                us.append(u.detach().numpy())

                # Control loss from the gym env
                th_normed = torch.remainder(th + np.pi, 2 * np.pi) - np.pi
                _control_loss = (
                    th_normed.pow(2) 
                    + .1 * thdot.pow(2)
                    + .001 * u.pow(2)
                )
                control_loss += _control_loss
                
                # Dynamics
                thdot = thdot + (c1 * torch.sin(th) + c2 * u) * dt 
                thdot = torch.clamp(thdot, -max_speed, max_speed)
                th = th + thdot * dt
                x = l * torch.cos(th)
                y = l * torch.sin(th)
                
                # Dynamics loss is difference in predicted and true x,y values
                if learn_dynamics:
                    dynamics_loss += (x_pred - x)**2 + (y_pred - y)**2

            # Compute loss for gradient descent.  We turn off the dynamics loss
            # term if not learning the parameters.
            loss = (control_loss + dynamics_loss) / max_steps
            loss = loss.mean()
            losses.append(loss.item())

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        
            # Gradient step
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Iteration info
            _l = np.mean(losses[-50:])
            _u = np.array(us).mean() # max abs action
            pb.set_description(f"{_l:1.2e}, {_u:1.2e}")
            
    except KeyboardInterrupt:
        print("stopped")
        
    if learn_dynamics:
        _c1 = model.c1.detach().numpy().squeeze()
        _c2 = model.c2.detach().numpy().squeeze()
        _l = model.l.detach().numpy().squeeze()
        print(f"TRUE: c1={c1:1.3f}, c2={c2:1.3f}, l={l:1.3f}")
        print(f"PRED: c1={_c1:1.3f}, c2={_c2:1.3f}, l={_l:1.3f}")
        
    return model, losses

    
if __name__ == "__main__":
    
    from gym_control import run_env
    from gym_control.args import parser
    
    args = parser.parse_args()
    
    logger.info("Training")
    controller, losses = train(learn_dynamics=False)
    
    ## Uncomment if you want to plot the training losses
    # import pandas as pd
    # import matplotlib.pyplot as plt
    #fig, ax = plt.subplots()
    #_ = pd.DataFrame(losses).plot(ax=ax, title="log training loss")
    # ax.set_yscale("log")
    #plt.show()
    
    logger.info("Evaluating")
    for seed in range(args.num_seeds):
        env = gym.make("Pendulum-v1")
        run_env(env, controller, render=True, seed=seed)

    