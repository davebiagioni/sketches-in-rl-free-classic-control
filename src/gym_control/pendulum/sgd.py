import numpy as np  # otherwise torch segfaults on my mac!?!

import gym
from tqdm import tqdm

import torch

from gym_control import GenericController


class SGDController(GenericController):
    
    def __init__(
        self, 
        lr: float = .1,
        horiz: int = 20,
        num_samples: int = 16,
        num_sgd_steps: int = 5,
        warm_start: bool = True
    ):
        """SGD based controller that uses batched initial solutions (actions)
        over a planning horizon, and performs a limited number of SGD update
        steps using the dynamics model to improve the action.  At the end of
        the update, the best action is applied.  This is basically an SGD 
        version of MPC.

        Parameters
        ----------
        lr, optional
            learning rate for SGD steps, by default .1.
        horiz, optional
            rollout horizon length, by default 20
        num_samples, optional
            number of initial guesses to use for the action sequence, by default 16
        num_sgd_steps, optional
            number of SGD steps to take before computing the best initial action, by default 5
        warm_start, optional
            whether to seed the initial guess with the previous solution, by default True
        """        
        self.lr = lr
        self.horiz = horiz
        self.num_samples = num_samples
        self.num_sgd_steps = num_sgd_steps
        self.warm_start = warm_start
        
        _env = gym.make("Pendulum-v1")
        g = _env.g
        l = _env.l
        m = _env.m
        self.max_speed = _env.max_speed
        self.max_torque = _env.max_torque
        self.dt = _env.dt
        self.c1 = 3 * g / (2 * l)
        self.c2 = 3 / (m * l**2)
        
    
    def reset(self):
        """Resets the controller for warm starting by setting last solution to None.
        """        
        self.last_u = None

    
    def solve(self, obs: np.ndarray) -> np.ndarray:
        """Compute the control input for a given starting state.

        Parameters
        ----------
        obs
            gym observation array

        Returns
        -------
            control input
        """        
        
        obs = torch.from_numpy(obs)
        th0 = torch.atan2(obs[1], obs[0])
        thdot0 = obs[2]

        # This is "warm start" if previous u is available, use it as one of the guesses
        if self.last_u is not None and self.warm_start:
            u = self.last_u.detach().clone()
        else:
            u0 = torch.zeros((1, self.horiz))
            u1 = self.max_torque * ( 2 * torch.rand((self.num_samples-1, self.horiz)) - 1)
            u = torch.cat((u0, u1), axis=0)
        u.requires_grad_(True)

        opt = torch.optim.Adam([u], lr=self.lr)
        
        for _ in range(self.num_sgd_steps):

            loss = []

            th = th0.detach().clone()
            thdot = thdot0.detach().clone()

            for t in range(self.horiz):

                # Step loss
                th_normed = torch.remainder(th + np.pi, 2 * np.pi) - np.pi
                loss.append(
                    th_normed**2 
                    + .1 * thdot.pow(2)
                    + .001 * u[:, t].pow(2)
                )

                # Dynamics
                thdot = thdot + (self.c1 * torch.sin(th) + self.c2 * u[:, t]) * self.dt 
                thdot = torch.clamp(thdot, -self.max_speed, self.max_speed)
                th = th + thdot * self.dt
                
            loss = torch.stack(loss, dim=1)
            opt_loss = loss.sum(dim=1).mean()
            opt.zero_grad()
            opt_loss.backward()
            opt.step()

        idx = torch.argmin(loss.mean(axis=1))
        u = u[idx, 0].unsqueeze(0)

        return u.detach().numpy()
    
    
if __name__ == "__main__":
    
    from gym_control.runner import run_env
    
    controller = SGDController(horiz=20, num_samples=16, lr=.1, num_sgd_steps=5)
    
    for seed in range(10):
        env = gym.make("Pendulum-v1")
        run_env(env, controller, render=True, seed=seed)
