import logging

from tqdm import tqdm

import numpy as np  # otherwise torch segfaults on my mac!?!

import gym

import torch
from torch import nn
import torch.nn.functional as F

from gym_control import GenericController


logger = logging.getLogger(__file__)


# class NeuralPredictiveModel(nn.Module):
    
#     def __init__(
#         self,
#         seq_len: int = None,
#         hidden_dim: int = 64,
#         max_torque: float = 2.
#     ):  
#         super().__init__()

#         self.seq_len = seq_len
#         self.hidden_dim = hidden_dim
#         self.max_torque = max_torque
        
#         self.output = nn.Sequential( 
#             nn.Linear(3, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )
        
#         for layer in self.output:
#             if isinstance(layer, nn.Linear):
#                 layer.weight.data.fill_(0.)


#     def forward(self, x: torch.Tensor, y: torch.Tensor, thdot: torch.Tensor):
#         u = torch.cat((x, y, thdot), axis=-1)
#         u = self.output(u)
#         u = self.max_torque * (2 * torch.sigmoid(u) - 1)
#         return u
    
    
class NeuralPredictiveController(GenericController, nn.Module):
    
    def __init__(
        self,
        hidden_dim: int = 64,
        max_torque: float = 2.,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_torque = max_torque
        
        self.policy = nn.Sequential( 
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        for layer in self.policy:
            if isinstance(layer, nn.Linear):
                layer.weight.data.fill_(0.)


    def forward(self, x: torch.Tensor, y: torch.Tensor, thdot: torch.Tensor):
        u = torch.cat((x, y, thdot), axis=-1)
        u = self.policy(u)
        u = self.max_torque * (2 * torch.sigmoid(u) - 1)
        return u


    def solve(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        
        x = torch.from_numpy(np.array([obs[0]])).reshape(-1, 1)        
        y = torch.from_numpy(np.array([obs[1]])).reshape(-1, 1)    
        thdot = torch.from_numpy(np.array([obs[2]])).reshape(-1, 1)

        u = self(x, y, thdot)
        
        return np.array([u.detach().numpy().squeeze()])


def train():

    _env = gym.make("Pendulum-v1")
    g = _env.g
    l = _env.l
    m = _env.m
    max_speed = _env.max_speed
    max_torque = _env.max_torque
    dt = _env.dt
    c1 = 3 * g / (2 * l)
    c2 = 3 / (m * l**2)

    bsz = 64
    hidden_dim = 128
    max_steps = 40
    num_epochs = 300
    
    torch.manual_seed(42)
    
    model = NeuralPredictiveController(hidden_dim=hidden_dim, max_torque=max_torque)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    losses = []
    pb = tqdm(range(num_epochs))
    try:
        for _ in pb:

            # Initialize simulation
            loss = 0.
            th = np.pi * (2 * torch.rand(bsz, 1) - 1)
            thdot = 2 * torch.rand(bsz, 1) - 1
            x = l * torch.cos(th)
            y = l * torch.sin(th)
            
            us = []  # keep a history of actions for each episode to monitor norm

            for t in range(max_steps):

                # Call the policy model
                u = model(x, y, thdot)
                us.append(u.detach().numpy())

                # Step loss
                th_normed = torch.remainder(th + np.pi, 2 * np.pi) - np.pi
                _loss = (
                    th_normed**2 
                    + .1 * thdot.pow(2)
                    + .001 * u.pow(2)
                )
                loss += _loss / max_steps

                # Dynamics
                thdot = thdot + (c1 * torch.sin(th) + c2 * u) * dt 
                thdot = torch.clamp(thdot, -max_speed, max_speed)
                th = th + thdot * dt
                x = l * torch.cos(th)
                y = l * torch.sin(th)

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
        
    return model, losses

    
if __name__ == "__main__":
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    from gym_control.runner import run_env
    
    
    controller, losses = train()
    
    fig, ax = plt.subplots()
    _ = pd.DataFrame(losses).plot(ax=ax, title="log training loss")
    ax.set_yscale("log")
    plt.show()
    
    for seed in range(10):
        env = gym.make("Pendulum-v1")
        run_env(env, controller, render=True, seed=seed)

    