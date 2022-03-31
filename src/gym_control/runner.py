from collections import defaultdict
import logging
import time
from typing import Tuple

import gym
from tqdm import tqdm

from gym_control import GenericController


logger = logging.getLogger(__file__)


def run_env(
    env: gym.Env,
    controller: GenericController,
    max_steps: int = 200,
    seed: int = None,
    control_int: int = 1,
    render: bool = False,
    **solve_kwargs,
) -> Tuple[dict, float]:
    """Function to run an environment with a controller.

    Parameters
    ----------
    env
        Target gym environment.
    controller
        Controller object that will compute actions.
    max_steps, optional
        Max number of episode steps before termination, by default 200
    seed, optional
        Random seed for env.reset(), by default None
    control_int, optional
        Compute actions every this many env steps, by default 1
    render, optional
        Render the env, by default False
    solve_kwargs, optional
        optional arguments needed for the controller when calling solve

    Returns
    -------
    traj
        Dict of environment trajectory.
    
    reward
        Episode reward.
    """

    obs = env.reset(seed=seed)
    controller.reset()
    
    # Initialize env and run the control loop.
    done = False
    reward = 0.
    traj = defaultdict(list)
    pb = tqdm(range(max_steps))
    try:
        # Main control loop.
        for t in pb:
            
            # Solve the control problem and extract the first action.
            # We do this once every 
            if t % control_int == 0:
                u = controller.solve(obs=obs, **solve_kwargs)
            
            # Aply the action and collect trajectory/reward.
            obs, rew, done, _ = env.step(u)
            reward += rew
            traj["u"].append(u)
            traj["obs"].append(obs)
            traj["rew"].append(rew)
    
            # Give some additional info on run time and current state. 
            pb.set_description(f"ep_rew={reward:1.1f}")
        
            if render:
                env.render()
        
            if done:
                break
            
    except KeyboardInterrupt:
        print("stopped by user")
    
    return traj, reward


