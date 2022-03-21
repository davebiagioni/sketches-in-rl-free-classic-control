from collections import defaultdict
import logging
import time
from typing import Tuple

import gym
from tqdm import tqdm

from gym_control import GenericController


logger = logging.getLogger(__file__)


def run_env(
    env: gym.Env,                   # target gym environment
    controller: GenericController,  # instance of controller with solve method
    max_steps: int = 200,           # max number of episode steps
    seed: int = None,           # random seed for env.reset()
    control_int: int = 1,       # number of steps between solves
    use_tqdm: bool = True,      # use tqdm for more info?
    render: bool = False,       # render the env?
    **controller_kwargs,
) -> Tuple[dict, float]:
    """Main control loop to run MPC against the gym environment. Returns the
    episode reward."""

    obs = env.reset(seed=seed)
    controller.reset()
    
    # Initialize env and run the control loop.
    done = False
    reward = 0.
    traj = defaultdict(list)
    rng = range(max_steps)
    pb = tqdm(rng) if use_tqdm else rng
    try:
        # Main control loop.
        for t in pb:
            
            # Solve the control problem and extract the first action.
            # We do this once every 
            if t % control_int == 0:
                u = controller.solve(obs=obs, **controller_kwargs)
            
            print(u)
            # Aply the action and collect trajectory/reward.
            obs, rew, done, _ = env.step(u)
            reward += rew
            traj["u"].append(u)
            traj["obs"].append(obs)
            traj["rew"].append(rew)
    
            # Give some additional info on run time and current state. 
            # It's not obvious from the rendering but the stable position
            # is at (x=1, y=0).
            if use_tqdm:
                pb.set_description(f"ep_rew={reward:1.1f}")
        
            if render:
                env.render()
        
            if done:
                break
            
    except KeyboardInterrupt:
        print("stopped by user")
    
    env.close()
    
    return traj, reward


