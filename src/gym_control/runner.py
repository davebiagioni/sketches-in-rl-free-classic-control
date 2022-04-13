from collections import defaultdict
import logging
import time
from typing import Tuple, List

import numpy as np
import pandas as pd
import gym
from tqdm import tqdm

from gym_control import GenericController


logger = logging.getLogger(__file__)


def run_env(
    env: gym.Env,
    controller: GenericController,
    max_steps: int = 200,
    seeds: List[int] = None,
    early_term_steps: int = None,
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
    seeds, optional
        List of seeds to evaluate; if None, just use one at random
    early_term_steps, optional
        Terminate the episode if the reward is unchanged for this many
        steps.  If None, do not terminate early.
    control_int, optional
        Compute actions every this many env steps, by default 1
    render, optional
        Render the env, by default False
    solve_kwargs, optional
        optional arguments needed for the controller when calling solve

    Returns
    -------
    rewards
        List of episode rewards.
        
    solve_times
        List of list of solve times (one list per episode)
    """
    
    seeds = seeds if seeds is not None else [None]
    early_term_steps = early_term_steps if early_term_steps else np.inf
    
    solve_times = []
    rewards = np.zeros((max_steps, len(seeds)))
    
    for trial_idx, seed in enumerate(seeds):

        obs = env.reset(seed=seed)
        controller.reset()
        
        # Initialize env and run the control loop.
        done = False
        pb = tqdm(range(max_steps))
        try:
            # Main control loop.
            for t in pb:
                
                # Solve the control problem and extract the first action.
                # We do this once every 
                if t % control_int == 0:
                    tic = time.time()
                    u = controller.solve(obs=obs, **solve_kwargs)
                    solve_times.append(time.time() - tic)
                
                # Aply the action and collect trajectory/reward.
                obs, rew, done, _ = env.step(u)
                rewards[t, trial_idx] = rew
        
                # Give some additional info on run time and current state. 
                pb.set_description(f"ep_rew={sum(rewards[:t, trial_idx]):1.1f}")
            
                if render:
                    env.render()
            
                if done:
                    break
                
                # Look for early termination, if avg of last early_term_steps
                # close to zero.
                rolling_reward = np.mean(np.abs(rewards[t-early_term_steps:t, trial_idx]))
                if t > early_term_steps and rolling_reward < 1e-4:
                        break
                
        except KeyboardInterrupt:
            print("stopped by user")

    env.close()
    
    # Generate some performance data to compare algorithms with
    rtot = pd.DataFrame(rewards).sum(axis=0)
    perf_data = {
        "reward_mean": rtot.mean(),
        "reward_std":  rtot.std(),
        "solve_time_min": min(solve_times),
        "solve_time_max": max(solve_times)
    }
                
    return perf_data


