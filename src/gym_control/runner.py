from collections import defaultdict
import logging
import time

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
) -> float:
    """Main control loop to run MPC against the gym environment. Returns the
    episode reward."""

    obs = env.reset(seed=seed)
    controller.reset()
    
    # Initialize env and run the control loop.
    tic = time.time()
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
    
    cpu_time = time.time() - tic
    
    return traj, reward, cpu_time



# if __name__ == "__main__":
    
#     import argparse
    
#     # Common parser args used by this script and distributed_trials.py
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--K",
#         default=20,
#         type=int,
#         help="Lookahead horizon (number of steps of length dt seconds)"
#     )
#     parser.add_argument(
#         "--max-steps",
#         default=200,
#         type=int,
#         help="Max number of episode steps (<= 200 for Pendulum-v1)"
#     )
#     parser.add_argument(
#         "--controller",
#         default="mpc",
#         choices=["mpc"],
#         type=str,
#         help="controller type to run"
#     )
#     parser.add_argument(
#         "--render",
#         action="store_true",
#         help="Flag to render env (rendering OFF by default)"
#     )
#     parser.add_argument(
#         "--seed",
#         default=None,
#         type=int,
#         help="random seed for env reset"
#     )
#     parser.add_argument(
#         "--tee",
#         action="store_true",
#         help="turn on IPOPT verbose output"
#     )
#     parser.add_argument(
#         "--warm-start",
#         action="store_true",
#         help="turn on IPOPT warm start"
#     )
#     parser.add_argument(
#         "--control-int",
#         type=int,
#         default=1,
#         help="compute the action every this many steps"
#     )
#     parser.add_argument(
#         "--dt",
#         type=float,
#         default=0.05,
#         help="integration step for model physics (higher -> lower accuracy)"
#     )
#     args = parser.parse_args()

#     if args.controller == "mpc-pyomo":
#         from mpc import ModelPredictiveController
#         controller = ModelPredictiveController(K=args.K, dt=args.dt)
#     else:
#         raise NotImplementedError

#     args = vars(args)
#     args["controller"] = controller
#     _, rew, cpu_time = run_trial(**args)
    
#     seed = args["seed"]
#     logger.info(
#         f"[worker {seed}] reward = {rew:1.3f}, cpu_time = {cpu_time:1.1f}s")

