import time

import pandas as pd
import ray

import mpc


@ray.remote(num_returns=2)
def run_mpc(K: int = 4, max_steps: int = 200, seed: int = None, **kwargs) -> float:
    """Returns the MPC episode reward for a single remote call."""
    reward, traj = mpc.run_mpc(
        K=K, max_steps=max_steps, seed=seed, render=False, use_tqdm=False)
    print(f"[{seed}]: {reward:1.2f}")
    return reward, traj


def main(*, K=None, seeds=None, max_steps=None) -> pd.DataFrame:
    """Returns a dataframe of episode rewards for all seeds."""
    
    # Create and execute the ray remote call.
    futures = [run_mpc.remote(K=K, max_steps=max_steps, seed=seed) for seed in seeds]
    rewards = ray.get([x[0] for x in futures])
    trajs = ray.get([x[1] for x in futures])
    
    # Create a dataframe with unique row indices and summarize.
    rewards = pd.DataFrame(rewards, columns=["reward"], index=range(len(seeds)))
    print(rewards.describe())
    
    return rewards, trajs


if __name__ == "__main__":
    
    import os
    import pickle
    
    parser = mpc.parser
    parser.add_argument(
        "--num-seeds",
        default=4,
        type=int,
        help="number of random seeds to evaluate"
    )
    parser.add_argument(
        "--num-cpus",
        default=4,
        type=int,
        help="number of cpus to use"
    )
    args = parser.parse_args()
    print(f"CLI args: {args}")
    
    ray.init(num_cpus=args.num_cpus)
    
    # Call the ray function for remote execution.
    tic = time.time()
    rewards, trajs = main(K=args.K, max_steps=args.max_steps, seeds=range(args.num_seeds))
    elapsed = time.time() - tic
    print(f"took {elapsed:1.0f} sec")
    
    # Save CSV file of episode rewards dataframe.
    path = os.path.join("results", f"{args.K}-{args.num_seeds}.csv")
    rewards.to_csv(path)
    print(f"wrote rewards to {path}")

    # Save pickle file of episode trajectories.
    path = path.replace(".csv", ".p")
    with open(path, "wb") as f:
        pickle.dump(trajs, f)
    print(f"wrote trajectories to {path}")


    ray.shutdown()
