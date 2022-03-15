import pandas as pd
import ray

import mpc


@ray.remote(num_returns=2)
def run_mpc(K: int = 4, max_steps: int = 200, seed: int = None, **kwargs) -> float:
    """Returns the MPC episode reward for a single remote call."""
    reward, cpu_time = mpc.run_mpc(
        K=K, max_steps=max_steps, seed=seed, render=False, use_tqdm=False)
    print(f"[{seed}]: {reward:1.2f}, {cpu_time:1.2f} s")
    return reward, cpu_time


def main(*, K=None, seeds=None, max_steps=None) -> pd.DataFrame:
    """Returns a dataframe of episode rewards for all seeds."""
    
    # Create and execute the ray remote call.
    futures = [run_mpc.remote(K=K, max_steps=max_steps, seed=seed) for seed in seeds]
    reward = ray.get([x[0] for x in futures])
    cpu_time = ray.get([x[1] for x in futures])
    
    # Create a single dataframes with unique row indices and summarize
    # reward and cpu time.
    reward = pd.DataFrame(reward, columns=["reward"], index=seeds)
    cpu_time = pd.DataFrame(cpu_time, columns=["cpu_time"], index=seeds)
    df = pd.concat((reward, cpu_time), axis=1)
    
    print(df.describe())
    
    return df


if __name__ == "__main__":
    
    import os
    
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
    df = main(K=args.K, max_steps=args.max_steps, seeds=range(args.num_seeds))
    
    # Save CSV file of episode rewards dataframe.
    k = str(args.K).zfill(3)
    path = os.path.join("results", f"K_{k}_n_{args.num_seeds}.csv")
    df.to_csv(path)
    print(f"wrote results to {path}")

    ray.shutdown()
