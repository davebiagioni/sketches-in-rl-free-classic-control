import argparse
from collections import defaultdict
import logging
import time

import numpy as np
import pandas as pd

from tqdm import tqdm
import gym

from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import pyomo.environ as pyo
from pyomo.environ import value

from pyutilib import subprocess


parser = argparse.ArgumentParser()
parser.add_argument(
    "--K",
    default=20,
    type=int,
    help="Lookahead horizon (number of steps of length 0.05 seconds)"
)
parser.add_argument(
    "--max-steps",
    default=200,
    type=int,
    help="Max number of episode steps (<= 200)"
)

subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

logger = logging.getLogger(__file__)


class ModelPredictiveController(object):
    
    def __init__(
        self,
        max_speed: float = 8.,     # max angular speed (rad/s)
        max_torque: float = 2.,    # maximum torque (N*m)
        dt: float = 0.05,          # time step (seconds)
        g: float = 10.,            # gravitational constant (m/s**2)
        m: float = 1.,             # mass (kg)
        l: float = 1.,             # pendulum length (m)
        K = 10,                    # number of MPC lookahead steps
        solver: str = "ipopt"      # NLP solver
    ):
    
        # These are parameters that stay constant for each time step.    
        self.max_speed = max_speed
        self.max_torque = max_torque
        self.dt = dt
        self.g = g
        self.m = m
        self.l = l
        
        # Solver-specific configuration.
        self.K = K
        self.solver = solver
        
        
    def create_model(self, initial_state: np.ndarray) -> pyo.ConcreteModel:
        """Returns a pyomo model that solves the MPC problem."""
        
        m = pyo.ConcreteModel()
        
        # Index sets.
        m.t = pyo.RangeSet(0, self.K)
        m.t_not_init = pyo.RangeSet(1, self.K)
        
        # Variables.
        m.th = pyo.Var(m.t)
        m.thdot = pyo.Var(m.t)
        m.u = pyo.Var(m.t, bounds=(-self.max_torque, self.max_torque))
        m.step_cost = pyo.Var(m.t)
        
        # Initial conditions.
        m.thdot_init_cons = pyo.Constraint(rule=lambda m: m.thdot[0] == initial_state[2])
        # This numpy function puts the angle in the right quadrant.  HOWEVER, it takes arguments 
        # y,x (not x,y) and if you don't do this you'll get stuck like I did for an hour trying 
        # to debug :P
        m.th_init_cons = pyo.Constraint(
            rule=lambda m: m.th[0] == np.arctan2(initial_state[1], initial_state[0]))

        # Update angular velocity using Newton (physics) and Euler (integration).
        # newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        g, mass, l, dt = self.g, self.m, self.l, self.dt
        m.thdot_cons = pyo.Constraint(
            m.t_not_init,
            rule=lambda m, t: \
                m.thdot[t] == m.thdot[t-1] + \
                    (3 * g / (2 * l) * pyo.sin(m.th[t-1]) + 3. / (mass * l**2) * m.u[t-1]) * dt
        )

        # Update angle based on current velocity using Euler's method.
        m.newth_cons = pyo.Constraint(
            m.t_not_init,
            rule=lambda m, t: m.th[t] == m.th[t-1] + m.thdot[t] * dt
        )

        # Stage costs. We avoid angular normalization as implemented in the gym 
        # because modulo (%) is non-differentiable and forces us into discrete 
        # optimization.  We get around this by using the xy-coordinates instead since 
        # theta=0 when (x,y) = (1,0).
        m.cost_cons = pyo.Constraint(
            m.t,
            rule=lambda m, t: \
                m.step_cost[t] == \
                    l**2 * (pyo.sin(m.th[t])**2 + (pyo.cos(m.th[t]) - 1)**2) \
                    + 0.1 * m.thdot[t]**2 \
                    + 0.001 * m.u[t]**2
        )

        # Objective fuction simply sums over stage costs.
        m.cost = pyo.Objective(rule=sum(m.step_cost[t] for t in m.t))
        
        return m
    
    
    def solve(
        self,
        initial_state: np.ndarray,   # initial state from gym env (gym obs)
        tee: bool = False            # verbose solver output?
    ) -> pd.DataFrame:
        """Returns a DataFrame with state/action variables corresponding to the 
        MPC solution."""
        
        # Create the model given the initial state.
        model = self.create_model(initial_state=initial_state)
        
        # Solve the MPC problem, tee turns on verbose output from IPOPT.
        opt = SolverFactory(self.solver, tee=tee)
        
        # Try to parse the solution.  This isn't super helpful traceback but at 
        # least should tell you high-level why the solved failed.  The solver 
        # really _shouldn't_ fail if you give it a feasible problem.
        try:
            solution = opt.solve(model, tee=tee)
            self._check_status(solution)
            return self.parse(model)
        except Exception as e:
            logger.error("solve failed: {}".format(e))
            return pd.DataFrame()
        
            
    def _check_status(self, solution):
        """Logs errors/warnings if solver status is fishy."""
        if solution.solver.status != SolverStatus.ok:
            logger.error("solver status: ", solution.solver.status)
        
        if solution.solver.termination_condition != TerminationCondition.optimal:
            logger.warning(
                "termination condition: ", solution.solver.termination_condition)
            
            
    def parse(self, m: pyo.ConcreteModel) -> pd.DataFrame:
        """Returns a dataframe containing MPC's model of the problem variables 
        over the lookahead horizon."""
        
        # Pull variable data from solved model.
        data = {
            "u": [value(m.u[t]) for t in m.t],
            "th": [value(m.th[t]) for t in m.t],
            "thdot": [value(m.thdot[t]) for t in m.t],
            "step_cost": [value(m.step_cost[t]) for t in m.t]
        }
        
        # For convenience, compute the x-y coordinate of pendulum head.
        data.update({
            "x": [self.l * np.cos(th) for th in data["th"]],
            "y": [self.l * np.sin(th) for th in data["th"]]
        })
                            
        return pd.DataFrame(data)


def run_mpc(
    K: int = 4,             # number of lookahead steps
    max_steps: int = 200,   # max number of episode steps
    seed: int = None,       # random seed for env.reset()
    use_tqdm: bool = True,  # use tqdm for more info?
    render: bool = False,   # render the env?
    tee: bool = False      # verbose solver output?
) -> float:
    """Main control loop to run MPC against the gym environment. Returns the
    episode reward."""
    
    # Create the gym environment.
    env = gym.make("Pendulum-v1")

    # Initialize the MPC controller with specified lookahead.
    c = ModelPredictiveController(K=K)

    # Main control loop.
    done = False
    reward = 0.
    traj = defaultdict(list)
    tic = time.time()
    obs = env.reset(seed=seed)
    rng = range(min(200, max_steps))     # control step index
    pb = tqdm(rng) if use_tqdm else rng
    try:
        for _ in pb:
            df = c.solve(initial_state=obs, tee=tee)
            u = df["u"].values[0]
            obs, rew, done, _ = env.step([u])
            reward += rew
            traj["u"].append(u)
            traj["obs"].append(obs)
            traj["rew"].append(rew)
            if use_tqdm:
                pb.set_description(
                    f"ep_rew={reward:1.2f}," +\
                    f"(x,y)=({obs[0]:1.2f},{obs[1]:1.2f})," +\
                    f"thdot={obs[2]:1.3f}")
            if render:
                env.render()
            if done: 
                break
    except KeyboardInterrupt:
        print("stopped by user")
    env.close()
    
    cpu_time = time.time() - tic
    
    return reward, cpu_time


if __name__ == "__main__":
    
    # Add the render argument for this script since it's just running one worker.
    parser.add_argument(
        "--render",
        action="store_true",
        help="Flag to render env (rendering OFF by default)"
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="random seed for env reset"
    )
    args = parser.parse_args()

    rew, cpu_time = run_mpc(**vars(args))
    logger.info(
        f"[worker {args.seed}] reward = {rew:1.3f}, cpu_time = {cpu_time:1.1f}s")
