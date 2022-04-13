import logging

import numpy as np
import pandas as pd
# from pyutilib import subprocess

import gym

from pyomo.opt import SolverFactory
import pyomo.environ as pyo
from pyomo.environ import value

from gym_control import GenericController


logger = logging.getLogger(__file__)

# Magic incantation needed by pyomo for -- I don't quite remember?  
# Multiprocessing?  Possibly unnecessary!
# subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False


class ModelPredictiveController(GenericController):
    
    def __init__(
        self,
        horiz: int = 10,       # number of MPC lookahead steps
        solver: str = "ipopt", # NLP solver
    ):
        super().__init__()
        
        # Solver-specific configuration.
        self.horiz = horiz
        self.solver = solver
        
        self._env = gym.make("Pendulum-v1")
        
        
    def create_model(self, initial_state: np.ndarray) -> pyo.ConcreteModel:
        """Returns a pyomo model that solves the MPC problem."""
        
        m = pyo.ConcreteModel()
        
        # Index sets.
        m.t = pyo.RangeSet(0, self.horiz)
        m.t_not_init = pyo.RangeSet(1, self.horiz)
        
        # Variables.
        m.th = pyo.Var(m.t)
        m.thdot = pyo.Var(m.t)
        m.u = pyo.Var(m.t, bounds=(-self._env.max_torque, self._env.max_torque))
        m.step_cost = pyo.Var(m.t)

        # Initial conditions.
        m.thdot_init_cons = pyo.Constraint(rule=lambda m: m.thdot[0] == initial_state[2])
        # This numpy function puts the angle in the right quadrant.  HOWEVER, 
        # it takes arguments y,x (not x,y) and if you don't do this you'll get stuck 
        # like I did for an hour trying to debug :P
        m.th_init_cons = pyo.Constraint(
            rule=lambda m: m.th[0] == np.arctan2(initial_state[1], initial_state[0]))

        # Update angular velocity using Newton (physics) and Euler (integration).
        # newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        g, mass, l, dt = self._env.g, self._env.m, self._env.l, self._env.dt
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
        m.cost = pyo.Objective(
            rule=sum(m.step_cost[t]/self.horiz for t in m.t))
        
        return m
    
    
    def solve(
        self,
        obs: np.ndarray,   # initial state from gym env (gym obs)
        tee: bool = False,           # verbose solver output?
        **kwargs
    ) -> pd.DataFrame:
        """Returns a DataFrame with state/action variables corresponding to the 
        MPC solution."""
        
        # Create the model given the initial state.
        model = self.create_model(initial_state=obs)
        
        # # Solve the MPC problem, tee turns on verbose output from IPOPT.
        opt = SolverFactory(self.solver, tee=tee)
        
        # Set solver options
        for key in kwargs:
            opt.options[key] = kwargs[key]

        # Try to parse the solution.  This isn't super helpful traceback but at 
        # least should tell you high-level why the solved failed.  The solver 
        # really _shouldn't_ fail if you give it a feasible problem.
        u = None
        try:
            _ = opt.solve(model, tee=tee)
            df = self._parse(model)
            return np.array([df["u"].values[0]])
        except Exception as e:
            logger.error("solve failed: {}".format(e))

            
    def _parse(self, m: pyo.ConcreteModel) -> pd.DataFrame:
        """Returns a dataframe containing MPC's model of the problem variables 
        over the lookahead horizon.

        Parameters
        ----------
        m
            pyomo model to be parsed

        Returns
        -------
            dataframe with model variables (columns) over the MPC horizon (rows)
        """        
                
        def _getval(name):
            return [value(getattr(m, name)[t]) for t in m.t]

        data = {k: _getval(k) for k in ["u", "th", "thdot", "step_cost"]}
        
        # For convenience, compute the x-y coordinate of pendulum head.
        data.update({
            "x": [self._env.l * np.cos(th) for th in data["th"]],
            "y": [self._env.l * np.sin(th) for th in data["th"]]
        })

        return pd.DataFrame(data)



if __name__ == "__main__":
    
    from gym.wrappers.record_video import RecordVideo
    
    from gym_control import run_env
    from gym_control.args import parser
    
    args = parser.parse_args()
    
    controller = ModelPredictiveController(horiz=20)

    # IPOPT options that let you solve faster but possibly less accurately.
    solve_kwargs = {}
    # Leave this commented to use the default settings.
    # solve_kwargs = {
    #     "max_iter": 500,        # fewer iterations before stopping
    #     "max_cpu_time": 0.05,   # time limited
    #     "tol": 1e-8             # convergence tolerance
    # }
    
    
    env = gym.make("Pendulum-v1")
    env = RecordVideo(env, "pendulum-ipopt")
    
    # Run with rendering and terminate early if reward is ~0 for 1 second
    seeds = list(range(args.num_seeds))
    perf_data = run_env(
        env, controller, render=True, seeds=seeds, early_term_steps=20)
    
    print(perf_data)
    