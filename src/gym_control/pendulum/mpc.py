import logging

import numpy as np
import pandas as pd
# from pyutilib import subprocess

from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import pyomo.environ as pyo
from pyomo.environ import value

from controller import PendulumController


logger = logging.getLogger(__file__)

# Magic incantation needed by pyomo for -- I don't quite remember?  
# Multiprocessing?  Possibly unnecessary!
# subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False


class ModelPredictiveController(PendulumController):
    
    def __init__(
        self,
        *args,
        K: int = 10,               # number of MPC lookahead steps
        gamma: float = 1.0,       # decay factor in cost function
        solver: str = "ipopt",     # NLP solver
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Solver-specific configuration.
        self.K = K
        self.gamma = gamma
        self.solver = solver
        self.num_solves = 0
        
        
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
        # This numpy function puts the angle in the right quadrant.  HOWEVER, 
        # it takes arguments y,x (not x,y) and if you don't do this you'll get stuck 
        # like I did for an hour trying to debug :P
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
        m.cost = pyo.Objective(
            rule=sum(self.gamma**t * m.step_cost[t]/self.K for t in m.t))
        
        return m
    
    async def async_solve(self, obs, **kwargs):
        if obs is None:
            return 0.
        return self.solve(obs, **kwargs)
    
    
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
        
        # Various IPOPT options to tinker with.
        for key in kwargs:
            opt.option[key] = kwargs.get(value)
        # opt.options["max_iter"] = 100
        # opt.options["max_cpu_time"] = 0.05  # force it to solve in real-time :)
        opt.options["tol"] = 1e-8
        # if "warm_start" in kwargs and self.num_solves > 1:
        #     opt.options["warm_start_init_point"] = "yes"
        #     opt.options["warm_start_same_structure"] = "yes"

        # Try to parse the solution.  This isn't super helpful traceback but at 
        # least should tell you high-level why the solved failed.  The solver 
        # really _shouldn't_ fail if you give it a feasible problem.
        u = None
        try:
            _ = opt.solve(model, tee=tee)
            self.num_solves += 1
            df = self.parse(model)
            u = df["u"].values[0]
        except Exception as e:
            logger.error("solve failed: {}".format(e))
            u = None
        finally:
            logger.debug(f"u={u}")
            return np.array([u])

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
        
        def _getval(name):
            return [value(getattr(m, name)[t]) for t in m.t]

        data = {k: _getval(k) for k in ["u", "th", "thdot", "step_cost"]}
        
        # For convenience, compute the x-y coordinate of pendulum head.
        data.update({
            "x": [self.l * np.cos(th) for th in data["th"]],
            "y": [self.l * np.sin(th) for th in data["th"]]
        })
                            
        return pd.DataFrame(data)

