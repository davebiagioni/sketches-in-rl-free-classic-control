import logging

import numpy as np
import pandas as pd
# from pyutilib import subprocess

import gym

from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
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
        *args,
        K: int = 10,               # number of MPC lookahead steps
        gamma: float = 1.0,        # decay factor in cost function
        control_int: int = 1,
        solver: str = "ipopt",     # NLP solver
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Solver-specific configuration.
        self.K = K
        self.gamma = gamma
        self.control_int = control_int
        self.solver = solver
        self.num_solves = 0
        
        self._env = gym.make("Acrobot-v1")
        
    
    def _parse_obs(self, obs: np.ndarray):
        th1 = np.arctan2(obs[1], (obs[0] + 1e-4))
        th2 = np.arctan2(obs[3], (obs[2] + 1e-4))
        return th1, th2, obs[4], obs[5]

        
    def create_model(self, obs: np.ndarray) -> pyo.ConcreteModel:
        """Returns a pyomo model that solves the MPC problem."""
        
        m1 = self._env.LINK_MASS_1
        m2 = self._env.LINK_MASS_2
        l1 = self._env.LINK_LENGTH_1
        lc1 = self._env.LINK_COM_POS_1
        lc2 = self._env.LINK_COM_POS_2
        I1 = self._env.LINK_MOI
        I2 = self._env.LINK_MOI
        g = 9.8
        dt = self.control_int * self._env.dt
        
        m = pyo.ConcreteModel()
        
        # Index sets.
        m.t = pyo.RangeSet(0, self.K)
        m.t_not_init = pyo.RangeSet(1, self.K)
        
        # Variables.
        m.th1 = pyo.Var(m.t, bounds=(-4*np.pi, 4*np.pi))
        m.th2 = pyo.Var(m.t, bounds=(-4*np.pi, 4*np.pi))
        m.dth1 = pyo.Var(m.t) #, bounds=(-4*np.pi, 4*np.pi))
        m.dth2 = pyo.Var(m.t) #, bounds=(-9*np.pi, 9*np.pi))
        m.ddth1 = pyo.Var(m.t)
        m.ddth2 = pyo.Var(m.t)
        m.d1 = pyo.Var(m.t)
        m.d2 = pyo.Var(m.t)
        m.phi1 = pyo.Var(m.t)
        m.phi2 = pyo.Var(m.t)
        m.u = pyo.Var(m.t, domain=pyo.Integers, bounds=(-1., 1.))
        m.step_cost = pyo.Var(m.t, bounds=(-10, 10))

        # Initial conditions.
        th1, th2, dth1, dth2 = self._parse_obs(obs)
        m.th1_init_cons = pyo.Constraint(rule=lambda m: m.th1[0] == th1)
        m.th2_init_cons = pyo.Constraint(rule=lambda m: m.th2[0] == th2)
        m.dth1_init_cons = pyo.Constraint(rule=lambda m: m.dth1[0] == dth1)
        m.dth2_init_cons = pyo.Constraint(rule=lambda m: m.dth2[0] == dth2)
        
        m.d1_cons = pyo.Constraint(
            m.t,
            rule=lambda m, t:
                m.d1[t] == (
                    m1 * lc1 ** 2
                    + m2 * m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * pyo.cos(m.th2[t]))
                    + I1
                    + I2
                )
        )
        
        m.d2_cons = pyo.Constraint(
            m.t,
            rule=lambda m, t:
                m.d2[t] == m2 * (lc2 ** 2 + l1 * lc2 * pyo.cos(m.th2[t])) + I2
        )
        
        m.phi1_cons = pyo.Constraint(
            m.t,
            rule=lambda m, t:
                m.phi1[t] == (
                    -m2 * l1 * lc2 * m.dth2[t] ** 2 * pyo.sin(m.th2[t])
                    - 2 * m2 * l1 * lc2 * m.dth2[t] * m.dth1[t] * pyo.sin(m.th2[t])
                    + (m1 * lc1 + m2 * l1) * g * pyo.cos(m.th1[t] - np.pi / 2)
                    + m.phi2[t]
                )
        )
        
        m.phi2_cons = pyo.Constraint(
            m.t,
            rule=lambda m, t:
                m.phi2[t] == (
                    m2 * lc2 * g * pyo.cos(m.th1[t] + m.th2[t] - np.pi / 2.)
                )
        )
        
        m.ddth2_cons = pyo.Constraint(
            m.t,
            rule=lambda m, t:
                m.ddth2[t] == (
                    (m.u[t] + m.d2[t] / (m.d1[t] + 1e-6) * m.phi1[t] - m2 * l1 * lc2 * m.dth1[t] ** 2 * pyo.sin(m.th2[t]) - m.phi2[t])
                    / (m2 * lc2 ** 2 + I2 - m.d2[t] ** 2 / (m.d1[t] + 1e-6))
                )
        )
        
        m.ddth1_cons = pyo.Constraint(
            m.t,
            rule=lambda m, t: 
                m.ddth1[t] == -(m.d2[t] * m.ddth2[t] + m.phi1[t]) / (m.d1[t] + 1e-6)
        )
        
        m.th1_cons = pyo.Constraint(
            m.t_not_init,
            rule=lambda m, t:
                m.th1[t] == m.th1[t-1] + m.dth1[t-1] * dt
        )
        
        m.th2_cons = pyo.Constraint(
            m.t_not_init,
            rule=lambda m, t:
                m.th2[t] == m.th2[t-1] + m.dth2[t-1] * dt
        )
        
        m.dth1_cons = pyo.Constraint(
            m.t_not_init,
            rule=lambda m, t:
                m.dth1[t] == m.dth1[t-1] + m.ddth1[t-1] * dt
        )
        
        m.dth2_cons = pyo.Constraint(
            m.t_not_init,
            rule=lambda m, t:
                m.dth2[t] == m.dth2[t-1] + m.ddth2[t-1] * dt
        )
        
        # Stage costs.  When the pendulum isn't swung up yet, we just 
        # try to get the first link to be up.  After that we try to solve
        # the problem.
        m.cost_cons = pyo.Constraint(
            m.t,
            rule=lambda m, t: \
                m.step_cost[t] == \
                    pyo.cos(m.th1[t]) + pyo.cos(m.th2[t] + m.th1[t])
        )

        # Objective fuction simply sums over stage costs.
        m.cost = pyo.Objective(
            #rule=sum(self.gamma**(self.K - t) * m.step_cost[t] for t in m.t))
            rule=m.step_cost[self.K])
        
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
        model = self.create_model(obs=obs)
        
        # # Solve the MPC problem.
        opt = SolverFactory(self.solver) #self.solver)
        
        # Various IPOPT options to tinker with.
        if self.solver == "ipopt":
            opt.options["max_iter"] = 500
            # opt.options["max_cpu_time"] = 0.05  # force it to solve in real-time :)
            # opt.options["tol"] = 1e-6
            # opt.options["halt_on_ampl_error"] = "yes"
            # if "warm_start" in kwargs and self.num_solves > 1:
            #     opt.options["warm_start_init_point"] = "yes"
            #     opt.options["warm_start_same_structure"] = "yes"

        # Try to parse the solution.  This isn't super helpful traceback but at 
        # least should tell you high-level why the solved failed.  The solver 
        # really _shouldn't_ fail if you give it a feasible problem.
        try:
            _ = opt.solve(model, tee=False) #, time_limit=2)
            self.num_solves += 1
            df = self.parse(model)
            u = df["u"].values[0]
        except Exception as e:
            logger.error("solve failed: {}".format(e))
            u = 0
        finally:
            return int(np.round(u)) + 1     # Map back to env's discrete action space


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

        data = {
            k: _getval(k) for k in [
                "u", "th1", "th2", "dth1", "dth2", "step_cost"
            ]
        }
                            
        return pd.DataFrame(data)



if __name__ == "__main__":
    
    from gym_control import run_env
    
    env = gym.make("Acrobot-v1")
    mpc = ModelPredictiveController(K=8*5, control_int=1./8)
    
    # Using ipopt results in relaxing the integer variables.  Use "mindtpy"
    # solver for mix integer.
    _ = run_env(env, mpc, render=True, max_steps=500, control_int=1,
                solver="mindtpy")
