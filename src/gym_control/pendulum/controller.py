from abc import abstractmethod
import logging

import numpy as np

from gym_control import GenericController


logger = logging.getLogger(__file__)



class PendulumController(GenericController):
    
    def __init__(
        self,
        max_speed: float = 8.,     # max angular speed (rad/s)
        max_torque: float = 2.,    # maximum torque (N*m)
        dt: float = 0.05,          # time step (seconds)
        g: float = 10.,            # gravitational constant (m/s**2)
        m: float = 1.,             # mass (kg)
        l: float = 1.,             # pendulum length (m)
        **kwargs
    ):
        
        # These are parameters that stay constant for each time step and
        # we assume is available to any controller.
        self.max_speed = max_speed
        self.max_torque = max_torque
        self.dt = dt
        self.g = g
        self.m = m
        self.l = l
