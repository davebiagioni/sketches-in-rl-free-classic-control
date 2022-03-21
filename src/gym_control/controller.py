from abc import abstractmethod
import logging

import numpy as np


logger = logging.getLogger(__file__)


class GenericController(object):
    
    
    @abstractmethod
    def compute_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        # Solve method to take state obs and return action array.
        raise NotImplemented


    @abstractmethod
    def reset(self, **kwargs):
        return

