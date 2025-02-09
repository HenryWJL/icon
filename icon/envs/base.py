import abc
import torch
import numpy as np
from typing import Union, Tuple, Dict


class Env:

    @abc.abstractmethod
    def reset(self) -> Dict:
        raise NotImplementedError
    
    @abc.abstractmethod
    def step(self, actions: Union[torch.Tensor, np.ndarray]) -> Tuple:
        raise NotImplementedError
    
    def terminate(self) -> None:
        pass