from typing import Union, Callable
from collections import namedtuple

import numpy as np
import control
import gym


class SystemEnv(gym.Env):


    def __init__(
        self, system: Union[Callable, control.LinearIOSystem], q=1, r=0, dt=1e-2,
        seed=None, constrained_actions=None,
        dtype=np.float32, custom_reward: bool=False, 
    ):
        super().__init__()
        self.system = system
        if isinstance(system, (control.LinearIOSystem, control.NonlinearIOSystem, control.InputOutputSystem)):
            self.dynamics = system.dynamics
        else:
            self.dynamics = system
        self.custom_reward = custom_reward
        self.dtype = dtype
        self.q = np.atleast_2d(q)
        self.r = np.atleast_2d(r)
        self.dt = dt
        self.x = None
        self.dxdt = 0
        self.n = 0
        self.t = 0
        self.random = np.random.RandomState(seed)
        self.seed(seed)


    @property
    def name(self) -> str:
        return self.__class__.__name__
    @property
    def state(self) -> np.ndarray:
        return self.x
    @state.setter
    def state(self, x: np.ndarray):
        self.x = np.asarray(x, dtype=self.dtype)


    def reset(self, x=None):
        self.x = (np.asarray(x) if x is not None else \
                  self.observation_space.sample()
                 ).astype(self.dtype)
        self.n = 0
        self.t = 0
        self.dxdt = 0
        return self.x


    def reward(self, xold, u, x):
        # squeeze in case x has a batch dimension [1, x] (while using torch.mpc)
        # x = np.atleast_1d(x.squeeze())
        # u = np.atleast_1d(u.squeeze())
        # return -(x.T @ self.q @ x + u.T @ self.r @ u).item()
        return None


    def step(self, u: np.ndarray, from_x: np.ndarray=None, persist=True, dt=None, is_velocity=False):
        u = np.asarray(u, dtype=self.dtype)
        old_dxdt = self.dxdt
        old_x = np.asarray(self.x if from_x is None else from_x, dtype=self.dtype)
        new_dxdt = np.asarray(self.dynamics(self.t, old_x, u), dtype=self.dtype)
        dt = self.dt if dt is None else dt
        new_x = (old_x + 0.5 * (old_dxdt + new_dxdt) * dt).astype(self.dtype)
        r = 0.
        if not self.custom_reward:
            r = self.reward(old_x, u, new_x)
        if persist:
            self.n += 1
            self.t += self.dt
            self.x = new_x
            self.dxdt = new_dxdt
        return new_x, r, False, {'u': u, 'dxdt': new_dxdt}



def functions(use_torch: bool=False) -> namedtuple:
    """
    Create a namespace of functions either from pytorch, or numpy such that
    they can be used transparently by other functions.

    Parameters
    ----------
    use_torch : bool, optional
        Whether to use pytorch functions, by default False

    Returns
    -------
    Namespace
        A namespace with functions, which can be used like np.sin, np.cos etc.
    """
    NameSpace = namedtuple('Functions', (
        'sin', 'cos', 'tan',
        'min', 'max', 'clip',
        'atleast_1d', 'atleast_2d',
        'zeros_like', 'zeros', 'sign',
        'dot', 'cross',
        'asarray',
        'float32'
    ))
    if use_torch:
        import torch
        ns = NameSpace(
            sin=torch.sin, cos=torch.cos, tan=torch.tan,
            min=torch.min, max=torch.max, clip=torch.clip,
            atleast_1d=torch.atleast_1d, atleast_2d=torch.atleast_2d,
            zeros_like=torch.zeros_like, zeros=torch.zeros, sign=torch.sign,
            dot=torch.dot, cross=torch.cross,
            asarray=torch.as_tensor,
            float32=torch.float32
        )
    else:
        ns = NameSpace(
            sin=np.sin, cos=np.cos, tan=np.tan,
            min=np.min, max=np.max, clip=np.clip,
            atleast_1d=np.atleast_1d, atleast_2d=np.atleast_2d,
            zeros_like=np.zeros_like, zeros=np.zeros, sign=np.sign,
            dot=np.dot, cross=np.cross,
            asarray=np.asarray,
            float32=np.float32
        )
    return ns