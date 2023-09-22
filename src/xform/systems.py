"""
Operations primarily on systems and environments.
"""

from typing import Union, Tuple, Callable
from types import SimpleNamespace

import numpy as np
import control
import gym

from .matrices import ab_xform_from_pseudo_matrix



def policy_transform(
    sys: Union[control.LinearIOSystem, Tuple[np.ndarray, np.ndarray]],
    xformA=None, xformB=None, ctrl_law=None,
    A_t=None, B_t=None
) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(sys, control.LinearIOSystem):
        A_s, B_s = sys
    else:
        A_s, B_s = sys.A, sys.B
    F_A = np.eye(len(A_s)) if xformA is None else xformA
    F_B = np.eye(len(A_s)) if xformB is None else xformB
    I = np.eye(len(A_s))
    if B_t is None:
        F_B_B_ = np.linalg.pinv(F_B @ B_s)
    else:
        F_B_B_ = B_t_ = np.linalg.pinv(B_t)
    if A_s is None:
        F_A_A = F_A @ A_s
    else:
        F_A_A = A_t
    
    state_xform = (F_B_B_@(F_A_A-A_s))
    action_xform = (F_B_B_@B_s)
    if ctrl_law is not None:
        # return K_F such that u = -K_F x
        K_F = state_xform + action_xform @ ctrl_law
        return K_F
    else:
        # return state, action transforms such that
        # u = K_x @ x + K_u @ u
        # which means
        # u = (-state_xform) @ x + action_xform @ (-law @ x)
        return -state_xform, action_xform



def pseudo_matrix(sys: control.LinearIOSystem, dt=1e-2):
    """
    The matrix [A.dt + I, B.dt] to be multiplied with [x;u] to give the next state
    """
    assert sys.A.shape[0]==sys.B.shape[0], 'State size of A=/=B matrix'
    nstates, nactions = sys.A.shape[0], sys.B.shape[1]
    shape = (nstates, nstates + nactions)
    m = np.zeros(shape, dtype=np.float32)
    m[:, :nstates] = sys.A * dt + np.eye(nstates)
    m[:, nstates:] = sys.B * dt
    return m



def pseudo_matrix_from_data(
    env: gym.Env, n, control_law=None, n_episodes_or_steps='episodes'
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    xu, x = get_env_samples(
        env=env, control_law=control_law,
        n=n, n_episodes_or_steps=n_episodes_or_steps
    )
    # P = (x @ xu.T) @ np.linalg.pinv(xu @ xu.T)
    P = (x) @ np.linalg.pinv(xu)
    err = np.linalg.norm(P @ xu - x, axis=0).mean()
    return P, err, xu, x



def get_env_samples(
    env: gym.Env, n, control_law: Union[np.ndarray, Callable, None]=None, n_episodes_or_steps='episodes'
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(control_law, np.ndarray):
        policy = lambda x: -control_law @ x
    elif control_law is None:
        policy = lambda x: env.action_space.sample()
    elif hasattr(control_law, 'predict'): # i.e. is ActorCriticPolicy from stable_baselines3
        policy = lambda x: control_law.predict(x, deterministic=True)[0]
    elif callable(control_law):
        policy = control_law
    xu, x = [], []
    i = 0
    while i < n:
        state = env.reset()
        if len(state)==2 and isinstance(state[1], dict):
            # conforming to the new gym spec, where reset returns a dict
            # as the second value
            state = state[0]
        done = False
        while not done and i < n:
            action = policy(state)
            nstate, _, done, *_, info = env.step(action)
            if n_episodes_or_steps == 'steps':
                i += 1
            xu.append(np.concatenate((state, info.get('u', action))))
            x.append(nstate)
            state = nstate
            if done:
                if n_episodes_or_steps == 'episodes':
                    i += 1
                break
    return (np.asarray(xu, dtype=np.float32).T,
           np.asarray(x, dtype=np.float32).T)



def transform_linear_system(sys: control.LinearIOSystem, xformA, xformB) -> control.LinearIOSystem:
    sys = sys.copy(name=sys.name)
    sys.A = xformA @ sys.A
    sys.B = xformB @ sys.B
    return sys



def get_transforms(
    agent, env_s, env_t,
    buffer_episodes=5,
    n_episodes_or_steps='episodes',
    xformA=None, xformB=None,
    data_driven_source=True,
    x0=None, u0=None,
) -> Tuple[np.ndarray, np.ndarray, SimpleNamespace]:
    # linearize a non-linear system to get A,B,C,D representation,
    # and the resulting pseudo matrix P_s of the source task
    if hasattr(env_s, 'system') and not data_driven_source:
        if isinstance(env_s.system, control.LinearIOSystem):
            _sys_linear = env_s.system
        elif isinstance(env_s.system, control.NonlinearIOSystem):
            if x0 is None:
                x0 = env_s.observation_space.sample() * 0
            if u0 is None:
                u0 = env_s.action_space.sample() * 0
            _sys_linear = env_s.system.linearize(x0, u0)
    else:
        # Learn environment model from data
        data_driven_source = True
    # get the pseudo matrix representing source system dynamics
    if data_driven_source:
        P_s, err_s, *_ = pseudo_matrix_from_data(env_s, buffer_episodes, agent, n_episodes_or_steps=n_episodes_or_steps)
    else:
        P_s, err_s, *_ = pseudo_matrix(_sys_linear, env_s.dt), 0.
    # get pseudo matrix representing target system dynamics
    P_t, err_t, xu, x = pseudo_matrix_from_data(env_t, buffer_episodes, agent, n_episodes_or_steps=n_episodes_or_steps)
    # get the relationship between source and target systems
    A_s, B_s, A_t, B_t, F_A, F_B = ab_xform_from_pseudo_matrix(P_s, P_t, env_s.dt)
    C_s, D_s = np.eye(len(A_s)), np.zeros_like(B_s)
    if xformA is not None:
        F_A = xformA
    if xformB is not None:
        F_B = xformB
    # generate policy transforms from the source system,
    # and its relationship to the target system
    if data_driven_source:
        source_system = control.ss(A_s, B_s, C_s, D_s)
    else:
        source_system = _sys_linear
    state_xform, action_xform = policy_transform(source_system, F_A, F_B, A_t=A_t, B_t=B_t)
    ns = SimpleNamespace()
    ns.x = x.T # convert column vectors to row vectors
    ns.A_s, ns.B_s = A_s, B_s
    ns.F_A, ns.F_B = F_A, F_B
    ns.err_s, ns.err_t = err_s, err_t
    return state_xform, action_xform, ns



def is_controllable(sys: control.LinearIOSystem):
    #https://www.mathworks.com/help/control/ref/ss.ctrb.html
    return np.linalg.matrix_rank(control.ctrb(sys.A, sys.B)) - \
           len(sys.A) == 0