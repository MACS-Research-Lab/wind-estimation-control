"""
Environment utilities for gym classes.
"""

from typing import Callable, Dict, List, Tuple
import multiprocessing as mp

import numpy as np
import gym



def runner(env: gym.Env, callback: Callable, agent_fn: Callable=None,
           episodes: int=1, max_steps: int=1e3, processes: int=None):
    """
    Run environment and execute callback with local variables at every step.

    Parameters
    ----------
    env : gym.Env
        Environment to run.
    callback : Callable
        A function which accepts a dictionary of local variables. Called at each
        step of the environment.
    agent_fn : Callable, optional
        Function that accepts state and returns action to take for Env, by default random.
    episodes : int, optional
        Number of episodes to run for, by default 1
    max_steps : int, optional
        Maximum number of steps to take. Both `episodes` and `max_steps`
        will cause function to terminate if exceeded, by default 1e3
    processes : int, optional
        Not implemented. TODO.

    Returns
    -------
    List[np.ndarray]
        A list of arrays. Each array contains rewards for one episode.
    """
    ep, steps = 0, 0
    act = (lambda s: env.action_space.sample()) if agent_fn is None else agent_fn
    while (ep < episodes):
        done = False
        state = env.reset()
        while not done and (steps < max_steps):
            action = act(state)
            nstate, reward, done, _ = env.step(action)
            steps += 1
            callback(locals())
            state = nstate
        ep += 1



def rewards(env: gym.Env, agent_fn: Callable=None, episodes: int=1,
            max_steps: int=1e3) -> List[np.ndarray]:
    """
    Run environment and gather rewards.

    Parameters
    ----------
    env : gym.Env
        Environment to run.
    agent_fn : Callable, optional
        Function that accepts state and returns action to take for Env, by default random.
    episodes : int, optional
        Number of episodes to run for, by default 1
    max_steps : int, optional
        Maximum number of steps to take. Both `episodes` and `max_steps`
        will cause function to terminate if exceeded, by default 1e3

    Returns
    -------
    List[np.ndarray]
        A list of arrays. Each array contains rewards for one episode.
    """
    rewards = [[]]
    def rgetter(lcl: Dict):
        reward = lcl.get('reward')
        done = lcl.get('done')
        rewards[-1].append(reward)
        if done:
            rewards.append([])
    runner(env, rgetter, agent_fn, episodes=episodes, max_steps=max_steps)
    if len(rewards[-1]) == 0:
        rewards.pop()
    return [np.asarray(r) for r in rewards]



def get_from_env(variables: Tuple[str], env: gym.Env, agent_fn: Callable=None, episodes: int=1,
                 max_steps: int=1e3) -> Dict[str, List[np.ndarray]]:
    collection = {variable: [[]] for variable in variables}
    def getter(lcl: Dict):
        for variable in variables:
            value = lcl.get(variable)
            collection[variable][-1].append(value)
        done = lcl.get('done')
        if done:
            for variable in variables:
                collection[variable][-1] = np.asarray(collection[variable][-1])
                collection[variable].append([])
    runner(env, getter, agent_fn, episodes=episodes, max_steps=max_steps)
    for _, list_of_arrs in collection.items():
        if len(list_of_arrs[-1]) == 0:
            list_of_arrs.pop()
    return collection