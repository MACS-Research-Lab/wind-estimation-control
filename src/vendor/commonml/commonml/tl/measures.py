"""
Similarity measures for tasks.
"""

from typing import Tuple

import numpy as np
from scipy.spatial import distance
import torch

from ..rl import Policy, Memory, DEVICE


def policy_distance(policy1: Policy, policy2: Policy, memory: Memory,
    method='jensenshannon') -> Tuple[float, float]:
    """
    Calculate distance measure between two policies on a sample of experiences.

    Parameters
    ----------
    policy1 : Policy
        First policy instance.
    policy2 : Policy
        Second policy instance
    memory : Memory
        A memory instance containing states and actions (`.actions`, `.states`)
    method : str, optional
        The distance measure to use, by default 'jensenshannon' metric

    Returns
    -------
    Tuple[float, float]
        The distance measure of [p1,p2] and [p2,p1]. If the measure is symmetric,
        both values are the same.
    """
    s = torch.tensor(memory.states).float()
    a = torch.tensor(memory.actions).float()

    logprobs1, _, _ = policy1.evaluate(s.to(policy1.device), a.to(policy1.device))
    p1 = torch.exp(logprobs1).detach()

    logprobs2, _, _ = policy2.evaluate(s.to(policy2.device), a.to(policy2.device))
    p2 = torch.exp(logprobs2).detach()

    if method=='jensenshannon':
        dist = distance.jensenshannon(p1.cpu(), p2.cpu())
    elif method == 'euclidean':
        dist = distance.euclidean(p1.cpu(), p2.cpu())
    elif method == 'l1':
        dist = np.linalg.norm(p1.cpu() - p2.cpu(), ord=1)
    elif method == 'cosine':
        dist = distance.cosine(p1.cpu(), p2.cpu())
    return dist, dist
