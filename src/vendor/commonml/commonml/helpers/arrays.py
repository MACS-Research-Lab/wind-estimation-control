"""
Array or tensor operations operations.
"""

from typing import List, Iterable, Dict, Union
from collections import OrderedDict

import numpy as np
import torch



def homogenous_array(arrays: List[Iterable], start_align=True, fillvalue=np.nan) -> np.ndarray:
    """
    Convert a list of 1D arrays of multiple lengths into a 2D array padded with
    zeros.

    Parameters
    ----------
    arrays : List[Iterable]
        List of 1D iterables.
    start_align : bool, optional
        Whether to align all arrays' start positions, by default True
    fillvalue : float, int
        The value to put in empty parts of the 2D array, by default NaN

    Returns
    -------
    np.ndarray
        A 2D array of size len(arrays) x max array length
    """
    maxlen = max(map(len, arrays))
    res = np.zeros((len(arrays), maxlen)) + fillvalue
    for i, arr in enumerate(arrays):
        if start_align:
            res[i, :len(arr)] = arr
        else:
            res[i, -len(arr):] = arr
    return res



def copy_tensor(t: Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]) \
    -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Make a copy of a tensor or a state_dict such that it is detached from the
    computation graph and does not share underlying data.

    Parameters
    ----------
    t : Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]
        A tensor or a dictionary of [name, torch.Tensor]

    Returns
    -------
    Union[torch.Tensor, Dict[str, torch.Tensor]]
        Same object as t
    """
    if isinstance(t, OrderedDict):
        return OrderedDict([(k, v.clone().detach()) for k, v in t.items()])
    elif isinstance(t, torch.Tensor):
        return t.clone().detach()
    elif isinstance(t, (list, tuple)):
        return [t_.clone().detach() for t_ in t]
    else:
        raise TypeError('Only OrderedDict or Tensor supported')
