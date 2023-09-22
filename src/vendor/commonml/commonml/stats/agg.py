"""
Aggregation functions over numpy arrays.
"""

from typing import Tuple

import numpy as np



def rolling_mean(arr: np.ndarray, window: int=None) -> np.ndarray:
    """
    Calculate rolling mean of a sequence. Resulting array is of length
    len(arr) - window + 1.

    Parameters
    ----------
    arr : np.ndarray
        Array to take average of.
    window : int, optional
        Size of averaging window, by default square root of array length

    Returns
    -------
    np.ndarray
        An array of means
    """
    arr = np.asarray(arr)
    window = int(np.sqrt(len(arr))) if window is None else window
    return np.convolve(arr, np.ones(window), 'valid') / window



def mean_std(arr: np.ndarray, axis: int=0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate means and standard deviations of arrays, ignoring NaNs.

    Parameters
    ----------
    arr : np.ndarray
        Array to calculate mean and standard deviation for.
    axis : int, optional
        Axis along which to reduce, by default 0 (i.e. mean of each element across
        rows of a 2D array for example)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two arrays of same shape as `arr` for mean and std
    """
    arr = np.asarray(arr)
    means = np.nanmean(arr, axis=axis)
    stds = np.nanstd(arr, axis=axis)
    return means, stds
