"""
Timeseries-specific processing.
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from tslearn.metrics import dtw
# from tslearn.utils import to_time_series



def similarity_matrix(timeseries: List[pd.DataFrame], columns: List[str]=None,
    beta: float=1) -> np.ndarray:
    """
    Calculate a similarity measure (not metric) between time series. The series
    do not have be of the same length. Used Dynamic Time Warping.

    Parameters
    ----------
    timeseries : List[pd.DataFrame]
        List of data frames, where each dataframe is a multivariate time series.
    columns : List[str], optional
        Columns to use to measure similarity, by default all columns are used
    beta : float, optional
        The factor in the exponential when normalizing the measures, by default 1

    Returns
    -------
    np.ndarray
        A square matrix of dimensions `len(timeseries), len(timeseries)`
    """
    columns = timeseries[0].columns if columns is None else columns
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(pd.concat(timeseries, axis=0)[columns])
    dissimilarities = np.zeros((len(timeseries), len(timeseries))) + np.inf # initially all tickers are dissimilar

    for i, arr1 in enumerate(timeseries):
        for j, arr2 in enumerate(timeseries):
            dissimilarities[i, j] = dtw(
                    to_time_series(scaler.transform(arr1[columns])),
                    to_time_series(scaler.transform(arr2[columns])))
    
    return np.exp(-beta * dissimilarities / dissimilarities.std())
