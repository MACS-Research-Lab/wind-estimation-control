"""
Timeseries operations
"""

from typing import Iterable
from datetime import datetime, timedelta



def contiguous_sequences(index: Iterable[datetime], interval: timedelta,
                        filter_min: int=1) -> Iterable[Iterable[datetime]]:
    """
    Breaks up a `DatetimeIndex` or a list of timestamps into a list of contiguous
    sequences.

    Parameters
    ----------
    index: Iterable[pd.datetime]
        An index/list of timestamps in chronoligical order,
    interval: pd.Timedelta
        A `Timedelta` object specifying the uniform intervals to determine
        contiguous indices.
    filter_min: int, optional
        Minimum size of subsequence to include in the result.

    Returns
    -------
    List[Iterable[pd.datetime]]
        A list of lists of `pd.datetime` objects.
    """
    indices = []
    j, k = 0, 1
    while k < len(index):           # for each subsequence
        seq = [index[j]]
        indices.append(seq)
        while k < len(index):       # for each element in subsequence
            diff = index[k] - index[j]
            if diff == interval:    # exact interval, add to subsequence
                seq.append(index[k])
                k += 1
                j += 1
            elif diff < interval:   # interval too small, look ahead
                k += 1
            else:                   # new subsequence
                j = k
                k += 1
                break
    return [i for i in indices if len(i) >= filter_min]