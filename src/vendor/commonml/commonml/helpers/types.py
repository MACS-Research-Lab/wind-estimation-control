"""
Type-checking, -conversion operations.
"""



import datetime
from typing import Any

import numpy as np



def is_datetype(d: Any) -> bool:
    """
    Check if an object represents a date.
    
    Arguments:
        d {Any} -- Any instance
    
    Returns:
        bool -- True if it represents a date or a time.
    """
    return isinstance(d, (datetime.datetime, datetime.date, datetime.time)) \
           or np.issubdtype(getattr(d, 'dtype', None), np.datetime64)
