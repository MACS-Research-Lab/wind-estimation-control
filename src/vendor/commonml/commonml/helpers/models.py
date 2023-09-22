from typing import Any, OrderedDict, Union, Dict
from copy import deepcopy

from torch.nn import Module
from sklearn.base import BaseEstimator



def clone(model: Union[BaseEstimator, Module, OrderedDict, Dict],
          attrs: Dict[str, Any]=None) -> Union[BaseEstimator, Module, OrderedDict, Dict]:
    """
    Copy a scikit-learn or pytorch model.

    Parameters
    ----------
    model : Union[BaseEstimator, Module]
        The model instance.
    attrs : Dict[str, Any], optional
        Any attributes to set in the copied model, by default None

    Returns
    -------
    Union[BaseEstimator, Module]
        The copied model with attributes set.
    """
    mcopy = deepcopy(model)
    if attrs is not None:
        for attr, val in attrs.items():
            setattr(mcopy, attr, val)
    return mcopy
