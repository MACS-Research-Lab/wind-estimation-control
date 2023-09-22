from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ..helpers import is_datetype


def plot_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, ax=None, fig_kwargs:Dict={}, **kwargs) \
    -> Axes3D:
    """
    Plot a 3D surface given x, y, z coordinates.

    Arguments:
        x {np.ndarray} -- A 1D or 2D array (indexed as [x, y]). Can be numeric or
            date/time-like.
        y {np.ndarray} -- A 1D or 2D array (indexed as [x, y]). Can be numeric or
            date/time-like.
        z {np.ndarray} -- A 2D array indexed as [x, y].

    Keyword Arguments:
        ax {Axes3D} -- The axes on which to plot surface. (default: {None})
        fig_kwargs {Dict} -- Dictionary of arguments for plt.figure() creation.
        **kwargs -- Passed to `ax.plot_surface()`

    Returns:
        Axes3D -- The axes on which the surface was plotted.
    """
    if ax is None:
        fig = plt.figure(**fig_kwargs)
        ax = fig.add_subplot(111, projection='3d')

    xtime, ytime = is_datetype(x[0]), is_datetype(y[0])
    xgrid, ygrid, zgrid = map(np.asarray, (x, y, z))

    xdim = x.shape[0]
    ydim = y.shape[0] if y.ndim == 1 else y.shape[1]

    if xgrid.ndim == 1:
        xlabels = xgrid
        xgrid = np.repeat(x[:, None], axis=1, repeats=ydim)
    else:
        xlabels = xgrid[:, 0]
    if ygrid.ndim == 1:
        ylabels = ygrid
        ygrid = np.repeat(y[None, :], axis=0, repeats=xdim)
    else:
        ylabels = ygrid[0, :]

    if xtime:
        xgrid = np.repeat(np.arange(xdim).reshape(-1, 1), axis=1, repeats=ydim)
    if ytime:
        ygrid = np.repeat(np.arange(ydim).reshape(1, -1), axis=0, repeats=xdim)

    ax.plot_surface(xgrid, ygrid, zgrid, **kwargs)

    if xtime:
        xticklocs = np.asarray(tuple(filter(lambda x: 0 <= x < xdim, \
                                            ax.get_xticks()))).astype(int)
        ax.set_xticks(xticklocs)
        ax.set_xticklabels(xlabels[xticklocs], rotation=20)
    if ytime:
        yticklocs = np.asarray(tuple(filter(lambda x: 0 <= x < xdim, \
                                            ax.get_yticks()))).astype(int)
        ax.set_yticks(yticklocs)
        ax.set_yticklabels(ylabels[yticklocs], rotation=20)

    return ax
