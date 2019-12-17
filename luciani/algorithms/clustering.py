##clustering_coef_wu(W)

import numpy as np
import scipy as sc
import pandas as pd

def clustering_coef_wu(W):
    '''
    The weighted clustering coefficient is the average "intensity" of
    triangles around a node.

    Parameters
    ----------
    W : NxN np.ndarray
        weighted undirected connection matrix

    Returns
    -------
    C : Nx1 np.ndarray
        clustering coefficient vector
    '''
    K = np.array(np.sum(np.logical_not(W == 0), axis=1), dtype=float)
    ws = np.cbrt(W)
    cyc3 = np.diag(np.dot(ws, np.dot(ws, ws)))
    K[np.where(cyc3 == 0)] = np.inf  # if no 3-cycles exist, set C=0
    C = cyc3 / (K * (K - 1))
    return C

