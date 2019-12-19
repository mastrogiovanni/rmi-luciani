
import numpy as np
import scipy as sc
import pandas as pd
import bct

"""
matching_ind
"""

def matching_ind(CIJ):
    '''
    For any two nodes u and v, the matching index computes the amount of
    overlap in the connection patterns of u and v. Self-connections and
    u-v connections are ignored. The matching index is a symmetric
    quantity, similar to a correlation or a dot product.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        adjacency matrix

    Returns
    -------
    Min : NxN np.ndarray
        matching index for incoming connections
    Mout : NxN np.ndarray
        matching index for outgoing connections
    Mall : NxN np.ndarray
        matching index for all connections

    '''
    n = len(CIJ)

    Min = np.zeros((n, n))
    Mout = np.zeros((n, n))
    Mall = np.zeros((n, n))

    # compare incoming connections
    for i in range(n - 1):
        for j in range(i + 1, n):
            c1i = CIJ[:, i]
            c2i = CIJ[:, j]
            usei = np.logical_or(c1i, c2i)
            usei[i] = 0
            usei[j] = 0
            nconi = np.sum(c1i[usei]) + np.sum(c2i[usei])
            if not nconi:
                Min[i, j] = 0
            else:
                Min[i, j] = 2 * \
                    np.sum(np.logical_and(c1i[usei], c2i[usei])) / nconi

            c1o = CIJ[i, :]
            c2o = CIJ[j, :]
            useo = np.logical_or(c1o, c2o)
            useo[i] = 0
            useo[j] = 0
            ncono = np.sum(c1o[useo]) + np.sum(c2o[useo])
            if not ncono:
                Mout[i, j] = 0
            else:
                Mout[i, j] = 2 * \
                    np.sum(np.logical_and(c1o[useo], c2o[useo])) / ncono

            c1a = np.ravel((c1i, c1o))
            c2a = np.ravel((c2i, c2o))
            usea = np.logical_or(c1a, c2a)
            usea[i] = 0
            usea[i + n] = 0
            usea[j] = 0
            usea[j + n] = 0
            ncona = np.sum(c1a[usea]) + np.sum(c2a[usea])
            if not ncona:
                Mall[i, j] = 0
            else:
                Mall[i, j] = 2 * \
                    np.sum(np.logical_and(c1a[usea], c2a[usea])) / ncona

    Min = Min + Min.T
    Mout = Mout + Mout.T
    Mall = Mall + Mall.T

    return Min, Mout, Mall

"""
edge_nei_overlap_bd
"""
def edge_nei_overlap_bd(CIJ):

    ik, jk = np.where(CIJ)
    lel = len(CIJ[ik, jk])
    n = len(CIJ)
    
    _, _, deg = bct.degrees_dir(CIJ)

    ec = np.zeros((lel,))
    degij = np.zeros((2, lel))
    for e in range(lel):
        neiik = np.setdiff1d(np.union1d(
            np.where(CIJ[ik[e], :]), np.where(CIJ[:, ik[e]])), (ik[e], jk[e]))
        neijk = np.setdiff1d(np.union1d(
            np.where(CIJ[jk[e], :]), np.where(CIJ[:, jk[e]])), (ik[e], jk[e]))
        ec[e] = len(np.intersect1d(neiik, neijk)) / \
            len(np.union1d(neiik, neijk))
        degij[:, e] = (deg[ik[e]], deg[jk[e]])

    EC = np.tile(np.inf, (n, n))
    EC[ik, jk] = ec
    return EC, ec, degij




