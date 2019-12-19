
import numpy as np
import scipy as sc
import pandas as pd
import bct
import networkx as nx

"""
distance_wei_floyd
"""
def distance_wei_floyd(adjacency, transform=None):
   

    if transform is not None:
        if transform == 'log':
            if np.logical_or(adjacency > 1, adjacency < 0).any():
                raise ValueError("Connection strengths must be in the " +
                                 "interval [0,1) to use the transform " +
                                 "-log(w_ij).")
            SPL = -np.log(adjacency)
        elif transform == 'inv':
            SPL = 1. / adjacency
        else:
            raise ValueError("Unexpected transform type. Only 'log' and " +
                             "'inv' are accepted")
    else:
        SPL = adjacency.copy().astype('float')
        SPL[SPL == 0] = np.inf

    n = adjacency.shape[1]

    flag_find_paths = True
    hops = np.array(adjacency != 0).astype('float')
    Pmat = np.repeat(np.atleast_2d(np.arange(0, n)), n, 0)

    for k in range(n):
        i2k_k2j = np.repeat(SPL[:, [k]], n, 1) + np.repeat(SPL[[k], :], n, 0)

        if flag_find_paths:
            path = SPL > i2k_k2j
            i, j = np.where(path)
            hops[path] = hops[i, k] + hops[k, j]
            Pmat[path] = Pmat[i, k]

        SPL = np.min(np.stack([SPL, i2k_k2j], 2), 2)

    eye = np.eye(n) > 0
    SPL[eye] = 0

    if flag_find_paths:
        hops[eye], Pmat[eye] = 0, 0

    return SPL, hops, Pmat

"""
retrieve_shortest_path
"""
def retrieve_shortest_path(s, t, hops, Pmat):
  path_length = hops[s, t]
  if path_length != 0:
    path = np.zeros((int(path_length + 1), 1), dtype='int')
    path[0] = s
    for ind in range(1, len(path)):
      s = Pmat[s, t]
      path[ind] = s
    else:
      path = []

    return path


"""
search_information
"""

def search_information(adjacency, transform=None, has_memory=False):
  

    N = len(adjacency)

    if np.allclose(adjacency, adjacency.T):
        flag_triu = True
    else:
        flag_triu = False

    T = np.linalg.solve(np.diag(np.sum(adjacency, axis=1)), adjacency)
    _, hops, Pmat = distance_wei_floyd(adjacency, transform)

    SI = np.zeros((N, N))
    SI[np.eye(N) > 0] = np.nan

    for i in range(N):
        for j in range(N):
            if (j > i and flag_triu) or (not flag_triu and i != j):
                path = retrieve_shortest_path(i, j, hops, Pmat)
                lp = len(path) - 1
                if flag_triu:
                    if np.any(path):
                        pr_step_ff = np.zeros(lp)
                        pr_step_bk = np.zeros(lp)
                        if has_memory:
                            pr_step_ff[0] = T[path[0], path[1]]
                            pr_step_bk[lp-1] = T[path[lp], path[lp-1]]
                            for z in range(1, lp):
                                pr_step_ff[z] = T[path[z], path[z+1]] /\
                                    (1 - T[path[z-1], path[z]])
                                pr_step_bk[lp-z-1] = T[path[lp-z],
                                                       path[lp-z-1]] /\
                                    (1 - T[path[lp-z+1], path[lp-z]])
                        else:
                            for z in range(lp):
                                pr_step_ff[z] = T[path[z], path[z+1]]
                                pr_step_bk[z] = T[path[z+1], path[z]]

                        prob_sp_ff = np.prod(pr_step_ff)
                        prob_sp_bk = np.prod(pr_step_bk)
                        SI[i, j] = -np.log2(prob_sp_ff)
                        SI[j, i] = -np.log2(prob_sp_bk)
                else:
                    if np.any(path):
                        pr_step_ff = np.zeros(lp)
                        if has_memory:
                            pr_step_ff[0] = T[path[0], path[1]]
                            for z in range(1, lp):
                                pr_step_ff[z] = T[path[z], path[z+1]] /\
                                                (1 - T[path[z-1], path[z]])
                        else:
                            for z in range(lp):
                                pr_step_ff[z] = T[path[z], path[z+1]]

                        prob_sp_ff = np.prod(pr_step_ff)
                        SI[i, j] = -np.log2(prob_sp_ff)
                    else:
                        SI[i, j] = np.inf

    return SI

"""
distance_inv
"""

def distance_inv(g):
	D = np.eye(len(g))
	n = 1
	nPATH = g.copy()
	L = (nPATH != 0)
	while np.any(L):
		D += n * L
		n += 1
		nPATH = np.dot(nPATH,g)
		L = (nPATH != 0) * (D == 0)
	D[np.logical_not(D)] = np.inf
	D = 1 / D
	np.fill_diagonal(D, 0)
	return D

"""
vulnerability_index
"""
def vulnerability_index(w):
	n = len(w)
	e = distance_inv(w)
	G = nx.Graph(w)
	E = nx.global_efficiency(G)
	for i in range(1,64):
		Ei = np.sum(e[i]) / (n * n - n)
		print(Ei)
	Vi = np.divide(np.subtract(E,Ei),E)
	return Vi

