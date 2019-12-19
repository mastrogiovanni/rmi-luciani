

import numpy as np
import scipy as sc
import pandas as pd

#binarize
def binarize(w, copy=True):    
    if copy:
        w = w.copy()
    w[w != 0] = 1
    return w

#get_components
def get_components(w, no_depend=False):

    w = binarize(w, copy=True)
    n = len(w)
    np.fill_diagonal(w, 1)

    edge_map = [{u,v} for u in range(n) for v in range(n) if w[u,v] == 1]
    union_sets = []
    for item in edge_map:
        temp = []
        for s in union_sets:

            if not s.isdisjoint(item):
                item = s.union(item)
            else:
                temp.append(s)
        temp.append(item)
        union_sets = temp

    comps = np.array([i+1 for v in range(n) for i in 
        range(len(union_sets)) if v in union_sets[i]])
    comp_sizes = np.array([len(s) for s in union_sets])

    return comps, comp_sizes
