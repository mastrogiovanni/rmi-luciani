
from luciani import *
import pickle
import numpy as np
import bct

pickle_file = '1.pickle'

with open(pickle_file, "rb") as f:
    matrix = pickle.load(f)

def get_binary_matrix(matrix, threshold):
    return np.where(np.abs(matrix) < threshold, 0, 1)

def get_weighted_matrix(matrix, threshold):
    adj_matrix = np.copy(matrix)
    adj_matrix[np.absolute(adj_matrix) < threshold] = 0
    return adj_matrix

w = get_weighted_matrix(matrix, 0.1)

bin_matrix = get_binary_matrix(matrix, 0.1)

#------------------------------
# Algorithms
#------------------------------

bct.betweenness_bin(bin_matrix)

