
from luciani import *
import pickle
import numpy as np
import bct
import networkx as nx
import math 
import functools


pickle_file = 'data/1.pickle'

with open(pickle_file, "rb") as f:
    matrix = pickle.load(f)
    print(matrix)

def get_binary_matrix(matrix, threshold):
    return np.where(np.abs(matrix) < threshold, 0, 1)

def get_weighted_matrix(matrix, threshold):
    adj_matrix = np.copy(matrix)
    adj_matrix[np.absolute(adj_matrix) < threshold] = 0
    return adj_matrix

print(matrix)



w = get_weighted_matrix(matrix, 0.1)
G = nx.Graph(w)
phi = (1+ math.sqrt(5))

bin_matrix = get_binary_matrix(matrix, 0.1)

print(bin_matrix)

#------------------------------
# Algorithms
#------------------------------

nx.single_source_shortest_path_length

#centrality
bct.betweenness_bin(bin_matrix)
bct.eigenvector_centrality_und(w)
bct.subgraph_centrality(bin_matrix)
bct.flow_coef_bd(bin_matrix)
bct.kcoreness_centrality_bu(bin_matrix)
bct.erange(w)
bct.pagerank_centrality(w, 0.85)
bct.kcoreness_centrality_bd(bin_matrix)
#nx.current_flow_betweenness_centrality(G)
nx.closeness_centrality(G)
nx.katz_centrality_numpy(G, 1/phi)
nx.communicability_betweenness_centrality(G)

#clustering
bct.clustering_coef_wu(w)
bct.transitivity_wu(w)
#bct.transitivity_bd(bin_matrix)  escono valori pari a inf
bct.transitivity_bu(bin_matrix)
bct.transitivity_wd(w)
#bct.clustering_coef_bd(bin_matrix)  valori pari a inf
bct.clustering_coef_bu(bin_matrix)
bct.clustering_coef_wd(w)
bct.clustering_coef_wu_sign(w)
#core
#bct.local_assortativity_wu_sign(w) da errori
bct.assortativity_wei(w)
bct.assortativity_bin(bin_matrix)
bct.core_periphery_dir(w)
bct.rich_club_wu(w)
bct.rich_club_wd(w)
bct.rich_club_bu(bin_matrix)
bct.rich_club_bd(bin_matrix)
bct.kcore_bd(bin_matrix, 1)
bct.kcore_bu(bin_matrix, 1)
#degree
bct.degrees_und(w)
bct.degrees_dir(w)
bct.strengths_und(w)
bct.strengths_dir(w)
bct.strengths_und_sign(w)
#distance
bct.efficiency_wei(w)
bct.efficiency_bin(matrix)
bct.distance_wei(w)
bct.findwalks(matrix)
bct.charpath(w)
bct.breadthdist(matrix)
bct.distance_bin(matrix)
#modularity
bct.modularity_und(w)
bct.partition_distance(w, 0.85)
bct.modularity_louvain_und_sign(bin_matrix, 1, 'sta')
bct.modularity_louvain_und_sign(w, 1, 'sta')
bct.modularity_probtune_und_sign(matrix, 'sta', 1, None, 0.45)
# bct.modularity_dir(w, 1, None)
bct.modularity_finetune_dir(bin_matrix)
bct.modularity_finetune_dir(w)
bct.modularity_finetune_und(bin_matrix, None, 1)
bct.modularity_finetune_und(w, None, 1)
bct.modularity_finetune_und_sign(w, 'sta', 1, None)
bct.modularity_finetune_und_sign(matrix, 'sta', 1, None)
bct.modularity_louvain_dir(w, 1, False)
bct.modularity_louvain_dir(matrix, 1, False)
bct.modularity_louvain_dir(matrix, 1, False)
bct.modularity_louvain_und(matrix, 1, False)
#physical_connectivity
bct.density_dir(w)   #calcolabile sia per la matrice binaria che pesata 
bct.density_dir(matrix)
bct.density_und(w)
bct.density_und(matrix)
#reference
bct.latmio_dir_connected(w,1)
#randmio_und(w, 1) #itr è un intero
bct.latmio_dir(w, 1)   #1 è scelto da me, è un parametro itr
bct.latmio_und(w, 1)
bct.randmio_dir(w, 1)   #int è un numero scelto da noi
bct.randmio_dir_connected(w, 1)
bct.randmio_dir_signed(w, 1)
bct.randmio_und_signed(w, 1)
#similarity
bct.edge_nei_overlap_bu(w)
bct.edge_nei_overlap_bu(bin_matrix)
bct.gtom(w,1)
bct.matching_ind_und(w)
bct.corr_flat_dir(w,bin_matrix)
bct.corr_flat_und(w,bin_matrix)
bct.dice_pairwise_und(w,bin_matrix)



