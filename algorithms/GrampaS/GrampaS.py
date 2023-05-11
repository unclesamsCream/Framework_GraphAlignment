# from algorithms import Grampa
# from Utils import adj_from_edges, compute_laplacian, embed_spectral, expand_matrix, decompose_spectral, spectral_embedding
# import numpy as np
# from numpy.linalg import inv
# from numpy.linalg import eigh,eig
# import networkx as nx 
# import warnings
# import itertools
# from sklearn.manifold import spectral_embedding as sk_specemb
# from lapjv import lapjv
# from sklearn.cluster import k_means
# from scipy.spatial.distance import cdist
# from sklearn.exceptions import ConvergenceWarning
# import time
# import traceback

from algorithms import Grampa
from Utils import adj_from_edges, compute_laplacian, embed_spectral, expand_matrix, decompose_spectral, spectral_embedding
import numpy as np
from numpy.linalg import inv
from numpy.linalg import eigh,eig
import networkx as nx 
import warnings
import itertools
from sklearn.manifold import spectral_embedding as sk_specemb
from lapjv import lapjv
from sklearn.cluster import k_means, KMeans
from sklearn.exceptions import ConvergenceWarning
import time
import traceback
# def split_graph_hyper(graph, clustering, weighting_scheme='ncut', rcon=True): # POTENTIAL: add mem_eff=False parameter
#     '''Split a graph into disjunct clusters and construct weighted graph where a node represents a cluster.
    
#     Description:
#         - Removes all edges between clusters.
#         - Adds a node per cluster to a new graph.
#         - Adds edges between new nodes with a weight corresponding to the amount of edges removed between them.

#     Parameters:
#         graph (np.array): nxn adjacency matrix

#         clustering: A clustering for the input graph.
#             /format/ Clustering is a dict {node (int): cluster (int)} containing the input graph clustering.
#                      Clustering values has to be consecutive, i.e. >>>np.unique(clustering.valuess()) yields [0,1,2,...,k-1]

#         weighting_scheme: The weighting scheme to be used. Current implementation allows for:
#                           'cut': based on number of connections between clusters.
#                           ###NOT IMPLEMENTED###'size': based on size of cluster relative to total number of nodes

#     Returns:
#         cluster_graph: (kxk np.array) A graph constructed as per above description.
#         subgraphs: A list of edgelist corresponding to graphs where each graph is a disjuncted cluster of the input graph.
#     '''
#     k = np.max(list(clustering.values())) + 1 # Assuming input format is respected
#     # new graph of clusters represented by its adjacency matrix
#     cluster_graph = np.zeros(shape=(k, k))
#     clusters = [[node for (node, c) in clustering.items() if c==i] for i in range(k)]
#     cluster_sizes = np.array([len(c) for c in clusters])

#     # subgraphs = [[] for _ in range(k)]
#     p_G = nx.from_edgelist(graph)
#     Gs = [p_G.subgraph(c).copy() for c in clusters]
#     subgraphs = [np.array(G.edges) for G in Gs]
#     cons = [list(nx.connected_components(G)) for G in Gs]
    
#     # Build hyper graph
#     for (e, f) in list(graph):
#         c1 = clustering[e]
#         c2 = clustering[f]
#         if c1 != c2:
#             if weighting_scheme == 'cut' or weighting_scheme == 'ncut':
#                 # add edge in cluster graph adjacency matrix
#                 cluster_graph[c1, c2] += 1
#                 cluster_graph[c2, c1] += 1

#     if weighting_scheme == 'size':
#         # e_AB = max(|A|, |B|) / |V|
#         for i in range(k):
#             for j in range(k):
#                 if i != j:
#                     cluster_graph[i, j] = cluster_graph[j, i] = max(cluster_sizes[i], cluster_sizes[j]) / len(np.unique(graph))
#     if weighting_scheme == 'ncut':
#         matrix_iterator = list(itertools.combinations(range(k), r=2))
#         for (i, j) in matrix_iterator:
#             cut = cluster_graph[i,j]
#             ncut = (cut / cluster_sizes[i]) + (cut / cluster_sizes[j])
#             cluster_graph[i,j] = cluster_graph[j,i] = ncut
#             # cluster_graph[i,j] = cluster_graph[j,i] = ncut + (len(cons[i]) / len(cons[j]))

#     return cluster_graph, subgraphs, cons # isolated

# def alhpa(src_graph, tar_graph, rsc=0, weighting_scheme='ncut', lap=False, gt=None, ki=False, lalpha=10000):
#     """
#     MAtching by Recursive Partition Alignment
#     ____
#     Summary or Description of the Function

#     Parameters:
#       src_graph (np.array): Edge list of source graph.
#       tar_graph (np.array): Edge list of target graph.

#     Returns:
#       matching (np.array): Array of 2-tuples each representing a matching pair nodes (n1, n2) where n1 is in src_graph and n2 is in tar_graph.
#     """

#     n = len(np.unique(src_graph))
#     if rsc == 0: rsc = np.sqrt(n)

#     eta = 0.2

#     matching = -1 * np.ones(shape=(n, ), dtype=int)
#     readjustment_accs = []
#     all_pos = []
#     def match_grampa(src, tar):
#         if isinstance(src, tuple) and isinstance(tar, tuple):
#             src_adj, src_map = src
#             tar_adj, tar_map = tar
#         else:
#             src_adj, src_map = adj_from_edges(src)
#             tar_adj, tar_map = adj_from_edges(tar)
#         diff = len(src_map) - len(tar_map)
#         #print(diff)
#         if diff < 0: # expand sub src
#             #print(f'size before expansion: {src_adj.shape}')
#             src_adj = expand_matrix(src_adj, abs(diff))
#             #print(f'size after expansion: {src_adj.shape}')
#             #print(f'src_map size before: {len(src_map)}')
#             src_map = list(src_map) + [-1] * abs(diff)
#             #print(f'src_map size after: {len(src_map)}')
#         if diff > 0: # expand sub tar
#             #print(f'size before expansion: {tar_adj.shape}')
#             tar_adj = expand_matrix(tar_adj, diff)
#             #print(f'size after expansion: {tar_adj.shape}')
#             #print(f'tar_map size before: {len(tar_map)}')
#             tar_map = list(tar_map) + [-1] * diff
#             #print(f'tar_map size after: {len(tar_map)}')

#         sub_sim = Grampa.grampa(src_adj, tar_adj, eta, lap=lap)
#         #print(f'adjacency matrices for src and tar (shapes): {src_adj.shape}, {tar_adj.shape})')
#         #print(sub_sim)
#         r, c, _ = lapjv(-sub_sim)
#         match = list(zip(range(len(c)), c))
#         #print(f'GRAMPA:{match}')
#         # translate with map and add to solution
#         for (n1, n2) in match:
#             matching[src_map[n1]] = tar_map[n2]

#     def cluster_recurse(src_e, tar_e, pos=[(0,0)]):
#         pos = pos.copy()
#         src_adj, src_nodes = adj_from_edges(src_e)
#         tar_adj, tar_nodes = adj_from_edges(tar_e)
        
#         #### 1. Spectrally embed graphs into 1 dimension.
#         # l, U, snz = decompose_spectral(src_adj)
#         # mu, V, tnz = decompose_spectral(tar_adj)

#         # src_embedding = U[:, snz[:e_dim]]
#         # tar_embedding = V[:, tnz[:e_dim]]
        
#         # l, src_embedding = spectral_embedding(src_adj, n_components=src_adj.shape[0])
#         # mu, tar_embedding = spectral_embedding(tar_adj, n_components=tar_adj.shape[0])
        
#         l, src_embedding = spectral_embedding(src_adj, n_components=10)
#         mu, tar_embedding = spectral_embedding(tar_adj, n_components=10)

#         # Find the number of clusters based on eigengap heuristics
#         # diffs = np.array([abs(l[i-1] - l[i]) for i in range(1, len(l))])
#         # diffs = np.abs(np.diff(l))
#         # diffs[0] = diffs[1] = diffs[2] = diffs[3] = 0
#         # K = diffs.argmax()
#         # b = K.copy()
#         # for cand in np.argsort(diffs)[::-1][1:]:
#         #     if cand < K and cand > 3:
#         #         if abs(diffs[cand]-diffs[b]) < diffs[b]*.9:
#         #             K = cand

#         diffs = np.abs(np.diff(l))
#         # diffs[0] = diffs[1] = diffs[2] = 0
#         K = max(diffs.argmax() + 1, 3)
#         # b = K.copy()
#         # for cand in np.argsort(diffs)[::-1][1:]:
#         #     if cand < K and cand > 2:
#         #         if abs(diffs[cand]-diffs[b]) < diffs[b]*.05:
#         #             K = cand
#         #             break
#         # K = K + 1

#         # K = max(diffs.argmax()-1, 3)
#         # # inc = 1/len(l)
#         # # weights = np.linspace(1, len(diffs), num=len(diffs))
#         # # wdiffs = diffs
#         # i = 2
#         # # while (diffs[i] * weights[i]) < diffs.mean():
#         # while diffs[i] < diffs.mean():
#         #     i += 1
#         # K = i + 1
#         # print(f'\nFound K={K} at position={pos}\ndiffs: {diffs}\nmean: {diffs.mean()}')
#         print(f'\nFound K={K} at position={pos}')
#         src_embedding = src_embedding.T[:K].T
#         tar_embedding = tar_embedding.T[:K].T

#         # Compute clusters on embedded data with kmeans and lloyd's algorithm
#         warnings.simplefilter('error', category=ConvergenceWarning)
#         try:
#             #print('computing k-means (src graph)')
#             src_centroids, _src_cluster, _ = k_means(src_embedding, n_clusters=K, n_init=10)
#         except Exception as e:
#             #print(f'Troublesome graph!!!\n Matching parent graph using GRAMPA')
#             # match_grampa((src_adj, src_nodes), (tar_adj, tar_nodes), decomp=[(l, U), (mu, V)])
#             match_grampa((src_adj, src_nodes), (tar_adj, tar_nodes))
#             return
#         try:
#             #print('computing k-means (tar graph)')
#             # Seed target graph kmeans by using the src centroids.
#             tar_centroids, _tar_cluster, _ = k_means(tar_embedding, n_clusters=K, init=src_centroids, n_init=1)
#             # tar_centroids, _tar_cluster, _ = k_means(tar_embedding, n_clusters=K, n_init=10)            
#         except Exception as e:
#             #print(f'Troublesome graph!!!\n Matching parent graph using GRAMPA')
#             # match_grampa((src_adj, src_nodes), (tar_adj, tar_nodes), decomp=[(l, U), (mu, V)])
#             match_grampa((src_adj, src_nodes), (tar_adj, tar_nodes))
#             return

#         #print(f'src_centroids:\n{src_centroids}')
#         #print(f'tar_centroids:\n{tar_centroids}')

#         src_cluster = dict(zip(src_nodes, _src_cluster))
#         tar_cluster = dict(zip(tar_nodes, _tar_cluster))
#         src_nodes_by_cluster = [[k for k,v in src_cluster.items() if v == i] for i in range(K)]
#         tar_nodes_by_cluster = [[k for k,v in tar_cluster.items() if v == i] for i in range(K)]
#         print(f'\ncluster numbers: src:{[len(x) for x in src_nodes_by_cluster]}, tar:{[len(x) for x in tar_nodes_by_cluster]}\n')

#         # 2. split graphs (src, tar) according to cluster.
#         src_cluster_graph, src_subgraphs, src_cons = split_graph_hyper(src_e, src_cluster, weighting_scheme=weighting_scheme)
#         tar_cluster_graph, tar_subgraphs, tar_cons = split_graph_hyper(tar_e, tar_cluster, weighting_scheme=weighting_scheme)
#         #print(f'src_cons: {[len(x) for x in src_cons]}\ntar_cons: {[len(x) for x in tar_cons]}')

#         C_UPDATED = False

#         # 3. match cluster_graph.
#         #print(f'\ncluster graphs (src, tar)\n{src_cluster_graph}\n{tar_cluster_graph}')

#         sim = Grampa.grampa(src_cluster_graph, tar_cluster_graph, eta, # ki=ki, lalpha=lalpha
#                             )
#         row, col, _ = lapjv(-sim) # row, col, _ = lapjv(-sim)
#         partition_alignment = list(zip(range(len(col)), col))


#         cur_part_acc = []
#         part_size_diff = {}
#         for (c_s, c_t) in partition_alignment:
#             # print(f'c_s_size: {c_s_size}, c_t_size: {c_t_size}')
#             part_size_diff[(c_s, c_t)] = len(src_nodes_by_cluster[c_s]) - len(tar_nodes_by_cluster[c_t])
#             s_trans_nodes = [np.argwhere(gt[0] == node)[0][0] for node in tar_nodes_by_cluster[c_t]]
#             acc_count = np.array([node in src_nodes_by_cluster[c_s] for node in s_trans_nodes], dtype=int)
#             #print(f'\nCLUSTER ACC (proportion of nodes in target cluster also present in src cluster)\n\n{acc_count.sum()}/{len(acc_count)}={acc_count.mean()}')
#             cur_part_acc.append(acc_count.sum())
#         cur_part_acc = np.array(cur_part_acc)

#         # for each positive entry in part_size_diff borrow from negative entries
#         # for (pp, size) in part_size_diff.items():
#         for i in range(len(part_size_diff)):
#             pp, size = list(part_size_diff.items())[i]
#             if size > 0:
#                 centroid = np.array([tar_centroids[pp[1]]])
#                 # find candidate clusters from which to borrow nodes
#                 candidate_clusters = [k for (k, v) in part_size_diff.items() if v < 0]
#                 # list of indices of tar_nodes that correspond to candidate cluster c_i for all c_i candidate (target) clusters.
#                 cand_idcs_list = [[i for i, v in enumerate(_tar_cluster) if v == j[1]] for j in candidate_clusters]
#                 cand_points = [tar_embedding[idcs] for idcs in cand_idcs_list]
#                 dists = [cdist(pts, centroid) for pts in cand_points]

#                 while size > 0:
#                     best_dist = np.inf
#                     best_idx = (0, 0)
#                     best_c = (None, None)

#                     for i, k in enumerate(candidate_clusters):
#                         dist = dists[i]
#                         min_idx = np.argmin(dist)
#                         d = dist[min_idx]
#                         if d < best_dist and part_size_diff[k] < 0:
#                             best_dist = d
#                             best_idx = (i, min_idx)
#                             best_c = k

#                     # Update loop variables
#                     size -= 1
#                     if best_c != (None, None):
#                         dists[best_idx[0]][best_idx[1]] = np.inf
#                         part_size_diff[best_c] += 1

#                         # Adjust clustering
#                         _tar_cluster[cand_idcs_list[best_idx[0]][best_idx[1]]] = pp[1]
#                         C_UPDATED = True

#         # Only recompute cluster if cluster was changed
#         if C_UPDATED:
#             tar_cluster = dict(zip(tar_nodes, _tar_cluster))
#             tar_nodes_by_cluster = [[k for k,v in tar_cluster.items() if v == i] for i in range(K)]
#             tar_cluster_graph, tar_subgraphs, _ = split_graph_hyper(tar_e, tar_cluster)
#             new_part_acc = []
#             for (c_s, c_t) in partition_alignment:
#                 # print(f'c_s_size: {c_s_size}, c_t_size: {c_t_size}')
#                 part_size_diff[(c_s, c_t)] = len(src_nodes_by_cluster[c_s]) - len(tar_nodes_by_cluster[c_t])
#                 s_trans_nodes = [np.argwhere(gt[0] == node)[0][0] for node in tar_nodes_by_cluster[c_t]]
#                 acc_count = np.array([node in src_nodes_by_cluster[c_s] for node in s_trans_nodes], dtype=int)
#                 #print(f'\nCLUSTER ACC --- AFTER UPDATE\n\n{acc_count.sum()}/{len(acc_count)}={acc_count.mean()}')
#                 new_part_acc.append(acc_count.sum())

#             new_part_acc = np.array(new_part_acc)
#             print(f'\ncluster numbers after: src:{[len(x) for x in src_nodes_by_cluster]}, tar:{[len(x) for x in tar_nodes_by_cluster]}\n')
#             print(f'\n cluster acc before: {cur_part_acc}, after: {new_part_acc}')
#             print(part_size_diff)
#             readjustment_accs.append((new_part_acc-cur_part_acc).sum())
#         # 4. recurse or match
#         for i, (c_s, c_t) in enumerate(partition_alignment):
#             #print(f'Iterating partition alignments -- Current pair = {(c_s, c_t)}')
#             sub_src = src_subgraphs[c_s]
#             sub_tar = tar_subgraphs[c_t]
#             c_s_nodes = np.unique(sub_src)
#             c_t_nodes = np.unique(sub_tar)
#             len_cs = len(c_s_nodes)
#             len_ct = len(c_t_nodes)

#             if len_cs == 0 or len_ct == 0:
#                 #print(f'c_s_nodes and c_t_nodes both empty: {c_s_nodes}, {c_t_nodes} -- BREAKING LOOP (Discard matching)')
#                 break

#             # if cluster size is smaller than sqrt(|V|) then align nodes.
#             #  else cluster recurse.
#             if len(c_s_nodes) <= rsc or len(c_t_nodes) <= rsc:
#                 #print(f'Smaller than rsc: Matching clusters directly\nclusters: {c_s}, {c_t} at pos={pos}')
#                 match_grampa(sub_src, sub_tar)
#                 all_pos.append(pos)
#             else:
#                 pos.append((c_s, c_t))
#                 cluster_recurse(sub_src, sub_tar, pos=pos)

#     cluster_recurse(src_graph, tar_graph)
#     if len(all_pos) == 0:
#         pos_res = {'max_depth': 0, 'avg_depth': 0}
#     else:
#         pos_res = {'max_depth': len(max(all_pos, key=lambda x: len(x))), 'avg_depth': np.array([len(x) for x in all_pos]).mean()}# , pos': all_pos
#     matching = np.c_[np.linspace(0, n-1, n).astype(int), matching].astype(int).T

#     return matching, pos_res, np.array(readjustment_accs)

def split_graph_hyper(graph, clustering, weighting_scheme='ncut'): # dists (for sil-comp))
    '''Split a graph into disjunct clusters and construct weighted graph where a node represents a cluster.
    
    Description:
        - Removes all edges between clusters.
        - Adds a node per cluster to a new graph.
        - Adds edges between new nodes with a weight corresponding to the amount of edges removed between them.

    Parameters:
        graph (np.array): nxn adjacency matrix

        clustering: A clustering for the input graph.
            /format/ Clustering is a dict {node (int): cluster (int)} containing the input graph clustering.
                     Clustering values has to be consecutive, i.e. >>>np.unique(clustering.valuess()) yields [0,1,2,...,k-1]

        weighting_scheme: The weighting scheme to be used. Current implementation allows for:
                          'cut': based on number of connections between clusters.
                          ###NOT IMPLEMENTED###'size': based on size of cluster relative to total number of nodes

    Returns:
        cluster_graph: (kxk np.array) A graph constructed as per above description.
        subgraphs: A list of edgelist corresponding to graphs where each graph is a disjuncted cluster of the input graph.
    '''
    k = np.max(list(clustering.values())) + 1 # Assuming input format is respected
    # new graph of clusters represented by its adjacency matrix
    cut_graph = np.zeros(shape=(k, k), dtype=int)
    cluster_graph = np.zeros(shape=(k, k))
    clusters = [[node for (node, c) in clustering.items() if c==i] for i in range(k)]
  

    # sils = [[] for _ in range(k)]
    # dist_sorts = np.argsort(dists, axis=1)
    # for i in range(len(dists)):
    #     #print(f'dists[i]={dists[i]}')
    #     c = list(clustering.items())[i][1]
    #     a = dists[i][c]
    #     j = 0
    #     while dist_sorts[i][j] == c:
    #         j += 1
    #     b = dists[i][dist_sorts[i][j]]
    #     sils[c].append((b - a) / max(a, b))

    # avg_sils = [(np.array(sil).mean()+1)/2 for sil in sils]
    # subgraphs = [[] for _ in range(k)]
    p_G = nx.from_edgelist(graph)
    Gs = [p_G.subgraph(max(nx.connected_components(p_G.subgraph(c)), key=len)) for c in clusters]
    subgraphs = [np.array(G.edges) for G in Gs]
    cons = [sorted(list(nx.connected_components(G)), key=len) for G in Gs]

    # Build hyper graph
    for (e, f) in list(graph):
        c1 = clustering[e]
        c2 = clustering[f]
        if c1 != c2:
            if weighting_scheme in ['cut', 'rcut', 'ncut']:
                # add edge in cluster graph adjacency matrix
                cut_graph[c1, c2] += 1
                cut_graph[c2, c1] += 1

    matrix_iterator = list(itertools.combinations(range(k), r=2))
                
    if weighting_scheme == 'size':
        cluster_sizes = np.array([len(c) for c in clusters])
        # e_AB = max(|A|, |B|) / |V|
        for i in range(k):
            for j in range(k):
                if i != j:
                    cut_graph[i, j] = cut_graph[j, i] = max(cluster_sizes[i], cluster_sizes[j]) / len(np.unique(graph))

    if weighting_scheme == 'rcut':
        cluster_sizes = np.array([len(c) for c in clusters])
        for (i, j) in matrix_iterator:
            cut = cut_graph[i,j]
            rcut = (cut / cluster_sizes[i]) + (cut / cluster_sizes[j])
            cut_graph[i,j] = cut_graph[j,i] = rcut

    with np.printoptions(linewidth=np.nan, precision=4):
        print(cut_graph)
    if weighting_scheme == 'ncut':
        for (i, j) in matrix_iterator:
            vol_i = len(subgraphs[i]) + sum(cut_graph[i, :])
            vol_j = len(subgraphs[j]) + sum(cut_graph[j, :])
            # print(f'\n\n{vol_i, len(subgraphs[i])}\n{vol_j, len(subgraphs[j])}\n\n')
            cut = cut_graph[i,j]
            ncut = (cut / vol_i) + (cut / vol_j)
            cluster_graph[i,j] = cluster_graph[j,i] = ncut
    with np.printoptions(linewidth=np.nan, precision=4):
        print(f'new cg:\n{cluster_graph}')

    return cluster_graph, subgraphs, # avg_sils # sils # cons # isolated

def alhpa(src_graph, tar_graph, rsc=0, weighting_scheme='ncut', lap=False, gt=None, ki=False, lalpha=10000):
    """
    MAtching by Recursive Partition Alignment
    ____
    Summary or Description of the Function

    Parameters:
      src_graph (np.array): Edge list of source graph.
      tar_graph (np.array): Edge list of target graph.

    Returns:
      matching (np.array): Array of 2-tuples each representing a matching pair nodes (n1, n2) where n1 is in src_graph and n2 is in tar_graph.
    """

    n = len(np.unique(src_graph))
    if rsc == 0: rsc = .1*n
    eta = 0.2

    matching = -1 * np.ones(shape=(n, ), dtype=int)
    readjustment_accs = []
    all_pos = []
    def match_grampa(src, tar):
        if isinstance(src, tuple) and isinstance(tar, tuple):
            src_adj, src_map = src
            tar_adj, tar_map = tar
        else:
            src_adj, src_map = adj_from_edges(src)
            tar_adj, tar_map = adj_from_edges(tar)
        diff = len(src_map) - len(tar_map)
        #print(diff)
        if diff < 0: # expand sub src
            #print(f'size before expansion: {src_adj.shape}')
            src_adj = expand_matrix(src_adj, abs(diff))
            #print(f'size after expansion: {src_adj.shape}')
            #print(f'src_map size before: {len(src_map)}')
            src_map = list(src_map) + [-1] * abs(diff)
            #print(f'src_map size after: {len(src_map)}')
        if diff > 0: # expand sub tar
            #print(f'size before expansion: {tar_adj.shape}')
            tar_adj = expand_matrix(tar_adj, diff)
            #print(f'size after expansion: {tar_adj.shape}')
            #print(f'tar_map size before: {len(tar_map)}')
            tar_map = list(tar_map) + [-1] * diff
            #print(f'tar_map size after: {len(tar_map)}')

        sub_sim = Grampa.grampa(src_adj, tar_adj, eta, lap=lap)
        #print(f'adjacency matrices for src and tar (shapes): {src_adj.shape}, {tar_adj.shape})')
        #print(sub_sim)
        r, c, _ = lapjv(-sub_sim)
        match = list(zip(range(len(c)), c))
        #print(f'GRAMPA:{match}')
        # translate with map and add to solution
        for (n1, n2) in match:
            matching[src_map[n1]] = tar_map[n2]

    def cluster_recurse(src_e, tar_e, pos=[(0,0)]):
        pos = pos.copy()
        src_adj, src_nodes = adj_from_edges(src_e)
        tar_adj, tar_nodes = adj_from_edges(tar_e)
        
        #### 1. Spectrally embed graphs into 1 dimension.
        # l, U, snz = decompose_spectral(src_adj)
        # mu, V, tnz = decompose_spectral(tar_adj)

        # src_embedding = U[:, snz[:e_dim]]
        # tar_embedding = V[:, tnz[:e_dim]]
        
        # l, src_embedding = spectral_embedding(src_adj, n_components=src_adj.shape[0])
        # mu, tar_embedding = spectral_embedding(tar_adj, n_components=tar_adj.shape[0])
        print('embedding src graph...')
        l, src_embedding = spectral_embedding(src_adj, n_components=10)
        print('embedding tar graph...')
        mu, tar_embedding = spectral_embedding(tar_adj, n_components=10)

        # Find the number of clusters based on eigengap heuristics
        # diffs = np.array([abs(l[i-1] - l[i]) for i in range(1, len(l))])
        # diffs = np.abs(np.diff(l))
        # diffs[0] = diffs[1] = diffs[2] = diffs[3] = 0
        # K = diffs.argmax()
        # b = K.copy()
        # for cand in np.argsort(diffs)[::-1][1:]:
        #     if cand < K and cand > 3:
        #         if abs(diffs[cand]-diffs[b]) < diffs[b]*.9:
        #             K = cand

        diffs = np.abs(np.diff(l))
        diffs_ = np.abs(np.diff(mu))
        diffs[0] = diffs[1] = diffs_[0] = diffs_[1] = 0
        # diffs[0] = diffs[1] = 0
        K = diffs.argmax() + 1
        K_ = diffs_.argmax() + 1
        print(f'diffs:\n{diffs}\n{diffs_}\n')
        print(f'K:{K}, K_: {K_}')
        # b = K.copy()
        # for cand in np.argsort(diffs)[::-1][1:]:
        #     if cand < K and cand > 2:
        #         if abs(diffs[cand]-diffs[b]) < diffs[b]*.05:
        #             K = cand
        #             break                    
        # K = K + 1
        K = max(K, K_, 3)
        # # inc = 1/len(l)
        # # weights = np.linspace(1, len(diffs), num=len(diffs))
        # # wdiffs = diffs
        # i = 2
        # # while (diffs[i] * weights[i]) < diffs.mean():
        # while diffs[i] < diffs.mean():
        #     i += 1
        # K = i + 1
        # print(f'\nFound K={K} at position={pos}\ndiffs: {diffs}\nmean: {diffs.mean()}')
        print(f'\nFound K={K} at position={pos}')
        src_embedding = src_embedding.T[:K].T
        tar_embedding = tar_embedding.T[:K].T
        # src_embedding2 = src_embedding.T[:2].T
        # tar_embedding2 = tar_embedding.T[:2].T        

        # Compute clusters on embedded data with kmeans and lloyd's algorithm
        warnings.simplefilter('error', category=ConvergenceWarning)
        try:
            print('computing k-means (src graph)')
            kmeans = KMeans(n_clusters=K, init='random', n_init=10).fit(src_embedding)
            src_centroids = kmeans.cluster_centers_
            # print(src_centroids)
            _src_cluster = kmeans.labels_
            # print(_src_cluster)
            src_dists = kmeans.transform(src_embedding)
            # print('2nd kmean...')
            # kmeans2 = KMeans(n_clusters=2, init='random', n_init=10).fit(src_embedding2)
            # src_cents2 = kmeans2.cluster_centers_
            # _src_cluster2 = kmeans2.labels_
            # print(src_dists.shape)
            # print(src_dists)
            # print(f' doing the thing: {np.all(_src_cluster == np.argsort(src_dists, axis=1)[:,0])}')
            # src_centroids, _src_cluster, _ = k_means(src_embedding, n_clusters=K, n_init=10)
        except Exception as e:
            traceback.print_exc()
            print(repr(e))
            print(f'Troublesome graph!!!\n Matching parent graph using GRAMPA')
            # match_grampa((src_adj, src_nodes), (tar_adj, tar_nodes), decomp=[(l, U), (mu, V)])
            match_grampa((src_adj, src_nodes), (tar_adj, tar_nodes))
            return
        try:
            print('computing k-means (tar graph)')
            # Seed target graph kmeans by using the src centroids.
            kmeans = KMeans(n_clusters=K, init=src_centroids, n_init=1).fit(tar_embedding)
            # kmeans = KMeans(n_clusters=K, init='random', n_init=10).fit(tar_embedding)
            tar_centroids = kmeans.cluster_centers_
            _tar_cluster = kmeans.labels_
            tar_dists = kmeans.transform(tar_embedding)
            # print('2nd kmean...')
            # kmeans2 = KMeans(n_clusters=2, init=src_cents2, n_init=1).fit(tar_embedding2)
            # _tar_cluster2 = kmeans2.labels_
            # print(f' doing the thing: {np.all(_tar_cluster == np.argsort(tar_dists, axis=1)[:,0])}')
            # tar_centroids, _tar_cluster, _ = k_means(tar_embedding, n_clusters=K, init=src_centroids, n_init=1)
            # tar_centroids, _tar_cluster, _ = k_means(tar_embedding, n_clusters=K, n_init=10)            
        except Exception as e:
            traceback.print_exc()
            print(repr(e))
            print(f'Troublesome graph!!!\n Matching parent graph using GRAMPA')
            # match_grampa((src_adj, src_nodes), (tar_adj, tar_nodes), decomp=[(l, U), (mu, V)])
            match_grampa((src_adj, src_nodes), (tar_adj, tar_nodes))
            return
        # k2src1 = len(_src_cluster2.nonzero()[0])
        # k2src0 = len(_src_cluster2) - k2src1
        # k2tar1 = len(_tar_cluster2.nonzero()[0])
        # k2tar0 = len(_tar_cluster2) - k2tar1
        # print(f'K=2 clusters numbers --- src: {(k2src0, k2src1)}, tar: {(k2tar0, k2tar1)}')
        #print(f'src_centroids:\n{src_centroids}')
        #print(f'tar_centroids:\n{tar_centroids}')

        src_cluster = dict(zip(src_nodes, _src_cluster))
        tar_cluster = dict(zip(tar_nodes, _tar_cluster))
        src_nodes_by_cluster = [[k for k,v in src_cluster.items() if v == i] for i in range(K)]
        src_cluster_sizes = [len(x) for x in src_nodes_by_cluster]
        tar_nodes_by_cluster = [[k for k,v in tar_cluster.items() if v == i] for i in range(K)]
        tar_cluster_sizes = [len(x) for x in tar_nodes_by_cluster]
        print(f'\ncluster numbers: src:{src_cluster_sizes}, tar:{tar_cluster_sizes}\n')

        # 2. split graphs (src, tar) according to cluster.
        # print('Splitting SRC')
        # src_cluster_graph, src_subgraphs, src_cons = split_graph_hyper(src_e, src_cluster, src_dists, weighting_scheme=weighting_scheme)
        # src_cluster_graph2, src_subgraphs2, src_cons2 = split_graph_hyper(src_e, src_cluster, src_dists, weighting_scheme='silcut')
        # src_cluster_graph, src_subgraphs, src_sils = split_graph_hyper(src_e, src_cluster, weighting_scheme=weighting_scheme)
        src_cluster_graph, src_subgraphs = split_graph_hyper(src_e, src_cluster, weighting_scheme=weighting_scheme)
        # print('Splitting TAR')
        # tar_cluster_graph, tar_subgraphs, tar_cons = split_graph_hyper(tar_e, tar_cluster, tar_dists, weighting_scheme=weighting_scheme)
        # tar_cluster_graph2, tar_subgraphs2, tar_cons2 = split_graph_hyper(tar_e, tar_cluster, tar_dists, weighting_scheme='silcut')
        tar_cluster_graph, tar_subgraphs = split_graph_hyper(tar_e, tar_cluster, weighting_scheme=weighting_scheme)
        #print(f'src_cons: {[len(x) for x in src_cons]}\ntar_cons: {[len(x) for x in tar_cons]}')

        C_UPDATED = False

        # 3. match cluster_graph.
        print(f'\ncluster graphs (src, tar)\n{src_cluster_graph}\n{tar_cluster_graph}')
        try:
            sim = Grampa.grampa(src_cluster_graph, tar_cluster_graph, eta, lap=True) # init=(src_sils, tar_sils ki=ki, lalpha=lalpha)
            # W = np.zeros((K,K)) # weights
            # for i in range(K):
            #     for j in range(K):
            #         if j >= i: # Construct weights based cluster cardinality, such that different sizes decreases similarity.
            #             W[i,j] = W[j,i] = min(src_cluster_sizes[i], tar_cluster_sizes[j]) / max(src_cluster_sizes[i], tar_cluster_sizes[j])
            # W = 1 - W
            # sim = sim / W
        except Exception as e:
            traceback.print_exc()
            print(repr(e))
            print('\n\n!!!!!!!!!!!HYPER ERROR!!!!!!!!!!!\n\n')

            # match_grampa((src_adj, src_nodes), (tar_adj, tar_nodes))
            return

        row, col, _ = lapjv(-sim) # row, col, _ = lapjv(-sim)
        partition_alignment = list(zip(range(len(col)), col))
        cur_part_acc = []
        part_size_diff = {}
        print(sim)
        for (c_s, c_t) in partition_alignment:
            # print(f'c_s_size: {c_s_size}, c_t_size: {c_t_size}')
            part_size_diff[(c_s, c_t)] = len(src_nodes_by_cluster[c_s]) - len(tar_nodes_by_cluster[c_t])
            s_trans_nodes = [np.argwhere(gt[0] == node)[0][0] for node in tar_nodes_by_cluster[c_t]]
            acc_count = np.array([node in src_nodes_by_cluster[c_s] for node in s_trans_nodes], dtype=int)
            #print(f'\nCLUSTER ACC (proportion of nodes in target cluster also present in src cluster)\n\n{acc_count.sum()}/{len(acc_count)}={acc_count.mean()}')
            cur_part_acc.append(acc_count.sum())
        cur_part_acc = np.array(cur_part_acc)

        # for each positive entry in part_size_diff borrow from negative entries
        # for (pp, size) in part_size_diff.items():
        for i in range(len(part_size_diff)):
            pp, size = list(part_size_diff.items())[i]
            if size > 0:
                # find candidate clusters from which to borrow nodes
                candidate_clusters = [k for (k, v) in part_size_diff.items() if v < 0]
                # list of indices of tar_nodes that correspond to candidate cluster c_i for all c_i candidate (target) clusters.
                cand_idcs_list = [[i for i, v in enumerate(_tar_cluster) if v == j[1]] for j in candidate_clusters]
                # cand_idcs_list = np.array([i for arr in cand_idcs_list for i in arr])
                # print(tar_dists.shape)
                # print(cand_idcs_list.shape)
                # cand_points = [tar_embedding[idcs] for idcs in cand_idcs_list]
                # dists = [cdist(pts, centroid) for pts in cand_points]
                dists = [tar_dists[idcs_list][:, pp[1]] for idcs_list in cand_idcs_list]

                while size > 0:
                    best_dist = np.inf
                    best_idx = (0, 0)
                    best_c = (None, None)

                    for i, k in enumerate(candidate_clusters):
                        dist = dists[i]
                        min_idx = np.argmin(dist)
                        d = dist[min_idx]
                        if d < best_dist and part_size_diff[k] < 0:
                            best_dist = d
                            best_idx = (i, min_idx)
                            best_c = k

                    # Update loop variables
                    size -= 1
                    if best_c != (None, None):
                        dists[best_idx[0]][best_idx[1]] = np.inf
                        part_size_diff[best_c] += 1

                        # Adjust clustering
                        _tar_cluster[cand_idcs_list[best_idx[0]][best_idx[1]]] = pp[1]
                        C_UPDATED = True

        print(f'\nPartition Alignment: {partition_alignment}\n')
        # Only recompute cluster if cluster was changed
        if C_UPDATED:
            tar_cluster = dict(zip(tar_nodes, _tar_cluster))
            tar_nodes_by_cluster = [[k for k,v in tar_cluster.items() if v == i] for i in range(K)]
            tar_cluster_graph, tar_subgraphs = split_graph_hyper(tar_e, tar_cluster)
            new_part_acc = []
            for (c_s, c_t) in partition_alignment:
                # print(f'c_s_size: {c_s_size}, c_t_size: {c_t_size}')
                part_size_diff[(c_s, c_t)] = len(src_nodes_by_cluster[c_s]) - len(tar_nodes_by_cluster[c_t])
                s_trans_nodes = [np.argwhere(gt[0] == node)[0][0] for node in tar_nodes_by_cluster[c_t]]
                acc_count = np.array([node in src_nodes_by_cluster[c_s] for node in s_trans_nodes], dtype=int)
                #print(f'\nCLUSTER ACC --- AFTER UPDATE\n\n{acc_count.sum()}/{len(acc_count)}={acc_count.mean()}')
                new_part_acc.append(acc_count.sum())

            new_part_acc = np.array(new_part_acc)
            print(f'\ncluster numbers after: src:{src_cluster_sizes}, tar:{tar_cluster_sizes}\n')
            print(f'\n cluster acc before: {cur_part_acc}, after: {new_part_acc}')
            print(part_size_diff)
            readjustment_accs.append((new_part_acc-cur_part_acc).sum())
        # 4. recurse or match
        for i, (c_s, c_t) in enumerate(partition_alignment):
            #print(f'Iterating partition alignments -- Current pair = {(c_s, c_t)}')
            sub_src = src_subgraphs[c_s]
            sub_tar = tar_subgraphs[c_t]
            c_s_nodes = np.unique(sub_src)
            c_t_nodes = np.unique(sub_tar)
            len_cs = len(c_s_nodes)
            len_ct = len(c_t_nodes)

            if len_cs == 0 or len_ct == 0:
                #print(f'c_s_nodes and c_t_nodes both empty: {c_s_nodes}, {c_t_nodes} -- BREAKING LOOP (Discard matching)')
                break

            # if cluster size is smaller than sqrt(|V|) then align nodes.
            #  else cluster recurse.
            if len(c_s_nodes) <= rsc or len(c_t_nodes) <= rsc:
                #print(f'Smaller than rsc: Matching clusters directly\nclusters: {c_s}, {c_t} at pos={pos}')
                match_grampa(sub_src, sub_tar)
                all_pos.append(pos)
            else:
                pos.append((c_s, c_t))
                cluster_recurse(sub_src, sub_tar, pos=pos)

    cluster_recurse(src_graph, tar_graph)
    if len(all_pos) == 0:
        pos_res = {'max_depth': 0, 'avg_depth': 0}
    else:
        pos_res = {'max_depth': len(max(all_pos, key=lambda x: len(x))), 'avg_depth': np.array([len(x) for x in all_pos]).mean()}# , pos': all_pos
    matching = np.c_[np.linspace(0, n-1, n).astype(int), matching].astype(int).T

    return matching, pos_res, np.array(readjustment_accs)

def main(data, eta, rsc, lap, ki, lalpha):
    print(f'\n\n\nstarting run:\nargs: (, eta, rsc, lap, ki, lalpha)={(eta, rsc, lap, ki, lalpha)}')
    Src = data['src_e']
    Tar = data['tar_e']
    gt = data["gt"]

    n = Src.shape[0]
    s = time.time()
    matching, pos, readj_accs = alhpa(Src, Tar, rsc=rsc, weighting_scheme='ncut', lap=lap, gt=gt, ki=ki, lalpha=lalpha)
    print(f'Produced matching in time: {time.time()-s}')
    print(f'readjustment accuracis: {readj_accs.mean()} (avg.)\n{readj_accs}')

    ma, mb = matching
    gmb, gmb1 = gt
    gmab = np.arange(gmb.size)
    gmab[ma] = mb
    gacc = np.mean(gmb == gmab)

    mab = gmb[ma]
    acc = np.mean(mb == mab)
    print(f'\nAccuracy: {acc}\n')

    for k, v in pos.items():
        print(f'{k}:\n{v}')
    return matching
