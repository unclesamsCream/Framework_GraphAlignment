from algorithms import Grampa
from Utils import adj_from_edges, compute_laplacian, embed_spectral, expand_matrix
import numpy as np
from numpy.linalg import inv
from numpy.linalg import eigh,eig
import networkx as nx 
import warnings
import itertools
from sklearn.manifold import spectral_embedding
from lapjv import lapjv
from sklearn.cluster import k_means
from scipy.spatial.distance import cdist
from sklearn.exceptions import ConvergenceWarning

def split_graph_hyper(graph, clustering, weighting_scheme='ncut', rcon=True): # POTENTIAL: add mem_eff=False parameter
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
    cluster_graph = np.zeros(shape=(k, k))
    clusters = [[node for (node, c) in clustering.items() if c==i] for i in range(k)]
    cluster_sizes = np.array([len(c) for c in clusters])

    # subgraphs = [[] for _ in range(k)]
    p_G = nx.from_edgelist(graph)
    Gs = [p_G.subgraph(c).copy() for c in clusters]
    subgraphs = [np.array(G.edges) for G in Gs]
    cons = [list(nx.connected_components(G)) for G in Gs]
    
    # Build hyper graph
    for (e, f) in list(graph):
        c1 = clustering[e]
        c2 = clustering[f]
        if c1 != c2:
            if weighting_scheme == 'cut' or weighting_scheme == 'ncut':
                # add edge in cluster graph adjacency matrix
                cluster_graph[c1, c2] += 1
                cluster_graph[c2, c1] += 1

    if weighting_scheme == 'size':
        # e_AB = max(|A|, |B|) / |V|
        for i in range(k):
            for j in range(k):
                if i != j:
                    cluster_graph[i, j] = cluster_graph[j, i] = max(cluster_sizes[i], cluster_sizes[j]) / len(np.unique(graph))
    if weighting_scheme == 'ncut':
        matrix_iterator = list(itertools.combinations(range(k), r=2))
        for (i, j) in matrix_iterator:
            cut = cluster_graph[i,j]
            ncut = (cut / cluster_sizes[i]) + (cut / cluster_sizes[j])
            cluster_graph[i,j] = cluster_graph[j,i] = ncut

    return cluster_graph, subgraphs, cons # isolated


def marpa(src_graph, tar_graph, K=3, rsc=0, weighting_scheme='ncut', lap=False, e_dim=1):
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
    if rsc == 0: rsc = np.sqrt(n)

    eta = 0.2
    
    matching = -1 * np.ones(shape=(n, ), dtype=int)
    all_pos = []
    def match_grampa(src, tar):
        if isinstance(src, tuple) and isinstance(tar, tuple):
            src_adj, src_map = src
            tar_adj, tar_map = tar
        else:
            src_adj, src_map = adj_from_edges(src)
            tar_adj, tar_map = adj_from_edges(tar)
        diff = len(src_map) - len(tar_map)
        if diff < 0: # expand sub src
            src_adj = expand_matrix(src_adj, abs(diff))
            src_map = list(src_map) + [-1] * abs(diff)
        if diff > 0: # expand sub tar
            tar_adj = expand_matrix(tar_adj, diff)
            tar_map = list(tar_map) + [-1] * diff

        sub_sim = Grampa.grampa(src_adj, tar_adj, eta, lap=lap)
        r, c, _ = lapjv(-sub_sim)
        match = list(zip(range(len(c)), c))
        # translate with map and add to solution
        for (n1, n2) in match:
            matching[src_map[n1]] = tar_map[n2]

    def cluster_recurse(src_e, tar_e, pos=[(0,0)]):
        pos = pos.copy()
        src_adj, src_nodes = adj_from_edges(src_e)
        tar_adj, tar_nodes = adj_from_edges(tar_e)

        #### 1. Spectrally embed graphs into 1 dimension.
        warnings.simplefilter('error', category=UserWarning)
        try:
            src_embedding = spectral_embedding(src_adj, n_components=e_dim)            
        except Exception as e:
            match_grampa((src_adj, src_nodes), (tar_adj, tar_nodes))
            return
        try:
            tar_embedding = spectral_embedding(tar_adj, n_components=e_dim)
        except Exception as e:
            match_grampa((src_adj, src_nodes), (tar_adj, tar_nodes))
            return
            
        # Compute clusters on embedded data with kmeans and lloyd's algorithm
        src_centroids, _src_cluster, _ = k_means(src_embedding, n_clusters=K, n_init=10)
        # Seed target graph kmeans by using the src centroids.
        # tar_centroids, _tar_cluster, _ = k_means(tar_embedding, n_clusters=K, init=src_centroids, n_init=1, random_state=SEED)
        tar_centroids, _tar_cluster, _ = k_means(tar_embedding, n_clusters=K, n_init=10)

        src_cluster = dict(zip(src_nodes, _src_cluster))
        tar_cluster = dict(zip(tar_nodes, _tar_cluster))

        # 2. split graphs (src, tar) according to cluster.
        src_cluster_graph, src_subgraphs, src_cons = split_graph_hyper(src_e, src_cluster, weighting_scheme=weighting_scheme)
        tar_cluster_graph, tar_subgraphs, tar_cons = split_graph_hyper(tar_e, tar_cluster, weighting_scheme=weighting_scheme)

        C_UPDATED = False

        # 3. match cluster_graph.
        sim = Grampa.grampa(src_cluster_graph, tar_cluster_graph, eta)
        row, col, _ = lapjv(-sim) # row, col, _ = lapjv(-sim)
        partition_alignment = list(zip(range(len(row)), row))

        # 4. Refine clusters
        # Find cluster sizes
        rc_size_diff = {}
        cr_size_diff = {}
        for (r, c) in partition_alignment:
            src_r = len([x for x in _src_cluster if x == r])
            src_c = len([x for x in _src_cluster if x == c])
            tar_r = len([x for x in _tar_cluster if x == r])
            tar_c = len([x for x in _tar_cluster if x == c])
            rc_size_diff[(r, c)] = src_r - tar_c
            cr_size_diff[(c, r)] = src_c - tar_r

        rc_sum = abs(np.array(list(rc_size_diff.values()))).sum()
        cr_sum = abs(np.array(list(cr_size_diff.values()))).sum()
        if rc_sum < cr_sum:
            part_size_diff = rc_size_diff
        else:
            part_size_diff = cr_size_diff
            partition_alignment = [(c, r) for (r,c) in partition_alignment]

        # for each positive entry in part_size_diff borrow from negative entries
        # for (pp, size) in part_size_diff.items():
        for i in range(len(part_size_diff)):
            pp, size = list(part_size_diff.items())[i]
            if size > 0:
                centroid = np.array([tar_centroids[pp[1]]])
                # find candidate clusters from which to borrow nodes
                candidate_clusters = [k for (k, v) in part_size_diff.items() if v < 0]
                # list of indices of tar_nodes that correspond to candidate cluster c_i for all c_i candidate (target) clusters.
                cand_idcs_list = [[i for i, v in enumerate(_tar_cluster) if v == j[1]] for j in candidate_clusters]
                cand_points = [tar_embedding[idcs] for idcs in cand_idcs_list]
                dists = [cdist(pts, centroid) for pts in cand_points]

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

        # Only recompute cluster if cluster was changed
        if C_UPDATED:
            tar_cluster = dict(zip(tar_nodes, _tar_cluster))
            tar_cluster_graph, tar_subgraphs, _ = split_graph_hyper(tar_e, tar_cluster)        
            new_part_size_diff = {}
            for (c_s, c_t) in partition_alignment:
                c_s_size = len([x for x in _src_cluster if x == c_s])
                c_t_size = len([x for x in _tar_cluster if x == c_t])
                new_part_size_diff[(c_s, c_t)] = c_s_size - c_t_size

        # 4. recurse or match
        for i, (c_s, c_t) in enumerate(partition_alignment):
            sub_src = src_subgraphs[c_s]
            sub_tar = tar_subgraphs[c_t]
            c_s_nodes = np.unique(sub_src)
            c_t_nodes = np.unique(sub_tar)
            len_cs = len(c_s_nodes)
            len_ct = len(c_t_nodes)

            if len_cs == 0 or len_ct == 0:
                break

            # if cluster size is smaller than sqrt(|V|) then align nodes.
            #  else cluster recurse.
            if len(c_s_nodes) <= rsc or len(c_t_nodes) <= rsc:
                match_grampa(sub_src, sub_tar)
                all_pos.append(pos)
            else:
                pos.append((c_s, c_t))
                cluster_recurse(sub_src, sub_tar, pos=pos)

    cluster_recurse(src_graph, tar_graph)
    if len(all_pos) == 0:
        pos_res = {'max_depth': 0, 'avg_depth': 0}
    else:
        pos_res = {'max_depth': len(max(all_pos, key=lambda x: len(x))), 'avg_depth': np.array([len(x) for x in all_pos]).mean()}
    matching = np.c_[np.linspace(0, n-1, n).astype(int), matching].T

    return matching, pos_res

def main(data, eta, k, rsc, lap, edim):
  Src = data['src_e']
  Tar = data['tar_e']
  n = Src.shape[0]
  # src = (Src, np.linspace(0, n-1, n).astype(int))
  # tar = (Tar, np.linspace(0, n-1, n).astype(int))
  # matching, pos = marpa(src, tar, K=k, rsc=rsc, weighting_scheme='ncut')
  matching, pos = marpa(Src, Tar, K=k, rsc=rsc, weighting_scheme='ncut', lap=lap, e_dim=edim)
  print(pos)
  return matching

