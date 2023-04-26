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
from sklearn.cluster import k_means
from scipy.spatial.distance import cdist
from sklearn.exceptions import ConvergenceWarning
import time
from threading import Lock

def printLog(*args, **kwargs):
    print(*args, **kwargs)
    # with open('/home/kb/Documents/Projects/Framework_GraphAlignment/logfiles/run_log.txt','a') as file:
    #     print(*args, **kwargs, file=file)

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
            # cluster_graph[i,j] = cluster_graph[j,i] = ncut + (len(cons[i]) / len(cons[j]))

    return cluster_graph, subgraphs, cons # isolated

def alhpa(src_graph, tar_graph, rsc=0, weighting_scheme='ncut', lap=False):
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

    mlock = Lock()
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
        #printLog(diff)
        if diff < 0: # expand sub src
            #printLog(f'size before expansion: {src_adj.shape}')
            src_adj = expand_matrix(src_adj, abs(diff))
            #printLog(f'size after expansion: {src_adj.shape}')
            #printLog(f'src_map size before: {len(src_map)}')
            src_map = list(src_map) + [-1] * abs(diff)
            #printLog(f'src_map size after: {len(src_map)}')
        if diff > 0: # expand sub tar
            #printLog(f'size before expansion: {tar_adj.shape}')
            tar_adj = expand_matrix(tar_adj, diff)
            #printLog(f'size after expansion: {tar_adj.shape}')
            #printLog(f'tar_map size before: {len(tar_map)}')
            tar_map = list(tar_map) + [-1] * diff
            #printLog(f'tar_map size after: {len(tar_map)}')

        sub_sim = Grampa.grampa(src_adj, tar_adj, eta, lap=lap)
        #printLog(f'adjacency matrices for src and tar (shapes): {src_adj.shape}, {tar_adj.shape})')
        #printLog(sub_sim)
        r, c, _ = lapjv(-sub_sim)
        match = list(zip(range(len(c)), c))
        #printLog(f'GRAMPA:{match}')
        # translate with map and add to solution
        with mlock:
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

        l, src_embedding = spectral_embedding(src_adj, n_components=25)
        mu, tar_embedding = spectral_embedding(tar_adj, n_components=25)

        # Find the number of clusters based on eigengap heuristics
        diffs = np.array([abs(l[i-1] - l[i]) for i in range(1, len(l))])
        # i = 2
        # while diffs[i] < diffs.mean():
        #     i += 1
        diffs[0] = diffs[1] = 0
        K = np.argmax(diffs)
        printLog(f'\nFound K={K} at position={pos}\ndiffs: {diffs}\nmean: {diffs.mean()}')
        src_embedding = src_embedding.T[:K].T
        tar_embedding = tar_embedding.T[:K].T

        # Compute clusters on embedded data with kmeans and lloyd's algorithm
        warnings.simplefilter('error', category=ConvergenceWarning)
        try:
            #printLog('computing k-means (src graph)')
            src_centroids, _src_cluster, _ = k_means(src_embedding, n_clusters=K, n_init=10)
        except Exception as e:
            #printLog(f'Troublesome graph!!!\n Matching parent graph using GRAMPA')
            # match_grampa((src_adj, src_nodes), (tar_adj, tar_nodes), decomp=[(l, U), (mu, V)])
            match_grampa((src_adj, src_nodes), (tar_adj, tar_nodes))
            return
        try:
            #printLog('computing k-means (tar graph)')
            # Seed target graph kmeans by using the src centroids.
            # tar_centroids, _tar_cluster, _ = k_means(tar_embedding, n_clusters=K, init=src_centroids, n_init=1)
            tar_centroids, _tar_cluster, _ = k_means(tar_embedding, n_clusters=K, n_init=10)            
        except Exception as e:
            #printLog(f'Troublesome graph!!!\n Matching parent graph using GRAMPA')
            # match_grampa((src_adj, src_nodes), (tar_adj, tar_nodes), decomp=[(l, U), (mu, V)])
            match_grampa((src_adj, src_nodes), (tar_adj, tar_nodes))
            return

        #printLog(f'src_centroids:\n{src_centroids}')
        #printLog(f'tar_centroids:\n{tar_centroids}')

        src_cluster = dict(zip(src_nodes, _src_cluster))
        tar_cluster = dict(zip(tar_nodes, _tar_cluster))

        cll = [[k for k,v in src_cluster.items() if v == i] for i in range(K)]
        clll = [[k for k,v in tar_cluster.items() if v == i] for i in range(K)]
        #printLog(f'cluster numbers: src:{[len(x) for x in cll]}, tar:{[len(x) for x in clll]}')

        # 2. split graphs (src, tar) according to cluster.
        src_cluster_graph, src_subgraphs, src_cons = split_graph_hyper(src_e, src_cluster, weighting_scheme=weighting_scheme)
        tar_cluster_graph, tar_subgraphs, tar_cons = split_graph_hyper(tar_e, tar_cluster, weighting_scheme=weighting_scheme)
        #printLog(f'src_cons: {[len(x) for x in src_cons]}\ntar_cons: {[len(x) for x in tar_cons]}')

        C_UPDATED = False

        # 3. match cluster_graph.
        #printLog(f'\ncluster graphs (src, tar)\n{src_cluster_graph}\n{tar_cluster_graph}')

        sim = Grampa.grampa(src_cluster_graph, tar_cluster_graph, eta)
        row, col, _ = lapjv(-sim) # row, col, _ = lapjv(-sim)
        partition_alignment = list(zip(range(len(row)), row))
        #printLog(f'cluster sim given by grampa:\n{sim}\n')
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

            #printLog(f'option1: src_r({r})_size: {src_r}, tar_c({c})_size: {tar_c}')
            #printLog(f'option2: src_c({c})_size: {src_c}, tar_r({r})_size: {tar_r}')

        rc_sum = abs(np.array(list(rc_size_diff.values()))).sum()
        cr_sum = abs(np.array(list(cr_size_diff.values()))).sum()
        if rc_sum < cr_sum:
            #printLog(f'min sum is {rc_sum}. Choosing rc_size_diff: {rc_size_diff}. (cr diff: {cr_size_diff})')
            part_size_diff = rc_size_diff
        else:
            #printLog(f'min sum is {cr_sum}. Choosing cr_size_diff: {cr_size_diff}. (rc diff: {rc_size_diff})')
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
            #printLog(f'new_part_size_diff:\n{new_part_size_diff}')

        # 4. recurse or match
        for i, (c_s, c_t) in enumerate(partition_alignment):
            #printLog(f'Iterating partition alignments -- Current pair = {(c_s, c_t)}')
            sub_src = src_subgraphs[c_s]
            sub_tar = tar_subgraphs[c_t]
            c_s_nodes = np.unique(sub_src)
            c_t_nodes = np.unique(sub_tar)
            len_cs = len(c_s_nodes)
            len_ct = len(c_t_nodes)

            if len_cs == 0 or len_ct == 0:
                #printLog(f'c_s_nodes and c_t_nodes both empty: {c_s_nodes}, {c_t_nodes} -- BREAKING LOOP (Discard matching)')
                break

            # if cluster size is smaller than sqrt(|V|) then align nodes.
            #  else cluster recurse.
            if len(c_s_nodes) <= rsc or len(c_t_nodes) <= rsc:
                #printLog(f'Smaller than rsc: Matching clusters directly\nclusters: {c_s}, {c_t} at pos={pos}')
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
    matching = np.c_[np.linspace(0, n-1, n).astype(int), matching].astype(int).T

    return matching, pos_res

def main(data, eta, rsc, lap):
    printLog(f'\n\n\nstarting run:\nargs: (eta, rsc, lap)={eta, rsc, lap}')
    Src = data['src_e']
    Tar = data['tar_e']
    n = Src.shape[0]
    # src = (Src, np.linspace(0, n-1, n).astype(int))
    # tar = (Tar, np.linspace(0, n-1, n).astype(int))
    # matching, pos = marpa(src, tar, K=k, rsc=rsc, weighting_scheme='ncut')
    s = time.time()
    matching, pos = alhpa(Src, Tar, rsc=rsc, weighting_scheme='ncut', lap=lap)
    printLog(f'Produced matching in time: {time.time()-s}')
    printLog(pos)
    return matching
