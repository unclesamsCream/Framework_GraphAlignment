from algorithms import Grampa, base_align
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

def split_graph(graph, clustering):
    '''Split a graph into disjunct clusters and construct weighted graph where a node represents a cluster.
    Parameters:
        graph (np.array): nxn adjacency matrix
        clustering: A clustering for the input graph.
            /format/ Clustering is a dict {node (int): cluster (int)} containing the input graph clustering.
                     Clustering values has to be consecutive, i.e. >>>np.unique(clustering.values()) yields [0,1,2,...,k-1]
    Returns:
        subgraphs: A list of edgelist corresponding to graphs where each graph is a disjuncted cluster of the input graph.
    '''
    k = np.max(list(clustering.values())) + 1 # Assuming input format is respected
    # new graph of clusters represented by its adjacency matrix
    clusters = [[node for (node, c) in clustering.items() if c==i] for i in range(k)]
    p_G = nx.from_edgelist(graph)
    Gs = [p_G.subgraph(max(nx.connected_components(p_G.subgraph(c)), key=len)) for c in clusters]
    subgraphs = [np.array(G.edges) for G in Gs]
    return subgraphs

def alhpa(src_graph, tar_graph, rsc=0, n_comp=10, gt=None):
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
    if rsc > 0 and rsc <= 1: rsc = int(n*rsc)
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

        sub_sim = Grampa.grampa(src_adj, tar_adj, eta, lap=True)
        #print(f'adjacency matrices for src and tar (shapes): {src_adj.shape}, {tar_adj.shape})')
        #print(sub_sim)
        r, c, _ = lapjv(-sub_sim)
        match = list(zip(range(len(c)), c))
        #print(f'GRAMPA:{match}')
        # translate with map and add to solution
        for (n1, n2) in match:
            si = src_map[n1]
            if si < n:
                matching[src_map[n1]] = tar_map[n2]
            else:
                print('!!!src node outside matching idx!!!')

    def cluster_recurse(src_e, tar_e, pos=[(0,0)]):
        pos = pos.copy()
        G_s = nx.from_edgelist(src_e)
        con_gs = list(nx.connected_components(G_s))
        # print(list(con_gs))
        print(f'\nconnected components in src graph: {[len(x) for x in con_gs]}')
        # src_e = G_s.subgraph(max(con_gs, key=len)).edges
        src_e = G_s.edges
        del G_s
        G_t = nx.from_edgelist(tar_e)
        con_gt = list(nx.connected_components(G_t))
        print(f'\nconnected components in tar graph: {[len(x) for x in con_gt]}')
        # tar_e = G_t.subgraph(max(con_gt, key=len)).edges
        tar_e = G_t.edges
        del G_t

        src_adj, src_nodes = adj_from_edges(src_e)
        tar_adj, tar_nodes = adj_from_edges(tar_e)
        
        print('embedding src graph...')
        l, src_embedding = spectral_embedding(src_adj, n_components=n_comp)
        print('embedding tar graph...')
        mu, tar_embedding = spectral_embedding(tar_adj, n_components=n_comp)

        diffs = np.abs(np.diff(l))
        diffs_ = np.abs(np.diff(mu))
        # diffs[0] = diffs[1] = diffs_[0] = diffs_[1] = 0 # We do not wish to find k<3
        K = diffs.argmax() + 2 # +1 due to diff[i] being [i+1] in original array. +1 as eigengap heuristic includes the l=0 vector.
        K_ = diffs_.argmax() + 2
        print(f'diffs:\n{diffs}\n{diffs_}\n')
        print(f'K:{K}, K_: {K_}')
        K = max(K, 2)
        # K = max(K, K_, 2) # At least two clusters.
        # K += (max(len(con_gs), len(con_gt)) - 1) # Add 1 for each connected component as spect. clust. should uncover these clusters.
        
        # K = 2
        # d = int(np.ceil(np.log2(K)))
        d = 2
        # print(f'\nFound K={K} at position={pos}')
        print(f'\nFound K={K}, d={d} at position={pos}')

        src_embedding = src_embedding.T[:d].T
        tar_embedding = tar_embedding.T[:d].T
        l = l[:d]
        mu = mu[:d]

        # I = np.eye(d)
        # B = base_align.optimize_AB(I, I, 0, src_embedding.T, tar_embedding.T, l, mu, d)
        # print(f'shapes of stuff:\nsrc_emb:{src_embedding.shape}\ntar_emb: {tar_embedding.shape}\nrot_mat: {B.shape}\n\nAlso B:\n{B}')
        # tar_embedding = (B @ tar_embedding.T).T

        # Compute clusters on embedded data with kmeans and lloyd's algorithm
        warnings.simplefilter('error', category=ConvergenceWarning)
        try:
            print('computing k-means (src graph)')
            kmeans = KMeans(n_clusters=K, init='k-means++', n_init=10).fit(src_embedding.T[:d].T)
            src_centroids = kmeans.cluster_centers_
            _src_cluster = kmeans.labels_
            src_dists = kmeans.transform(src_embedding.T[:d].T)
        except Exception as e:
            traceback.print_exc()
            print(repr(e))
            print(f'Troublesome graph!!!\n Matching parent graph using GRAMPA')
            match_grampa((src_adj, src_nodes), (tar_adj, tar_nodes))
            return
        try:
            print('computing k-means (tar graph)')
            # Seed target graph kmeans by using the src centroids.
            kmeans = KMeans(n_clusters=K, init=src_centroids, n_init=1).fit(tar_embedding.T[:d].T)
            # kmeans = KMeans(n_clusters=K, init='k-means++', n_init=10).fit(tar_embedding.T[:2].T)
            
            tar_centroids = kmeans.cluster_centers_
            _tar_cluster = kmeans.labels_
            tar_dists = kmeans.transform(tar_embedding.T[:d].T)
        except Exception as e:
            traceback.print_exc()
            print(repr(e))
            print(f'Troublesome graph!!!\n Matching parent graph using GRAMPA')

            match_grampa((src_adj, src_nodes), (tar_adj, tar_nodes))
            return

        src_cluster = dict(zip(src_nodes, _src_cluster))
        tar_cluster = dict(zip(tar_nodes, _tar_cluster))
        src_nodes_by_cluster = [[k for k,v in src_cluster.items() if v == i] for i in range(K)]
        src_cluster_sizes = [len(x) for x in src_nodes_by_cluster]
        tar_nodes_by_cluster = [[k for k,v in tar_cluster.items() if v == i] for i in range(K)]
        tar_cluster_sizes = [len(x) for x in tar_nodes_by_cluster]
        print(f'\ncluster numbers (2-emb): src:{src_cluster_sizes}, tar:{tar_cluster_sizes}\n')

        # 2. split graphs (src, tar) according to cluster.
        # print('Splitting SRC')
        src_subgraphs = split_graph(src_e, src_cluster)
        # print('Splitting TAR')
        tar_subgraphs = split_graph(tar_e, tar_cluster)

        C_UPDATED = False
        # row, col, _ = lapjv(-sim) # row, col, _ = lapjv(-sim)
        partition_alignment = list(zip(range(K), range(K)))
        cur_part_acc = []
        part_size_diff = {}

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
        for i, (pp, size) in enumerate(part_size_diff.items()):
            if size > 0:
                # find candidate clusters from which to borrow nodes
                candidate_clusters = [k for (k, v) in part_size_diff.items() if v < 0]
                # list of indices of tar_nodes that correspond to candidate cluster c_i for all c_i candidate (target) clusters.
                cand_idcs_list = [[i for i, v in enumerate(_tar_cluster) if v == j[1]] for j in candidate_clusters]
                # distance to current
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
            tar_cluster_sizes = [len(x) for x in tar_nodes_by_cluster]
            tar_subgraphs = split_graph(tar_e, tar_cluster)
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
                print(f'c_s_nodes or c_t_nodes empty: {c_s_nodes}, {c_t_nodes} -- CONTINUING LOOP (Discard matching)')
                continue

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

def main(data, rsc, n_comp):
    print(f'\n\n\nstarting run:\nargs: (rsc, n_comp)={(rsc, n_comp)}')
    Src = data['src_e']
    Tar = data['tar_e']
    gt = data["gt"]

    n = Src.shape[0]
    s = time.time()
    matching, pos, readj_accs = alhpa(Src, Tar, rsc=rsc, n_comp=n_comp, gt=gt)
    print(f'readjustment accuracis: {readj_accs.mean()} (avg.)\n{readj_accs}')

    for k, v in pos.items():
        print(f'{k}:\n{v}')
    return matching
