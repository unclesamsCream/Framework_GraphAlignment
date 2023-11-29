from algorithms import Grampa, base_align
from Utils import adj_from_edges, compute_laplacian, embed_spectral, expand_matrix, decompose_spectral, spectral_embedding
import numpy as np
from numpy.linalg import inv
from numpy.linalg import eigh,eig
import networkx as nx 
import warnings
import itertools
from sklearn.manifold import spectral_embedding as sk_specemb
import lapjv
from sklearn.cluster import k_means, KMeans
from sklearn.exceptions import ConvergenceWarning
import time
import traceback
from pprint import pprint

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
    #initilize the n array to -1
    clustering_accs_pre = []
    clustering_accs_post = []
    readjustment_accs = []
    all_pos = []
    def match_grampa(src, tar):
        if isinstance(src, tuple) and isinstance(tar, tuple):
            src_adj, src_map = src
            tar_adj, tar_map = tar
        else:
            src_adj, src_map = adj_from_edges(src)
            tar_adj, tar_map = adj_from_edges(tar)
        # get the  adjecency matrix and the map of the index of the nodes

        diff = len(src_map) - len(tar_map)
        #calculate the difference between the number of nodes in the two graphs
        #print(diff)
        if diff < 0: # expand sub src
            #print(f'size before expansion: {src_adj.shape}')
            src_adj = expand_matrix(src_adj, abs(diff))
            #print(f'size after expansion: {src_adj.shape}')
            #print(f'src_map size before: {len(src_map)}')
            src_map = list(src_map) + [-1] * abs(diff)
            #print(f'src_map size after: {len(src_map)}')
        # if the difference is less than 0, then expand the matrix of the src graph
        if diff > 0: # expand sub tar
            #print(f'size before expansion: {tar_adj.shape}')
            tar_adj = expand_matrix(tar_adj, diff)
            #print(f'size after expansion: {tar_adj.shape}')
            #print(f'tar_map size before: {len(tar_map)}')
            tar_map = list(tar_map) + [-1] * diff
            #print(f'tar_map size after: {len(tar_map)}')

        sub_sim = Grampa.grampa(src_adj, tar_adj, eta, lap=True)
        # get the similarity matrix of the two graphs that is n by n, each element is the similarity between the two nodes
        
        return sub_sim

        #print(f'adjacency matrices for src and tar (shapes): {src_adj.shape}, {tar_adj.shape})')
        #print(sub_sim)
        # r, c, _ = lapjv(-sub_sim)
        # match = list(zip(range(len(c)), c))
        # #print(f'GRAMPA:{match}')
        # # translate with map and add to solution
        # for (n1, n2) in match:    
        #     si = src_map[n1]
        #     if si < n:
        #         matching[src_map[n1]] = tar_map[n2]
        #     else:
        #         print('!!!src node outside matching idx!!!')
        #get the matching of the two graphs

    def cluster_recurse(src_e, tar_e, pos=[(0,0)]):
        pos = pos.copy()
        G_s = nx.from_edgelist(src_e)
        con_gs = list(nx.connected_components(G_s))
        # print(list(con_gs))
        print(f'\nconnected components in src graph: {[len(x) for x in con_gs]}')
        # src_e = G_s.subgraph(max(con_gs, key=len)).edges
        src_e = G_s.edges
        # print(f'INFO: Graph -- |V|={len(list(G_s.nodes))}, |E|={len(list(G_s.edges))}, #/\={round(sum(list(nx.triangles(G_s).values()))/1000, 4)}, CC={round(nx.average_clustering(G_s) / 1000, 5)}')
        del G_s
        G_t = nx.from_edgelist(tar_e)
        con_gt = list(nx.connected_components(G_t))
        print(f'\nconnected components in tar graph: {[len(x) for x in con_gt]}')
        # tar_e = G_t.subgraph(max(con_gt, key=len)).edges
        tar_e = G_t.edges
        del G_t

        src_adj, src_nodes = adj_from_edges(src_e)
        tar_adj, tar_nodes = adj_from_edges(tar_e)
        
        # print('embedding src graph...')
        # l, src_embedding = spectral_embedding(src_adj, n_components=n_comp)
        # # the default value of n_comp is 10
        # print('embedding tar graph...')
        # mu, tar_embedding = spectral_embedding(tar_adj, n_components=n_comp)
        
        # diffs = np.abs(np.diff(l))
        # diffs_ = np.abs(np.diff(mu))
        # # diffs[0] = diffs[1] = diffs_[0] = diffs_[1] = 0 # We do not wish to find k<3
        # K = diffs.argmax() + 2 # +1 due to diff[i] being [i+1] in original array. +1 as eigengap heuristic includes the l=0 vector.
        # K_ = diffs_.argmax() + 2
        # print(f'diffs:\n{diffs}\n{diffs_}\n')
        # print(f'K:{K}, K_: {K_}')
        # K = max(K, 2)
        # K = max(K, K_, 2) # At least two clusters.
        # K += (max(len(con_gs), len(con_gt)) - 1) # Add 1 for each connected component as spect. clust. should uncover these clusters.
        
        K = 5
        # d = int(np.ceil(np.log2(K)))
        # d = 2
        # # print(f'\nFound K={K} at position={pos}')
        # print(f'\nFound K={K}, d={d} at position={pos}')

        # src_embedding = src_embedding.T[:d].T
        # tar_embedding = tar_embedding.T[:d].T
        # l = l[:d]
        # mu = mu[:d]

        # I = np.eye(d)
        # B = base_align.optimize_AB(I, I, 0, src_embedding.T, tar_embedding.T, l, mu, d)
        # print(f'shapes of stuff:\nsrc_emb:{src_embedding.shape}\ntar_emb: {tar_embedding.shape}\nrot_mat: {B.shape}\n\nAlso B:\n{B}')
        # tar_embedding = (B @ tar_embedding.T).T

        sub_sim = match_grampa((src_adj, src_nodes), (tar_adj, tar_nodes))
        print(f'sub_sim:\n{sub_sim}')
        warnings.simplefilter('error', category=ConvergenceWarning)
        try:
            print('computing k-means')
            kmeans = KMeans(n_clusters=K, init='k-means++', n_init=1).fit(sub_sim)
            src_centroids = kmeans.cluster_centers_
            _src_cluster = kmeans.labels_
            src_dists = kmeans.transform(sub_sim)
        # Compute clusters on embedded data with kmeans and lloyd's algorithm
        except Exception as e:
            traceback.print_exc()
            print(repr(e))
            print(f'Troublesome graph!!!\n Matching parent graph using GRAMPA')
            # match_grampa((src_adj, src_nodes), (tar_adj, tar_nodes))
            return
        # print('k-means')
        # print(f'k-means centroids:\n{src_centroids}')
        # print(f'k-means labels:\n{_src_cluster}')
        src_cluster = dict(zip(src_nodes, _src_cluster))
        src_nodes_by_cluster = [[k for k,v in src_cluster.items() if v == i] for i in range(K)]
        # print(f'\ncluster\n{src_cluster}')
        # print(f'\ncluster nodes\n{src_nodes_by_cluster}')
        # print(f'\ncluster numbers (2-emb): src:{[len(x) for x in src_nodes_by_cluster]}\n')

        sub_matrices = []

        for cluster in src_nodes_by_cluster:
            indices = np.array(cluster) - 1
            sub_matrix = sub_sim[np.ix_(indices, indices)]
            sub_matrices.append(sub_matrix)

        # print(f'sub_matrices:\n{sub_matrices}')

        def jv(dist):

            # print('hungarian_matching: calculating distance matrix')



            # dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)

            n = dist.shape[0]

            # print(np.shape(dist))

            # print('hungarian_matching: calculating matching')

            cols, rows, _ = lapjv.lapjv(dist)

            # print(cols)

            # print(rows)

            matching = np.c_[rows, np.linspace(0, n-1, n).astype(int)]

            # print(matching)

            matching = matching[matching[:, 0].argsort()]

            # print(matching)

            return matching.astype(int).T

        results = []
        for sub_matrix in sub_matrices:
            # print(f'sub_matrix:\n{sub_matrix}')
            result = jv(sub_matrix)
            # print(f'result:\n{result}')
            # print(f'result shape:\n{result.shape}')
            results.append(result)
        print(f'results:\n{results}')
        final_result = np.concatenate(results, axis=1)
        # print(f'final_result shape:\n{final_result.shape}')
        # print("final_result")
        # pprint(final_result)
        return final_result

        #     # if cluster size is smaller than sqrt(|V|) then align nodes.
        #     #  else cluster recurse.
        #     if len(c_s_nodes) <= rsc or len(c_t_nodes) <= rsc:
        #         #print(f'Smaller than rsc: Matching clusters directly\nclusters: {c_s}, {c_t} at pos={pos}')
        #         match_grampa(sub_src, sub_tar)
        #         all_pos.append(pos)
        #     else:
        #         pos.append((c_s, c_t))
        #         cluster_recurse(sub_src, sub_tar, pos=pos)

    matching = cluster_recurse(src_graph, tar_graph)
    if len(all_pos) == 0:
        pos_res = {'max_depth': 0, 'avg_depth': 0}
    else:
        pos_res = {'max_depth': len(max(all_pos, key=lambda x: len(x))), 'avg_depth': np.array([len(x) for x in all_pos]).mean()}# , pos': all_pos
    # matching = np.c_[np.linspace(0, n-1, n).astype(int), matching].astype(int).T

    return matching, pos_res, np.array(readjustment_accs), np.array(clustering_accs_pre), np.array(clustering_accs_post)

def main(data, rsc, n_comp):
    print(f'\n\n\nstarting run:\nargs: (rsc, n_comp)={(rsc, n_comp)}')
    Src = data['src_e']
    Tar = data['tar_e']
    gt = data["gt"]

    n = Src.shape[0]
    s = time.time()
    matching, pos, readj_accs, cacc_pre, cacc_post = alhpa(Src, Tar, rsc, n_comp, gt)
    # print(f'readjustment accuracis: {readj_accs.mean()} (avg.)\n{readj_accs}')
    # print(f'average cluster acc.: (pre,post): {cacc_pre.mean()},{cacc_post.mean()}')

    print(f'ALHPA_S: {matching}')
    # for k, v in pos.items():
    #     print(f'{k}:\n{v}')
    return matching
