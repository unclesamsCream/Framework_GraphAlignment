import numpy as np
import numba as nb
import scipy as sp
import scipy.sparse

def adj_from_edges(edge_list):
    unique = np.unique(edge_list)
    N = unique.shape[0]
    node_map = {node: idx for idx, node in enumerate(unique)}
    idx_map = np.array(list(node_map.keys())) # Allows node name retrieval.

    adj = np.zeros(shape=(N, N))
    for (e1, e2) in edge_list:
        adj[node_map[e1]][node_map[e2]] += 1
        if e1 != e2:
            adj[node_map[e2]][node_map[e1]] += 1

    return adj, idx_map

    
def graph_eig(L): 
    """
      Takes a graph Laplacian and returns sorted the eigenvalues and vectors.
    """
    lambdas, eigenvectors = np.linalg.eigh(L)
    lambdas = np.real(lambdas)
    eigenvectors = np.real(eigenvectors)

    order = np.argsort(lambdas)
    lambdas = lambdas[order]
    eigenvectors = eigenvectors[:, order]

    return lambdas, eigenvectors

def compute_laplacian(G):
    A = G
    d = A.sum(axis=1)
    D = np.diag(d)
    L = D - A
    D12 = np.diag(pow(d, -0.5))

    return D12 @ L @ D12

def embed_spectral(G, e_dim=5):
    """
      Takes a graph (nx.Graph) or adjacency matrix (nxn np.array) and returns a clustering
    """
    
    L = compute_laplacian(G)
    lambdas, eig_vectors = np.linalg.eigh(L)
    # select only eigen vectors corresponding to non-zero eigenvalues
    nonzero_lambdas = np.arange(len(lambdas))[np.where(lambdas > 1e-15)]
    selection = nonzero_lambdas[:e_dim] # use e_dim number vectors for embedding

    return eig_vectors[:, selection]

def decompose_spectral(G):
    """
      Takes a graph adjacency matrix (nxn np.array) and returns a clustering
    """
    L = compute_laplacian(G)
    l, U = np.linalg.eigh(L)
    # select only eigen vectors corresponding to non-zero eigenvalues
    nz = np.arange(len(l))[np.where(l > 1e-15)]

    return l, U, nz

def expand_matrix(M, size):
    expanded = M.copy()
    for i in range(size):
        n, _ = expanded.shape
        expanded = np.c_[
            (np.r_[(expanded, np.zeros(shape=(1, n)))],
             np.zeros(shape=(n+1, 1)))
        ]
        expanded[-1,-1] = 1

    return expanded
