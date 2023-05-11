import numpy as np
import numba as nb
import scipy as sp
import warnings

def adj_from_edges(edge_list):
    unique = np.unique(edge_list)
    N = unique.shape[0]
    node_map = {node: idx for idx, node in enumerate(unique)}
    idx_map = np.array(list(node_map.keys())) # Allows node name retrieval.

    adj = np.zeros(shape=(N, N), dtype=np.float32)
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

def compute_laplacian(A):
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

def _graph_connected_component(graph, node_id):
    """Find the largest graph connected components that contains one
    given node.

    Parameters
    ----------
    graph : array-like of shape (n_samples, n_samples)
        Adjacency matrix of the graph, non-zero weight means an edge
        between the nodes.

    node_id : int
        The index of the query node of the graph.

    Returns
    -------
    connected_components_matrix : array-like of shape (n_samples,)
        An array of bool value indicating the indexes of the nodes
        belonging to the largest connected components of the given query
        node.
    """
    n_node = graph.shape[0]
    if sp.sparse.issparse(graph):
        # speed up row-wise access to boolean connection mask
        graph = graph.tocsr()
    connected_nodes = np.zeros(n_node, dtype=bool)
    nodes_to_explore = np.zeros(n_node, dtype=bool)
    nodes_to_explore[node_id] = True
    for _ in range(n_node):
        last_num_component = connected_nodes.sum()
        np.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break
        indices = np.where(nodes_to_explore)[0]
        nodes_to_explore.fill(False)
        for i in indices:
            if sp.sparse.issparse(graph):
                neighbors = graph[i].toarray().ravel()
            else:
                neighbors = graph[i]
            np.logical_or(nodes_to_explore, neighbors, out=nodes_to_explore)
    return connected_nodes

def _graph_is_connected(graph):
    """Return whether the graph is connected (True) or Not (False).

    Parameters
    ----------
    graph : {array-like, sparse matrix} of shape (n_samples, n_samples)
        Adjacency matrix of the graph, non-zero weight means an edge
        between the nodes.

    Returns
    -------
    is_connected : bool
        True means the graph is fully connected and False means not.
    """
    if sp.sparse.isspmatrix(graph):
        # sparse graph, find all the connected components
        n_connected_components, _ = connected_components(graph)
        return n_connected_components == 1
    else:
        # dense graph, find all connected components start from node 0
        return _graph_connected_component(graph, 0).sum() == graph.shape[0]


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )

def _deterministic_vector_sign_flip(u):
    """
    https://github.com/scikit-learn/scikit-learn/blob/2a2772a87b6c772dc3b8292bcffb990ce27515a8/sklearn/utils/extmath.py#L1093
    
    Modify the sign of vectors for reproducibility.

    Flips the sign of elements of all the vectors (rows of u) such that
    the absolute maximum element of each vector is positive.

    Parameters
    ----------
    u : ndarray
        Array with vectors as its rows.

    Returns
    -------
    u_flipped : ndarray with same shape as u
        Array with the sign flipped vectors as its rows.
    """
    max_abs_rows = np.argmax(np.abs(u), axis=1)
    signs = np.sign(u[range(u.shape[0]), max_abs_rows])
    u *= signs[:, np.newaxis]
    return u

def spectral_embedding(adjacency, *, n_components=25, random_state=None):
    """Project the sample on the first eigenvectors of the graph Laplacian.

    The adjacency matrix is used to compute a normalized graph Laplacian
    whose spectrum (especially the eigenvectors associated to the
    smallest eigenvalues) has an interpretation in terms of minimal
    number of cuts necessary to split the graph into comparably sized
    components.

    This embedding can also 'work' even if the ``adjacency`` variable is
    not strictly the adjacency matrix of a graph but more generally
    an affinity or similarity matrix between samples (for instance the
    heat kernel of a euclidean distance matrix or a k-NN matrix).

    However care must taken to always make the affinity matrix symmetric
    so that the eigenvector decomposition works as expected.

    Note : Laplacian Eigenmaps is the actual algorithm implemented here.

    Read more in the :ref:`User Guide <spectral_embedding>`.

    Parameters
    ----------
    adjacency : {array-like, sparse graph} of shape (n_samples, n_samples)
        The adjacency matrix of the graph to embed.

    n_components : int, default=8
        The dimension of the projection subspace.

    Returns
    -------
    l : ndarray of shape (n_samples,)
        The lambda values of the spectral decomposition
    embedding : ndarray of shape (n_samples, n_components)
        The reduced samples.

    Notes
    -----
    Spectral Embedding (Laplacian Eigenmaps) is most useful when the graph
    has one connected component. If there graph has many components, the first
    few eigenvectors will simply uncover the connected components of the graph.

    References
    ----------
    * https://en.wikipedia.org/wiki/LOBPCG

    * :doi:`"Toward the Optimal Preconditioned Eigensolver: Locally Optimal
      Block Preconditioned Conjugate Gradient Method",
      Andrew V. Knyazev
      <10.1137/S1064827500366124>`
    """

    random_state = check_random_state(random_state)

    n_nodes = adjacency.shape[0]
    n_components = n_components + 1

    if not _graph_is_connected(adjacency):
        warnings.warn(
            "Graph is not fully connected, spectral embedding may not work as expected."
        )

    laplacian, dd = sp.sparse.csgraph.laplacian(
        adjacency, normed=True, return_diag=True
    )
    tol = 0
    laplacian *= -1

    v0 = random_state.uniform(-1, 1, laplacian.shape[0])
    l, diffusion_map = sp.sparse.linalg.eigsh(
        laplacian, k=n_components, sigma=1.0, which="LM", tol=tol, v0=v0
    )
    embedding = diffusion_map.T[n_components::-1]
    l = l[::-1]
    print(f'\nlambdas:\n{l}\n1-l:{1-l}')
    embedding = embedding / dd
    embedding = _deterministic_vector_sign_flip(embedding)

    return l[1:n_components], embedding[1:n_components].T
