import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from .pl import _plot_diffusion_map
from scipy import sparse


def compute_diffusion_map(reference_data, neigen=2, epsilon=None, pca_comps=None, k=None):
    """
    Compute a diffusion map embedding from reference (training) data.

    Parameters:
      reference_data : np.ndarray, shape (n_samples_ref, n_features)
          Input training data.
      neigen : int
          Number of non-trivial diffusion components to return.
      epsilon : float or None
          Gaussian kernel width parameter. If None, set to the square of the median of non-zero pairwise distances.
      pca_comps : int or None
          If provided, apply PCA to reduce to this many dimensions before computing distances.
      k : int or None
          Number of nearest neighbors to consider when building the affinity matrix.  
          If provided, a sparse k-NN graph is used instead of the full pairwise 
          distance matrix. Use together with a X_pca embedding key or other lower 
          dimensional spaces to speed up computation on larger datasets.

    Returns:
      dict with keys:
        'ref_diffusion'        : Diffusion coordinates (n_samples_ref x neigen).
        'eigenvalues'          : Selected eigenvalues (length neigen).
        'distance_matrix_ref'  : Pairwise reference distance matrix (dense array).
        'neighbors_matrix_ref' : k-NN sparse graph (if k specified), else None.
        'ref_proc'             : Processed reference data after optional PCA.
        'epsilon'              : Kernel width used.
        'pca'                  : PCA object if used, else None.
    """
    # Step 1: Optional PCA dimensionality reduction
    if pca_comps is not None:
        from sklearn.decomposition import PCA
        pca_obj = PCA(n_components=pca_comps)
        ref_proc = pca_obj.fit_transform(reference_data)
    else:
        ref_proc = reference_data
        pca_obj = None

    # Step 2: Build Gaussian affinity and symmetric normalization
    if k is None:
        D_ref = squareform(pdist(ref_proc, metric='euclidean'))  # full dense distances
        if epsilon is None:
            nonzero = D_ref[D_ref > 0]
            epsilon = np.median(nonzero) ** 2
        K = np.exp(-np.square(D_ref) / epsilon)
        row_sums = np.sum(K, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(row_sums))  # density normalization
        A = D_inv_sqrt @ K @ D_inv_sqrt
        eigenvals, eigenvecs = np.linalg.eigh(A)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        phi = D_inv_sqrt @ eigenvecs
        distance_matrix_ref = D_ref
        neighbors_matrix_ref = None
    else:
        from sklearn.neighbors import kneighbors_graph
        from scipy.sparse import diags
        from scipy.sparse.linalg import eigsh
        D_sparse = kneighbors_graph(ref_proc, n_neighbors=k+1, mode='distance', metric='euclidean')
        if epsilon is None:
            nonzero = D_sparse.data[D_sparse.data > 0]
            epsilon = np.median(nonzero) ** 2
        K_sparse = D_sparse.copy()
        K_sparse.data = np.exp(-np.square(K_sparse.data) / epsilon)
        K_sparse = 0.5 * (K_sparse + K_sparse.T)
        p_vals = np.array(K_sparse.sum(axis=1)).flatten()
        D_inv_sqrt = diags(1.0 / np.sqrt(p_vals))
        A_sparse = D_inv_sqrt @ K_sparse @ D_inv_sqrt
        n_eigs = neigen + 1
        eigenvals, eigenvecs = eigsh(A_sparse, k=n_eigs, which='LM')
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        phi = D_inv_sqrt.dot(eigenvecs)
        distance_matrix_ref = D_sparse.toarray()
        neighbors_matrix_ref = D_sparse

    # Step 3: Select diffusion coordinates (drop trivial first mode if necessary)
    if np.var(phi[:, 0]) < 1e-10:
        ref_coords = phi[:, 1:neigen+1]
        chosen_eigenvals = eigenvals[1:neigen+1]
    else:
        ref_coords = phi[:, :neigen]
        chosen_eigenvals = eigenvals[:neigen]

    return {
        'ref_diffusion': ref_coords,
        'eigenvalues': chosen_eigenvals,
        'distance_matrix_ref': distance_matrix_ref,
        'neighbors_matrix_ref': neighbors_matrix_ref,
        'ref_proc': ref_proc,
        'epsilon': epsilon,
        'pca': pca_obj
    }


def nystrom_extension(query_data, diffusion_obj, k=None):
    """
    Extend diffusion map to new query points using the Nyström method.
    """
    ref_proc = diffusion_obj['ref_proc']
    epsilon = diffusion_obj['epsilon']
    ref_diffusion = diffusion_obj['ref_diffusion']
    eigenvalues = diffusion_obj['eigenvalues']

    if diffusion_obj.get('pca') is not None:
        query_proc = diffusion_obj['pca'].transform(query_data)
    else:
        query_proc = query_data

    if k is None:
        D_new = cdist(query_proc, ref_proc, metric='euclidean')
        K_new = np.exp(-np.square(D_new) / epsilon)
        K_new_norm = K_new / np.sum(K_new, axis=1, keepdims=True)
        distance_matrix_query = D_new
        neighbors_matrix_query = None
    else:
        from sklearn.neighbors import NearestNeighbors
        from scipy.sparse import diags
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(ref_proc)
        D_sparse = nbrs.kneighbors_graph(query_proc, mode='distance')
        K_sparse = D_sparse.copy().tocsr()
        K_sparse.data = np.exp(-np.square(K_sparse.data) / epsilon)
        row_sums = np.array(K_sparse.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        D_inv = diags(1.0 / row_sums)
        K_new_norm = (D_inv @ K_sparse).toarray()
        distance_matrix_query = D_sparse.toarray()
        neighbors_matrix_query = D_sparse

    inv_eigs = np.diag(1.0 / eigenvalues)
    query_diffusion = K_new_norm.dot(ref_diffusion).dot(inv_eigs)

    return {
        'query_diffusion': query_diffusion,
        'distance_matrix_query': distance_matrix_query,
        'neighbors_matrix_query': neighbors_matrix_query
    }


def run_diffusion_map(ref_adata, query_adata, embedding_key="X",
                      neigen=2, k=None, pca_comps=None, epsilon=None, plot=True):
    """
    Full workflow:
      1. Compute and store diffusion map for reference.
      2. Extend mapping to query via Nyström.
      3. Store both embeddings under .obsm['X_diffmap'].

    The .uns['diffusion_results'] in ref_adata holds intermediate outputs.
    """
    # Handle raw X conversion only if using 'X' representation
    if embedding_key == "X":
        if k is not None:
            if not sparse.issparse(ref_adata.X):
                ref_adata.X = sparse.csr_matrix(ref_adata.X)
            if not sparse.issparse(query_adata.X):
                query_adata.X = sparse.csr_matrix(query_adata.X)
        else:
            if sparse.issparse(ref_adata.X):
                ref_adata.X = ref_adata.X.toarray()
            if sparse.issparse(query_adata.X):
                query_adata.X = query_adata.X.toarray()

    reference_data = ref_adata.obsm.get(embedding_key, ref_adata.X)
    query_data = query_adata.obsm.get(embedding_key, query_adata.X)

    print(f"Computing diffusion map ({neigen} components) on reference...")
    diff_map = compute_diffusion_map(reference_data, neigen=neigen,
                                     epsilon=epsilon, pca_comps=pca_comps, k=k)

    print("Extending to query via Nyström extension...")
    nys = nystrom_extension(query_data, diff_map, k=k)

    # Store the diffusion embeddings explicitly under 'X_diffmap'
    ref_adata.obsm['X_diffmap'] = diff_map['ref_diffusion']
    query_adata.obsm['X_diffmap'] = nys['query_diffusion']


    if plot and neigen >= 2:
        _plot_diffusion_map(ref_adata, query_adata)

    results = {
        'eigenvalues': diff_map['eigenvalues'],
        'ref_proc':    diff_map['ref_proc'],
        'epsilon':     diff_map['epsilon'],
        'pca':         diff_map['pca'],
        'distance_matrix_ref':   diff_map['distance_matrix_ref'],
        'neighbors_matrix_ref':  diff_map['neighbors_matrix_ref'],
        'distance_matrix_query': nys['distance_matrix_query'],
        'neighbors_matrix_query':nys['neighbors_matrix_query']
    }
    # Store all intermediate results for reproducibility and diagnostics
    ref_adata.uns['diffusion_results'] = results

    print("Stored embeddings in .obsm['X_diffmap'] and details in .uns['diffusion_results'].")
