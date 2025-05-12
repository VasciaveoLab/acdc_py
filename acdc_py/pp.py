### ---------- IMPORT DEPENDENCIES ----------
from ._pp import _corr_distance, _neighbors_knn, _neighbors_graph, _compute_diffusion_map, _nystrom_extension
import numpy as np

### ---------- EXPORT LIST ----------
__all__ = []

# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# ---------------------------- ** DISTANCE FUNCS ** ----------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
def corr_distance(adata,
                  use_reduction=True,
                  reduction_slot="X_pca",
                  key_added="corr_dist",
                  batch_size=1000,
                  dtype=np.int16,
                  verbose=True):
    """\
    A tool for computing a distance matrix based on pearson correlation.

    Parameters
    ----------
    adata
        An anndata object containing a signature in adata.X
    use_reduction : default: True
        Whether to use a reduction (True) (highly recommended - accurate & much faster)
        or to use the direct matrix (False) for computing distance.
    reduction_slot : default: "X_pca"
        If reduction is TRUE, then specify which slot for the reduction to use.
    key_added : default: "corr_dist"
        Slot in obsp to store the resulting distance matrix.
    batch_size : default: 1000
        Reduce total memory usage by running data in batches.
    dtype : default: np.int16
        Data type used to represent the distance values. np.int16 (default) is
        a compromise between smaller memory size while not reducing information
        so much as to affect clustering. dtypes include np.int8, np.int16 (default) np.int32, np.int64, np.float16, np.float32, and np.float64.
    verbose : default: True
        Show a progress bar for each batch of data.

    Returns
    -------
    Adds fields to the input adata, such that it contains a distance matrix
    stored in adata.obsp[key_added].
    """
    # returns if isinstance(adata, np.ndarray) or isinstance(adata, pd.DataFrame):
    return _corr_distance(
        adata,
        use_reduction,
        reduction_slot,
        key_added,
        batch_size,
        dtype,
        verbose
    )

# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# ---------------------------- ** KNN ARRAY FUNC ** ----------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
def neighbors_knn(adata,
                  max_knn=101,
                  dist_slot="corr_dist",
                  key_added="knn",
                  batch_size = 1000,
                  verbose = True,
                  njobs = 1):
    """\
    A tool for computing a KNN array used to then rapidly generate connectivity
    graphs with acdc.pp.neighbors_graph for clustering.

    Parameters
    ----------
    adata
        An anndata object containing a distance object in adata.obsp.
    max_knn : default: 101
        The maximum number of k-nearest neighbors (knn) to include in this array.
        acdc.pp.neighbors_graph will only be able to compute KNN graphs with
        knn <= max_knn.
    dist_slot : default: "corr_dist"
        The slot in adata.obsp where the distance object is stored. One way of
        generating this object is with adata.pp.corr_distance.
    key_added : default: "knn"
        Slot in uns to store the resulting knn array.
    batch-size : default: 1000
        Size of the batches used to reduce memory usage.
    verbose : default: True
        Whether to display a progress bar of the batches completed.
    njobs : default: 1
        Paralleization option that allows users to speed up runtime.

    Returns
    -------
    Adds fields to the input adata, such that it contains a knn array stored in
    adata.uns[key_added].
    """
    # returns if isinstance(adata, np.ndarray) or isinstance(adata, pd.DataFrame):
    return _neighbors_knn(
        adata,
        max_knn,
        dist_slot,
        key_added,
        batch_size,
        verbose,
        njobs
    )


# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# -------------------------- ** NEIGHBOR GRAPH FUNC ** -------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
def neighbors_graph(adata,
                    n_neighbors=15,
                    knn_slot='knn',
                    batch_size=1000,
                    verbose = True):
    """\
    A tool for rapidly computing a k-nearest neighbor (knn) graph (i.e.
    connectivities) that can then be used for clustering.

    graphs with acdc.pp.neighbors_graph for clustering.

    Parameters
    ----------
    adata
        An anndata object containing a distance object in adata.obsp.
    n_neighbors : default: 15
        The number of nearest neighbors to use to build the connectivity graph.
        This number must be less than the total number of knn in the knn array
        stored in adata.uns[knn_slot].
    knn_slot : default: 101
        The slot in adata.uns where the knn array is stored. One way of
        generating this object is with acdc.pp.neighbors_knn.
    batch-size : default: 1000
        Size of the batches used to reduce memory usage.
    verbose : default: True
        Whether to display a progress bar of the batches completed.

    Returns
    -------
    Adds fields to the input adata, such that it contains a knn graph stored in
    adata.obsp['connectivities'] along with metadata in adata.uns["neighbors"].
    """
    # returns if isinstance(adata, np.ndarray) or isinstance(adata, pd.DataFrame):
    return _neighbors_graph(
        adata,
        n_neighbors,
        knn_slot,
        batch_size,
        verbose
    )


def compute_diffusion_map(reference_data, neigen=2, epsilon=None, pca_comps=None):
    """\
    Compute a diffusion map embedding from reference (training) data.

    Parameters
    ----------
    reference_data : array-like, shape (n_samples_ref, n_features)
        Input training data (dense or sparse; will be densified internally).
    neigen : int, optional
        Number of non-trivial diffusion components to return. Default is 2.
    epsilon : float or None, optional
        Gaussian kernel width parameter. If None, set to the square of the median of
        non-zero pairwise distances. Default is None.
    pca_comps : int or None, optional
        If provided, apply PCA to reduce to this many dimensions before computing
        distances. Default is None.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'ref_diffusion'       : Diffusion coordinates (n_samples_ref × neigen).
        - 'eigenvalues'         : Selected eigenvalues (length neigen).
        - 'distance_matrix_ref' : Pairwise distance matrix (dense).
        - 'ref_proc'            : Processed reference data after optional PCA.
        - 'epsilon'             : Kernel width used.
        - 'pca'                 : PCA object if used, else None.
    """
    return _compute_diffusion_map(reference_data, neigen=neigen,
                                  epsilon=epsilon, pca_comps=pca_comps)
    

def nystrom_extension(query_data, diffusion_obj):
    """\
    Extend a reference diffusion map to new query data using the Nyström method.

    Parameters
    ----------
    query_data : array-like, shape (n_query, n_features)
        Feature matrix for query samples, preprocessed to match the reference.
    diffusion_obj : dict
        Output of _compute_diffusion_map, containing:
          - 'ref_proc'      : ndarray of shape (n_ref, n_features_proc)
          - 'epsilon'       : float
          - 'ref_diffusion' : ndarray of shape (n_ref, neigen)
          - 'eigenvalues'   : ndarray of length neigen
          - 'pca'           : PCA object or None

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'query_diffusion'      : Mapped diffusion coordinates (n_query × neigen).
        - 'distance_matrix_query': Pairwise distance matrix (dense).
    """
    return _nystrom_extension(query_data, diffusion_obj)


