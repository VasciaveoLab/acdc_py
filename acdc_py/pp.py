### ---------- IMPORT DEPENDENCIES ----------
from ._pp import _corr_distance, _neighbors_knn, _neighbors_graph

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
                  key_added="corr_dist"):
    """\
    A tool for computing a distance matrix based on pearson correlation.

    Parameters
    ----------
    adata
        An anndata object containing a signature in adata.X
    use_reduction (default: True)
        Whether to use a reduction (True) (highly recommended - accurate & much faster)
        or to use the direct matrix (False) for computing distance.
    reduction_slot (default: "X_pca")
        If reduction is TRUE, then specify which slot for the reduction to use.
    key_added (default: "corr_dist")
        Slot in obsp to store the resulting distance matrix.

    Returns
    -------
    Adds fields to the input adata, such that it contains a distance matrix
    stored in adata.obsp[key_added].
    """
    # returns if isinstance(adata, np.ndarray) or isinstance(adata, pd.DataFrame):
    return _corr_distance(adata, use_reduction, reduction_slot, key_added)

# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# ---------------------------- ** KNN ARRAY FUNC ** ----------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
def neighbors_knn(adata,
                  max_knn=101,
                  dist_slot="corr_dist",
                  key_added="knn",
                  njobs = 1):
    """\
    A tool for computing a KNN array used to then rapidly generate connectivity
    graphs with acdc.pp.neighbors_graph for clustering.

    Parameters
    ----------
    adata
        An anndata object containing a distance object in adata.obsp.
    max_knn (default: 101)
        The maximum number of k-nearest neighbors (knn) to include in this array.
        acdc.pp.neighbors_graph will only be able to compute KNN graphs with
        knn <= max_knn.
    dist_slot (default: "corr_dist")
        The slot in adata.obsp where the distance object is stored. One way of
        generating this object is with adata.pp.corr_distance.
    key_added (default: "knn")
        Slot in uns to store the resulting knn array.
    njobs (default: 1)
        Paralleization option that allows users to speed up runtime.

    Returns
    -------
    Adds fields to the input adata, such that it contains a knn array stored in
    adata.uns[key_added].
    """
    # returns if isinstance(adata, np.ndarray) or isinstance(adata, pd.DataFrame):
    return _neighbors_knn(adata, max_knn, dist_slot, key_added, njobs)


# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# -------------------------- ** NEIGHBOR GRAPH FUNC ** -------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
def neighbors_graph(adata,
                    n_neighbors=15,
                    knn_slot='knn'):
    """\
    A tool for rapidly computing a k-nearest neighbor (knn) graph (i.e.
    connectivities) that can then be used for clustering.

    graphs with acdc.pp.neighbors_graph for clustering.

    Parameters
    ----------
    adata
        An anndata object containing a distance object in adata.obsp.
    n_neighbors (default: 15)
        The number of nearest neighbors to use to build the connectivity graph.
        This number must be less than the total number of knn in the knn array
        stored in adata.uns[knn_slot].
    knn_slot (default: 101)
        The slot in adata.uns where the knn array is stored. One way of
        generating this object is with acdc.pp.neighbors_knn.

    Returns
    -------
    Adds fields to the input adata, such that it contains a knn graph stored in
    adata.obsp['connectivities'] along with metadata in adata.uns["neighbors"].
    """
    # returns if isinstance(adata, np.ndarray) or isinstance(adata, pd.DataFrame):
    return _neighbors_graph(adata, n_neighbors, knn_slot)
