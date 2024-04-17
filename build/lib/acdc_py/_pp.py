### ---------- IMPORT DEPENDENCIES ----------
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed

### ---------- EXPORT LIST ----------
__all__ = []

# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# ---------------------------- ** DISTANCE FUNCS ** ----------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# SA_GS_subfunctions.R
def _corr_distance_matrix(data):
    # Equivalent to the following in R: d = sqrt(1 - stats::corr(X))
    # Computing the correlation
    corr_matrix = np.corrcoef(data)
    # Calculating sqrt_one_minus_corr_matrix
    sqrt_one_minus_corr_matrix = np.sqrt(1 - corr_matrix)
    # Ensuring diagonal contains 0 values
    np.fill_diagonal(sqrt_one_minus_corr_matrix, 0)
    return(sqrt_one_minus_corr_matrix)

def __add_row_column_names_to_dist_mat(dist_mat, adata):
    # Turn into a dataframe with row and column names
    df_dist = pd.DataFrame(
        dist_mat,
        columns = adata.obs_names,
        index = adata.obs_names
    )
    return(df_dist)

def _corr_distance(adata,
                   use_reduction=True,
                   reduction_slot="X_pca",
                   key_added="corr_dist"):
    if isinstance(adata, np.ndarray) or isinstance(adata, pd.DataFrame):
        return _corr_distance_matrix(adata)

    if use_reduction == False:
        # use original features
        d = _corr_distance_matrix(adata.X)
    elif use_reduction == True:
        # use principal components
        X = adata.obsm[reduction_slot]
        d = _corr_distance_matrix(X)
    else:
        raise ValueError("reduction must be logical.")
    d = __add_row_column_names_to_dist_mat(d, adata)
    adata.obsp[key_added] = d

# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# ---------------------------- ** KNN ARRAY FUNC ** ----------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-

def __rank_column(column): return rankdata(column, method='ordinal')

def __get_top_n_indices(dist_array_ranked, knn, njobs = 1):
    # Sort each row of the input array and get the indices of the sorted elements
    if njobs == 1:
        sorted_indices = np.argsort(dist_array_ranked, axis=1)
    else:
        sorted_indices = np.array(Parallel(n_jobs=njobs)(
            delayed(np.argsort)(row) for row in dist_array_ranked
        ))

    # Slice the sorted indices to keep only the top knn indices for each row
    top_n_indices = sorted_indices[:, :knn]

    return top_n_indices

def _get_knn_array(dist_df, max_knn, njobs = 1):
    # For each sample, sort neighbors by distance.
    if njobs ==1:
        dist_array_ranked = np.apply_along_axis(
            __rank_column,
            axis=1,
            arr=np.array(dist_df)
        )
    else:
        dist_array_ranked = np.array(Parallel(n_jobs=njobs)(
            delayed(__rank_column)(column) for column in dist_df
        ))

    # Trim this ranking to MaxNN
    max_knn_sorted_indices_array = __get_top_n_indices(dist_array_ranked, max_knn, njobs)

    return max_knn_sorted_indices_array

# ------------------------------ ** HELPER FUNC ** -----------------------------
def _neighbors_knn(adata,
                   max_knn=101,
                   dist_slot="corr_dist",
                   key_added="knn",
                   njobs = 1):
    if isinstance(adata, np.ndarray) or isinstance(adata, pd.DataFrame):
        return _get_knn_array(dist_df = adata, max_knn = max_knn, njobs = njobs)

    # Get the distance DataFrame
    dist_df = adata.obsp[dist_slot]

    # Comptue the KNN array
    max_knn_sorted_indices_array = _get_knn_array(dist_df, max_knn, njobs)

    # Add this to the adata
    adata.uns[key_added] = max_knn_sorted_indices_array

# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# -------------------------- ** NEIGHBOR GRAPH FUNC ** -------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-

# ------------------------------ ** HELPER FUNC ** -----------------------------

def __get_connectivity_graph(max_knn_sorted_indices_array, knn):
    num_rows = max_knn_sorted_indices_array.shape[0]

    # Create a square array A filled with zeros
    A = np.zeros((num_rows, num_rows), dtype=int)

    # Slice max_knn_sorted_indices_array to keep only the top knn columns
    B = max_knn_sorted_indices_array[:, :knn]

    # Create row indices for indexing A
    row_indices = np.arange(num_rows).reshape(-1, 1)

    # Broadcast row indices to match the shape of B
    row_indices = np.tile(row_indices, (1, knn))

    # Use advanced indexing to set the values in A to 1 based on indices from B
    A[row_indices, B] = 1

    return A

# ------------------------------- ** MAIN FUNC ** ------------------------------
def _neighbors_graph(adata,
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
    if isinstance(adata, np.ndarray) or isinstance(adata, pd.DataFrame):
        return __get_connectivity_graph(adata, n_neighbors)

    new_graph = __get_connectivity_graph(adata.uns[knn_slot], n_neighbors)
    adata.obsp['connectivities'] = csr_matrix(new_graph)
    # adata.uns["neighbors"] = {'connectivities':adata.obsp['connectivities']}
    adata.uns["neighbors"] = {
        'connectivities_key': 'connectivities',
        'params': {'n_neighbors': n_neighbors}
    }
