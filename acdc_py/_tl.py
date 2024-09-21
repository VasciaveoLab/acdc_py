### ---------- IMPORT DEPENDENCIES ----------
import numpy as np
import pandas as pd
from ._pp import _corr_distance, _neighbors_knn, _neighbors_graph
from ._SA_GS_subfunctions import _cluster_adata
from ._condense_diffuse_funcs import __diffuse_subsample_labels

### ---------- EXPORT LIST ----------
__all__ = []


def _cluster_final_internal(adata,
                            res,
                            knn,
                            dist_slot = None,
                            clust_alg="Leiden",
                            seed=0,
                            approx={
                                "run": False,
                                "size": 1000,
                                "exact_size": False
                            },
                            key_added="clusters",
                            knn_slot='knn',
                            verbose=True,
                            njobs = 1):
    if approx["run"] is True: adata = get_approx_anndata(adata, approx, seed, verbose, njobs)

    if verbose is True: print("Computing neighbor graph with " + str(knn) + " neighbors...")
    if not (knn_slot in adata.uns.keys()):
        _neighbors_knn(
            adata,
            max_knn=knn,
            dist_slot = dist_slot,
            key_added = knn_slot,
            njobs = njobs
        )
    elif not (adata.uns[knn_slot].shape[1] >= knn):
        _neighbors_knn(
            adata,
            max_knn=knn,
            dist_slot = dist_slot,
            key_added = knn_slot,
            njobs = njobs
        )
    _neighbors_graph(
        adata,
        n_neighbors = knn,
        knn_slot = knn_slot,
        verbose = False
    )

    if verbose is True: print("Clustering with resolution " + str(res) + " using " + str(clust_alg) + "...")
    adata = _cluster_adata(adata,
                           seed,#my_random_seed,
                           res,
                           clust_alg,
                           key_added)

    if approx["run"] is True:
        if verbose is True: print("Diffusing clustering results...")
        adata = __diffuse_subsample_labels(
            adata,
            res,
            knn,
            dist_slot,
            use_reduction,
            reduction_slot,
            key_added = key_added,
            knn_slot = knn_slot,
            verbose = verbose,
            seed = seed,
            njobs = njobs)

    return adata

def _cluster_final(adata,
                  res,
                  knn,
                  dist_slot=None,
                  use_reduction=True,
                  reduction_slot="X_pca",
                  clust_alg="Leiden",
                  seed=0,
                  approx={
                      "run": False,
                      "size": 1000,
                      "exact_size": False
                  },
                  key_added="clusters",
                  knn_slot='knn',
                  verbose=True,
                  njobs = 1):
    if dist_slot is None:
        if verbose: print("Computing distance object...")
        dist_slot = "corr_dist"
        _corr_distance(adata,
                      use_reduction,
                      reduction_slot,
                      key_added=dist_slot,
                      verbose=verbose)

    adata = _cluster_final_internal(adata,
                                    res,
                                    knn,
                                    dist_slot,
                                    clust_alg,
                                    seed,
                                    approx,
                                    key_added,
                                    knn_slot,
                                    verbose,
                                    njobs)
    return adata

def _extract_clusters(adata, obs_column, clust_names):
    sample_indices = np.isin(adata.obs[obs_column], clust_names)
    adata_subset = adata[sample_indices].copy()
    return adata_subset

# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# ---------------------------- ** merge: HELPERS ** ----------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-

def __relabel_subcluster_labels_by_group_size(subcluster_labels):
    subcluster_labels_groups_ordered_by_alphabet = \
        np.sort(np.unique(subcluster_labels).astype(int)).astype('str')
    subcluster_labels_groups_ordered_by_cluster_size = pd.DataFrame({
        "names":np.unique(subcluster_labels, return_counts = True)[0],
        "sizes":np.unique(subcluster_labels, return_counts = True)[1]
    }).sort_values('sizes').iloc[::-1]['names'].values

    subcluster_labels_orderedGroups = np.zeros(len(subcluster_labels)).astype('str')

    for i in range(len(subcluster_labels_groups_ordered_by_alphabet)):
        subcluster_labels_orderedGroups[subcluster_labels == \
            subcluster_labels_groups_ordered_by_cluster_size[i]] = \
            subcluster_labels_groups_ordered_by_alphabet[i]
    subcluster_labels = subcluster_labels_orderedGroups
    return subcluster_labels

def __relabel_subcluster_labels_by_incr_ints(subcluster_labels):
    subcluster_labels_groups_ordered_by_alphabet = np.sort(np.unique(subcluster_labels).astype(int)).astype(str)
    n_subclusters = len(np.unique(subcluster_labels))
    for i in range(n_subclusters):
        subcluster_labels[subcluster_labels==subcluster_labels_groups_ordered_by_alphabet[i]] = str(i)
    return subcluster_labels

def __merge_int_labels(
    cluster_labels,
    clust_names,
    merged_clust_name,
    update_numbers
):
    clust_names = np.sort(np.array(clust_names).astype(int)).astype(str)
    cluster_labels = cluster_labels.astype(str)
    min_clust = clust_names[0]

    if merged_clust_name is None:
        cluster_labels[np.isin(cluster_labels, clust_names)] = min_clust

        if update_numbers:
            cluster_labels = __relabel_subcluster_labels_by_group_size(
                cluster_labels
            )
            cluster_labels = __relabel_subcluster_labels_by_incr_ints(
                cluster_labels
            )
    else:
        cluster_labels[np.isin(cluster_labels, clust_names)] = merged_clust_name

    return cluster_labels

def __merge_string_labels(
    cluster_labels,
    clust_names,
    merged_clust_name,
    update_numbers
):
    is_digit_labels = np.all([elem.isdigit() for elem in cluster_labels])
    if is_digit_labels:
        return __merge_int_labels(
            cluster_labels,
            clust_names,
            merged_clust_name,
            update_numbers
        )

    if merged_clust_name is None:
        merged_clust_name = "&".join(np.sort(clust_names))

    max_len_str = np.max(np.vectorize(len)(cluster_labels))
    max_len_str = np.max([max_len_str, len(merged_clust_name)])
    cluster_labels = cluster_labels.astype('<U'+str(max_len_str))

    indices = np.isin(cluster_labels, clust_names)
    cluster_labels[indices] = merged_clust_name

    return cluster_labels

# def __merge_float_labels(cluster_labels, clust_names, update_numbers):
#     return __merge_string_labels(
#         cluster_labels.astype(str), clust_names.astype(str), update_numbers
#     )

# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# ------------------------------ ** merge: MAIN ** -----------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-

def _merge(
    adata,
    obs_column,
    clust_names,
    merged_clust_name = None,
    update_numbers = True,
    key_added = "clusters",
    return_as_series = False
):

    if isinstance(obs_column, str):
        cluster_labels = adata.obs[obs_column].values.copy().astype('str')
    else:
        cluster_labels = obs_column.copy().astype('str')

    if isinstance(clust_names, list):
        clust_names = np.array(clust_names)
    elif not isinstance(clust_names, np.ndarray):
        clust_names = np.array([clust_names])

    if merged_clust_name is not None:
        merged_clust_name = str(merged_clust_name)

    # Check if array is full of integers
    is_all_int = np.issubdtype(cluster_labels.dtype, np.integer)

    # Check if array is full of floats
    is_all_float = np.issubdtype(cluster_labels.dtype, np.floating)

    # Check if array is full of strings
    is_all_string = np.issubdtype(cluster_labels.dtype, np.str_)

    cluster_labels = cluster_labels.astype('str')
    clust_names = clust_names.astype('str')

    if is_all_string:
        cluster_labels = __merge_string_labels(
            cluster_labels,
            clust_names,
            merged_clust_name,
            update_numbers
        )

    elif is_all_float:
        cluster_labels = __merge_string_labels(
            cluster_labels,
            clust_names,
            merged_clust_name,
            update_numbers
        )

    elif is_all_int:
        cluster_labels = __merge_int_labels(
            cluster_labels,
            clust_names,
            merged_clust_name,
            update_numbers
        )

    else:
        raise ValueError("cluster labels must be str, float or int.")

    if return_as_series:
        return pd.Series(cluster_labels, index = adata.obs_names)

    adata.obs[key_added] = cluster_labels
