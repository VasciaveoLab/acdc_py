import numpy as np
from ._tl import _extract_clusters, _cluster_final
from ._SA_GS_subfunctions import get_approx_anndata
from ._condense_diffuse_funcs import __diffuse_subsample_labels
from ._SA_GS_subfunctions import __merge_subclusters_into_clusters

### ---------- EXPORT LIST ----------
__all__ = []

# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# --------------------------- ** HELPER FUNCTIONS ** ---------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-

def _results_metric_search_data(
    results,
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    n_clusts = None
):
    search_df = results["search_df"]
    if n_clusts is not None:
        n_clusts_unique = np.unique(search_df['n_clust'].values).astype(int)
        if not np.isin(n_clusts, n_clusts_unique):
            n_clusts_options = ", ".join(n_clusts_unique.astype('str'))
            raise ValueError(
                "A solution with " + \
                str(n_clusts) + \
                " is not available. " + \
                "Choose from the following n_clusts:\n\t" + \
                str(n_clusts_options) + \
                "."
            )
        search_df = search_df[search_df['n_clust'] == n_clusts]

    if(opt_metric_dir == "max"):
        max_opt_metric = np.nanmax(search_df[opt_metric])
        search_df_opt_row = search_df[search_df[opt_metric] >= max_opt_metric].iloc[0]
    elif(opt_metric_dir == "min"):
        min_opt_metric = np.nanmin(search_df[opt_metric])
        search_df_opt_row = search_df[search_df[opt_metric] <= min_opt_metric].iloc[0]
    else:
        ValueError('Unsupported opt_metric_dir:' + str(opt_metric_dir) +
                    '\n\t opt_metric_dir must be "max" or "min".')
    return search_df_opt_row

def _generic_clustering(
    adata,
    dist_slot=None,
    use_reduction=True,
    reduction_slot="X_pca",
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    cluster_labels = None,
    cluster_name = None,
    n_clusts = None,
    seed = 0,
    approx_size = None,
    key_added = "clusters",
    knn_slot = 'knn',
    verbose = True,
    njobs = 1,
    mode = "GS"
):
    if approx_size is None:
        approx = {"run":False}
    else:
        approx = {"run":True, "size":approx_size, "exact_size":True}

    if mode == "GS":
        opt_params = _GS_params(
            adata,
            opt_metric,
            opt_metric_dir,
            n_clusts
        )
    else:
        opt_params = _SA_params(
            adata,
            opt_metric,
            opt_metric_dir,
            n_clusts
        )

    if cluster_name is not None:
        if cluster_labels is None:
            raise ValueError("cluster_name provided but not cluster_labels.")
        else:
            adata_original = adata
            adata = _extract_clusters(
                adata_original, cluster_labels, cluster_name
            )

    if approx["run"] is True:
        adata = get_approx_anndata(adata, approx, seed, verbose, njobs)

    adata = _cluster_final(
        adata,
        res = opt_params["opt_res"],
        knn = opt_params["opt_knn"],
        dist_slot = dist_slot,
        use_reduction = use_reduction,
        reduction_slot = reduction_slot,
        seed = seed,
        approx_size = approx_size,
        key_added = key_added,
        knn_slot = knn_slot,
        verbose = verbose,
        njobs = njobs
    )
    if dist_slot is None: dist_slot = "corr_dist"

    if approx["run"] is True:
        if verbose is True: print("Diffusing clustering results...")
        adata = __diffuse_subsample_labels(
            adata,
            res = opt_params["opt_res"],
            knn = opt_params["opt_knn"],
            dist_slot = dist_slot,
            key_added = key_added,
            knn_slot = knn_slot,
            verbose = verbose,
            seed = seed,
            njobs = njobs
        )

    if cluster_name is not None:
        merged_clusters = __merge_subclusters_into_clusters(
            cluster_labels = adata_original.obs[cluster_labels].values,
            subcluster_labels = adata.obs[key_added].values,
            subcluster_name = cluster_name
        )
        adata_original.obs[key_added] = merged_clusters

# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# -------------------------- ** OPTIMMIZATION FUNCS ** -------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-

def _SA_clustering(
    adata,
    dist_slot=None,
    use_reduction=True,
    reduction_slot="X_pca",
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    cluster_labels = None,
    cluster_name = None,
    n_clusts = None,
    seed = 0,
    approx_size = None,
    key_added = "clusters",
    knn_slot = 'knn',
    verbose = True,
    njobs = 1
):
    _generic_clustering(
        adata,
        dist_slot,
        use_reduction,
        reduction_slot,
        opt_metric,
        opt_metric_dir,
        cluster_labels,
        cluster_name,
        n_clusts,
        seed,
        approx_size,
        key_added,
        knn_slot,
        verbose,
        njobs,
        mode = "SA"
    )

def _SA_params(
    adata,
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    n_clusts = None
):
    search_df_opt_row = _SA_metric_search_data(
        adata,
        opt_metric,
        opt_metric_dir,
        n_clusts
    )
    opt_res = search_df_opt_row["resolution"]
    opt_knn = int(search_df_opt_row["knn"])
    opt_params = {"opt_res": opt_res, "opt_knn": opt_knn}
    return opt_params

def _SA_metric_value(
    adata,
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    n_clusts = None
):
    search_df_opt_row = _SA_metric_search_data(
        adata,
        opt_metric,
        opt_metric_dir,
        n_clusts
    )
    return search_df_opt_row[opt_metric]

def _SA_metric_search_data(
    adata,
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    n_clusts = None
):
    results = adata.uns['SA_results_dict']
    search_df_opt_row = _results_metric_search_data(
        results,
        opt_metric,
        opt_metric_dir,
        n_clusts
    )
    return search_df_opt_row

def _GS_clustering(
    adata,
    dist_slot=None,
    use_reduction=True,
    reduction_slot="X_pca",
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    cluster_labels = None,
    cluster_name = None,
    n_clusts = None,
    seed = 0,
    approx_size = None,
    key_added = "clusters",
    knn_slot = 'knn',
    verbose = True,
    njobs = 1
):
    _generic_clustering(
        adata,
        dist_slot,
        use_reduction,
        reduction_slot,
        opt_metric,
        opt_metric_dir,
        cluster_labels,
        cluster_name,
        n_clusts,
        seed,
        approx_size,
        key_added,
        knn_slot,
        verbose,
        njobs,
        mode = "GS"
    )

def _GS_params(
    adata,
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    n_clusts = None
):
    search_df_opt_row = _GS_metric_search_data(
        adata,
        opt_metric,
        opt_metric_dir,
        n_clusts
    )
    opt_res = search_df_opt_row["resolution"]
    opt_knn = int(search_df_opt_row["knn"])
    opt_params = {"opt_res": opt_res, "opt_knn": opt_knn}
    return opt_params

def _GS_metric_value(
    adata,
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    n_clusts = None
):
    search_df_opt_row = _GS_metric_search_data(
        adata,
        opt_metric,
        opt_metric_dir,
        n_clusts
    )
    return search_df_opt_row[opt_metric]

def _GS_metric_search_data(
    adata,
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    n_clusts = None
):
    results = adata.uns['GS_results_dict']
    search_df_opt_row = _results_metric_search_data(
        results,
        opt_metric,
        opt_metric_dir,
        n_clusts
    )
    return search_df_opt_row
