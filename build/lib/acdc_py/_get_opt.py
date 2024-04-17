import numpy as np
from .tl import cluster_final
from ._SA_GS_subfunctions import get_approx_anndata
from ._condense_diffuse_funcs import __diffuse_subsample_labels

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
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    dist_slot=None,
    use_reduction=True,
    reduction_slot="X_pca",
    clust_alg = "Leiden",
    n_clusts = None,
    seed = 0,
    approx = {
        "run":False,
        "size":1000,
        "exact_size":True
    },
    key_added = "clusters",
    knn_slot = 'knn',
    verbose = True,
    njobs = 1,
    mode = "GS"
):
    if "size" not in approx.keys(): approx["size"] = 1000
    if "exact_size" not in approx.keys(): approx["exact_size"] = False

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

    if approx["run"] is True: adata = get_approx_anndata(adata, approx, seed, verbose, njobs)

    adata = cluster_final(adata,
                          res = opt_params["opt_res"],
                          knn = opt_params["opt_knn"],
                          dist_slot = dist_slot,
                          use_reduction = use_reduction,
                          reduction_slot = reduction_slot,
                          clust_alg = clust_alg,
                          seed = seed,
                          approx = {"run":False},
                          key_added = key_added,
                          knn_slot = knn_slot,
                          verbose = verbose,
                          njobs = njobs)
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
    return adata


# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# -------------------------- ** OPTIMMIZATION FUNCS ** -------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-

def _SA_clustering(
    adata,
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    dist_slot=None,
    use_reduction=True,
    reduction_slot="X_pca",
    clust_alg = "Leiden",
    n_clusts = None,
    seed = 0,
    approx = {
        "run":False,
        "size":1000,
        "exact_size":True
    },
    key_added = "clusters",
    knn_slot = 'knn',
    verbose = True,
    njobs = 1
):
    _generic_clustering(
        adata,
        opt_metric,
        opt_metric_dir,
        dist_slot,
        use_reduction,
        reduction_slot,
        clust_alg,
        n_clusts,
        seed,
        approx,
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
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    dist_slot=None,
    use_reduction=True,
    reduction_slot="X_pca",
    clust_alg = "Leiden",
    n_clusts = None,
    seed = 0,
    approx = {
        "run":False,
        "size":1000,
        "exact_size":True
    },
    key_added = "clusters",
    knn_slot = 'knn',
    verbose = True,
    njobs = 1
):
    _generic_clustering(
        adata,
        opt_metric,
        opt_metric_dir,
        dist_slot,
        use_reduction,
        reduction_slot,
        clust_alg,
        n_clusts,
        seed,
        approx,
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
