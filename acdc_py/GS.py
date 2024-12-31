### ---------- IMPORT DEPENDENCIES ----------
from ._SA_GS_subfunctions import *
from ._SA_GS_subfunctions import _cluster_adata
from ._SA_GS_subfunctions import __merge_subclusters_into_clusters
from ._condense_diffuse_funcs import __diffuse_subsample_labels
from ._tl import _cluster_final_internal, _extract_clusters
from ._pp import _corr_distance, _neighbors_knn, _neighbors_graph
from tqdm import tqdm
from datetime import datetime
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from .config import config

### ---------- EXPORT LIST ----------
__all__ = ['GS']

# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# --------------------------- ** GRIDSEARCH FUNCS ** ---------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
def get_results_for_knn(
    a_nn,
    adata,
    NN_vector,
    res_vector,
    n_pcs,
    metrics,
    seed,#my_random_seed,
    clust_alg,
    SS_weights,
    SS_exp_base,
    key_added,
    n_subsamples,
    subsamples_pct_cells,
    dist_slot,
    batch_size
):
    n_iters = len(res_vector)*n_subsamples
    sil_df = getEmptySilDF(n_iters, metrics)
    curr_iter = np.where(np.isin(NN_vector, a_nn))[0][0] * len(res_vector)
    sil_df.index = np.arange(n_iters) + curr_iter

    _neighbors_graph(
        adata,
        n_neighbors = a_nn,
        batch_size=batch_size,
        verbose = False
    )
    for a_res in res_vector:
        adata = _cluster_adata(adata,
                               seed,#my_random_seed,
                               a_res,
                               clust_alg,
                               key_added)
        # run 100 times, change the seed
        silhouette_avgs = []
        for i in range(1,n_subsamples+1):
            sil_df = add_clustering_results_to_sil_df_using_subsampling(
                               adata.obsp[dist_slot],
                               adata,
                               i,
                               subsamples_pct_cells,
                               sil_df,
                               n_pcs,
                               a_res,
                               a_nn,
                               SS_weights,
                               SS_exp_base,
                               curr_iter,
                               metrics,
                               update_method="loc",
                               key_added = key_added
            )
            curr_iter = curr_iter + 1
    return sil_df

def get_gs_results_1core(
    adata,
    NN_vector,
    res_vector,
    n_pcs,
    metrics,
    seed,#my_random_seed,
    clust_alg,
    SS_weights,
    SS_exp_base,
    key_added,
    n_subsamples,
    subsamples_pct_cells,
    dist_slot,
    batch_size,
    show_progress_bar,
    verbose
):
    n_iters = len(NN_vector)*len(res_vector)*n_subsamples
    sil_df = getEmptySilDF(n_iters, metrics)
    curr_iter = 0
    if verbose: print("Beginning GridSearch clustering...")

    if show_progress_bar: pbar = tqdm(desc = "GridSearch", total = n_iters, position=0, leave=True)
    for a_nn in NN_vector:
        _neighbors_graph(
            adata,
            n_neighbors = a_nn,
            batch_size=batch_size,
            verbose = False
        )
        for a_res in res_vector:
            adata = _cluster_adata(adata,
                                   seed,#my_random_seed,
                                   a_res,
                                   clust_alg,
                                   key_added)
            # run 100 times, change the seed
            silhouette_avgs = []
            for i in range(1,n_subsamples+1):
                sil_df = add_clustering_results_to_sil_df_using_subsampling(
                                   adata.obsp[dist_slot],
                                   adata,
                                   i,
                                   subsamples_pct_cells,
                                   sil_df,
                                   n_pcs,
                                   a_res,
                                   a_nn,
                                   SS_weights,
                                   SS_exp_base,
                                   curr_iter,
                                   metrics,
                                   update_method="loc",
                                   key_added = key_added
                )
                if show_progress_bar: pbar.update(1)
                curr_iter = curr_iter + 1
    if show_progress_bar: pbar.close()
    return sil_df

def get_gs_results_multicore(
    adata,
    NN_vector,
    res_vector,
    n_pcs,
    metrics,
    seed,#my_random_seed,
    clust_alg,
    SS_weights,
    SS_exp_base,
    key_added,
    n_subsamples,
    subsamples_pct_cells,
    dist_slot,
    batch_size,
    verbose,
    njobs
):
    if verbose: print("Beginning GridSearch clustering...")
    sil_df = Parallel(njobs)(
        delayed(get_results_for_knn)(
            a_nn,
            adata,
            NN_vector,
            res_vector,
            n_pcs,
            metrics,
            seed,#my_random_seed,
            clust_alg,
            SS_weights,
            SS_exp_base,
            key_added,
            n_subsamples,
            subsamples_pct_cells,
            dist_slot,
            batch_size
        ) for a_nn in NN_vector
    )
    sil_df = pd.concat(sil_df)
    return sil_df

def get_gs_results(
    adata,
    res_vector=np.arange(0.1, 2, 0.2),
    NN_vector=np.arange(11, 102, 10),
    dist_slot=None,
    use_reduction=True,
    reduction_slot="X_pca",
    metrics="sil_mean",
    SS_weights="unitary",
    SS_exp_base=2.718282,
    verbose=True,
    show_progress_bar=True,
    clust_alg="Leiden",
    n_subsamples=1,
    subsamples_pct_cells=100,
    seed = 0,
    key_added = "clusters",
    batch_size = 1000,
    njobs = 1
):
    # Consistency so that metrics is a list
    if isinstance(metrics, str): metrics = [metrics]

    if dist_slot is None:
        if verbose: print("Computing distance object...")
        dist_slot = "corr_dist"
        _corr_distance(adata,
                       use_reduction,
                       reduction_slot,
                       key_added=dist_slot,
                       batch_size=batch_size,
                       dtype=config['corr_distance_dtype'],
                       verbose=verbose)

    if use_reduction == True:
        n_pcs = adata.obsm[reduction_slot].shape[1]
    else:
        n_pcs = None

    # ---------------- SUBSAMPLING HERE ----------------
    if verbose: print("Computing neighbors...")
    _neighbors_knn(
        adata,
        max_knn=np.max(NN_vector),
        dist_slot=dist_slot,
        batch_size=batch_size,
        verbose=verbose,
        njobs = njobs
    )

    if njobs == 1:
        sil_df = get_gs_results_1core(
            adata,
            NN_vector,
            res_vector,
            n_pcs,
            metrics,
            seed,#my_random_seed,
            clust_alg,
            SS_weights,
            SS_exp_base,
            key_added,
            n_subsamples,
            subsamples_pct_cells,
            dist_slot,
            batch_size,
            show_progress_bar,
            verbose
        )
    else:
        sil_df = get_gs_results_multicore(
            adata,
            NN_vector,
            res_vector,
            n_pcs,
            metrics,
            seed,#my_random_seed,
            clust_alg,
            SS_weights,
            SS_exp_base,
            key_added,
            n_subsamples,
            subsamples_pct_cells,
            dist_slot,
            batch_size,
            verbose,
            njobs
        )

    sil_df["resolution"] = np.around(sil_df["resolution"].astype(np.double),3)#prevent 0.3 being 0.300000000004

    run_params = {
        "res_vector": res_vector,
        "NN_vector": NN_vector,
        "use_reduction": use_reduction,
        "reduction_slot": reduction_slot,
        "SS_weights": SS_weights,
        "SS_exp_base": SS_exp_base,
        "clust_alg": clust_alg,
        "n_subsamples": n_subsamples,
        "subsamples_pct_cells": subsamples_pct_cells,
    }
    gs_results = {
        "search_df": sil_df,
        "run_params": run_params
    }
    adata.uns["GS_results_dict"] = gs_results
    return(adata)

def get_opt_res_knn_from_gs_results(
    gs_results,
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    n_clusts = None
):
    search_df = gs_results["search_df"]
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

    opt_res = search_df_opt_row["resolution"]
    opt_knn = int(search_df_opt_row["knn"])
    opt_params = {"opt_res": opt_res, "opt_knn": opt_knn}
    return(opt_params)

# -------------------------- ** MAIN RUN FUNCTION ** ---------------------------
def GS(
    adata,
    res_vector = np.arange(0.1, 2, 0.2),
    NN_vector = np.arange(11, 102, 10),
    dist_slot = None,
    use_reduction=True,
    reduction_slot="X_pca",
    # clust_alg = "Leiden",
    metrics = "sil_mean", #["sil_mean", "sil_mean_median", "tot_sil_neg", "lowest_sil_clust","max_sil_clust"]
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    cluster_labels = None,
    cluster_name = None,
    # SS_weights = "unitary",
    # SS_exp_base = 2.718282,
    # n_subsamples = 1,
    # subsamples_pct_cells = 100,
    seed = 0,
    key_added = "clusters",
    approx_size = None,
    verbose = True,
    show_progress_bar = True,
    batch_size = 1000,
    njobs = 1
):
    """\
    A tool for the optimization-based unsupervised clustering of large-scale
    data. Grid Search (GS) allows for deterministic optimization of several
    variables—Nearest Neighbors and resolution–with several objective
    functions—e.g. Silhouette Score. An approximation method we call subsampling
    and diffusion is included to allow fast and accurate clustering of hundreds
    of thousands of cells.

    Parameters
    ----------
    adata
        An anndata object containing a gene expression signature in adata.X and
        gene expression counts in adata.raw.X.
    res_vector : default: np.arange(0.1, 2, 0.2)
         sequence of values of the resolution parameter.
    NN_vector : default: np.arange(11, 102, 10)
         sequence of values for the number of nearest neighbors.
    dist_slot : default: None
        Slot in adata.obsp where a pre-generated distance matrix computed across
        all cells is stored in adata for use in construction of NN. (Default =
        None, i.e. distance matrix will be automatically computed as a
        correlation distance and stored in "corr_dist").
    use_reduction : default: True
        Whether to use a reduction (True) (highly recommended - accurate & much
        faster) or to use the direct matrix (False) for clustering.
    reduction_slot : default: "X_pca"
        If reduction is TRUE, then specify which slot for the reduction to use.
    metrics : default: "sil_mean"
        A metric or a list of metrics to be computed at each iteration of the
        GridSearch. Possible metrics to use include "sil_mean",
        "sil_mean_median", "tot_sil_neg", "lowest_sil_clust", "max_sil_clust",
        "ch" and "db".
    opt_metric : default: "sil_mean"
        A metric from metrics to use to optimize parameters for the clustering.
    opt_metric_dir : default: "max"
        Whether opt_metric is more optimal by maximizing ("max") or
        by minimizing ("min").
    cluster_labels : default: None
        A column in adata.obs with a set of cluster labels containing a
        cluster to subcluster. Specify the cluster with the cluster_name
        parameter.
    cluster_name : default: None
        A cluster from cluster_labels to subcluster. When None, cluster whole
        dataset.
    seed : default: 0
        Random seed to use.
    key_added : default: "clusters"
        Slot in obs to store the resulting clusters.
    approx_size : default: None
        When set to a positive integer, instead of running GS on the entire
        dataset, perform GS on a subsample and diffuse those results. This will
        lead to an approximation of the optimal solution for cases where the
        dataset is too large to perform GS on due to time or memory constraints.
    verbose : default: True
        Include additional output with True. Alternative = False.
    show_progress_bar : default: True
        Show a progress bar to visualize the progress of the algorithm.
    batch_size : default: 1000
        The size of each batch. Larger batches result in more memory usage. If
        None, use the whole dataset instead of batches.
    njobs : default: 1
        Paralleization option that allows users to speed up runtime.
    Returns
    -------
    A object of :class:~anndata.Anndata containing a clustering vector
    "clusters" in the .obs slot and a dictionary "GS_results_dict" with
    information on the run in the .uns slot.
    """
    if isinstance(NN_vector, int): NN_vector = np.array([NN_vector])
    if isinstance(res_vector, int): res_vector = np.array([res_vector])

    if opt_metric not in metrics:
        raise ValueError("opt_metric (" + str(opt_metric) + ") is missing from metrics.")
    n_max_cores = cpu_count()
    if njobs > n_max_cores:
        raise ValueError('njobs (' + str(njobs) + ') is larger than the ' +
                         'number of CPU cores (' + str(n_max_cores) + ').')

    if cluster_name is not None:
        if cluster_labels is None:
            raise ValueError("cluster_name provided but not cluster_labels.")
        else:
            adata_original = adata
            adata = _extract_clusters(
                adata_original, cluster_labels, cluster_name
            )

    if approx_size is None:
        approx = {"run":False}
    else:
        approx = {"run":True, "size":approx_size, "exact_size":True}
        adata = get_approx_anndata(adata, approx, seed, verbose, njobs)

    adata = get_gs_results(
        adata,
        res_vector,
        NN_vector,
        dist_slot,
        use_reduction,
        reduction_slot,
        metrics,
        config['SS']['SS_weights'],
        config['SS']['SS_exp_base'],
        verbose,
        show_progress_bar,
        config['clust_alg'],
        config['SS']['n_subsamples'],
        config['SS']['subsamples_pct_cells'],
        seed,
        key_added,
        batch_size,
        njobs
    )

    opt_params = get_opt_res_knn_from_gs_results(
        adata.uns['GS_results_dict'],
        opt_metric,
        opt_metric_dir,
        n_clusts = None
    )
    adata = _cluster_final_internal(
        adata,
        opt_params["opt_res"],
        opt_params["opt_knn"],
        dist_slot,
        config['clust_alg'],
        seed,
        approx = {"run":False},
        key_added = key_added,
        knn_slot = "knn",
        verbose = False,
        batch_size = batch_size,
        njobs = njobs
    )

    if approx["run"] is True:
        if verbose is True: print("Diffusing clustering results...")
        adata = __diffuse_subsample_labels(
            adata,
            res = opt_params["opt_res"],
            knn = opt_params["opt_knn"],
            dist_slot = dist_slot,
            use_reduction = use_reduction,
            reduction_slot = reduction_slot,
            key_added = key_added,
            knn_slot = 'knn',
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
        adata_original.uns['GS_results_dict'] = adata.uns['GS_results_dict']
