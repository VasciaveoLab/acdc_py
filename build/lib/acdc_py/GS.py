### ---------- IMPORT DEPENDENCIES ----------
from ._SA_GS_subfunctions import *
from ._SA_GS_subfunctions import _cluster_adata
from ._condense_diffuse_funcs import __diffuse_subsample_labels
from ._tl import _cluster_final_internal
from .pp import corr_distance, neighbors_knn, neighbors_graph
from tqdm import tqdm
from datetime import datetime
from multiprocessing import cpu_count
from joblib import Parallel, delayed

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
    dist_slot
):
    n_iters = len(res_vector)*n_subsamples
    sil_df = getEmptySilDF(n_iters, metrics)
    curr_iter = np.where(np.isin(NN_vector, a_nn))[0][0] * len(res_vector)
    sil_df.index = np.arange(n_iters) + curr_iter

    neighbors_graph(adata, n_neighbors = a_nn)
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
    show_progress_bar,
    verbose
):
    n_iters = len(NN_vector)*len(res_vector)*n_subsamples
    sil_df = getEmptySilDF(n_iters, metrics)
    curr_iter = 0
    if verbose: print("Beginning GridSearch clustering...")

    if show_progress_bar: pbar = tqdm(desc = "GridSearch", total = n_iters, position=0, leave=True)
    for a_nn in NN_vector:
        neighbors_graph(adata, n_neighbors = a_nn)
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
            dist_slot
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
    njobs = 1
):
    # Consistency so that metrics is a list
    if isinstance(metrics, str): metrics = [metrics]

    if dist_slot is None:
        if verbose: print("Computing distance object...")
        dist_slot = "corr_dist"
        corr_distance(adata,
                      use_reduction,
                      reduction_slot,
                      key_added=dist_slot)

    if use_reduction == True:
        n_pcs = adata.obsm[reduction_slot].shape[1]
    else:
        n_pcs = None

    # ---------------- SUBSAMPLING HERE ----------------
    if verbose: print("Computing neighbors...")
    neighbors_knn(adata, max_knn=np.max(NN_vector), dist_slot=dist_slot, njobs = njobs)

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
    clust_alg = "Leiden",
    metrics = "sil_mean", #["sil_mean", "sil_mean_median", "tot_sil_neg", "lowest_sil_clust","max_sil_clust"]
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    SS_weights = "unitary",
    SS_exp_base = 2.718282,
    n_subsamples = 1,
    subsamples_pct_cells = 100,
    seed = 0,
    key_added = "clusters",
    approx = {
        "run":False,
        "size":1000,
        "exact_size":True
    },
    verbose = True,
    show_progress_bar = True,
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
    res_vector (default: np.arange(0.1, 2, 0.2))
         sequence of values of the resolution parameter.
    NN_vector (default: np.arange(11, 102, 10))
         sequence of values for the number of nearest neighbors.
    dist_slot (default: None)
        Slot in adata.obsp where a pre-generated distance matrix computed across
        all cells is stored in adata for use in construction of NN. (Default =
        None, i.e. distance matrix will be automatically computed as a
        correlation distance and stored in "corr_dist").
    use_reduction (default: True)
        Whether to use a reduction (True) (highly recommended - accurate & much faster)
        or to use the direct matrix (False) for clustering.
    reduction_slot (default: "X_pca")
        If reduction is TRUE, then specify which slot for the reduction to use.
    clust_alg (default: "Leiden")
        Clustering algorithm. Choose among: "Leiden" (default) or  "Louvain".
    metrics (default: "sil_mean")
        A metric or a list of metrics to be computed at each iteration of the
        GridSearch. Possible metrics to use include "sil_mean", "sil_mean_median",
        "tot_sil_neg", "lowest_sil_clust", "max_sil_clust", "ch" and "db".
    opt_metric (default: "sil_mean")
        A metric from metrics to use to optimize parameters for the clustering.
    opt_metric_dir (default: "max")
        Whether opt_metric is more optimal by maximizing ("max") or
        by minimizing ("min").
    SS_weights (default: "unitary")
        Negative silhouette scores can be given more weight by exponentiation ("exp").
        Otherwise, leave SS_weights as "unitary".
    SS_exp_base (default: 2.718282.)
        If SS_weights is set to "exp", then set the base for exponentiation.
    n_subsamples (default=1)
        Number of subsamples per bootstrap.
    subsamples_pct_cells (default: 100)
        Percentage of cells sample at each bootstrap iteration.
        i.e. when 100, 100%, all cells are used).
    seed (default: 0)
        Random seed to use.
    key_added (default: "clusters")
        Slot in obs to store the resulting clusters.
    approx (default: {"run":False, "size":1000, "exact_size":True})
        A diciontary object containing three parameters to control subsampling and diffusion
            "run": True or False whether to use subsampling and diffusion. Default=False
            "size": the number of cells to use in the subsampling. Default=1000.
            "exact_size": whether to get the exact size "size" of subsampling (True) or
            be more inclusive during the representative subsampling (False, recommended).
    verbose (default: True)
        Include additional output with True. Alternative = False.
    show_progress_bar (default: True)
        Show a progress bar to visualize the progress of the algorithm.
    njobs (default: 1)
        Paralleization option that allows users to speed up runtime.
    Returns
    -------
    A object of :class:~anndata.Anndata containing a clustering vector
    "clusters" in the .obs slot and a dictionary "GS_results_dict" with
    information on the run in the .uns slot.
    """
    if opt_metric not in metrics:
        raise ValueError("opt_metric (" + str(opt_metric) + ") is missing from metrics.")
    n_max_cores = cpu_count()
    if njobs > n_max_cores:
        raise ValueError('njobs (' + str(njobs) + ') is larger than the ' +
                         'number of CPU cores (' + str(n_max_cores) + ').')

    if approx is False: approx = {"run":False}
    if approx is True: approx = {"run":False}
    if "size" not in approx.keys(): approx["size"] = 1000
    if "exact_size" not in approx.keys(): approx["exact_size"] = False

    if approx["run"] is True: adata = get_approx_anndata(adata, approx, seed, verbose, njobs)

    adata = get_gs_results(
        adata,
        res_vector,
        NN_vector,
        dist_slot,
        use_reduction,
        reduction_slot,
        metrics,
        SS_weights,
        SS_exp_base,
        verbose,
        show_progress_bar,
        clust_alg,
        n_subsamples,
        subsamples_pct_cells,
        seed,
        key_added,
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
        clust_alg,
        seed,
        approx = {"run":False},
        key_added = key_added,
        knn_slot = "knn",
        verbose = False,
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
