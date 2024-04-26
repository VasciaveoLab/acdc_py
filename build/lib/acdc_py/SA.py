### ---------- IMPORT DEPENDENCIES ----------
from ._SA_GS_subfunctions import *
from ._SA_GS_subfunctions import _cluster_adata
from ._dual_annealing_with_progress_bar import *
from ._tl import _cluster_final_internal
from .pp import corr_distance, neighbors_knn, neighbors_graph
from ._condense_diffuse_funcs import __diffuse_subsample_labels
from sys import maxsize
from multiprocessing import cpu_count

### ---------- EXPORT LIST ----------
__all__ = ['SA']

# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# ------------------------- ** DUAL ANNEALING FUNCS ** -------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
def get_sa_results(adata,
                   res_range = [0.1, 1.9],
                   NN_range = [11,101],
                   dist_slot = None,
                   use_reduction = True,
                   reduction_slot = "X_pca",
                   metrics = "sil_mean", #["sil_mean", "sil_mean_median", "tot_sil_neg", "lowest_sil_clust","max_sil_clust"]
                   opt_metric = "sil_mean",
                   opt_metric_dir = "max",
                   SS_weights = "unitary",
                   SS_exp_base = 2.718282,
                   verbose = True,
                   show_progress_bar = True,
                   clust_alg = "Leiden",
                   n_subsamples = 1,
                   subsamples_pct_cells=100,
                   maxiter = 20,#20,
                   initial_temp = 5230,
                   restart_temp_ratio = 2e-5,
                   visit = 2.62,
                   accept = -5.0,
                   maxfun = 1e7,
                   seed = 0,
                   key_added = "clusters",
                   njobs = 1):
    # par_init = NULL,
    # control = NULL,
    # lq = 0,
    # rng_seeds = c(1234,0)):
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

    if verbose: print("Computing neighbors...")
    neighbors_knn(adata, max_knn=np.max(NN_range), dist_slot=dist_slot, njobs = njobs)

    # In order to use global inside a nested function, we have to declare a
        # variable with a global keyword inside a nested function
    global sil_df
    global curr_iter
    sil_df = getEmptySilDF(nrow = 0, metrics = metrics)
    curr_iter = 1
    bounds = [res_range, NN_range]

    def objective(v, adata, n_subsamples, subsamples_pct_cells,
                  n_pcs, clust_alg, SS_weights, SS_exp_base,
                  metrics, opt_metric, opt_metric_dir, dist_slot):
        a_res, a_nn = v
        a_nn = int(np.floor(a_nn))
        global sil_df
        global curr_iter
        neighbors_graph(adata, n_neighbors = a_nn)
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
                               update_method = "concat",
                               key_added = key_added
            )
        curr_iter = curr_iter + 1
        opt_metric_i = np.mean(sil_df[opt_metric].tail(n_subsamples).values) #Get the mean silouette score for these subsamples

        if opt_metric_dir == "max":
            if np.isnan(opt_metric_i):
                return 0
            else:
                return(opt_metric_i*-1)
        else:
            if np.isnan(opt_metric_i):
                return maxsize
            else:
                return(opt_metric_i)

    if verbose: print("Beginning Simulated Annealing clustering...")
    # perform the dual annealing search
    # opt_result = dual_annealing(func = objective,
    opt_result = dual_annealing_with_progress_bar(func = objective,
                                bounds = bounds,
                                args = (adata,
                                        n_subsamples,
                                        subsamples_pct_cells,
                                        n_pcs,
                                        clust_alg,
                                        SS_weights,
                                        SS_exp_base,
                                        metrics,
                                        opt_metric,
                                        opt_metric_dir,
                                        dist_slot),
                                maxiter = maxiter,
                                initial_temp = initial_temp,
                                restart_temp_ratio = restart_temp_ratio,
                                visit = visit,
                                accept = accept,
                                maxfun = maxfun,
                                seed = seed,
                                show_progress_bar = show_progress_bar)
    run_params = {
        "res_range": res_range,
        "NN_range": NN_range,
        "use_reduction": use_reduction,
        "reduction_slot": reduction_slot,
        "SS_weights": SS_weights,
        "SS_exp_base": SS_exp_base,
        "clust_alg": clust_alg,
        "n_subsamples": n_subsamples,
        "subsamples_pct_cells": subsamples_pct_cells,
        "maxiter": maxiter,
        "initial_temp": initial_temp,
        "restart_temp_ratio": restart_temp_ratio,
        "visit": visit,
        "accept": accept,
        "maxfun": maxfun,
        "seed": seed
    }
    sa_results = {
        "search_df": sil_df,
        "opt_result": opt_result,
        "run_params": run_params
    }
    adata.uns["SA_results_dict"] = sa_results
    return(adata)

# -------------------------- ** MAIN RUN FUNCTION ** ---------------------------
def SA(
    adata,
    res_range=[0.1, 1.9],
    NN_range=[11, 101],
    dist_slot=None,
    use_reduction=True,
    reduction_slot="X_pca",
    clust_alg="Leiden",
    metrics = "sil_mean", #["sil_mean", "sil_mean_median", "tot_sil_neg", "lowest_sil_clust","max_sil_clust"]
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    SS_weights="unitary",
    SS_exp_base=2.718282,
    n_subsamples=1,
    subsamples_pct_cells=100,
    maxiter=20,#20,
    initial_temp=5230,
    restart_temp_ratio=2e-5,
    visit=2.62,
    accept=-5.0,
    maxfun=1e7,
    seed=0,
    key_added = "clusters",
    approx = {
        "run":False,
        # "mode":"subsample", #subsample, metacell_standard,
        "size":1000,
        "exact_size":True
    },
    verbose=True,
    show_progress_bar = True,
    njobs = 1
):
    """\
    A tool for the optimization-based unsupervised clustering of large-scale
    data. Simulated Annealing (SA) allows for stochastic optimization of several
    variables—Nearest Neighbors and resolution–with several objective
    functions—e.g. Silhouette Score. An approximation method we call subsampling
    and diffusion is included to allow fast and accurate clustering of hundreds
    of thousands of cells.

    Parameters
    ----------
    adata
        An anndata object containing a gene expression signature in adata.X and
        gene expression counts in adata.raw.X.
    res_range (default: [0.1, 1.9])
         edge values of the search space for the resolution parameter.
    NN_range (default: [11, 101])
        edge values of the search space for the nearest neighbors parameter.
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
    maxiter : (default: 20)
        The maximum number of global search iterations. If None, value is 1000.
    minimizer_kwargs : dict, optional
        Extra keyword arguments to be passed to the local minimizer
        (minimize). Some important options could be:
        method for the minimizer method to use and args for
        objective function additional arguments.
    initial_temp : float, optional
        The initial temperature, use higher values to facilitates a wider
        search of the energy landscape, allowing dual_annealing to escape
        local minima that it is trapped in. Default value is 5230. Range is
        (0.01, 5.e4].
    restart_temp_ratio : float, optional
        During the annealing process, temperature is decreasing, when it
        reaches initial_temp * restart_temp_ratio, the reannealing process
        is triggered. Default value of the ratio is 2e-5. Range is (0, 1).
    visit : float, optional
        Parameter for visiting distribution. Default value is 2.62. Higher
        values give the visiting distribution a heavier tail, this makes
        the algorithm jump to a more distant region. The value range is (1, 3].
    accept : float, optional
        Parameter for acceptance distribution. It is used to control the
        probability of acceptance. The lower the acceptance parameter, the
        smaller the probability of acceptance. Default value is -5.0 with
        a range (-1e4, -5].
    maxfun : int, optional
        Soft limit for the number of objective function calls. If the
        algorithm is in the middle of a local search, this number will be
        exceeded, the algorithm will stop just after the local search is
        done. Default value is 1e7.
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

    adata = get_sa_results(adata,
                           res_range,
                           NN_range,
                           dist_slot,
                           use_reduction,
                           reduction_slot,
                           metrics,
                           opt_metric,
                           opt_metric_dir,
                           SS_weights,
                           SS_exp_base,
                           verbose,
                           show_progress_bar,
                           clust_alg,
                           n_subsamples,
                           subsamples_pct_cells,
                           maxiter,
                           initial_temp,
                           restart_temp_ratio,
                           visit,
                           accept,
                           maxfun,
                           seed,
                           key_added,
                           njobs)
    sa_results = adata.uns["SA_results_dict"]

    opt_res = sa_results["opt_result"].x[0]
    opt_knn = int(np.floor(sa_results["opt_result"].x[1]))
    adata = _cluster_final_internal(
        adata,
        opt_res,
        opt_knn,
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
            res = opt_res,
            knn = opt_knn,
            dist_slot = dist_slot,
            use_reduction = use_reduction,
            reduction_slot = reduction_slot,
            key_added = key_added,
            knn_slot = 'knn',
            verbose = verbose,
            seed = seed,
            njobs = njobs
        )

    return(adata)
