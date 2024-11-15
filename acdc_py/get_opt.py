from ._get_opt import _GS_clustering, _GS_metric_search_data, _GS_metric_value, _GS_params, _SA_clustering, _SA_metric_search_data, _SA_metric_value, _SA_params

### ---------- EXPORT LIST ----------
__all__ = []

# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# -------------------------- ** OPTIMMIZATION FUNCS ** -------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-

def SA_clustering(
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
    """\
    Get the clustering using the parameters found by optimizing a particular
    metric using the results produced by the SA function. Note that this
    requires running the acdc.SA function first in order to produce
    adata.uns['SA_results_dict'].

    Parameters
    ----------
    adata
        An anndata object containing a distance object in adata.obsp.
    dist_slot : default: None
        Slot in adata.obsp where a pre-generated distance matrix computed across
        all cells is stored in adata for use in construction of NN. (Default =
        None, i.e. distance matrix will be automatically computed as a
        correlation distance and stored in "corr_dist").
    use_reduction : default: True
        Whether to use a reduction (True) (highly recommended - accurate & much faster)
        or to use the direct matrix (False) for clustering.
    reduction_slot : default: "X_pca"
        If reduction is TRUE, then specify which slot for the reduction to use.
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
    n_clusts : default: None
        If not None, restrict the search space to the number of clusters equal
        to n_clusts in order to compute the optimal clustering solution with
        this many clusters.
    seed : default: 0
        Random seed to use.
    approx_size : default: None
        When set to a positive integer, instead of running GS on the entire
        dataset, perform GS on a subsample and diffuse those results. This will
        lead to an approximation of the optimal solution for cases where the
        dataset is too large to perform GS on due to time or memory constraints.
    key_added : default: "clusters"
        Slot in obs to store the resulting clusters.
    knn_slot : default: "knn"
        Slot in uns that stores the KNN array used to compute a neighbors graph
        (i.e. adata.obs['connectivities']).
    verbose : default: True
        Include additional output with True. Alternative = False.
    njobs : default: 1
        Paralleization option that allows users to speed up runtime.

    Returns
    -------
    Adds fields to the input adata, such that it contains the clustering stored
    in adata.obs[key_added].
    """
    _SA_clustering(
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
        njobs
    )

def SA_params(
    adata,
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    n_clusts = None
):
    """\
    Get the optimal parameters for clustering found by optimizing a particular
    metric using the results produced by the SA function. Note that this
    requires running the acdc.SA function first in order to produce
    adata.uns['SA_results_dict'].

    Parameters
    ----------
    adata
        An anndata object containing the results of the acdc.SA function in
        adata.uns['SA_results_dict'].
    opt_metric : default: "sil_mean"
        A metric from metrics to use to optimize parameters for the clustering.
    opt_metric_dir : default: "max"
        Whether opt_metric is more optimal by maximizing ("max") or
        by minimizing ("min").
    n_clusts : default: None
        If not None, restrict the search space to the number of clusters equal
        to n_clusts in order to retrieve the parameters for optimal clustering
        with this many clusters.

    Returns
    -------
    A dictionary with keys opt_res and opt_knn and the corresponding values that
    produce the requested clustering solution.
    """
    return _SA_params(
        adata,
        opt_metric,
        opt_metric_dir,
        n_clusts
    )

def SA_metric_value(
    adata,
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    n_clusts = None
):
    """\
    Get the optimal value for a particular metric found when using parameters
    for clustering that optimize said metric. This will be identified in using
    the results produced by the SA function. Note that this therefore requires
    running acdc.GS first in order to produce adata.uns['SA_results_dict'].

    Parameters
    ----------
    adata
        An anndata object containing the results of the acdc.SA function in
        adata.uns['SA_results_dict'].
    opt_metric : default: "sil_mean"
        A metric from metrics to use to optimize parameters for the clustering.
    opt_metric_dir : default: "max"
        Whether opt_metric is more optimal by maximizing ("max") or
        by minimizing ("min").
    n_clusts : default: None
        If not None, restrict the search space to the number of clusters equal
        to n_clusts in order to retrieve the parameters for optimal clustering
        with this many clusters.

    Returns
    -------
    The value of the metric when optimized.
    """
    return _SA_metric_value(
        adata,
        opt_metric,
        opt_metric_dir,
        n_clusts
    )

def SA_metric_search_data(
    adata,
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    n_clusts = None
):
    """\
    Get the optimal parameters for clustering along with all their associated
    statistics. These will be found by optimizing a particular metric using the
    results produced by the SA function. Note that this requires running the
    acdc.SA function first in order to produce adata.uns['SA_results_dict'].

    Parameters
    ----------
    adata
        An anndata object containing the results of the acdc.SA function in
        adata.uns['SA_results_dict'].
    opt_metric : default: "sil_mean"
        A metric from metrics to use to optimize parameters for the clustering.
    opt_metric_dir : default: "max"
        Whether opt_metric is more optimal by maximizing ("max") or
        by minimizing ("min").
    n_clusts : default: None
        If not None, restrict the search space to the number of clusters equal
        to n_clusts in order to retrieve the parameters for optimal clustering
        with this many clusters.

    Returns
    -------
    A pandas series containing the resolution and knn that produce the requested
    clustering solution along with all other metrics.
    """
    return _SA_metric_search_data(
        adata,
        opt_metric,
        opt_metric_dir,
        n_clusts
    )

def GS_clustering(
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
    """\
    Get the clustering using the parameters found by optimizing a particular
    metric using the results produced by the GS function. Note that this
    requires running the acdc.GS function first in order to produce
    adata.uns['GS_results_dict'].

    Parameters
    ----------
    adata
        An anndata object containing a distance object in adata.obsp.
    dist_slot : default: None
        Slot in adata.obsp where a pre-generated distance matrix computed across
        all cells is stored in adata for use in construction of NN. (Default =
        None, i.e. distance matrix will be automatically computed as a
        correlation distance and stored in "corr_dist").
    use_reduction : default: True
        Whether to use a reduction (True) (highly recommended - accurate & much faster)
        or to use the direct matrix (False) for clustering.
    reduction_slot : default: "X_pca"
        If reduction is TRUE, then specify which slot for the reduction to use.
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
    n_clusts : default: None
        If not None, restrict the search space to the number of clusters equal
        to n_clusts in order to compute the optimal clustering solution with
        this many clusters.
    seed : default: 0)
        Random seed to use.
    approx_size : default: None
        When set to a positive integer, instead of running GS on the entire
        dataset, perform GS on a subsample and diffuse those results. This will
        lead to an approximation of the optimal solution for cases where the
        dataset is too large to perform GS on due to time or memory constraints.
    key_added : default: "clusters"
        Slot in obs to store the resulting clusters.
    knn_slot : default: "knn"
        Slot in uns that stores the KNN array used to compute a neighbors graph
        (i.e. adata.obs['connectivities']).
    verbose : default: True
        Include additional output with True. Alternative = False.
    njobs : default: 1
        Paralleization option that allows users to speed up runtime.

    Returns
    -------
    Adds fields to the input adata, such that it contains the clustering stored
    in adata.obs[key_added].
    """
    _GS_clustering(
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
        njobs
    )

def GS_params(
    adata,
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    n_clusts = None
):
    """\
    Get the optimal parameters for clustering found by optimizing a particular
    metric using the results produced by the GS function. Note that this
    requires running the acdc.GS function first in order to produce
    adata.uns['GS_results_dict'].

    Parameters
    ----------
    adata
        An anndata object containing the results of the acdc.GS function in
        adata.uns['GS_results_dict'].
    opt_metric : default: "sil_mean"
        A metric from metrics to use to optimize parameters for the clustering.
    opt_metric_dir : default: "max"
        Whether opt_metric is more optimal by maximizing ("max") or
        by minimizing ("min").
    n_clusts : default: None
        If not None, restrict the search space to the number of clusters equal
        to n_clusts in order to retrieve the parameters for optimal clustering
        with this many clusters.

    Returns
    -------
    A dictionary with keys opt_res and opt_knn and the corresponding values that
    produce the requested clustering solution.
    """
    return _GS_params(
        adata,
        opt_metric,
        opt_metric_dir,
        n_clusts
    )

def GS_metric_value(
    adata,
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    n_clusts = None
):
    """\
    Get the optimal value for a particular metric found when using parameters
    for clustering that optimize said metric. This will be identified in using
    the results produced by the GS function. Note that this therefore requires
    running acdc.GS first in order to produce adata.uns['GS_results_dict'].

    Parameters
    ----------
    adata
        An anndata object containing the results of the acdc.GS function in
        adata.uns['GS_results_dict'].
    opt_metric : default: "sil_mean"
        A metric from metrics to use to optimize parameters for the clustering.
    opt_metric_dir : default: "max"
        Whether opt_metric is more optimal by maximizing ("max") or
        by minimizing ("min").
    n_clusts : default: None
        If not None, restrict the search space to the number of clusters equal
        to n_clusts in order to retrieve the parameters for optimal clustering
        with this many clusters.

    Returns
    -------
    The value of the metric when optimized.
    """
    return _GS_metric_value(
        adata,
        opt_metric,
        opt_metric_dir,
        n_clusts
    )

def GS_metric_search_data(
    adata,
    opt_metric = "sil_mean",
    opt_metric_dir = "max",
    n_clusts = None
):
    """\
    Get the optimal parameters for clustering along with all their associated
    statistics. These will be found by optimizing a particular metric using the
    results produced by the GS function. Note that this requires running the
    acdc.GS function first in order to produce adata.uns['GS_results_dict'].

    Parameters
    ----------
    adata
        An anndata object containing the results of the acdc.SA function in
        adata.uns['GS_results_dict'].
    opt_metric : default: "sil_mean"
        A metric from metrics to use to optimize parameters for the clustering.
    opt_metric_dir : default: "max"
        Whether opt_metric is more optimal by maximizing ("max") or
        by minimizing ("min").
    n_clusts : default: None
        If not None, restrict the search space to the number of clusters equal
        to n_clusts in order to retrieve the parameters for optimal clustering
        with this many clusters.

    Returns
    -------
    A pandas series containing the resolution and knn that produce the requested
    clustering solution along with all other metrics.
    """
    return _GS_metric_search_data(
        adata,
        opt_metric,
        opt_metric_dir,
        n_clusts
    )
