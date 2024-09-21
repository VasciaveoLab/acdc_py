from ._tl import _cluster_final, _extract_clusters, _merge

### ---------- EXPORT LIST ----------
__all__ = []

def cluster_final(adata,
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
    """\
    A tool for replicating the final optimization-based unsupervised clustering
    of large-scale data performed by the Grid Search (GS) or Simulated Annealing
    (SA) functions. This includes replicating the approximation method we call
    subsampling and diffusion that enables fast and accurate clustering of
    hundreds of thousands of cells.

    Parameters
    ----------
    adata
        An anndata object containing a gene expression signature in adata.X and
        gene expression counts in adata.raw.X.
    res
         sequence of values of the resolution parameter.
    knn
         sequence of values for the number of nearest neighbors.
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
    clust_alg : default: "Leiden"
        Clustering algorithm. Choose among: "Leiden" (default) or  "Louvain".
    seed : default: 0
        Random seed to use.
    key_added : default: "clusters"
        Slot in obs to store the resulting clusters.
    knn_slot : default: "knn"
        Slot in uns that stores the KNN array used to compute a neighbors graph
        (i.e. adata.obs['connectivities']).
    approx : default: {"run":False, "size":1000, "exact_size":False}
        A diciontary object containing three parameters to control subsampling and diffusion
            "run": True or False whether to use subsampling and diffusion. Default=False
            "size": the number of cells to use in the subsampling. Default=1000.
            "exact_size": whether to get the exact size "size" of subsampling (True) or
            be more inclusive during the representative subsampling (False, recommended).
    verbose : default: True
        Include additional output with True. Alternative = False.

    Returns
    -------
    A object of :class:~anndata.Anndata containing a clustering vector
    "clusters" in the .obs slot.
    """
    return _cluster_final(
      adata,
      res,
      knn,
      dist_slot,
      use_reduction,
      reduction_slot,
      clust_alg,
      seed,
      approx,
      key_added,
      knn_slot,
      verbose,
      njobs
    )

def extract_clusters(adata, obs_column, clust_names):
    """\
    Extract clusters as a new AnnData object. Useful for subclustering.

    Parameters
    ----------
    adata
        An anndata object containing a gene expression signature in adata.X and
        gene expression counts in adata.raw.X.
    obs_column
        A name of the column in adata.obs.
    clust_names
        Clusters in adata.obs[obs_column] to extract.
    """
    return _extract_clusters(adata, obs_column, clust_names)

def merge(
    adata,
    obs_column,
    clust_names,
    merged_clust_name = None,
    update_numbers = True,
    key_added = "clusters",
    return_as_series = False
):
    """\
    Merge clusters together and, if desired, renumber the clusters based on
    cluster size.

    Parameters
    ----------
    adata
        An anndata object containing a gene expression signature in adata.X and
        gene expression counts in adata.raw.X.
    obs_column
        A name of the column in adata.obs.
    clust_names
        Clusters in adata.obs[obs_column] to extract.
    merged_clust_name : default: None
        The name of the new cluster. If None with digit clusters, the new
        cluster will be named after the smallest of the merged. If None with
        non-digit clusters, the new cluster will be named by joining the names
        of the clusters.
    update_numbers : default: True
        If clusters are digits, renumber the clusters based on cluster size.
    key_added : default: "clusters"
        Store the new clustering in adata.obs[key_added].
    return_as_series : default: False
        Rather than storing the clusters, return them as a pd.Series object.
    """
    return _merge(
        adata,
        obs_column,
        clust_names,
        merged_clust_name,
        update_numbers,
        key_added,
        return_as_series
    )
