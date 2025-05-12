from ._tl import _cluster_final, _extract_clusters, _merge, _diffusion_reference_mapping, _transfer_labels

### ---------- EXPORT LIST ----------
__all__ = []

def cluster_final(adata,
                  res,
                  knn,
                  dist_slot=None,
                  use_reduction=True,
                  reduction_slot="X_pca",
                  seed=0,
                  approx_size=None,
                  key_added="clusters",
                  knn_slot='knn',
                  verbose=True,
                  batch_size=1000,
                  njobs = 1):
    """\
    A tool for replicating the final optimization-based unsupervised clustering
    of large-scale data performed by the Grid Search (GS) or Simulated Annealing
    (SA) functions.

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
    seed : default: 0
        Random seed to use.
    key_added : default: "clusters"
        Slot in obs to store the resulting clusters.
    knn_slot : default: "knn"
        Slot in uns that stores the KNN array used to compute a neighbors graph
        (i.e. adata.obs['connectivities']).
    approx_size : default: None
        When set to a positive integer, instead of running GS on the entire
        dataset, perform GS on a subsample and diffuse those results. This will
        lead to an approximation of the optimal solution for cases where the
        dataset is too large to perform GS on due to time or memory constraints.
    batch_size : default: 1000
        The size of each batch. Larger batches result in more memory usage. If
        None, use the whole dataset instead of batches.
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
      seed,
      approx_size,
      key_added,
      knn_slot,
      verbose,
      batch_size,
      njobs
    )

def extract(adata, groupby, clusters):
    """\
    Extract clusters as a new AnnData object. Useful for subclustering.

    Parameters
    ----------
    adata
        An anndata object containing a gene expression signature in adata.X and
        gene expression counts in adata.raw.X.
    groupby
        A name of the column in adata.obs.
    clusters
        Names of clusters in adata.obs[groupby] to extract.
    """
    return _extract_clusters(adata, groupby, clusters)

def merge(
    adata,
    groupby,
    clusters,
    merged_name = None,
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
    groupby
        A name of the column in adata.obs.
    clusters
        Names of clusters in adata.obs[groupby] to extract.
    merged_name : default: None
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
        groupby,
        clusters,
        merged_name,
        update_numbers,
        key_added,
        return_as_series
    )

def rename(adata, groupby, name_dict):
    """\
    Rename clusters within adata.obs[groupby] using name_dict to specify
    the mapping between old and new names.
    """
    # Check if the column exists in adata.obs
    if groupby not in adata.obs:
        raise ValueError(f"Column '{groupby}' not found in adata.obs")

    # Get the current column values
    adata.obs[groupby] = adata.obs[groupby].replace(name_dict)


def diffusion_reference_mapping(ref_adata,
                                query_adata,
                                embedding_key="X",
                                neigen=2,
                                pca_comps=None,
                                epsilon=None,
                                plot=True):
    """\
    Compute diffusion map embedding on reference data and project query data via Nyström extension.

    Parameters
    ----------
    ref_adata : AnnData
        Reference single-cell dataset.
    query_adata : AnnData
        Query single-cell dataset.
    embedding_key : str, optional
        Key for the input data representation in .obsm or "X". Default is "X".
    neigen : int, optional
        Number of diffusion components to compute and retain. Default is 2.
    pca_comps : int or None, optional
        Number of PCA components for preprocessing. Default is None.
    epsilon : float or None, optional
        Bandwidth for the Gaussian kernel. Default is None.
    plot : bool, optional
        If True and neigen >= 2, generates scatter plots. Default is True.

    Returns
    -------
    None
        Side effects:
        - ref_adata.obsm['X_diffmap'], query_adata.obsm['X_diffmap']
        - ref_adata.uns['diffusion_results']
    """
    _diffusion_reference_mapping(ref_adata, query_adata,
                                 embedding_key=embedding_key,
                                 neigen=neigen,
                                 pca_comps=pca_comps,
                                 epsilon=epsilon,
                                 plot=plot)

def transfer_labels(ref_adata,
                    query_adata,
                    embedding_key='diffmap',
                    label_key='cell_type',
                    n_neighbors=15,
                    pca_comps=None,
                    ground_truth_label=None,
                    plot_labels=False,
                    plot_embedding_key='X_umap'
):  
    """\
    Transfer cluster (or other) labels from a reference AnnData to a query AnnData via K-nearest neighbours.

    This function fits a KNN classifier on the chosen embedding of `ref_adata`, predicts labels
    for `query_adata`, and stores them under `query_adata.obs["transf_" + label_key]`.  Optionally,
    it computes accuracy against known query labels and plots a comparison.

    Parameters
    ----------
    ref_adata : AnnData
        Reference dataset. Must contain true labels in `ref_adata.obs[label_key]`.
    query_adata : AnnData
        Query dataset to be annotated. Will be modified in place.
    embedding_key : str, optional (default: 'X_diffmap')
        Key in `.obsm` to use for KNN. Common choices:
        - 'X_diffmap': diffusion map coordinates  
        - 'X_pca': PCA coordinates (see `pca_comps`)  
        - 'X': data from AnnData.X
    label_key : str, optional (default: 'cell_type')
        Key in `ref_adata.obs` containing training labels. Predicted labels for query
        will be stored in `query_adata.obs["transf_" + label_key]`.
    n_neighbors : int, optional (default: 15)
        Number of neighbours for the KNN classifier.
    pca_comps : int or None, optional (default: None)
        If not `None`, perform PCA with this many components on `.X` and store results
        under `.obsm['X_pca']` in both datasets (regardless of `embedding_key`).
    ground_truth_label : str or None, optional (default: None)
        If provided, key in `query_adata.obs` with known labels. Accuracy will be
        computed and printed.
    plot_labels : bool, optional (default: False)
        If `True` *and* `ground_truth_label` is not `None`, generate an embedding plot
        (using `plot_embedding_key`) colored by predicted vs. true labels.
    plot_embedding_key : str, optional (default: 'X_umap')
        Key in `.obsm` to use for plotting when `plot_labels=True`.

    Returns
    -------
    None
        Side effects:
        - Adds `query_adata.obs["transf_" + label_key]`.
        - Sets `ref_adata.obs["dataset"] = 'reference'` and `query_adata.obs["dataset"] = 'query'`.
        - Optionally adds `.obsm['X_pca']` if `pca_comps` is specified.
    """
    
    _transfer_labels(ref_adata,
                    query_adata,
                    embedding_key,
                    label_key,
                    n_neighbors,
                    pca_comps,
                    ground_truth_label,
                    plot_labels,
                    plot_embedding_key)