### ---------- IMPORT DEPENDENCIES ----------
import numpy as np
import pandas as pd
import scanpy as sc
from ._pp import _corr_distance, _neighbors_knn, _neighbors_graph, _compute_diffusion_map, _nystrom_extension
from ._SA_GS_subfunctions import _cluster_adata
from ._condense_diffuse_funcs import __diffuse_subsample_labels
from .config import config
from .pl import plot_diffusion_map
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


### ---------- EXPORT LIST ----------
__all__ = []

def _cluster_final_internal(adata,
                            res,
                            knn,
                            dist_slot = None,
                            clust_alg = "Leiden",
                            seed=0,
                            approx={
                                "run": False,
                                "size": 1000,
                                "exact_size": False
                            },
                            key_added="clusters",
                            knn_slot='knn',
                            verbose=True,
                            batch_size=1000,
                            njobs = 1):
    if approx["run"] is True:
        adata = get_approx_anndata(adata, approx, seed, verbose, njobs)

    if verbose is True: print("Computing neighbor graph with " + str(knn) + " neighbors...")
    if not (knn_slot in adata.uns.keys()):
        _neighbors_knn(
            adata,
            max_knn=knn,
            dist_slot = dist_slot,
            key_added = knn_slot,
            verbose = verbose,
            batch_size=batch_size,
            njobs = njobs
        )
    elif not (adata.uns[knn_slot].shape[1] >= knn):
        _neighbors_knn(
            adata,
            max_knn=knn,
            dist_slot = dist_slot,
            key_added = knn_slot,
            verbose = verbose,
            batch_size=batch_size,
            njobs = njobs
        )
    _neighbors_graph(
        adata,
        n_neighbors = knn,
        knn_slot = knn_slot,
        batch_size=batch_size,
        verbose = verbose
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
                  seed=0,
                  approx_size=None,
                  key_added="clusters",
                  knn_slot='knn',
                  verbose=True,
                  batch_size=1000,
                  njobs = 1):
    if dist_slot is None:
        if verbose: print("Computing distance object...")
        dist_slot = "corr_dist"
        _corr_distance(adata,
                      use_reduction,
                      reduction_slot,
                      key_added=dist_slot,
                      batch_size=batch_size,
                      verbose=verbose)
    if approx_size is None:
      approx = {"run":False}
    else:
      approx = {"run":True, "size":approx_size, "exact_size":True}

    adata = _cluster_final_internal(adata,
                                    res,
                                    knn,
                                    dist_slot,
                                    config['clust_alg'],
                                    seed,
                                    approx,
                                    key_added,
                                    knn_slot,
                                    verbose,
                                    batch_size,
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

def _diffusion_reference_mapping(ref_adata, query_adata,
                                 embedding_key="X",
                                 neigen=2,
                                 pca_comps=None,
                                 epsilon=None,
                                 plot=True):
    # Always densify AnnData.X if sparse
    if embedding_key == "X" and sparse.issparse(ref_adata.X):
        ref_adata.X = ref_adata.X.toarray()
    if embedding_key == "X" and sparse.issparse(query_adata.X):
        query_adata.X = query_adata.X.toarray()

    # Extract matrices
    reference_data = ref_adata.obsm.get(embedding_key, ref_adata.X)
    query_data = query_adata.obsm.get(embedding_key, query_adata.X)

    print(f"Computing diffusion map ({neigen} components) on reference...")
    diff_map = _compute_diffusion_map(reference_data,
                                      neigen=neigen,
                                      epsilon=epsilon,
                                      pca_comps=pca_comps)

    print("Extending to query via Nyström extension...")
    nys = _nystrom_extension(query_data, diff_map)

    # Store embeddings
    ref_adata.obsm['X_diffmap'] = diff_map['ref_diffusion']
    query_adata.obsm['X_diffmap'] = nys['query_diffusion']

    if plot and neigen >= 2:
        plot_diffusion_map(ref_adata, query_adata)

    # Save diagnostics
    results = {
        'eigenvalues': diff_map['eigenvalues'],
        'ref_proc': diff_map['ref_proc'],
        'epsilon': diff_map['epsilon'],
        'pca': diff_map['pca'],
        'distance_matrix_ref': diff_map['distance_matrix_ref'],
        'distance_matrix_query': nys['distance_matrix_query'],
    }
    ref_adata.uns['diffusion_results'] = results
    print("Stored embeddings in .obsm['X_diffmap'] and details in .uns['diffusion_results'].")

    
def _transfer_labels(
    ref_adata,
    query_adata,
    embedding_key='diffmap',
    label_key='cell_type',
    n_neighbors=15,
    pca_comps=None,
    ground_truth_label=None,
    plot_labels=False,
    plot_embedding_key='X_umap'
):
    # Optional PCA preprocessing: compute and store PC scores in .obsm
    if pca_comps is not None:
        pca = PCA(n_components=pca_comps)
        # Fit PCA on reference X and transform both datasets
        ref_adata.obsm['X_pca'] = pca.fit_transform(ref_adata.X)
        query_adata.obsm['X_pca'] = pca.transform(query_adata.X)

    # Ensure raw X is dense matrix if using 'X' embedding
    if embedding_key == 'X':
        # Convert sparse X to dense arrays for both AnnData objects
        if hasattr(ref_adata.X, 'toarray'):
            ref_adata.obsm['X'] = ref_adata.X.toarray()
        else:
            ref_adata.obsm['X'] = ref_adata.X
        if hasattr(query_adata.X, 'toarray'):
            query_adata.obsm['X'] = query_adata.X.toarray()
        else:
            query_adata.obsm['X'] = query_adata.X

    # Extract feature matrices for KNN
    X_ref = ref_adata.obsm.get(embedding_key)
    X_query = query_adata.obsm.get(embedding_key)
    # Validate embeddings are present
    if X_ref is None or X_query is None:
        raise ValueError(f"Embedding '{embedding_key}' not found in .obsm of one or both AnnData objects.")

    # Train KNN classifier on reference embedding and labels
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_ref, ref_adata.obs[label_key].astype(str).values)

    transf_key = "transf_" + label_key

    predicted = knn.predict(X_query)
    query_adata.obs[transf_key] = pd.Categorical(predicted)

    # Annotate dataset origin for combined plotting or inspection
    ref_adata.obs['dataset'] = 'reference'
    query_adata.obs['dataset'] = 'query'

    # Align category ordering consistently
    def reorder(obs_df, key):
        cats = sorted(obs_df.obs[key].cat.categories)
        obs_df.obs[key] = obs_df.obs[key].cat.reorder_categories(cats)

    reorder(ref_adata, label_key)
    reorder(query_adata, transf_key)
    reorder(query_adata, ground_truth_label)

    print(f"Labels transferred to query .obs['{transf_key}'] using {embedding_key} embedding.")

    # Compute and print accuracy if ground truth provided
    if ground_truth_label is not None:
        true = query_adata.obs[ground_truth_label].astype(str)
        pred = query_adata.obs[transf_key].astype(str)
        accuracy = (true.values == pred.values).mean()
        print(f"Accuracy against '{ground_truth_label}': {accuracy:.2f}")

        # Optionally plot predicted vs ground truth
        if plot_labels:
            # Generate the plot and get the figure
            fig = sc.pl.embedding(
                query_adata,
                basis=plot_embedding_key,
                color=[transf_key, ground_truth_label],
                title=[f"Predicted {label_key}", f"{ground_truth_label}"],
                return_fig=True
            )

            # Access the axes of the figure
            axs = fig.axes  # This gives you a list of AxesSubplot objects

            # Add accuracy to the top right of the first subplot (predicted labels)
            ax_pred = axs[0]
            ax_pred.text(
                0.95, 0.95,
                f"Accuracy: {accuracy:.2f}",
                transform=ax_pred.transAxes,
                fontsize=12,
                ha='right',
                va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5)
            )

