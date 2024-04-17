### ---------- IMPORT DEPENDENCIES ----------
import numpy as np
import pandas as pd
import anndata
from .pp import corr_distance, neighbors_knn, neighbors_graph
from tqdm import tqdm
from math import log2
import random
import leidenalg
from sklearn.metrics import silhouette_samples
import igraph as ig


# from scanpy._utils import get_igraph_from_adjacency
def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""
    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es["weight"] = weights
    except KeyError:
        pass
    if g.vcount() != adjacency.shape[0]:
        logg.warning(
            f"The constructed graph has only {g.vcount()} nodes. "
            "Your adjacency matrix contained redundant nodes."
        )
    return g

### ---------- EXPORT LIST ----------
__all__ = ["_correct_diffusion_with_leiden"]


# Compute silhouette scores for all samples
def __compute_sil_with_missing_values(adata_full, dist_slot = None, key_added = "clusters"):
    n_samples = adata_full.shape[0]
    cluster_labels = adata_full.obs[key_added].values.astype(int)
    if dist_slot not in adata_full.obsp.keys():
        corr_distance(adata_full)
    indices_nonmissing_vals = np.array(np.where(cluster_labels != -1))[0]
    nonmissing_samples_silhouette_values = silhouette_samples(
        X = np.array(adata_full.obsp[dist_slot])[indices_nonmissing_vals[:, None], indices_nonmissing_vals],
        labels = cluster_labels[indices_nonmissing_vals],
        metric="precomputed"
    )
    samples_silhouette_values = np.zeros(shape = (n_samples,))
    samples_silhouette_values[indices_nonmissing_vals] = nonmissing_samples_silhouette_values
    return samples_silhouette_values

def __get_cluster_labels_for_correction_and_fixed_membership(adata_full, dist_slot = None, key_added = "clusters"):
    samples_silhouette_values = __compute_sil_with_missing_values(adata_full, dist_slot, key_added)

    n_samples = adata_full.shape[0]
    cluster_labels = adata_full.obs[key_added].values.astype(int)

    # Fix membership of high confidence samples so they won't be altered during Leiden correction
    clusts_names = np.unique(cluster_labels[cluster_labels != -1])
    is_membership_fixed = np.zeros(shape = (n_samples,)).astype(bool)
    for clust_name in clusts_names:
        # Within each cluster get the silhouette threshold
        clust_sil = samples_silhouette_values[cluster_labels == clust_name]
        sil_threshold = np.percentile(clust_sil, 25)
        # Samples of those clusters above the threshold should have fixed membership
        is_membership_fixed[(samples_silhouette_values >= sil_threshold) & (cluster_labels == clust_name)] = True

    # Fix membership only for high SS
    indices_below_sil_threshold = ~is_membership_fixed
    n_samples_below_sil_threshold = np.count_nonzero(indices_below_sil_threshold)

    # For all samples that have low SS and no fixed membership, replace their cluster label with a unique integer
    cluster_labels_for_correction = np.array(cluster_labels).astype(int)
    cluster_labels_for_correction[indices_below_sil_threshold] = range(0, n_samples_below_sil_threshold)+np.max(cluster_labels)
    return cluster_labels_for_correction, is_membership_fixed

def __get_adjacency_graph(adata_full, knn, dist_slot = None, knn_slot = 'knn', njobs = 1):
    if not (knn_slot in adata_full.uns.keys()):
        knn_array = neighbors_knn(
            adata_full.obsp[dist_slot],
            max_knn=knn,
            njobs = njobs
        )
    elif not (adata_full.uns[knn_slot].shape[0] >= knn):
        knn_array = neighbors_knn(
            adata_full.obsp[dist_slot],
            max_knn=knn,
            njobs = njobs
        )
    else:
        knn_array = adata_full.uns[knn_slot]

    new_graph = neighbors_graph(knn_array, knn)
    g = get_igraph_from_adjacency(new_graph, directed=True)
    return g

def __get_partition_from_diffused_clusters(cluster_labels_for_correction, g, res, seed = 0):
    partition_type = leidenalg.RBConfigurationVertexPartition
    partition_weights = np.array(g.es['weight']).astype(np.float64)
    # Create the leidenalg partition,
    # but do n_iterations=0 so we just get the partition object

    partition = leidenalg.find_partition(
        graph = g,
        partition_type = partition_type,
        initial_membership=cluster_labels_for_correction.astype(int),#adata_reclust.obs['small_clusters_forClusts'].values.astype(int),
        weights=partition_weights,#None,
        n_iterations=0,#-1,#2,
        max_comm_size=0,
        seed=seed,
        resolution_parameter = res
    )

    return partition

def __get_corrected_clusters_from_leiden_optimization_of_parition(
    partition,
    is_membership_fixed,
    seed = 0
):
    # Get the Leiden algorithm optimizer
    opt = leidenalg.Optimiser()
    # https://github.com/vtraag/leidenalg/issues/89
    # If you then turn off the option to consider also empty communities for
    # moving nodes, the optimiser only considers existing communities, i.e.
    # it will never go beyond the number of communities already in our partition.
    opt.consider_empty_community = False
    opt.set_rng_seed(seed) # For consistent results
    opt.optimise_partition(partition, n_iterations=-1, is_membership_fixed=list(is_membership_fixed))

    corrected_clusters = np.array(partition.membership).astype(str)
    return corrected_clusters

def __merge_newly_created_cluster_into_original_clusters(adata_full, dist_slot, corrected_clusters, diffused_clusters):
    corrected_clusts_names = np.unique(corrected_clusters)
    old_clusts_names = np.unique(diffused_clusters)
    new_clusts_names = corrected_clusts_names[~np.isin(corrected_clusts_names.astype(str), old_clusts_names.astype(str))]
    n_new_clusts = len(new_clusts_names)

    if n_new_clusts > 0:
        for new_clust in new_clusts_names:
            max_total_sil = -1
            old_clust_for_merge = -1
            for old_clust in old_clusts_names:
                corrected_clusters_tmp = corrected_clusters.copy()
                # See what the silhouette is when merging new_clust into old_clust
                corrected_clusters_tmp[corrected_clusters == new_clust] = old_clust
                samples_silhouette_values_tmp = silhouette_samples(
                                X = adata_full.obsp[dist_slot],
                                labels = corrected_clusters_tmp,
                                metric="precomputed"
                            )
                total_sil = np.sum(samples_silhouette_values_tmp)
                if total_sil > max_total_sil:
                    old_clust_for_merge = old_clust
            corrected_clusters[corrected_clusters == new_clust] = old_clust_for_merge

    return corrected_clusters

def _correct_diffusion_with_leiden(adata_full,
                                   res,
                                   knn,
                                   dist_slot = None,
                                   key_added = 'clusters',
                                   knn_slot = 'knn',
                                   verbose = True,
                                   seed = 0,
                                   njobs = 1):
    if verbose: pbar = tqdm(desc = "Correction", total = 4, position=0, leave=True)

    diffused_clusters = adata_full.obs[key_added].values.astype(int)

    cluster_labels_for_correction, is_membership_fixed = \
        __get_cluster_labels_for_correction_and_fixed_membership(adata_full, dist_slot, key_added)
    if verbose: pbar.update(1)

    g = __get_adjacency_graph(adata_full, knn, dist_slot, knn_slot, njobs)
    if verbose: pbar.update(1)

    partition = __get_partition_from_diffused_clusters(cluster_labels_for_correction, g, res, seed)

    corrected_clusters = \
        __get_corrected_clusters_from_leiden_optimization_of_parition(partition, is_membership_fixed, seed)
    if verbose: pbar.update(1)

    corrected_clusters = \
        __merge_newly_created_cluster_into_original_clusters(adata_full, dist_slot, corrected_clusters, diffused_clusters)
    if verbose: pbar.update(1)
    if verbose: pbar.close()

    adata_full.obs[key_added] = corrected_clusters
    return adata_full
