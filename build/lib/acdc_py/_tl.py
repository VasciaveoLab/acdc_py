### ---------- IMPORT DEPENDENCIES ----------
import numpy as np
import pandas as pd
from ._pp import _corr_distance, _neighbors_knn, _neighbors_graph
from ._SA_GS_subfunctions import _cluster_adata
from ._condense_diffuse_funcs import __diffuse_subsample_labels

### ---------- EXPORT LIST ----------
__all__ = []


def _cluster_final_internal(adata,
                            res,
                            knn,
                            dist_slot = None,
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
    if approx["run"] is True: adata = get_approx_anndata(adata, approx, seed, verbose, njobs)

    if verbose is True: print("Computing neighbor graph with " + str(knn) + " neighbors...")
    if not (knn_slot in adata.uns.keys()):
        _neighbors_knn(adata, max_knn=knn, dist_slot = dist_slot, key_added = knn_slot, njobs = njobs)
    elif not (adata.uns[knn_slot].shape[1] >= knn):
        _neighbors_knn(adata, max_knn=knn, dist_slot = dist_slot, key_added = knn_slot, njobs = njobs)
    _neighbors_graph(adata, n_neighbors = knn, knn_slot = knn_slot)

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
    if dist_slot is None:
        if verbose: print("Computing distance object...")
        dist_slot = "corr_dist"
        _corr_distance(adata,
                      use_reduction,
                      reduction_slot,
                      key_added=dist_slot)

    adata = _cluster_final_internal(adata,
                                    res,
                                    knn,
                                    dist_slot,
                                    clust_alg,
                                    seed,
                                    approx,
                                    key_added,
                                    knn_slot,
                                    verbose,
                                    njobs)
    return adata
