### ---------- IMPORT DEPENDENCIES ----------
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
import scanpy as sc
import pandas as pd
from random import sample
from random import seed
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import datetime
from ._condense_diffuse_funcs import subsample_anndata
import pickle

# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# ----------------------------- ** SIL DF FUNCS ** -----------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
def getEmptySilDF(nrow = 0, metrics = ['sil_mean']):
    if isinstance(metrics, str): metrics = [metrics]
    sil_df = pd.DataFrame(
        columns=['iter','n_pcs','resolution','knn','n_clust','subsamp_iter','seed'] + metrics,
        index=np.arange(nrow)
    )
    return(sil_df)
def at_least_min(n, min):
    if n < min:
        return(min)
    else:
        return(n)
def update_sil_df(sil_df,
                  n_pcs,
                  a_res,
                  a_nn,
                  tot_clusters,
                  subsamp_iter,
                  metrics_i,
                  seed,
                  curr_iter,
                  update_method = "iloc"):

    row_dict = metrics_i
    row_dict['iter'] = curr_iter
    row_dict['n_pcs'] = n_pcs
    row_dict['resolution'] = a_res
    row_dict['knn'] = a_nn
    row_dict['n_clust'] = tot_clusters
    row_dict['subsamp_iter'] = subsamp_iter
    row_dict['seed'] = seed

    new_iter_results = np.vectorize(row_dict.get, otypes=[float])(sil_df.columns.values)

    if(update_method == "loc"):
        sil_df.loc[curr_iter] = new_iter_results
    elif(update_method == "concat"):
        new_iter_results = pd.DataFrame([new_iter_results], columns=sil_df.columns.values)
        # pd.DataFrame([[curr_iter,
        #                                   n_pcs,
        #                                   a_res,
        #                                   a_nn,
        #                                   tot_clusters,
        #                                   subsamp_iter,
        #                                   silhouette_avgs_i,
        #                                   seed]],
        #                            columns=['iter',
        #                                     'n_pcs',
        #                                     'resolution',
        #                                     'knn',
        #                                     'n_clust',
        #                                     'subsamp_iter',
        #                                     'sil_avg',
        #                                     'seed'])
        sil_df = pd.concat([sil_df, new_iter_results], ignore_index = True)
    return(sil_df)
def compute_silhouette_samples(
    dist_mat,
    adata,
    pct_cells,
    SS_weights,
    SS_exp_base,
    i,
    cluster_labels
):
    # In case Leiden only finds 1 cluster, set SS=0 so it won't be selected as optimal.
    sil = silhouette_samples(X = dist_mat,
                             labels = cluster_labels,
                             metric="precomputed").tolist()
    if(pct_cells < 100):
        seed(i)
        sil = np.array(sample(sil, int(pct_cells/100*len(cluster_labels))), dtype=float)
    else:
        sil = np.array(sil, dtype=float)
    if SS_weights == 'exp':
        neg_sil_indices = np.where(sil < 0)
        sil[neg_sil_indices] = -1 * (SS_exp_base ** np.abs(sil[neg_sil_indices]))
    return sil
def add_clustering_results_to_sil_df_using_subsampling(dist_mat,
                                                       adata,
                                                       i,
                                                       pct_cells,
                                                       sil_df,
                                                       n_pcs,
                                                       a_res,
                                                       a_nn,
                                                       SS_weights,
                                                       SS_exp_base,
                                                       curr_iter,
                                                       metrics = "sil_mean",
                                                       update_method="iloc",
                                                       key_added = "clusters"):
    cluster_labels = adata.obs[key_added]
    if(len(adata.obs[key_added].unique())==1):
        metrics_i = {key: np.NaN for key in metrics}
    else:
        metrics_i = {}
        if np.any(np.isin(["sil_mean", "sil_mean_median", "tot_sil_neg", "lowest_sil_clust", "max_sil_clust"], metrics)):
            silhouette_samples_i = compute_silhouette_samples(dist_mat, adata, pct_cells, SS_weights, SS_exp_base, i, cluster_labels)
        if "sil_mean" in metrics:
            metrics_i["sil_mean"] = np.mean(silhouette_samples_i)
        if "sil_mean_median" in metrics:
            # Compute the silhouette scores for each cluster
            cluster_silhouette_scores = []
            for cluster in np.unique(cluster_labels):
                mask = (cluster_labels == cluster)
                cluster_silhouette = silhouette_samples_i[mask]
                cluster_silhouette_scores.append(cluster_silhouette)
            cluster_medians = [np.median(cluster) for cluster in cluster_silhouette_scores]
            metrics_i["sil_mean_median"] = np.mean(cluster_medians)  # Mean of all clusters' median silhouette scores
        if "tot_sil_neg" in metrics:
            metrics_i["tot_sil_neg"] = np.sum(silhouette_samples_i[silhouette_samples_i < 0])
        if "lowest_sil_clust" in metrics:
            metrics_i["lowest_sil_clust"] = np.min(silhouette_samples_i)
        if "max_sil_clust" in metrics:
            metrics_i["max_sil_clust"] = np.max(silhouette_samples_i)
        if "ch" in metrics:
            metrics_i["ch"] = calinski_harabasz_score(adata.X, cluster_labels)
        if "db" in metrics:
            metrics_i["db"] = davies_bouldin_score(adata.X, cluster_labels)

    tot_clusters = len(adata.obs[key_added].cat.categories)
    sil_df = update_sil_df(sil_df,
                           n_pcs,
                           a_res,
                           a_nn,
                           tot_clusters,
                           i,
                           metrics_i,
                           i,
                           curr_iter,
                           update_method)
    return(sil_df)

# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# ----------------------------- ** CLUSTER FUNCS ** ----------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
def get_clust_func(clust_alg):
    if(str.lower(clust_alg) == "louvain"):
        clust_func = sc.tl.louvain
    else:
        clust_func = sc.tl.leiden
    return(clust_func)
def _cluster_adata(adata,
                   my_random_seed,
                   res,
                   clust_alg,
                   key_added="clusters"):
    clust_alg_func = get_clust_func(clust_alg)
    clust_alg_func(adata,
                   random_state=my_random_seed,
                   resolution=res,
                   key_added=key_added)
    return(adata)

# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# -------------------------- ** CLUSTERING FUNC ** -------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-

def get_approx_anndata(adata, approx, seed, verbose, njobs):
    approx["mode"] = "subsample"
    if approx['mode'] == "subsample":
        if verbose: print("Creating a representative subsample by condensing data...")
        adata = subsample_anndata(adata, None, approx['size'], approx['exact_size'], seed, verbose, njobs)
    elif approx['mode'] == "metacell_standard":
        raise ValueError("metacell_standard is not yet implemented.")
        # if verbose: print("Condensing data and running a standard Scanpy pipeline...")
        # adata = condense_anndata(adata, None, approx['size'], approx['exact_size'], seed, verbose, recompute_signature = True)
    else:
        raise ValueError('Unsupported approx["mode"]:' + str(approx['mode']))
    return adata
