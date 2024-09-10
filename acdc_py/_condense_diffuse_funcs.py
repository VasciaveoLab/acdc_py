### ---------- IMPORT DEPENDENCIES ----------
import numpy as np
import pandas as pd
import anndata
import random
from .pp import corr_distance, neighbors_knn
from ._diffuse_correction_funcs import _correct_diffusion_with_leiden
from tqdm import tqdm
from math import log2

### ---------- EXPORT LIST ----------
# __all__ = ["subsample_anndata", "condense_anndata", "diffuse_subsample_labels"]

# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# ----------------------- ** CONDENSING ANNDATA FUNCS ** -----------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-

# ----------------------------- ** HELPER FUNCS ** -----------------------------

def _representative_subsample_pca(pca_array, size, njobs = 1):
    dist_mat = corr_distance(pca_array)
    return representative_subsample(dist_mat, size, njobs)

def representative_subsample(dist_mat, size, njobs = 1):
    n_total_samples = dist_mat.shape[0]
    NN_array = neighbors_knn(
        dist_mat,
        n_total_samples,
        njobs=njobs
    )

    # If sample size is greater than half the number of samples, we take a
    # representative subsample of size (total # cells - sample size) and then
    # remove that representative subsample from all the possible samples. What's
    # left is our final representative subsample that we return.
    if size > int(n_total_samples/2):
        sample_size = n_total_samples - size
    else:
        sample_size = size

    rep_MC_sampling_df = more_equal_sampling_with_nn(NN_array, sample_size, seed = 0)
    first_MC_sample_representative_indices = rep_MC_sampling_df[rep_MC_sampling_df['sample_index'] != -1]['sample_index'].tolist()

    if size > int(n_total_samples/2):
        all_indices = list(range(0, n_total_samples))
        remaining_MC_indices = list(np.setdiff1d(all_indices, first_MC_sample_representative_indices))
        repr_sample = remaining_MC_indices
    else:
        repr_sample = first_MC_sample_representative_indices

    return repr_sample

def more_equal_sampling_with_nn(NN_array, size, seed = 0):
    # We select metacells in a representative way: after selecting each metacell, we eliminate its nearest
    # unselected neighbor from the sampling pool. This way, if you sample more from one subpopulation/cluster of
    # metacells, you are less likely to do so and therefore more likely to sample from the other subpopulations.
    # The reason we use this approach and not just random.sample is because if we have a small population,
    # random.sample may miss it, while using more_equal_sampling_with_nn will capture it.
    n_cells = NN_array.shape[0]
    if size > n_cells:
        raise ValueError("size of " + str(size) + " is larger than the number of samples in NN_array, " + str(n_cells))
    df = pd.DataFrame({'sample_index': [-1] * size, 'nn_index': [-1] * size})
    random.seed(seed)
    randomized_indices = random.sample(list(range(0, n_cells)), n_cells)
    # indices_in_sample = np.array(np.zeros(size), dtype = int)
    excluded_from_sampling = list()
    n_added_samps = 0 # n_collected_samples
    for i in range(0, n_cells):
        index_i = randomized_indices[i]
        if not index_i in excluded_from_sampling:
            # Add index_i as our newest sample
            df.loc[n_added_samps]["sample_index"] = index_i
            # Prevent index_i from being treated as a neighbor
            # of a future sample
            excluded_from_sampling.append(index_i)
            # Identify the NN of index_i that has not
            # yet been added to excluded_from_sampling
            index_i_NN_sorted = NN_array[index_i,:]
            NN_not_excluded = index_i_NN_sorted[~np.isin(index_i_NN_sorted, excluded_from_sampling)]
            first_NN_not_excluded = NN_not_excluded[0]
            df.loc[n_added_samps]["nn_index"] = first_NN_not_excluded
            # Prevent index_i from being treated as a sample or
            # a different sample's neighbor
            excluded_from_sampling.append(first_NN_not_excluded)
            # If we've collected enough samples return our results
            n_added_samps += 1
            if n_added_samps == size:
                break
    return(df)

def condense_in_half(pca_array, metacell_indices_df, seed = 0, njobs = 1):
    pca_array = pca_array.copy()
    n_samples = pca_array.shape[0]
    if n_samples % 2 == 1: # check if odd number of samples
        # Generate a random index for the row to remove
        random.seed(seed)
        random_index = random.randint(0, pca_array.shape[0] - 1)
        # Remove the random row, so we now have an even number of samples
        pca_array = np.delete(pca_array, random_index, axis=0)
        metacell_indices_df = pd.DataFrame(np.delete(metacell_indices_df.values, random_index, axis=0))
        n_samples -= 1

    dist_array = corr_distance(pca_array)

    # Slowest step
    NN_array = neighbors_knn(dist_array, n_samples, njobs=njobs)

    n_half_samples = int(n_samples/2)
    sampled_indices_df = more_equal_sampling_with_nn(NN_array, size = n_half_samples, seed = seed)

    # Extract sample_index and nn_index as NumPy arrays
    sample_indices = sampled_indices_df['sample_index'].to_numpy()
    nn_indices = sampled_indices_df['nn_index'].to_numpy()

    # Use NumPy to index rows in pca_array and calculate the mean
    pca_array = np.mean(pca_array[np.vstack((sample_indices, nn_indices))], axis=0)

    # Use NumPy to extract the corresponding rows from metacell_indices_df
    sample_rows = metacell_indices_df.iloc[sample_indices].to_numpy()
    nn_rows = metacell_indices_df.iloc[nn_indices].to_numpy()
    # Stack the rows side by side to create a new array with the
    # original indices in each row, so we can later create the metacells
    metacell_indices_df = pd.DataFrame(np.hstack((sample_rows, nn_rows)))

    return {"pca_array":pca_array, "metacell_indices_df":metacell_indices_df}

def get_mc_indices_by_condensing(pca_array, size = 1000, exact_size = False, seed = 0, verbose = True, njobs = 1): #mode = "exact"
    n_samples = pca_array.shape[0]
    metacell_indices_df = pd.DataFrame(list(range(0, n_samples)))
    # Calculate the number of iterations necessary
    # n_samples/2^n = size ==> n = log2(n_samples / size)
    from math import log2
    total_iters = int(log2(n_samples / size)) # we floor because we want > size
    if verbose: progress_bar = tqdm(total=total_iters, desc="Condensing")
    for _ in range(total_iters): #i.e. while n_samples > size
        condense_results = condense_in_half(pca_array, metacell_indices_df, seed, njobs)
        pca_array = condense_results["pca_array"]
        metacell_indices_df = condense_results["metacell_indices_df"]
        if verbose: progress_bar.update(1)

    if exact_size:
        repr_MC_sample_indices = _representative_subsample_pca(pca_array, size, njobs = 1)
        metacell_indices_df = metacell_indices_df.loc[repr_MC_sample_indices].reset_index(drop=True)

    if verbose: progress_bar.close()

    return metacell_indices_df

# ------------------------------ ** MAIN FUNCS ** ------------------------------

def subsample_anndata(adata, pca_array = None, size = 1000, exact_size = False, seed = 0, verbose = True, njobs = 1):
    if pca_array is None:
        pca_array = adata.obsm["X_pca"].copy()
    metacell_indices_df = get_mc_indices_by_condensing(pca_array, size, exact_size, seed, verbose, njobs)
    sample_indices = np.array(metacell_indices_df.loc[:,0])

    adata_subsample = adata[sample_indices,:].copy()
    adata_subsample.uns["adata_full"] = adata
    adata_subsample.var = adata.var
    adata_subsample.uns["sample_indices"] = sample_indices
    adata_subsample.uns["metacell_indices_df"] = metacell_indices_df
    return adata_subsample

# def condense_anndata(adata, pca_array = None, size = 1000, exact_size = False, seed = 0, verbose = True, recompute_signature = False):
#     if adata.raw is None:
#         raise ValueError("adata.raw is None. Raw counts required to condense anndata.")
#     if pca_array is None:
#         pca_array = adata.obsm["X_pca"].copy()
#     metacell_indices_df = get_mc_indices_by_condensing(pca_array, size, exact_size, seed, verbose)
#     counts = adata.raw.X.copy()
#     # Initialize a new array to store the result
#     metacells = np.zeros((metacell_indices_df.shape[0], counts.shape[1]))
#     # Loop through each row in sampled_indices_df and sum the corresponding rows in counts
#     for i, row_indices in enumerate(metacell_indices_df.values):
#         metacells[i, :] = np.sum(counts[row_indices, :], axis=0)
#     mc_data = anndata.AnnData(metacells, dtype=np.float64)
#     mc_data.var = adata.raw.var
#     mc_data.uns["adata_full"] = adata
#     mc_data.uns["metacell_indices_df"] = metacell_indices_df
#
#     if recompute_signature:
#             sc.pp.filter_cells(adata, min_genes=200)
#             sc.pp.filter_genes(adata, min_cells=3)
#             sc.pp.normalize_total(adata, target_sum=1e4)
#             sc.pp.log1p(adata)
#             adata = adata[:, adata.uns['adata_full'].var.index]
#             sc.pp.scale(adata, max_value=10)
#             sc.tl.pca(adata, svd_solver='arpack', random_state=0)
#
#     return mc_data
#


# def get_rep_subsample_via_condensed_metacells(pca_array, size = 1000, exact_size = False, seed = 0, verbose = True):
#     metacell_indices_df = get_mc_indices_by_condensing(pca_array, size, exact_size, seed, verbose)
#     return {"sample_indices":np.array(metacell_indices_df.loc[:,0]), "metacell_indices_df":metacell_indices_df}




# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# ------------------------ ** DIFFUSING ANNDATA FUNCS ** -----------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-

# ----------------------------- ** HELPER FUNCS ** -----------------------------

# Diffuse clusters via metacell construction
def diffuse_subsample_labels_by_metacell_diffusion(
    adata_subsample,
    adata_full,
    metacell_indices_df,
    key_added = "clusters",
    verbose = True
):
    adata_full.obs[key_added] = "-1"
    n_metacells = adata_subsample.shape[0]
    if verbose: pbar = tqdm(desc = "Diffusion", total = n_metacells, position=0, leave=True)
    for i in range(0, n_metacells):
        # Take the samples used to construct the metacell our subsample was in
        subcell_indices = metacell_indices_df.loc[i,:]
        # Get the subsample's labels and give it to its neighbors in the metacell
        adata_full.obs[key_added].iloc[subcell_indices] = adata_subsample.obs[key_added].iloc[i]
        if verbose: pbar.update(1)
    if verbose: pbar.close()
    return adata_full
# def diffuse_across_remaining_unlabeled_by_distance_diffusion(adata, key_added = "clusters"):
#     # For those samples that were lost during metacell construction
#     # diffuse using neighbor votes, weighted by distance
#     pca_array = adata.obsm["X_pca"].copy()
#     dist_array = compute_corr_distance(pca_array)
#     adata.obsm['dist_obj'] = compute_corr_distance(pca_array)
#
#     # Create new clusters DF so we don't affect the old clustering
#     # as we use it to update our labels
#     new_clusters_df = adata.obs[key_added].copy()
#     for index in tqdm(range(0,adata.shape[0]), desc="Correction"):#samples_missing_diffusion:
#         # Set up voting DataFrame
#         voting_df = pd.DataFrame({"distance":dist_array[index,:], "cluster":adata.obs[key_added].values})
#         voting_df = voting_df.drop(index)
#         voting_df = voting_df[voting_df['cluster'] != -1]
#         voting_df['voting_weight'] = (1/voting_df['distance'])**2
#
#         # Group voting by cluster
#         grouped = voting_df.groupby('cluster', observed=True)
#
#         # Calculate the sum of 'voting_weight' for each cluster
#         sum_voting_weight = grouped['voting_weight'].sum()
#
#         # Calculate the total number of samples in each cluster
#         total_samples = grouped.size()
#
#         # Divide the sum of 'voting_weight' by the total number of samples for each cluster
#         result_df = (sum_voting_weight / total_samples).reset_index()
#
#         # Rename the resulting column to 'voting_weight_per_sample'
#         result_df.rename(columns={0: 'voting_weight_per_sample'}, inplace=True)
#
#         # Get cluster with highest vote weighted by size
#         clust_with_highest_vote = result_df.sort_values(by='voting_weight_per_sample', ascending=False).iloc[0]['cluster']
#
#         # Assign this cluster to the sample
#         new_clusters_df.iloc[index] = clust_with_highest_vote
#
#     adata.obs[key_added] = new_clusters_df
#     return adata

# ------------------------------- ** MAIN FUNC ** ------------------------------

def __diffuse_subsample_labels(adata_subsample,
                               res,
                               knn,
                               dist_slot = None,
                               use_reduction = True,
                               reduction_slot="X_pca",
                               adata_full=None,
                               metacell_indices_df=None,
                               key_added='clusters',
                               knn_slot='knn',
                               verbose=True,
                               seed = 0,
                               njobs = 1):
    if adata_full is None:
        adata_full = adata_subsample.uns["adata_full"]
    if metacell_indices_df is None:
        metacell_indices_df = adata_subsample.uns["metacell_indices_df"]
    adata_full = diffuse_subsample_labels_by_metacell_diffusion(
        adata_subsample,
        adata_full,
        metacell_indices_df,
        key_added,
        verbose
    )
    if dist_slot is None:
        if verbose: print("Computing distance object...")
        dist_slot = "corr_dist"
        corr_distance(adata_full,
                      use_reduction,
                      reduction_slot,
                      key_added=dist_slot)

    adata_full = _correct_diffusion_with_leiden(
        adata_full,
        res,
        knn,
        dist_slot,
        key_added,
        knn_slot,
        verbose,
        seed,
        njobs
    )

    if "SA_results_dict" in adata_subsample.uns.keys():
        adata_full.uns["SA_results_dict"] = adata_subsample.uns["SA_results_dict"]
    if "GS_results_dict" in adata_subsample.uns.keys():
        adata_full.uns["GS_results_dict"] = adata_subsample.uns["GS_results_dict"]
    return adata_full
