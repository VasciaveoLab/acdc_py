# config = {
#     "n_cores": 1,
#     "pval_threshold": 0.1,
#     "null_iters": 100,
#     "n_top_mrs": 25,
#     "tcm_size": 50,
#     "GS" : {
#         'res_vector': [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9],
#         'NN_vector': [ 11,  21,  31,  41,  51,  61,  71,  81,  91, 101],
#         'njobs':4,
#         'metrics':['sil_mean', 'sil_mean_median'],
#         'opt_metric':"sil_mean"#"sil_mean_median"
#     }
# }


# from .GS import GS
# import scanpy as sc
# import pyviper
# import numpy as np
# import pandas as pd
# import anndata
# from tqdm import tqdm
# import warnings
# import networkx as nx
# from scipy.stats import norm
# import random
#
# def __get_next_in_alphabet(my_letter):
#     ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#     index = ALPHABET.index(my_letter)
#     if index < len(ALPHABET) - 1:
#         return ALPHABET[index + 1]
#     else:
#         return None  # or you could return ALPHABET[0] if you want to cycle back to 'A'
#
# def __get_init_subclustering_df(obs_names, max_subcluster_iterations, AllSamps_cluster_labels_old):
#     column_names = np.concatenate((
#         np.array(["initial_clustering"]),
#         np.char.add("subcluster_iter_", np.arange(max_subcluster_iterations).astype(str))
#     ))
#     subclustering_df = pd.DataFrame(
#         index = obs_names,
#         columns = column_names
#     )
#     subclustering_df.loc[:,'initial_clustering'] = AllSamps_cluster_labels_old
#     return subclustering_df
#
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# # ------------------------------------------------------------------------------
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
#
# def __get_init_AllClusts_metrics_df(n_cells):
#     # Create an empty DataFrame with the specified columns
#     AllClusts_metrics_df = pd.DataFrame(columns=[
#         'clust',
#         'n_cells',
#         'parent',
#         'pval',
#         'pval_threshold',
#         'significant',
#         'kept',
#         'post_merge_clust',
#         'post_merge_pval'
#     ])
#     # Define the data for the new row
#     first_row = pd.DataFrame([{
#         'clust': 'A0',
#         'n_cells': n_cells,
#         'parent': None,
#         'pval': None,
#         'pval_threshold': None,
#         'significant': True,#'NA',
#         'kept': True,
#         'post_merge_clust': None,
#         'post_merge_pval': None
#     }])
#     AllClusts_metrics_df = pd.concat([AllClusts_metrics_df, first_row], ignore_index=True)
#
#     AllClusts_metrics_df = AllClusts_metrics_df.astype({
#         'clust': 'str',                    # Assuming 'clust' is categorical
#         'n_cells': 'int64',                # Convert to integer
#         'parent': 'str',                   # Assuming 'parent' is categorical
#         'pval': 'float64',                 # Convert to float
#         'pval_threshold': 'float64',       # Convert to float
#         'significant': 'bool',             # Convert to boolean
#         'kept': 'bool',                    # Convert to boolean
#         'post_merge_clust' : 'str',        # Convert to 'str'
#         'post_merge_pval' : 'float64'      # Convert to float
#     })
#     return AllClusts_metrics_df
#
# def __get_pax_data(ParentClust_gex_data, net_unPruned, store_input_data=False, store_raw=False):
#     if isinstance(ParentClust_gex_data, pd.DataFrame):
#         ParentClust_gex_data = anndata.AnnData(ParentClust_gex_data)
#     elif not isinstance(ParentClust_gex_data, anndata.AnnData):
#         raise ValueError('ParentClust_gex_data must be pd.DataFrame or anndata.AnnData.')
#
#     if store_raw:
#         ParentClust_gex_data.raw = ParentClust_gex_data
#
#     sc.pp.filter_genes(ParentClust_gex_data, min_cells=3)
#     sc.pp.normalize_total(ParentClust_gex_data, target_sum=1e4)
#     sc.pp.log1p(ParentClust_gex_data)
#     sc.pp.scale(ParentClust_gex_data, max_value=10)
#
#     # Filter and prune the network
#     net_Pruned = net_unPruned.copy()
#     net_Pruned.filter_targets(ParentClust_gex_data.var_names, verbose=False)
#     net_Pruned.prune(verbose=False)
#
#     # Compute the VIPER signature
#     ParentClust_pax_data = pyviper.viper(
#         ParentClust_gex_data,
#         interactome = net_Pruned,
#         mvws=10,
#         verbose=False,
#         store_input_data=store_input_data
#     )
#     return ParentClust_pax_data
#
#
#
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# # ------------------------------------------------------------------------------
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
#
# def __run_GS(ParentClust_pax_data):
# #     try:
#         acdc.GS(
#             ParentClust_pax_data,
#             res_vector=config['GS']['res_vector'],
#             NN_vector=config['GS']['NN_vector'],
#             njobs=config['GS']['njobs'],
#             metrics=['sil_mean', 'sil_mean_median'],
#             opt_metric="sil_mean_median",
#             verbose=False,
#             show_progress_bar = False
#         )
# #     except:
# #         ParentClust_pax_data.obs['clusters'] = None
#
# def __get_pax_data_from_gex(ParentClust_gex_data, net_unPruned):
#     # Compute the gExpr signature
#     counts = ParentClust_gex_data.to_df()
#     ParentClust_pax_data = __get_pax_data(ParentClust_gex_data, net_unPruned)
#     ParentClust_pax_data.obsm['counts'] = counts
#
#     # Compute the PCA for clustering
#     sc.tl.pca(ParentClust_pax_data)
#
#     return ParentClust_pax_data
#
# def __get_pax_data_with_clusters_from_gex(ParentClust_gex_data, net_unPruned):
#     ParentClust_pax_data = __get_pax_data_from_gex(
#         ParentClust_gex_data,
#         net_unPruned
#     )
#
#     # Get the cluster labels
#     __run_GS(ParentClust_pax_data)
#
#     return ParentClust_pax_data
#
# def __get_pax_data_with_clusters_from_pax(ParentClust_pax_data):
#     # Compute the PCA for clustering
#     sc.tl.pca(ParentClust_pax_data)
#
#     # Get the cluster labels
#     __run_GS(ParentClust_pax_data)
#
#     return ParentClust_pax_data
#
# def __get_pax_data_with_clusters(
#     counts,
#     net_unPruned,
#     vpmat,
#     AllSamps_cluster_labels_old,
#     ParentClust_name
# ):
#     if counts is None:
#         # Use the precomputed vpmat to get the pAct of the ParentClust samples
#         ParentClust_pax_data = __get_pax_data_with_clusters_from_pax(
#             anndata.AnnData(vpmat[AllSamps_cluster_labels_old == ParentClust_name])
#         )
#     else:
#         # Use the counts and net_unPruned to generate new pAct of the ParentClust samples
#         ParentClust_pax_data = __get_pax_data_with_clusters_from_gex(
#             ParentClust_gex_data = anndata.AnnData(counts[AllSamps_cluster_labels_old == ParentClust_name]),
#             net_unPruned = net_unPruned
#         )
#     return ParentClust_pax_data
#
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# # ------------------------------------------------------------------------------
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
#
# def __update_AllClusts_metrics_df(
#     AllClusts_metrics_df,
#     ParentClust_pax_data,
#     ParentClust_name,
#     parent_letter,
#     ChildClusts_pvals,
#     child_letter
# ):
#     ChildClusts_names = np.unique(ParentClust_pax_data.obs['clusters'].values)
#     for cl in range(len(ChildClusts_names)):
#         new_row = pd.DataFrame([{
#             'clust': child_letter + ChildClusts_names[cl],
#             'n_cells': np.sum(ParentClust_pax_data.obs['clusters']==ChildClusts_names[cl]),
#             'parent': parent_letter + ParentClust_name,
#             'pval': ChildClusts_pvals[cl],
#             'pval_threshold': config['pval_threshold'],
#             'significant': ChildClusts_pvals[cl] <= config['pval_threshold'],
#             'kept': ChildClusts_pvals[cl] <= config['pval_threshold'],
#             'post_merge_clust': None,
#             'post_merge_pval': None
#         }])
#         AllClusts_metrics_df = pd.concat([AllClusts_metrics_df, new_row], ignore_index=True)
#     return AllClusts_metrics_df
#
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# # ------------------------------------------------------------------------------
# # ------------ ** pairwise_merging_of_non_signif_clusts: HELPERS ** ------------
# # ------------------------------------------------------------------------------
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
#
# def __sort_sib_names_by_dist(ParentClust_pax_data, MainClust_name, SibClusts_names):
#     MainClust_SibClusts_dists = []
#     for SibClust_name in SibClusts_names:
#         MainClust_pca_center = np.mean(ParentClust_pax_data.obsm['X_pca'][
#             ParentClust_pax_data.obs['clusters']==MainClust_name
#         ], axis = 0)
#
#         SibClust_pca_center = np.mean(ParentClust_pax_data.obsm['X_pca'][
#             ParentClust_pax_data.obs['clusters']==SibClust_name
#         ], axis = 0)
#         MainClust_SibClusts_dists += [np.sqrt(1-np.corrcoef(MainClust_pca_center, SibClust_pca_center)[0,1])]
#     dist_closest_to_farthest = np.argsort(MainClust_SibClusts_dists)
#     return SibClusts_names[dist_closest_to_farthest]
#
# def __get_decision_to_merge_sibs(
#     ParentClust_pax_data,
#     net_unPruned,
#     MainClust_name,
#     SibClust_name,
# ):
#     ChildClusts_indices = np.isin(ParentClust_pax_data.obs['clusters'], [MainClust_name, SibClust_name])
#
#     # If the user provided counts and net_unPruned
#     if net_unPruned is not None: #'counts' in ParentClust_pax_data.obsm:
#         ChildClusts_M_S_gex_data = anndata.AnnData(
#             ParentClust_pax_data.obsm['counts'].loc[ChildClusts_indices]
#         )
#         ChildClusts_M_S_gex_data.obs['clusters'] = \
#             ParentClust_pax_data.obs['clusters'][ChildClusts_indices]
#
#         ChildClusts_M_S_pax_data = __get_pax_data_from_gex(
#             ChildClusts_M_S_gex_data,
#             net_unPruned
#         )
#         ChildClusts_M_S_pax_data.obs['clusters'] = \
#             ChildClusts_M_S_gex_data.obs['clusters']
#
#     # If the user provided vpmat instead
#     else:
#         ChildClusts_M_S_pax_data = ParentClust_pax_data.copy()
#         ChildClusts_M_S_pax_data._inplace_subset_obs(ChildClusts_indices)
#
#     # Compute P-Value for cluster-cluster differences
#     pval = __get_ChildClusts_pvals(ChildClusts_M_S_pax_data, verbose = False)
#
#     # Merge decision
#     return np.max(pval) > config['pval_threshold']
#
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# # ------------------------------------------------------------------------------
# # -------------- ** pairwise_merging_of_non_signif_clusts: MAIN ** -------------
# # ------------------------------------------------------------------------------
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
#
# def __get_merge_tracker_df(ChildClusts_names):
#     merge_tracker_df = pd.DataFrame({
#         'SibClust_name' : ChildClusts_names,
#         'merged_into' : ChildClusts_names
#     })
#     merge_tracker_df = merge_tracker_df.astype({
#         'SibClust_name': 'str',
#         'merged_into': 'str'
#     })
#     return merge_tracker_df
#
# def __update_merge_tracker_df(merge_tracker_df, SibClust_name, MainClust_name):
#     merge_tracker_df.loc[
#         merge_tracker_df['SibClust_name'] == SibClust_name, 'merged_into'
#     ] = MainClust_name
#     return merge_tracker_df
#
# def __pairwise_merging_of_non_signif_clusts(
#     ParentClust_pax_data,
#     net_unPruned,
#     ChildClusts_pvals,
#     AllClusts_metrics_df
# ):
#     ChildClusts_labels = ParentClust_pax_data.obs['clusters'].values.astype('str')
#     ChildClusts_names = np.unique(ChildClusts_labels)
#     ChildClusts_nonSig_names = ChildClusts_names[ChildClusts_pvals > config['pval_threshold']]
#
#     cur = 0
#     prevMainClust_names = []
#
#     # While we haven't yet checked whether all the non-significant clusters should each be merged
#     # and while we have at least 1 sibling to check against
#     # (i.e. if all ChildClusts are nonSig and we're checking the last nonSig ChildClust,
#           # we have no siblings => terminate loop)
#
#     merge_tracker_df = __get_merge_tracker_df(ChildClusts_names)
#
#     while cur < len(ChildClusts_nonSig_names) and cur < (len(ChildClusts_names)-1):
#
#         MainClust_name = ChildClusts_nonSig_names[cur]
#         prevMainClust_names = prevMainClust_names + [MainClust_name]
#
#         # Check all siblings we have not yet compared against
#         SibClusts_names = np.setdiff1d(ChildClusts_names, prevMainClust_names)
#         SibClusts_names_sorted = __sort_sib_names_by_dist(
#             ParentClust_pax_data, MainClust_name, SibClusts_names
#         )
#         for SibClust_name in SibClusts_names_sorted:
#             merge_decision = __get_decision_to_merge_sibs(
#                 ParentClust_pax_data,
#                 net_unPruned,
#                 MainClust_name,
#                 SibClust_name,
#             )
#             if merge_decision==True:
#                 ChildClusts_labels[ChildClusts_labels == SibClust_name] = MainClust_name
#                 ChildClusts_names = np.setdiff1d(ChildClusts_names, SibClust_name)
#                 merge_tracker_df = __update_merge_tracker_df(
#                     merge_tracker_df,
#                     SibClust_name,
#                     MainClust_name
#                 )
#                 # If we merge non-signif cluster "3" into non-signif cluster "1",
#                   # we don't want to loop through the missing identity "3" as a MainClust_name
#                 ChildClusts_nonSig_names = np.setdiff1d(ChildClusts_nonSig_names, SibClust_name)
#         cur = cur + 1
#
#     subcluster_labels_groups_ordered_by_cluster_size = pd.DataFrame({
#         "names":np.unique(ChildClusts_labels, return_counts = True)[0],
#         "sizes":np.unique(ChildClusts_labels, return_counts = True)[1]
#     }).sort_values('sizes').iloc[::-1]['names'].values
#     n_unique_clusts = len(np.unique(ChildClusts_labels))
#
#     ChildClusts_labels_original = ChildClusts_labels.copy()
#     for i in range(n_unique_clusts):
#         lab = subcluster_labels_groups_ordered_by_cluster_size[i]
#         ChildClusts_labels[ChildClusts_labels_original==lab]=str(i)
#         merge_tracker_df.loc[merge_tracker_df['merged_into'].values==lab,'merged_into']=str(i)
#
#     ParentClust_pax_data.obs['clusters'] = ChildClusts_labels
#     return ParentClust_pax_data, AllClusts_metrics_df
#
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# # ------------------------------------------------------------------------------
# # -------- ** update_subcluster_label_group_names: HELPERS ** ---------
# # ------------------------------------------------------------------------------
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
#
# def __rename_subcluster_labels_by_cluster_label_groups(subcluster_labels, cluster_labels, clusters_to_subcluster):
#     # Imagine in cluster_labels you have clusters 0, 1, 2, 3, 4, 5.
#     # Imagine you subclustered clusters 1 and 2 (clusters_to_subcluster) in your cluster_labels
#     # Now you have subclusters of clusters 1 and 2 in your subcluster_labels that contain subclusters 0, 1, 2, 3
#     # You don't want the subcluster names to overlap with 0, 1, 2, 3, 4, 5.
#     # So we take the max of our cluster labels (5) + 1 and add it to our subcluster labels,
#         # so they become 6, 7, 8, 9
#     # However, we will be overwriting 1 and 2, so we take the subcluster labels with the "largest" names
#         # and assign 1 and 2 to them. So our subcluster labels have groups 6, 7, 1, 2.
#     # relabel_subcluster_labels_by_group_size will correct the order of these.
#
#     cluster_labels = cluster_labels.astype('int')
#     subcluster_labels = subcluster_labels.astype('int')
#     clusters_to_subcluster = clusters_to_subcluster.astype('int')
#
#     max_of_cluster_labels = np.max(cluster_labels)
#     subcluster_labels = subcluster_labels + max_of_cluster_labels + 1 #extra +1 due to starting at 0 for cluster names
#     j = 0
#     for i in np.flip(np.arange(len(clusters_to_subcluster))):
#         subcluster_labels[subcluster_labels == (max(subcluster_labels)-i)] = clusters_to_subcluster[j]
#         j = j + 1
#     return subcluster_labels.astype('str')
#
# def __relabel_subcluster_labels_by_group_size(subcluster_labels):
#     subcluster_labels_groups_ordered_by_alphabet = \
#         np.sort(np.unique(subcluster_labels).astype(int)).astype('str')
#     subcluster_labels_groups_ordered_by_cluster_size = pd.DataFrame({
#         "names":np.unique(subcluster_labels, return_counts = True)[0],
#         "sizes":np.unique(subcluster_labels, return_counts = True)[1]
#     }).sort_values('sizes').iloc[::-1]['names'].values
#
#     subcluster_labels_orderedGroups = np.zeros(len(subcluster_labels)).astype('str')
#
#     for i in range(len(subcluster_labels_groups_ordered_by_alphabet)):
#         subcluster_labels_orderedGroups[subcluster_labels == \
#             subcluster_labels_groups_ordered_by_cluster_size[i]] = \
#             subcluster_labels_groups_ordered_by_alphabet[i]
#     subcluster_labels = subcluster_labels_orderedGroups
#     return subcluster_labels
#
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# # ------------------------------------------------------------------------------
# # -------------- ** update_subcluster_label_group_names: MAIN ** ---------------
# # ------------------------------------------------------------------------------
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
#
# def __update_subcluster_label_group_names(subcluster_labels, cluster_labels, clusters_to_subcluster):
#     subcluster_labels = __rename_subcluster_labels_by_cluster_label_groups(
#         subcluster_labels,
#         cluster_labels,
#         clusters_to_subcluster
#     )
#     subcluster_labels = __relabel_subcluster_labels_by_group_size(subcluster_labels)
#     return subcluster_labels
#
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# # ------------------------------------------------------------------------------
# # -------------- ** get_ChildClusts_pvals: HELPERS ** ---------------
# # ------------------------------------------------------------------------------
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
#
# def __generate_interactome_from_stouffer(stouffer_sig,
#                                         interactome_name="vpmat",
#                                         n_top=50,
#                                         is_symmetric=True):
#
#     n_mrs = len(stouffer_sig)
#
#     # For each sample, we calculate index arragement that would sort the vector
#     sorted_order_array = np.flip(stouffer_sig.sort_values())
#     # We then get the MRs ranked for each sample by indexing with this sorted order
#     mrs_ranked_array = sorted_order_array.index
#
#     if is_symmetric:
#         # Slice the top n_top/2 rows and bottom n_top/2 rows
#         # Get the top 25 and bottom 25 rows
#         n_top_half = int(n_top/2)
#         selected_column_indices = list(range(0,n_top_half)) + list(range(n_mrs-n_top_half,n_mrs))
#         cell_i_mor = np.concatenate((np.ones(n_top_half), np.full(n_top_half, -1)))
#     else:
#         # Slice the top n_top rows
#         selected_column_indices = list(range(0,n_top))
#         cell_i_mor = np.ones(n_top)
#
#     top_mrs_ranked_array = mrs_ranked_array[selected_column_indices]
#
#     regulator = np.repeat(interactome_name, n_top)
#     target = top_mrs_ranked_array
#     mor = np.tile(cell_i_mor, 1)
#
#     net_table = pd.DataFrame({
#             'regulator': regulator,
#             'target': top_mrs_ranked_array,
#             'mor': mor,
#             'likelihood': 1
#         })
#
#     return pyviper.Interactome(interactome_name, net_table)
#
# def __get_net_table_from_clust_vpmat(ChildClust_i_vpmat, clust_name, tcm_size):
#     ChildClust_i_stouffer = ChildClust_i_vpmat.sum(axis=0) / np.sqrt(ChildClust_i_vpmat.shape[0])
#     ChildClust_i_net_table = __generate_interactome_from_stouffer(
#             ChildClust_i_stouffer,
#             interactome_name = "clust_" + clust_name,
#             n_top = tcm_size
#     ).net_table
#     return ChildClust_i_net_table
#
# def __get_ChildClust_i_score(ParentClust_data, ChildClusts_labels, ChildClust_i_name, tcm_size):
#     ChildClust_i_vpmat = ParentClust_data[ChildClusts_labels==ChildClust_i_name]
#     ChildClust_i_net_table = __get_net_table_from_clust_vpmat(
#         ChildClust_i_vpmat, ChildClust_i_name, tcm_size
#     )
#     clust_vpscores = pyviper.viper(
#         ChildClust_i_vpmat,
#         pyviper.Interactome(ChildClust_i_name, ChildClust_i_net_table),
#         min_targets=0,
#         return_as_df=True,
#         transfer_obs=False,
#         store_input_data=False,
#         verbose = False
#     )
#     score = clust_vpscores.mean()
#     return score
#
# def __get_ChildClust_i_null_net_table_indices(
#     ParentClust_data,
#     ChildClusts_labels,
#     ChildClust_i_name,
#     tcm_size,
#     n_null
# ):
#     ChildClust_i_net_table_null = pd.DataFrame(
#         columns=['regulator', 'target', 'mor', 'likelihood']
#     )
#     ChildClust_i_size = np.sum(np.char.equal(ChildClusts_labels, ChildClust_i_name))
#     ChildClust_i_null_model_indices = np.zeros((ChildClust_i_size,n_null))
#     ChildClusts_labels_rand = ChildClusts_labels.copy()
#     for k in range(n_null):
#         np.random.seed(k)
#         np.random.shuffle(ChildClusts_labels_rand)
#         indices_k = np.where(ChildClusts_labels_rand==ChildClust_i_name)[0]
#         ChildClust_i_vpmat_null_k = ParentClust_data.iloc[indices_k]
#         ChildClust_i_null_model_indices[:,k] = indices_k
#         ChildClust_i_net_table_null_k = __get_net_table_from_clust_vpmat(
#             ChildClust_i_vpmat_null_k,
#             ChildClust_i_name + "_null_iter_" + str(k),
#             tcm_size
#         )
#         ChildClust_i_net_table_null = pd.concat([
#             ChildClust_i_net_table_null,
#             ChildClust_i_net_table_null_k
#         ])
#     return ChildClust_i_net_table_null, ChildClust_i_null_model_indices
#
# def __get_ChildClust_i_null_model(
#     ParentClust_data,
#     ChildClusts_labels,
#     ChildClust_i_name,
#     tcm_size,
#     n_null
# ):
#     ChildClust_i_net_table_null, ChildClust_i_null_model_indices = \
#         __get_ChildClust_i_null_net_table_indices(
#             ParentClust_data,
#             ChildClusts_labels,
#             ChildClust_i_name,
#             tcm_size,
#             n_null
#         )
#     null_vpmat = pyviper.viper(
#         ParentClust_data,
#         pyviper.Interactome(ChildClust_i_name + "_null", ChildClust_i_net_table_null),
#         min_targets=0,
#         return_as_df=True,
#         transfer_obs=False,
#         store_input_data=False,
#         verbose = False
#     )
#     null_model = np.mean(null_vpmat.values[
#         ChildClust_i_null_model_indices.astype(int),
#         np.arange(null_vpmat.shape[1])
#     ],axis=0)
#
#     return null_model
#
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# # ------------------------------------------------------------------------------
# # -------------- ** get_ChildClusts_pvals: MAIN ** ---------------
# # ------------------------------------------------------------------------------
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
#
# def __get_ChildClusts_pvals(ParentClust_pax_data, verbose = False):
#     ChildClusts_labels = ParentClust_pax_data.obs['clusters'].values.astype('str')
#     ParentClust_data = ParentClust_pax_data.to_df()
#     ChildClusts_names = np.sort(np.unique(ChildClusts_labels).astype(int)).astype('str')
#     ChildClusts_n = len(ChildClusts_names)
#     ChildClusts_pvals = np.zeros(ChildClusts_n)
#
#     for i in range(ChildClusts_n):
#         ChildClust_i_name = ChildClusts_names[i]
#         score = __get_ChildClust_i_score(
#             ParentClust_data,
#             ChildClusts_labels,
#             ChildClust_i_name,
#             config['tcm_size']
#         )
#         null_model = __get_ChildClust_i_null_model(
#             ParentClust_data,
#             ChildClusts_labels,
#             ChildClust_i_name,
#             config['tcm_size'],
#             config['null_iters']
#         )
#         mean_est = null_model.mean()
#         sd_est = np.std(null_model)
#         pval = 1 - norm.sf(-(score - mean_est) / sd_est)
#
#         ChildClusts_pvals[i] = pval
#     return ChildClusts_pvals
#
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# # ------------------------------------------------------------------------------
# # -------------- ** __get_keep_subclusters_decision: MAIN ** ---------------
# # ------------------------------------------------------------------------------
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
#
# def __get_keep_subclusters_decision(ChildClusts_pvals):
#     n_ChildClusts = len(ChildClusts_pvals)
#
#     # If we have exactly two clusters, we return our original result.
#     # We have a pairwise comparison here. Doing another would be redundant
#     # Case 1: Both are significant ==> keep subclustering
#     # Case 2: One is significant and one is not ==> keep subclustering
#     # Case 3: Neither significant ==> do not keep subclustering
#
#     if n_ChildClusts==1:
# #         print("No clusters. Remerging.")
#         return False
#
#     elif np.all(ChildClusts_pvals > config['pval_threshold']):
# #         print("No significant clusters. Remerging.")
#         return False
#
#     elif n_ChildClusts == 2 and ( np.min(ChildClusts_pvals) > config['pval_threshold'] ): #or np.max?
# #         print("No significant clusters. Remerging.")
#         return False
#
#     else:
#         return True
#
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# # ------------------------------------------------------------------------------
# # -------------- ** iter_cluster: HELPERS ** ---------------
# # ------------------------------------------------------------------------------
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
#
# def __get_AllSamps_names(counts, vpmat):
#     if counts is None:
#         obs_names = vpmat.index
#     else:
#         obs_names = counts.index
#     return obs_names
#
# def __get_AllSamps_n(counts, vpmat):
#     if counts is None:
#         n_cells = vpmat.shape[0]
#     else:
#         n_cells = counts.shape[0]
#     return n_cells
#
# def __add_ChildClusts_to_graph(G, ParentClust_name, ChildClusts_names, parent_letter, child_letter):
#     for c_n in ChildClusts_names:
#         G.add_edge(
#             parent_letter + ParentClust_name,
#             child_letter + c_n
#         )
#
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# # ------------------------------------------------------------------------------
# # -------------- ** iter_cluster: MAIN ** ---------------
# # ------------------------------------------------------------------------------
# # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
#
# def iter_cluster(
#     counts = None,
#     net_unPruned = None,
#     pax_data = None,
#     max_subcluster_iterations = 3,
#     test_subclusts = True,
#     key_added = "clusters"#,
# #     clusters_to_subcluster = '0'
# ):
#     # -------- USER INPUTS --------
#     if pax_data is None:
#         vpmat = None
#         if counts is None or net_unPruned is None:
#             raise ValueError("Either pax_data alone, or both counts and net_unPruned must be supplied.")
#     else:
#         if counts is not None or net_unPruned is not None:
#             warnings.warning("pax_data provided. counts and net_unPruned ignored.")
#             counts = None
#             net_unPruned = None
#         if isinstance(pax_data, anndata.AnnData):
#             vpmat = pax_data.to_df()
#         elif isinstance(pax_data, pd.DataFrame):
#             vpmat = pax_data.copy()
#             pax_data = anndata.AnnData(pax_data)
#         else:
#             raise ValueError("vpmat must be pd.DataFrame or anndata.AnnData.")
#
#     # -------- SET UP LOOP VARIABLES --------
#     G = nx.DiGraph()
#
#     parent_letter = "A"
#     child_letter = "B"
#     AllSamps_n = __get_AllSamps_n(counts, vpmat)
#     AllSamps_names = __get_AllSamps_names(counts, vpmat)
#
#     AllSamps_cluster_labels_old = pd.Series(
#         np.zeros(AllSamps_n).astype('int').astype('str'),
#         index = AllSamps_names
#     )
#     AllSamps_cluster_labels_new = AllSamps_cluster_labels_old.copy()
#
#     subclustering_df = __get_init_subclustering_df(
#         AllSamps_names, max_subcluster_iterations, AllSamps_cluster_labels_old
#     )
#     AllClusts_metrics_df = __get_init_AllClusts_metrics_df(AllSamps_n)
#     ToSubClst_names_old = np.array([0]) #parent_letter + ParentClust_name
#     ToSubClst_names_new = np.array([])
#
#     # -------- BEGIN LOOP --------
#     progress_bar = tqdm(range(max_subcluster_iterations))
#     for subclust_iter in progress_bar:
#         # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@
#         # @-@-@-@-@-@-@-@-@-@-@-@- START OF OUTER LOOP @-@-@-@-@-@-@-@-@-@-@-@-
#         # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@
#     #     subclust_iter = 0
# #         print("subclust_iter: " + str(subclust_iter))
#
#         # If no more clusters to subcluster
#         if(len(ToSubClst_names_old))==0:
#             # Keep only the current columns
#             subclustering_df = subclustering_df.iloc[:, 0:subclust_iter]
#             # Update the progress bar to look complete
# #             progress_bar.n = progress_bar.total
# #             progress_bar.refresh()
#             progress_bar.update(max_subcluster_iterations - subclust_iter)
#             progress_bar.close()
#             break
#
#         for i in range(len(ToSubClst_names_old)):
# #             print('i: ' + str(i))
#
#             # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@
#             # @-@-@-@-@-@-@-@-@-@-@-@- START OF INNER LOOP @-@-@-@-@-@-@-@-@-@-@-@-
#             # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@
#             # i = 0
#
#             # ~~~~~~~~~~ First Identify The Parent Cluster ~~~~~~~~~~
#             ParentClust_name = np.sort(np.array(ToSubClst_names_old).astype('int')).astype('str')[i]
# #             print('ParentClust_name: ' + str(ParentClust_name))
#
#             # -------- START ANALYSIS HERE --------
#
#             # !!! ADD CODE HERE: CHECK IF THERE ARE FEWER THAN 50 CELLS !!!
#
#
#             # Compute gExpr sig, VIPER sig, GS clusters
#             ParentClust_pax_data = __get_pax_data_with_clusters(
#                 counts,
#                 net_unPruned,
#                 vpmat,
#                 AllSamps_cluster_labels_old,
#                 ParentClust_name
#             )
#
# #             print("subclust_iter")
# #             print(subclust_iter)
# #             print("i")
# #             print(i)
# #             print("AllClusts_metrics_df")
# #             print(AllClusts_metrics_df)
# #             print("ParentClust_pax_data.obs['clusters'] is None")
# #             print(ParentClust_pax_data.obs['clusters'] is None)
# #             print("None in ParentClust_pax_data.obs['clusters'].values")
# #             print(None in ParentClust_pax_data.obs['clusters'].values)
# #             print("ParentClust_pax_data.obs['clusters']")
# #             print(ParentClust_pax_data.obs['clusters'].values)
#
#             if ParentClust_pax_data.obs['clusters'] is None: continue
#
#             if None in ParentClust_pax_data.obs['clusters'].values: continue
#
#
#
#             n_ChildClusts = len(np.unique(ParentClust_pax_data.obs['clusters'].values))
#             if test_subclusts:
#                 # Compute the p-values between the clusters
#                 ChildClusts_pvals = __get_ChildClusts_pvals(
#                     ParentClust_pax_data,
#                     verbose = False
#                 )
#             else:
#                 ChildClusts_pvals = [-1]*n_ChildClusts
#
#             # Update the names of the groups of subclusters so that they don't have
#             # the same name as other clusters we already have
#             ParentClust_pax_data.obs['clusters'] = __update_subcluster_label_group_names(
#                 cluster_labels = AllSamps_cluster_labels_old.copy(),
#                 subcluster_labels = ParentClust_pax_data.obs['clusters'].values,
#                 clusters_to_subcluster = np.array([ParentClust_name])
#             )
#
#             # Store initial metrics
#             AllClusts_metrics_df = __update_AllClusts_metrics_df(
#                 AllClusts_metrics_df,
#                 ParentClust_pax_data,
#                 ParentClust_name,
#                 parent_letter,
#                 ChildClusts_pvals,
#                 child_letter
#             )
#
#             # Merge non-significant clusters
#             if n_ChildClusts > 2 and ( np.any(ChildClusts_pvals > config['pval_threshold']) ):
#                 ParentClust_pax_data, AllClusts_metrics_df = __pairwise_merging_of_non_signif_clusts(
#                     ParentClust_pax_data,
#                     net_unPruned,
#                     ChildClusts_pvals,
#                     AllClusts_metrics_df
#                 )
#                 n_ChildClusts = len(np.unique(ParentClust_pax_data.obs['clusters'].values))
#                 keep_ChildClusts = ( n_ChildClusts > 1 )
#             else:
#                 keep_ChildClusts = __get_keep_subclusters_decision(ChildClusts_pvals)
#
#             # If keeping subclusters, update the overall clustering
#             ChildClusts_names = np.unique(ParentClust_pax_data.obs['clusters'].values)
#             if keep_ChildClusts:
#                 AllSamps_cluster_labels_new[ParentClust_pax_data.obs_names] =\
#                     ParentClust_pax_data.obs['clusters'].values
#         #         AllSamps_cluster_labels_new = merge_subcluster_labels_into_original_labels(
#         #             cluster_labels = AllSamps_cluster_labels_new,
#         #             subcluster_labels = ParentClust_pax_data.obs['clusters'].values,
#         #             clusters_to_subcluster = np.array([ParentClust_name])
#         #         )
#                 ToSubClst_names_new = np.concatenate((ToSubClst_names_new, ChildClusts_names))
#                 __add_ChildClusts_to_graph(
#                     G,
#                     ParentClust_name,
#                     ChildClusts_names,
#                     parent_letter,
#                     child_letter
#                 )
#
#             # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@
#             # @-@-@-@-@-@-@-@-@-@-@-@- END OF INNER LOOP @-@-@-@-@-@-@-@-@-@-@-@-@-
#             # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@
#
#         # ~~~~~~~~~~ Update OuterForLoop Variables ~~~~~~~~~~
#         # Update the results variables for this OuterFor Loop iteration
#         subclustering_df.iloc[:,subclust_iter+1] = AllSamps_cluster_labels_new
#         ToSubClst_names_old = ToSubClst_names_new
#         # Reset the loop variables for the next OuterFor Loop iteration
#         ToSubClst_names_new = list()
#         parent_letter = __get_next_in_alphabet(parent_letter)
#         child_letter = __get_next_in_alphabet(child_letter)
#
#         AllSamps_cluster_labels_old = AllSamps_cluster_labels_new
#
#         subclust_iter = subclust_iter + 1
#
#         # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@
#         # @-@-@-@-@-@-@-@-@-@-@-@- END OF OUTER LOOP @-@-@-@-@-@-@-@-@-@-@-@-@-
#         # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@
#
#     # If the user provided counts and net_unPruned
#     if counts is not None:
#         AllSamps_pax_data = __get_pax_data(
#             counts,
#             net_unPruned,
#             store_input_data=True,
#             store_raw=True
#         )
#     # If the user provided vpmat instead
#     else:
#         AllSamps_pax_data = pax_data
#
#     AllSamps_pax_data.uns['iter_cluster_metrics_df'] = AllClusts_metrics_df
# #     AllSamps_pax_data.uns['iter_cluster_labels_df'] = subclustering_df
#     AllSamps_pax_data.uns['iter_cluster_graph'] = G
#     AllSamps_pax_data.obs[key_added] = subclustering_df.iloc[:, -1]
#
#     for col in subclustering_df.columns:
#         AllSamps_pax_data.obs[col] = subclustering_df[col]
#
#     return AllSamps_pax_data
