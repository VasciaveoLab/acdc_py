### ---------- IMPORT DEPENDENCIES ----------
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns
import pandas as pd
import numpy as np

### ---------- EXPORT LIST ----------
__all__ = []

# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# --------------------------- ** HELPER FUNCTIONS ** ---------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
def _get_vega_30():
    vega_10_dark = ['#1f77b4',
                   '#ff7f0e',
                   '#2ca02c',
                   '#d62728',
                   '#9467bd',
                   '#8c564b',
                   '#e377c2',
                   '#7f7f7f',
                   '#bcbd22',
                   '#17becf']
    vega_10_light = ['#aec7e8',
                    '#ffbb78',
                    '#98df8a',
                    '#ff9896',
                    '#c5b0d5',
                    '#c49c94',
                    '#f7b6d2',
                    '#c7c7c7',
                    '#dbdb8d',
                    '#9edae5']
    vega_10_medium = ["#1f497d", # (medium dark blue)
                      "#d2691e", # (medium dark orange)
                      "#228b22", # (medium dark green)
                      "#a52a2a", # (medium dark red/brown)
                      "#483d8b", # (medium dark purple)
                      "#7b7b7b", # (medium grey)
                      "#ffd700", # (medium yellow/gold)
                      "#008080", # (medium teal)
                      "#da70d6", # (medium pink/purple)
                      "#ffa07a"] # (medium salmon/orange))
    vega_30_complete = vega_10_dark + vega_10_light + vega_10_medium
    return vega_30_complete
def _get_inferno_10():
    inferno_colors_10 = ["#000004FF",
                         "#1B0C42FF",
                         "#4B0C6BFF",
                         "#781C6DFF",
                         "#A52C60FF",
                         "#CF4446FF",
                         "#ED6925FF",
                         "#FB9A06FF",
                         "#F7D03CFF",
                         "#FCFFA4FF"]
    return(inferno_colors_10)
def _get_viridis_10():
    virids_colors_10 = ["#440154FF",
                        "#482878FF",
                        "#3E4A89FF",
                        "#31688EFF",
                        "#26828EFF",
                        "#1F9E89FF",
                        "#35B779FF",
                        "#6DCD59FF",
                        "#B4DE2CFF",
                        "#FDE725FF"]
    return(virids_colors_10)
def _get_YlOrRd_9():
    YlOrRd_9 = ["#FFFFCC", "#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026", "#800026"]
    return(YlOrRd_9)

def _get_BWR_3():
    BWR_3 = ["blue", "white", "red"]
    return(BWR_3)

def _get_YlGnBu_9():
    YlGnBu_9 = ["#FFFFD9", "#EDF8B1", "#C7E9B4", "#7FCDBB", "#41B6C4", "#1D91C0", "#225EA8", "#253494", "#081D58"]
    return(YlGnBu_9)

def _get_RdPu_9():
    RdPu_9 = ["#FFF7F3", "#FDE0DD", "#FCC5C0", "#FA9FB5", "#F768A1", "#DD3497", "#AE017E", "#7A0177", "#49006A"]
    return(RdPu_9)
def _get_cmap_for_sa_search_plot(plot_type):
    if(plot_type == "sil_avg"):
        cmap = LinearSegmentedColormap.from_list("", _get_YlOrRd_9())#["red","violet","blue"])
    elif(plot_type == "iter"):
        cmap = LinearSegmentedColormap.from_list("", _get_YlGnBu_9())#["red","violet","blue"])
    elif(plot_type == "n_clust"):
        cmap = LinearSegmentedColormap.from_list("", _get_RdPu_9())
    else:
        cmap = LinearSegmentedColormap.from_list("", _get_inferno_10())
    return(cmap)
def _get_cbar_label_for_sa_search_plot(plot_type):
    if(plot_type == "sil_avg"):
        cbar_label = "Ave Sil Score"
    elif(plot_type == "iter"):
        cbar_label = "Iteration"
    elif(plot_type == "n_clust"):
        cbar_label = "Clusters"
    else:
        cbar_label = None
    return(cbar_label)
def _create_scatter_plot_for_sa_search_plot(ax, search_df, plot_type):
    cmap = _get_cmap_for_sa_search_plot(plot_type)
    ax.scatter(search_df["resolution"], search_df["knn"].astype(int), c=search_df[plot_type], cmap = cmap)
    ax.set_xlabel('Resolution')
    ax.set_ylabel('Nearest Neighbors')
def _create_cbar_for_sa_search_plot(ax, plot_type):
    cbar_label = _get_cbar_label_for_sa_search_plot(plot_type)
    PCM=ax.get_children()[0] #matplotlib.collections.PathCollection
    cbar = plt.colorbar(PCM, ax=ax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(cbar_label, rotation=270)
def _create_countour_layer_for_sa_search_plot(ax, search_df):
    countour_layer = sns.kdeplot(x=search_df["resolution"],
                                 y=search_df["knn"],
                                 ax = ax,
                                 clip = [ax.get_xlim(),ax.get_ylim()])


# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# ---------------------- ** SEARCHSPACE PLOTTING FUNCS ** ----------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-

def _GS_search_space(adata, plot_type = "sil_mean"):
    heatmap_table = pd.pivot_table(adata.uns["GS_results_dict"]["search_df"],
                                   values=plot_type,
                                   index=['knn'],
                                   columns=['resolution'],
                                   aggfunc=np.sum).astype(np.float64)
    # So that KNN are displayed as ints instead of numerics with a .0
    heatmap_table.index = heatmap_table.index.astype(int)

    if(plot_type == "sil_mean"):
        color_map = "YlOrRd"#"inferno"
    else: #plot_type = "n_clust"
        color_map = "YlGnBu"#"viridis"
    cbar_label = None
    if(plot_type == "sil_mean"):
        cbar_label = "Ave Sil Score"
    elif(plot_type == "n_clust"):
        cbar_label = "Clusters"

    fig = plt.figure()
    ax = sns.heatmap(heatmap_table,
                     cmap = color_map,
                     cbar_kws={'label': cbar_label},
                     linewidths=1,
                     linecolor='black')
    ax.invert_yaxis()
    # for _, spine in ax.spines.items():
        # spine.set_visible(True)
    plt.xlabel('Resolution')
    plt.ylabel('Nearest Neighbors')
    plt.close()
    return(fig)

def _SA_search_space(adata, plot_type = "sil_avg", plot_density = True):
    # https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale
    search_df = adata.uns["SA_results_dict"]["search_df"]
    fig, ax = plt.subplots()
    _create_scatter_plot_for_sa_search_plot(ax, search_df, plot_type)
    _create_cbar_for_sa_search_plot(ax, plot_type)
    if(plot_density == True):
        _create_countour_layer_for_sa_search_plot(ax, search_df)
    plt.close()
    return(fig)

def _metric_vs_n_clusts(
    adata,
    metric = "sil_mean",
    width = 5,
    height = 5,
    xlabel = 'number of clusters',
    ylabel = None,
    axis_fontsize = 14
):
    unique_n_clusts = np.unique(
        adata.uns['GS_results_dict']['search_df']['n_clust'].astype(int)
    )
    ss = np.zeros(len(unique_n_clusts))

    k = 0
    from ._get_opt import _GS_metric_value
    for n_clusts in unique_n_clusts:
        ss[k] = _GS_metric_value(adata, n_clusts=n_clusts)
        k+=1

    df = pd.DataFrame({"n_clusts":unique_n_clusts,"ss":ss})

    plt.figure(figsize=(width, height))
    sns.scatterplot(x='n_clusts', y='ss', data=df, color='blue', s=100) # Scatter plot
    plt.plot(df['n_clusts'], df['ss'], color='blue') # Line plot

    if ylabel is None: ylabel = metric

    plt.xlabel(xlabel, fontsize=axis_fontsize)  # X-axis label font size
    plt.ylabel(ylabel, fontsize=axis_fontsize)  # Y-axis label font size
    plt.xticks(fontsize=12)  # X-axis tick labels font size
    plt.yticks(fontsize=12)  # Y-axis tick labels font size

# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# --------------------------- ** SS PLOTTING FUNC ** ---------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-


def __plot_ss(X, cluster_labels, palette, ylab, show):
    if palette is None:
        my_colors = _get_vega_30()
    else:
        cmap = plt.colormaps[palette]
        my_colors = [cmap(i) for i in range(cmap.N)]

    n_clusters = len(np.unique(cluster_labels))

    # Calculate silhouette scores
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    # Create a subplot with 1 row and 1 column
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(7, 7)

    # Set the range of x-axis
    ax1.set_xlim([-0.6, 1])

    # Set the range of y-axis
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])



    y_lower = 10
    for i in range(n_clusters):
        clust_i_name = np.unique(cluster_labels)[i]

        # Aggregate the silhouette scores for samples belonging to cluster i
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == clust_i_name]

        # Sort the silhouette scores
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = my_colors[i] #cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        #ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, clust_i_name)

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    # Labeling the axes
    ax1.set_title("Silhouette Scores For Clusters")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel(ylab)

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    # Clear the y-axis labels
    ax1.set_yticks([])
    ax1.set_xticks(np.arange(-0.6, 1.1, 0.1))

    if show: plt.show()

def _silhouette_scores(
    adata,
    groupby,
    dist_slot,
    palette=None,
    ylab = None,
    show = True
):
    if ylab is None: ylab = groupby
    __plot_ss(
        adata.obsp[dist_slot],
        adata.obs[groupby],
        palette,
        ylab,
        show
    )

def _plot_diffusion_map(
    ref_adata, 
    query_adata):

    ref_diffmap = ref_adata.obsm["X_diffmap"]
    query_diffmap = query_adata.obsm["X_diffmap"]

    # Create a figure with three subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # First row: Separate plots for reference and query
    # Plot reference diffusion coordinates
    axes[0, 0].scatter(ref_diffmap[:, 0], ref_diffmap[:, 1], s=10, alpha=0.7, label="Reference", color="blue")
    axes[0, 0].set_title("Reference Diffusion Map")
    axes[0, 0].set_xlabel("Diffusion Component 1")
    axes[0, 0].set_ylabel("Diffusion Component 2")
    axes[0, 0].legend()

    # Plot query diffusion coordinates
    axes[0, 1].scatter(query_diffmap[:, 0], query_diffmap[:, 1], s=10, alpha=0.7, label="Query", color="orange")
    axes[0, 1].set_title("Query Diffusion Map")
    axes[0, 1].set_xlabel("Diffusion Component 1")
    axes[0, 1].set_ylabel("Diffusion Component 2")
    axes[0, 1].legend()

    # Second row: Combined plot of reference and query
    axes[1, 0].scatter(ref_diffmap[:, 0], ref_diffmap[:, 1], c='blue', label='Reference', alpha=0.6)
    axes[1, 0].scatter(query_diffmap[:, 0], query_diffmap[:, 1], c='orange', label='Query', alpha=0.6)
    axes[1, 0].set_title("Combined Diffusion Map (Reference + Mapped Query)")
    axes[1, 0].set_xlabel("Diffusion Component 1")
    axes[1, 0].set_ylabel("Diffusion Component 2")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Remove the unused subplot in the second row, second column
    fig.delaxes(axes[1, 1])

    # Adjust layout
    plt.tight_layout()
    plt.show()

