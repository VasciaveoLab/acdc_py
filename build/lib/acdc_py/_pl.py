### ---------- IMPORT DEPENDENCIES ----------
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
# ---------------------------- ** PLOTTING FUNCS ** ----------------------------
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
