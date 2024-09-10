from ._pl import _GS_search_space, _SA_search_space

### ---------- EXPORT LIST ----------
__all__ = []

# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# ------------------------------------------------------------------------------
# ---------------------------- ** PLOTTING FUNCS ** ----------------------------
# ------------------------------------------------------------------------------
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-

def GS_search_space(adata, plot_type = "sil_mean"):
    """\
    Get a heatmap of the search space traversed by Grid Search (GS).

    Parameters
    ----------
    adata
        An anndata object that was previously given to GS
    plot_type : default: "sil_mean"
         A column name in adata.uns["GS_results_dict"]["search_df"].
         Among other, options include "sil_mean" and "n_clust".
    """
    return _GS_search_space(adata, plot_type)

def SA_search_space(adata, plot_type = "sil_mean", plot_density = True):
    # https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale
    """\
    Get a dot plot of the search space traversed by Simulated Annealing (SA).

    Parameters
    ----------
    adata
        An anndata object that was previously given to GS
    plot_type : default: "sil_mean"
         A column name in adata.uns["GS_results_dict"]["search_df"].
         Among other, options include "sil_mean" and "n_clust".
    plot_density : default: True
        Whether to plot density on the dotplot to identify regions that were
        highly traversed by SA.
    """
    return _SA_search_space(adata, plot_type, plot_density)

def metric_vs_n_clusts(
    adata,
    metric = "sil_mean",
    width = 5,
    height = 5,
    xlabel = 'number of clusters',
    ylabel = None,
    axis_fontsize = 14
):
    """\
    Get a dot plot of the search space traversed by Simulated Annealing (SA).

    Parameters
    ----------
    adata
        An anndata object that was previously given to GS
    metric : default: "sil_mean"
         A column name in adata.uns["GS_results_dict"]["search_df"].
         Among other, options include "sil_mean".
    width : default: 5
        Figure width (inches)
    height : default: 5
        Figure height (inches)
    xlabel : default: 'number of clusters'
        x-axis label
    ylabel : default: None
        When None, ylabel will be metric.
    axis_fontsize : default: 14
        Fontsize for xlabel and ylabel.
    """
    return _metric_vs_n_clusts(
        adata,
        metric,
        width,
        height,
        xlabel,
        ylabel,
        axis_fontsize
    )
