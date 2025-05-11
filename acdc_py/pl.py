from ._pl import _GS_search_space, _SA_search_space, _metric_vs_n_clusts, _silhouette_scores, _plot_diffusion_map

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

    Returns
    -------
    A object of :class:~matplotlib.figure.Figure containing the plot.
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

    Returns
    -------
    A object of :class:~matplotlib.figure.Figure containing the plot.
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

def silhouette_scores(
    adata,
    groupby,
    dist_slot,
    palette=None,
    ylab = None,
    show = True
):
    """\
    Get a dot plot of the search space traversed by Simulated Annealing (SA).

    Parameters
    ----------
    adata
        An anndata object.
    groupby
        A name of the column in adata.obs that contains the clustering that you
        want to calculate silhouette scores for.
    dist_slot
        The slot in adata.obsp where the distance object that will be used to
        calculate the silhouette score is stored.
    palette : default: None
        The name of a Matplotlib qualitative colormap. If None, use ACDC
        default palette.
    ylab : default: None
        The label to put on the y-axis.
    show : default: True
        Whether to show the plot.
    """
    _silhouette_scores(adata, groupby, dist_slot, palette, ylab, show)


def plot_diffusion_map(ref_adata, query_adata):
    """\
    Visualize reference and query diffusion embeddings on a 2D scatter plot grid.

    Generates a 2x2 grid of plots displaying the first two diffusion components:
    - Top-left: reference dataset alone.
    - Top-right: query dataset alone.
    - Bottom-left: combined reference and query datasets.
    - Bottom-right: unused (blank).

    Parameters
    ----------
    ref_adata : AnnData
        Annotated data containing diffusion coordinates in `ref_adata.obsm['X_diffmap']`.
    query_adata : AnnData
        Annotated data containing diffusion coordinates in `query_adata.obsm['X_diffmap']`.

    Returns
    -------
    None
        Displays the matplotlib figure with diffusion map plots.
    """
    _plot_diffusion_map(ref_adata, query_adata)