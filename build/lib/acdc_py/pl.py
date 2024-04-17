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
    plot_type (default: "sil_mean")
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
    plot_type (default: "sil_mean")
         A column name in adata.uns["GS_results_dict"]["search_df"].
         Among other, options include "sil_mean" and "n_clust".
    plot_density (default: True)
        Whether to plot density on the dotplot to identify regions that were
        highly traversed by SA.
    """
    return _SA_search_space(adata, plot_type, plot_density)
