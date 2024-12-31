import numpy as np

config = {
    "clust_alg":"Leiden",
    "SS":{
        "SS_weights": "unitary",
        "SS_exp_base": 2.718282,
        "n_subsamples": 1,
        "subsamples_pct_cells": 100
    },
    "corr_distance_dtype":np.int16
}

def set_clust_alg(clust_alg = "Leiden"):
    """\
    clust_alg : default: "Leiden"
    Clustering algorithm. Choose among: "Leiden" (default) or "Louvain".
    """
    config['clust_alg'] = clust_alg

def set_SS_weights(SS_weights = "unitary", SS_exp_base = 2.718282):
    """\
    SS_weights : default: "unitary"
        Negative silhouette scores can be given more weight by exponentiation
        ("exp"). Otherwise, leave SS_weights as "unitary".
    SS_exp_base : default: 2.718282.
        If SS_weights is set to "exp", then set the base for exponentiation.
    """
    config['SS']["SS_weights"] = SS_weights
    config['SS']["SS_exp_base"] = SS_exp_base

def set_SS_bootstraps(n_subsamples = 1, subsamples_pct_cells = 100):
    """\
    n_subsamples : default: 1
        Number of subsamples per bootstrap.
    subsamples_pct_cells : default: 100
        Percentage of cells sample at each bootstrap iteration.
        i.e. when 100, 100%, all cells are used).
    """
    config['SS']["n_subsamples"] = n_subsamples
    config['SS']["subsamples_pct_cells"] = subsamples_pct_cells

def set_corr_distance_dtype(dtype=np.int16):
    """\
    dtype : default: np.int16
        Data type used to represent the distance values. np.int16 (default) is
        a compromise between smaller memory size while not reducing information
        so much as to affect clustering. dtypes include np.int8, np.int16 (default) np.int32, np.int64, np.float16, np.float32, and np.float64.
    """
    config['corr_distance_dtype'] = dtype


# def set_regulators_filepath(group, species, new_filepath):
#     """\
#     Allows the user to use a custom list of regulatory proteins instead of the
#     default ones within pyVIPER's data folder.
#
#     Parameters
#     ----------
#     group
#         A group of regulatory proteins of either: "tfs", "cotfs", "sig" or "surf".
#     species
#         The species to which the group of proteins belongs to: "human" or "mouse".
#     new_filepath
#         The new filepath that should be used to retrieve these sets of proteins.
#
#     Returns
#     -------
#     None
#     """
#     if not species in ["human", "mouse"]:
#         raise ValueError("Unsupported species: " + str(species))
#     if not group in ["tfs", "cotfs", "sig", "surf"]:
#         raise ValueError("Unsupported species: " + str(group))
#     config['regulators_filepaths'][species][group] = new_filepath
#
# def set_regulators_species_to_use(species):
#     """\
#     Allows the user to specify which species they are currently studying, so the
#     correct sets of regulatory proteins will be used during analysis.
#
#     Parameters
#     ----------
#     species
#         The species to which the group of proteins belongs to: "human" or "mouse".
#
#     Returns
#     -------
#     None
#     """
#     if not species in ["human", "mouse"]:
#         raise ValueError("Unsupported species: " + str(species))
#     config['regulators_species'] = species
