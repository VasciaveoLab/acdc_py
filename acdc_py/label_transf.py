import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

def transfer_labels_anndata(
    ref_adata,
    query_adata,
    embedding_key='diffmap',
    label_key='cell_type',
    n_neighbors=15,
    pca_comps=None,
    ground_truth_label=None,
    plot_labels=False
):
    """
    Transfer cell-type labels from a reference AnnData to query AnnData using KNN.

    Parameters
    ----------
    ref_adata : AnnData
        Annotated data with known labels in .obs[label_key]. For 'diffmap', expects
        the diffusion map in .obsm[embedding_key]; for 'pca' or 'X', uses .X.
    query_adata : AnnData
        Annotated data to annotate; should have the same representation available.
    embedding_key : str, optional
        Which embedding to use: 'diffmap', 'pca', or 'X'. Default is 'diffmap'.
    label_key : str, optional
        Key in .obs for storing predicted labels. Default is 'cell_type'.
    n_neighbors : int, optional
        Number of neighbors for the KNN classifier. Default is 15.
    pca_comps : int or None, optional
        If specified and embedding_key=='pca', number of PCA components to compute.
    ground_truth_label : str or None, optional
        If provided, key in query_adata.obs for true labels used to compute accuracy.
    plot_labels : bool, optional
        If True, generate an embedding plot colored by predicted and ground-truth labels.
    """
    # Optional PCA preprocessing: compute and store PC scores in .obsm
    if pca_comps is not None:
        pca = PCA(n_components=pca_comps)
        # Fit PCA on reference X and transform both datasets
        ref_adata.obsm['X_pca'] = pca.fit_transform(ref_adata.X)
        query_adata.obsm['X_pca'] = pca.transform(query_adata.X)

    # Ensure raw X is dense matrix if using 'X' embedding
    if embedding_key == 'X':
        # Convert sparse X to dense arrays for both AnnData objects
        if hasattr(ref_adata.X, 'toarray'):
            ref_adata.obsm['X'] = ref_adata.X.toarray()
        else:
            ref_adata.obsm['X'] = ref_adata.X
        if hasattr(query_adata.X, 'toarray'):
            query_adata.obsm['X'] = query_adata.X.toarray()
        else:
            query_adata.obsm['X'] = query_adata.X

    # Extract feature matrices for KNN
    X_ref = ref_adata.obsm.get(embedding_key)
    X_query = query_adata.obsm.get(embedding_key)
    # Validate embeddings are present
    if X_ref is None or X_query is None:
        raise ValueError(f"Embedding '{embedding_key}' not found in .obsm of one or both AnnData objects.")

    # Train KNN classifier on reference embedding and labels
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_ref, ref_adata.obs[label_key].astype(str).values)

    # Predict query labels and store as categorical
    predicted = knn.predict(X_query)
    query_adata.obs[label_key] = pd.Categorical(predicted)

    # Annotate dataset origin for combined plotting or inspection
    ref_adata.obs['dataset'] = 'reference'
    query_adata.obs['dataset'] = 'query'

    # Align category ordering consistently
    def reorder(obs_df, key):
        cats = sorted(obs_df.obs[key].cat.categories)
        obs_df.obs[key] = obs_df.obs[key].cat.reorder_categories(cats)

    reorder(ref_adata, label_key)
    reorder(query_adata, label_key)
    reorder(query_adata, ground_truth_label)


    print(f"Labels transferred to query .obs['{label_key}'] using {embedding_key} embedding.")

    # Compute and print accuracy if ground truth provided
    if ground_truth_label is not None:
        true = query_adata.obs[ground_truth_label].astype(str)
        pred = query_adata.obs[label_key].astype(str)
        accuracy = (true.values == pred.values).mean()
        print(f"Accuracy against '{ground_truth_label}': {accuracy:.2f}")

        # Optionally plot predicted vs ground truth
        if plot_labels:
            # Generate the plot and get the figure
            fig = sc.pl.embedding(
                query_adata,
                basis=embedding_key,
                color=[label_key, ground_truth_label],
                title=[f"Predicted {label_key}", f"{ground_truth_label}"],
                return_fig=True
            )

            # Access the axes of the figure
            axs = fig.axes  # This gives you a list of AxesSubplot objects

            # Add accuracy to the top right of the first subplot (predicted labels)
            ax_pred = axs[0]
            ax_pred.text(
                0.95, 0.95,
                f"Accuracy: {accuracy:.2f}",
                transform=ax_pred.transAxes,
                fontsize=12,
                ha='right',
                va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5)
            )
