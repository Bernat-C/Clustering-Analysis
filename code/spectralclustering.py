import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering


def spectral_clustering(data, n_clusters=3, affinity='rbf', n_neighbors=None, assign_labels='kmeans',  eigen_solver
= 'arpack'):

    # Configure affinity
    if affinity == 'nearest_neighbors' and n_neighbors:
        model= SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            assign_labels=assign_labels,
            eigen_solver=eigen_solver,
            n_neighbors=n_neighbors,
            random_state=42
        )
    else:
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            assign_labels=assign_labels,
            eigen_solver=eigen_solver,
            random_state=42
        )

    # Perform clustering
    labels = model.fit_predict(data)


    return labels
