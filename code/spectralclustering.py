import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


def spectral_clustering(dataset, n_clusters=3, affinity='rbf', assign_labels='kmeans', n_neighbors=None, plot=True):
    # Standardize the data
    data = StandardScaler().fit_transform(dataset)

    # Configure affinity
    if affinity == 'nearest_neighbors' and n_neighbors:
        connectivity = kneighbors_graph(data, n_neighbors=n_neighbors, include_self=False)
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            assign_labels=assign_labels,
            connectivity=connectivity
        )
    else:
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            assign_labels=assign_labels
        )

    # Perform clustering
    labels = model.fit_predict(data)

    # Calculate Silhouette Score
    if len(set(labels)) > 1:
        silhouette = silhouette_score(data, labels)
        print(f"Silhouette Score: {silhouette:.4f}")
    else:
        silhouette = -1  # Invalid clustering
        print("Only one cluster was formed. Silhouette Score not calculated.")

    # Visualization
    if plot:
        plt.figure(figsize=(8, 6))

        # Generate unique colors for each cluster
        unique_labels = np.unique(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

        # Plot each cluster with a specific color
        for label, color in zip(unique_labels, colors):
            plt.scatter(
                data[labels == label, 0], data[labels == label, 1],
                color=color, label=f'Cluster {label}', s=50, alpha=0.7
            )

        plt.title('Spectral Clustering Visualization')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.show()

    return labels, silhouette
