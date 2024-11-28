import numpy as np
from scipy.stats import normaltest
from kmeans import CustomKMeans
from utils import compute_final_clusters
from sklearn.decomposition import PCA
import pandas as pd
from metrics import get_metrics


class GMeans:
    def __init__(self, max_clusters=10, max_iters=100, tolerance=1e-4, distance='euclidean', alpha=0.05):
        self.max_clusters = max_clusters
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.alpha = alpha
        self.centroids = []
        if distance == 'euclidean':
            self.distance = self.euclidean_distance
        elif distance == 'manhattan':
            self.distance = self.manhattan_distance
        elif distance == 'cosine':
            self.distance = self.cosine_distance
    
    def fit(self, data):
        n_samples, n_features = data.shape
        initial_center = [data.mean(axis=0)]  # Step 1: Start with the mean as the initial center
        self.centroids = initial_center
        while len(self.centroids) < self.max_clusters:
            # Step 2: Apply K-Means
            kmeans = CustomKMeans(n_clusters=len(self.centroids), init=self.centroids, distance=self.distance)
            kmeans.fit(data)
            self.centroids = kmeans.centroids  # Initialize with the first centroid
            new_centroids = []
            split = False

            for j, centroid in enumerate(self.centroids):
                # Step 3: Get points assigned to the current centroid
                cluster_points = data[self.predict(data) == j]
                if cluster_points.shape[0] < 16:  # Skip small clusters
                    new_centroids.append(centroid)

                # Step 4: Check normality
                direction = np.random.randn(n_features)  # Random direction vector
                projected = np.dot(cluster_points - centroid, direction)
                stat, p_value = normaltest(projected)

                if p_value < self.alpha:  # Step 5: Not Gaussian, split cluster
                    split = True
                    # Compute principal component
                    pca = PCA(n_components=1)
                    pca.fit(cluster_points)
                    principal_component = pca.components_[0]
                    eigenvalue = pca.explained_variance_[0]

                    # Define m based on the principal component
                    m = principal_component * np.sqrt(2 * eigenvalue)

                    # Add new centroids c ± m
                    new_centroids.append(centroid + m)
                    new_centroids.append(centroid - m)
                else:
                    new_centroids.append(centroid)
                if len(new_centroids) >= self.max_clusters:
                    break

            if not split:  # Step 6: Stop if no new centers are added
                break

            self.centroids = np.array(new_centroids)

        self.centroids = self.centroids

    def predict(self, data):
        distances = self.distance(data, self.centroids)
        return np.argmin(distances, axis=1)

    def euclidean_distance(self, X, centers):
        return np.linalg.norm(X[:, np.newaxis] - centers, axis=2)

    def manhattan_distance(self, X, centers):
        return np.sum(np.abs(X[:, np.newaxis] - centers), axis=2)

    def cosine_distance(self, X, centers):
        norm_X = np.linalg.norm(X, axis=1)[:, np.newaxis]
        norm_centers = np.linalg.norm(centers, axis=1)
        similarity = np.dot(X, centers.T) / (norm_X * norm_centers)
        return 1 - similarity


def run_gmeans(data, max_clusters, distance='euclidean'):
    data = np.array(data)
    kmeans = GMeans(max_clusters=max_clusters,distance=distance)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    centers = kmeans.centroids
    clusters = compute_final_clusters(data, labels, centers)
    return clusters

def run_all_kmeans(data_X, data_y):
    results = []
    data_X = np.array(data_X)
    data_y = np.array(data_y)
    for k in range(2, 8):
        for dist in ['euclidean', 'manhattan', 'cosine']:
            kmeans = GMeans(max_clusters=k,distance=dist)
            kmeans.fit(data_X)
            labels_pred = kmeans.predict(data_X)
            results_kmeans = get_metrics(data_X, data_y, labels_pred, k, dist)
            results.append(results_kmeans)

    # Convert to DataFrame
    results = pd.DataFrame(results)
    return results