import numpy as np
from kmeans import CustomKMeans
from utils import compute_final_clusters, plot_clusters
from sklearn.decomposition import PCA
import pandas as pd
from metrics import get_metrics_general
from scipy.stats import anderson
import time


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
        self.centroids = [data.mean(axis=0)]  # Start with the mean of the data

        while len(self.centroids) < self.max_clusters:
            kmeans = CustomKMeans(n_clusters=len(self.centroids), init=self.centroids)
            kmeans.fit(data)
            self.centroids = kmeans.centroids
            new_centroids = []
            split = False

            for j, centroid in enumerate(self.centroids):
                if len(self.centroids[j:])+len(new_centroids)>=self.max_clusters:
                    new_centroids.extend(self.centroids[j:])
                    break
                if len(new_centroids) +1 >= self.max_clusters:
                    new_centroids.append(centroid)
                    break
                # Get points in the cluster
                cluster_points = data[self.predict(data) == j]
                if cluster_points.shape[0] < 16:  # Skip small clusters
                    new_centroids.append(centroid)
                    continue

                # Step 1: Initialize child centers using PCA
                c1, c2 = self.initialize_centers_pca(cluster_points, centroid)

                # Step 2: Run K-Means with two centers
                child_kmeans = CustomKMeans(n_clusters=2, init=[c1, c2])
                child_kmeans.fit(cluster_points)
                c1, c2 = child_kmeans.centroids

                # Step 3: Project data onto the direction defined by c1 and c2
                v = c1 - c2
                v_norm = np.linalg.norm(v)
                projection = np.dot(cluster_points - centroid, v / v_norm)

                # Step 4: Normalize the projection (mean 0, variance 1)
                projection = (projection - projection.mean()) / projection.std()

                # Step 5: Apply Anderson-Darling test
                stat, critical_values, significance_levels = anderson(projection)
                if stat > critical_values[np.searchsorted(significance_levels, self.alpha * 100)]:
                    # Reject H0: Split the cluster
                    split = True
                    new_centroids.extend([c1, c2])
                else:
                    # Accept H0: Keep the original center
                    new_centroids.append(centroid)



            self.centroids = np.array(new_centroids)

    def initialize_centers_pca(self, cluster_points, centroid):
        # Compute PCA on the cluster points
        pca = PCA(n_components=1)
        pca.fit(cluster_points - centroid)
        principal_component = pca.components_[0]
        eigenvalue = pca.explained_variance_[0]

        # Calculate m = s * sqrt(2 * eigenvalue)
        m = principal_component * np.sqrt(2 * eigenvalue)

        # Initialize child centers
        c1 = centroid + m
        c2 = centroid - m
        return c1, c2

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
    return clusters, labels

def run_all_gmeans(data_X, data_y):
    results = []
    data_X = np.array(data_X)
    data_y = np.array(data_y)
    for k in range(2, 13):
        for dist in ['euclidean', 'manhattan', 'cosine']:
            start = time.time()
            kmeans = GMeans(max_clusters=k,distance=dist)
            kmeans.fit(data_X)
            labels_pred = kmeans.predict(data_X)
            execution_time = time.time()-start
            k_found = len(np.unique(labels_pred))
            results_kmeans = get_metrics_general(data_X, data_y, labels_pred, f"gmeans_k{k}_distance-{dist}_kfound{k_found}", execution_time)
            results.append(results_kmeans)

    # Convert to DataFrame
    results = pd.DataFrame(results)
    return results