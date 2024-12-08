import numpy as np
from utils import compute_final_clusters
from metrics import get_metrics_general
import pandas as pd
import time

class CustomKMeans:
    def __init__(self, n_clusters, init=None, distance='euclidean', max_iters=100, tolerance=1e-4):

        self.n_clusters = n_clusters
        self.init = init
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.centroids = None
        if distance == 'euclidean':
            self.distance = self.euclidean_distance
        elif distance == 'manhattan':
            self.distance = self.manhattan_distance
        elif distance == 'cosine':
            self.distance = self.cosine_distance

    def fit(self, data):

        n_samples, n_features = data.shape

        # Initialize centroids
        if self.init is not None:
            self.centroids = np.array(self.init)
        else:
            np.random.seed(42)
            random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            self.centroids = data[random_indices]

        for iteration in range(self.max_iters):
            # Assignment step: Compute distances and assign points to the nearest cluster
            distances = self.euclidean_distance(data, self.centroids)
            cluster_ids = np.argmin(distances, axis=1)

            # Update step: Compute new centers as the mean of points assigned to each cluster
            new_centroids = np.array([
                data[cluster_ids == i].mean(axis=0) if np.any(cluster_ids == i) else self.centroids[i]
                for i in range(self.n_clusters)
            ])

            # Check for convergence
            center_shift = np.sum(np.linalg.norm(new_centroids - self.centroids, axis=1))
            if center_shift < self.tolerance:
                break

            self.centroids = new_centroids

        # Store final cluster assignments and distances
        self.cluster_ids_ = cluster_ids
        self.distances_ = distances

    def predict(self, data):
        distances = self.distance(data, self.centroids)
        cluster_ids = np.argmin(distances, axis=1)
        return cluster_ids

    def transform(self, data):
        return self.distance(data, self.centroids)
    
    def euclidean_distance(self, X, centers):
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        return distances

    def manhattan_distance(self, X, centers):
        distances = np.sum(np.abs(X[:, np.newaxis] - centers), axis=2)
        return distances

    def cosine_distance(self, X, centers):
        norm_X = np.linalg.norm(X, axis=1)[:, np.newaxis]
        norm_centers = np.linalg.norm(centers, axis=1)
        similarity = np.dot(X, centers.T) / (norm_X * norm_centers)  # Cosine similarity
        return 1 - similarity  # Return 1 - cosine similarity for clustering

def run_kmeans(data, n_clusters, init=None, distance='euclidean'):
    data = np.array(data)
    kmeans = CustomKMeans(n_clusters=n_clusters, init=init, distance=distance, max_iters=100, tolerance=1e-4)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    centers = kmeans.centroids
    clusters = compute_final_clusters(data, labels, centers)
    return clusters, labels

def run_all_kmeans(data_X, data_y):
    results = []
    data_X = np.array(data_X)
    data_y = np.array(data_y)
    for k in range(2, 13):
        for dist in ['euclidean', 'manhattan', 'cosine']:
            for i in range(20):
                start = time.time()
                random_indices = np.random.choice(len(data_X), k, replace=False)
                centroids = data_X[random_indices]
                kmeans = CustomKMeans(n_clusters=k, init=centroids, distance=dist, max_iters=100, tolerance=1e-4)
                kmeans.fit(data_X)
                labels_pred = kmeans.predict(data_X)
                execution_time = time.time()-start
                results_kmeans = get_metrics_general(data_X, data_y, labels_pred, f"kmeans_k{k}_distance-{dist}", execution_time)
                results.append(results_kmeans)

    # Convert to DataFrame
    results = pd.DataFrame(results)
    return results

