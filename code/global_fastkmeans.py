

import numpy as np
from kmeans import CustomKMeans
from utils import compute_final_clusters
import pandas as pd
from metrics import get_metrics_general
import time


class FastGlobalKMeans:
    def __init__(self, max_clusters, distance):
        self.max_clusters = max_clusters
        self.distance_metric = distance
        
        # Map distance metrics to functions
        if distance == 'euclidean':
            self.distance = self.euclidean_distance
        elif distance == 'manhattan':
            self.distance = self.manhattan_distance
        elif distance == 'cosine':
            self.distance = self.cosine_distance

    def fit(self, data):
        N, d = data.shape
        self.centroids = np.empty((0, d))  # Initialize as an empty NumPy array

        # Pairwise squared distances matrix for efficiency
        pairwise_distances = self.pairwise_squared_distances(data)

        for k in range(1, self.max_clusters + 1):
            if k == 1:
                # Start with the centroid of all points for k=1
                initial_center = np.mean(data, axis=0).reshape(1, -1)
                kmeans = CustomKMeans(n_clusters=1, init=initial_center, distance=self.distance)
                kmeans.fit(data)
                self.centroids = kmeans.centroids
                continue

            # Compute b_n values for all data points
            d_k_minus_1 = self.compute_closest_distances(data, self.centroids)
            b_values = self.compute_bn(data, d_k_minus_1, pairwise_distances)

            # Select the point that maximizes b_n
            best_point_index = np.argmax(b_values)
            best_initial_center = data[best_point_index].reshape(1, -1)

            # Initialize k-means with k-1 fixed centers + the new center
            initial_centers = np.vstack([self.centroids, best_initial_center])
            kmeans = CustomKMeans(n_clusters=k, init=initial_centers, distance=self.distance)
            kmeans.fit(data)

            # Update centroids for k clusters
            self.centroids = kmeans.centroids

    def compute_bn(self, data, d_k_minus_1, pairwise_distances):
        """
        Compute b_n values for each data point using the upper bound formula.
        """
        N = data.shape[0]
        b_values = np.zeros(N)

        for i in range(N):
            b_sum = 0
            for j in range(N):
                reduction = max(d_k_minus_1[j] - pairwise_distances[i, j], 0)
                b_sum += reduction
            b_values[i] = b_sum
        return b_values

    def compute_closest_distances(self, data, centers):
        """
        Compute the squared distance between each point and its closest center.
        """
        distances = self.distance(data, centers)
        closest_distances = np.min(distances ** 2, axis=1)  # Square distances
        return closest_distances

    def pairwise_squared_distances(self, data):
        """
        Precompute all pairwise squared distances for efficiency.
        """
        N = data.shape[0]
        pairwise_distances = np.zeros((N, N))

        for i in range(N):
            for j in range(i, N):  # Symmetric matrix
                pairwise_distances[i, j] = np.sum((data[i] - data[j]) ** 2)
                pairwise_distances[j, i] = pairwise_distances[i, j]
        return pairwise_distances

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

    def predict(self, data):
        distances = self.distance(data, self.centroids)
        cluster_ids = np.argmin(distances, axis=1)
        return cluster_ids

    def transform(self, data):
        return self.distance(data, self.centroids)

def run_global_kmeans(data, max_clusters, distance='euclidean'):
    data = np.array(data)
    kmeans = FastGlobalKMeans(max_clusters=max_clusters,distance=distance)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    centers = kmeans.centroids
    clusters = compute_final_clusters(data, labels, centers)
    return clusters

def run_all_global_kmeans(data_X, data_y):
    results = []
    data_X = np.array(data_X)
    data_y = np.array(data_y)
    for k in range(2, 13):
        for dist in ['euclidean', 'manhattan', 'cosine']:
            start = time.time()
            kmeans = FastGlobalKMeans(max_clusters=k,distance=dist)
            kmeans.fit(data_X)
            labels_pred = kmeans.predict(data_X)
            execution_time = time.time()-start
            k_found = len(np.unique(labels_pred))
            results_kmeans = get_metrics_general(data_X, data_y, labels_pred, f"GlobalKmeans_k{k}_distance-{dist}_kfound{k_found}", execution_time)
            results.append(results_kmeans)

    # Convert to DataFrame
    results = pd.DataFrame(results)
    return results

"""data = pd.read_csv('./datasets_processed/grid.csv')
data = data.iloc[:,:-1]
run_global_kmeans(data, 3, distance='euclidean')"""