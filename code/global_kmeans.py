

import numpy as np
from kmeans import CustomKMeans
from utils import compute_final_clusters
import pandas as pd
from metrics import get_metrics

class GlobalKMeans:
    def __init__(self, max_clusters, distance):
        self.distance = distance
        self.max_clusters = max_clusters
        if distance == 'euclidean':
            self.distance = self.euclidean_distance
        elif distance == 'manhattan':
            self.distance = self.manhattan_distance
        elif distance == 'cosine':
            self.distance = self.cosine_distance

    def fit(self, data):
        N, d = data.shape
        self.centroids = np.empty((0, d))  # Initialize as an empty NumPy array
        
        for k in range(1, self.max_clusters + 1):
            best_inertia = float('inf')
            best_centers = None

            if k == 1:
                # Start with the centroid of all points for k=1
                initial_center = np.mean(data, axis=0).reshape(1, -1)
                kmeans = CustomKMeans(n_clusters=1, init=initial_center, distance=self.distance)
                kmeans.fit(data)
                self.centroids = kmeans.centroids  # Initialize with the first centroid
                continue

            for i in range(N):
                # Fix k-1 centers from the previous solution
                fixed_centers = self.centroids
                # Add a new cluster center at the position of data point X[i]
                initial_centers = np.vstack([fixed_centers, data[i].reshape(1, -1)])
                kmeans = CustomKMeans(n_clusters=k, init=initial_centers, distance=self.distance)
                kmeans.fit(data)

                # Compare inertia (sum of squared distances to the centroids)
                if np.sum(kmeans.distances_) < best_inertia:
                    best_inertia = np.sum(kmeans.distances_)
                    best_centers = kmeans.centroids

            # Store the best centroids for this k value
            self.centroids = np.vstack([self.centroids, best_centers[-1]])  # Append the new centroid as NumPy array

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
    kmeans = GlobalKMeans(max_clusters=max_clusters,distance=distance)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    centers = kmeans.centroids
    clusters = compute_final_clusters(data, labels, centers)
    return clusters

def run_all_global_kmeans(data_X, data_y):
    results = []
    data_X = np.array(data_X)
    data_y = np.array(data_y)
    for k in range(2, 8):
        for dist in ['euclidean', 'manhattan', 'cosine']:
            kmeans = GlobalKMeans(max_clusters=k,distance=dist)
            kmeans.fit(data_X)
            labels_pred = kmeans.predict(data_X)
            results_kmeans = get_metrics(data_X, data_y, labels_pred, k, dist)
            results.append(results_kmeans)

    # Convert to DataFrame
    results = pd.DataFrame(results)
    return results