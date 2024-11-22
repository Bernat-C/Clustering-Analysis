import numpy as np

class CustomKMeans:
    def __init__(self, n_clusters, init=None, max_iters=100, tolerance=1e-4):

        self.n_clusters = n_clusters
        self.init = init
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.centroids = None

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
        distances = self.euclidean_distance(data, self.centroids)
        cluster_ids = np.argmin(distances, axis=1)
        return cluster_ids

    def transform(self, data):
        return self.euclidean_distance(data, self.centroids)
    
    def euclidean_distance(self, X, centers):
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        return distances