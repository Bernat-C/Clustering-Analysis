import numpy as np

def update_centers(data, member_matrix, m):
    """
    Update the cluster centers based on the membership matrix.
    """
    n_clusters = member_matrix.shape[0]
    n_features = data.shape[1]

    centers = np.zeros((n_clusters, n_features))

    for i in range(n_clusters):
        um = member_matrix[i, :] ** m  # Raise membership values to the power of m
        centers[i] = np.dot(um, data) / np.sum(um)  # Compute new center as weighted average

    return centers

def update_membership_matrix(data, centers, m):
    """
    Update the membership matrix based on the current centers.
    """
    n_samples = data.shape[0]
    n_clusters = centers.shape[0]

    # Compute the distance between each data point and each center
    dist = np.zeros((n_clusters, n_samples))
    for i in range(n_clusters):
        dist[i, :] = np.linalg.norm(data - centers[i], axis=1)

    # Compute the inverse distance raised to the power of (m-1)
    dist = np.maximum(dist, 1e-10)  # Avoid division by zero
    inv_dist = 1.0 / dist
    inv_dist_m = inv_dist ** (m - 1)

    # Normalize to get membership matrix
    membership_matrix = inv_dist_m / np.sum(inv_dist_m, axis=0)

    return membership_matrix

def compute_final_clusters(data, centers, membership_matrix, n_clusters):
    """
    Compute the final fuzzy clusters based on the membership matrix and centers.
    """
    clusters = {i: {'points': [], 'center': centers[i]} for i in range(n_clusters)}

    # Assign points to clusters based on the highest membership
    for el in range(len(data)):
        cluster_idx = np.argmax(membership_matrix[:, el])
        clusters[cluster_idx]['points'].append(data.iloc[el])

    return clusters

def gs_fcm(data, n_clusters, m=2, max_iter=100, tolerance=1e-5):

    n_samples, n_features = data.shape

    # Randomly initialize membership matrix
    member_matrix = np.random.rand(n_clusters, n_samples)
    member_matrix /= np.sum(member_matrix, axis=0)  # Normalize so that each column sums to 1
    centers = np.dot(member_matrix, data) / np.sum(member_matrix, axis=1)[:, None]  # Compute initial centers

    prev_centers = np.zeros_like(centers)

    # Optimization loop
    for iter in range(max_iter):
        # Update centers based on membership matrix
        centers = update_centers(data, member_matrix, m)

        # Update membership matrix based on new centers
        member_matrix = update_membership_matrix(data, centers, m)

        # Check for convergence (if centers do not change significantly)
        if np.linalg.norm(centers - prev_centers) < tolerance:
            break

        prev_centers = centers.copy()

    return compute_final_clusters(data, centers, member_matrix, n_clusters)