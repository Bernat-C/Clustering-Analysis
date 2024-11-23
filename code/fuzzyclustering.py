import numpy as np

def update_centers(data, member_matrix, m):
    """
    Update the cluster centers based on the membership matrix.
    """
    n_clusters = member_matrix.shape[0]
    n_features = data.shape[1]

    centers = np.zeros((n_clusters, n_features)) # N clusters centers, and N features will describe each center.

    for i in range(n_clusters):
        member_center_m = member_matrix[i, :] ** m  # Raise membership values to the power of m
        centers[i] = np.dot(member_center_m, data) / np.sum(member_center_m)  # vi = sum_k(u^m*x) / sum_k(u^m)

    return centers

def update_membership_matrix(data, centers, m):
    """
    Update the membership matrix based on the current centers.
    """
    n_samples = data.shape[0]
    n_clusters = centers.shape[0]

    # Compute the distance between each data point and each center
    dist = np.zeros((n_clusters, n_samples)) # N clusters, and each cluster will have a probability for each sample

    for i in range(n_clusters):
        dist[i, :] = np.linalg.norm(data - centers[i], axis=1) # Axis = 1 as we are performing the norm between the samples and the cluster center

    # Compute the inverse distance raised to the power of (m-1)
    dist = np.maximum(dist, 1e-10)  # Avoid division by zero
    dist_m = dist ** (-2 / (m - 1))

    # Normalize to get membership matrix
    membership_matrix = dist_m / np.sum(dist_m, axis=0) # Axis = 0 as we are normalizing across clusters

    return membership_matrix

def compute_final_clusters(data, centers, membership_matrix, n_clusters):
    """
    Compute the final clusters (not fuzzy anymore) based on the membership matrix and centers.
    """
    clusters = {i: {'points': [], 'center': centers[i]} for i in range(n_clusters)}

    # Assign points to clusters based on the highest membership
    for el in range(len(data)):
        cluster_idx = np.argmax(membership_matrix[:, el])
        clusters[cluster_idx]['points'].append(data.iloc[el])

    return clusters

def suppress_membership_matrix(membership_matrix, alpha=0.5):
    """
    Perform the suppression step of the suppressed fuzzy c-means (s-FCM) algorithm.

    Parameters:
    - member_matrix: current membership matrix (n_clusters x n_samples).
    - alpha: suppression factor (0-1, default 0.5).
    """
    n_clusters, n_samples = membership_matrix.shape

    suppressed_u = np.zeros_like(membership_matrix)

    # Perform suppression on membership values
    for k in range(n_samples):  # Iterate through samples
        max_idx = np.argmax(membership_matrix[:,k])  # Find the cluster with the max membership
        for i in range(n_clusters):
            if i == max_idx:
                suppressed_u[i,k] = 1 - alpha + alpha * membership_matrix[i,k]  # Apply suppression to all memberships
            else:
                suppressed_u[i,k] = alpha * membership_matrix[i,k]  # Adjust the max membership
    return suppressed_u

def s_fcm(data, n_clusters, m=2, max_iter=1000, tolerance=1e-5, suppress=False, alpha=0.5):
    """
    :param data: Samples x features
    :param n_clusters: Number of clusters
    :param m: Fuzzification parameter
    :param max_iter: Maximum number of iterations run without convergence
    :param tolerance: Tolerance for convergence
    :param suppress: Perform suppression or not
    :param alpha: Suppression paramter (0-1, default 0.5)
    :return:
    """

    n_samples = data.shape[0]

    # Randomly initialize membership matrix
    member_matrix = np.random.rand(n_clusters, n_samples)
    member_matrix /= np.sum(member_matrix, axis=0)  # Normalize so that each column sums to 1
    centers = np.dot(member_matrix, data) / np.sum(member_matrix, axis=1)[:, None]  # Compute initial centers

    prev_centers = np.zeros_like(centers)

    # Optimization loop
    for iter in range(max_iter):
        # Update membership matrix based on new centers
        member_matrix = update_membership_matrix(data, centers, m)

        # Perform suppression if indicated
        if suppress:
            member_matrix = suppress_membership_matrix(member_matrix, alpha=alpha)

        # Update centers based on membership matrix
        centers = update_centers(data, member_matrix, m)

        # Check for convergence (if centers do not change significantly)
        if np.sum(np.linalg.norm(centers - prev_centers, axis=0)) < tolerance: # Check if the sum of individual norms for each cluster changes significantly
            print(f"Congence after {iter} iterations.")
            break

        prev_centers = centers.copy()

    return compute_final_clusters(data, centers, member_matrix, n_clusters)