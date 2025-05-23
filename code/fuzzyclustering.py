import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from metrics import get_metrics_general, get_metrics_fuzzy, xie_beni


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

    dist_m = (dist + 1e-5) ** (-2 / (m - 1)) # Avoid infinity

    # Normalize to get membership matrix
    membership_matrix = dist_m / np.sum(dist_m, axis=0) # Axis = 0 as we are normalizing across clusters

    return membership_matrix

def defuzzyfy(membership_matrix):
    # Assign points to clusters based on the highest membership
    cluster_memberships = np.argmax(membership_matrix, axis=0)
    return cluster_memberships

def get_cluster_list(data, centers, assignments, n_clusters):
    """
    Compute the final clusters (not fuzzy anymore) based on the membership matrix and centers.
    """
    clusters = {i: {'points': [], 'center': centers[i]} for i in range(n_clusters)}

    data = np.array(data)

    for point, cluster_id in zip(data, assignments):
        clusters[cluster_id]['points'].append(point)

    return clusters

## GENERALIZATION

def update_alpha_theta(u_w, m, eta=0.5):
    """
    Returns the updated value for alpha according to the first generalized suppression rule presented in the paper.

    Parameters:
    - u_w (float): The largest membership value for the sample.
    - m (float): Fuzziness parameter (m > 1).
    - eta (float): Suppression parameter (0 < eta < 1).
    """


    alpha_k = 1 / (1 - u_w + u_w * (1 - eta) ** (2 / (1 - m)))

    return alpha_k

## SUPPRESSION

def suppress_membership_matrix(membership_matrix, alpha, m, generalized=False, eta=0.5):
    """
    Perform the suppression step of the suppressed fuzzy c-means (s-FCM) algorithm. If generalized = True, it also includes alpha updating according to generalized suppressed.

    Parameters:
    - member_matrix: current membership matrix (n_clusters x n_samples).
    - alpha: suppression factor (0-1). If generalized=True it is ignored.
    """
    n_clusters, n_samples = membership_matrix.shape

    suppressed_u = np.zeros_like(membership_matrix)

    # Perform suppression on membership values
    for k in range(n_samples):  # Iterate through samples
        winner_idx = np.argmax(membership_matrix[:,k])  # Find the cluster with the max membership (winner cluster)
        if generalized:
            u_w = membership_matrix[winner_idx, k]
            alpha = update_alpha_theta(u_w, m, eta)
        for i in range(n_clusters):
            if i == winner_idx:
                suppressed_u[i,k] = 1 - alpha + alpha * membership_matrix[i,k]  # Apply suppression to all memberships
            else:
                suppressed_u[i,k] = alpha * membership_matrix[i,k]  # Adjust the max membership

    # Normalize to get membership matrix
    suppressed_u = suppressed_u / np.sum(suppressed_u, axis=0)  # Axis = 0 as we are normalizing across clusters


    return suppressed_u

def gs_fcm(data, n_clusters, m=2, max_iter=1000, tolerance=1e-5, suppress=False, alpha=0.5, generalized=False, eta=0.5):
    """
    Performs the generalized suppressed fuzzy c-means gs_fcm as explained in Szilágyi, L., Szilágyi, S.M.: Generalization rules for the suppressed fuzzy c-means clustering
    algorithm. Neurocomput. 139, 298–309 (2014). The generalization corresponds to the first one presented in the paper, where eta = theta.
    If suppressed is False, it performs the typical Fuzzy C-Means algorithm.
    If suppressed is True and Generalized is False it performs the Suppressed Fuzzy C-Means algorithm.
    If both suppressed and Generalized are True, it performs the Generalized Suppressed Fuzzy C-Means algorithm.

    Parameters:
    :param data: Samples x features
    :param n_clusters: Number of clusters
    :param m: Fuzzification parameter (m>1)
    :param max_iter: Maximum number of iterations run without convergence
    :param tolerance: Tolerance for convergence
    :param suppress: Perform suppression or not
    :param alpha: Suppression parameter (0-1, default 0.5)
    :param generalized: Perform generalized suppression or not. If suppress=False this parameter is ignored. If this parameter is set to true, the value for alpha is ignored as it is iteratively calculated using the generalization parameter eta.
    :param eta: Generalization parameter
    :return:
    """

    n_samples = data.shape[0]

    # Randomly initialize membership matrix
    member_matrix = np.random.rand(n_clusters, n_samples)
    member_matrix /= np.sum(member_matrix, axis=0)  # Normalize so that each column sums to 1
    centers = np.dot(member_matrix, data) / np.sum(member_matrix, axis=1)[:, None]  # Compute initial centers

    prev_centers = np.zeros_like(centers)
    n_iters = 0

    # Optimization loop
    for iter in range(max_iter):
        # Update membership matrix based on new centers
        member_matrix = update_membership_matrix(data, centers, m)

        # Perform suppression if necessary
        if suppress:
            member_matrix = suppress_membership_matrix(member_matrix, alpha, m, generalized, eta)

        # Update centers based on membership matrix
        centers = update_centers(data, member_matrix, m)

        # Check for convergence (if centers do not change significantly)
        if np.sum(np.linalg.norm(centers - prev_centers, axis=0)) < tolerance: # Check if the sum of individual norms for each cluster changes significantly
            #print(f"Convergence after {iter} iterations.")
            n_iters = iter
            break

        prev_centers = centers.copy()

    return member_matrix, n_iters, centers

def run_all_gs_fcm(data_X, data_y, dataset=""):
    """
    Run all grid search evaluations for fuzzy c-means clustering.
    """
    results = []
    for k in range(12,13): # Between 2 and 12
        for m in [1.05, 1.5, 1.75, 2]: # 1 < m < 2
            eta_bar = tqdm([0.1, 0.5, 0.9], desc="", position=0, leave=False)
            for eta in eta_bar: # 0 < eta < 1
                best_results = {"Xie-Beni": np.Inf}
                for _ in range(10):
                    eta_bar.set_description(f"Processing DS={dataset} k={k}, m={m:.2f}, eta={eta:.1f}")
                    start_time = time.time()
                    u, n_iterations, centers = gs_fcm(data_X, k, m=m, suppress=True, generalized=True, eta=eta)
                    end_time = time.time()
                    clusters = defuzzyfy(u) # Get final clusters
                    elapsed_time = end_time - start_time
                    temp_results = get_metrics_fuzzy(data_X, data_y, clusters, f"GS-FCM_k{k}_m{m}_eta{eta}", elapsed_time, n_iterations,u,centers,m)
                    if temp_results["Xie-Beni"] < best_results["Xie-Beni"]: # In xie_beni smaller is better
                        best_results = temp_results
                results.append(best_results)

    # Convert to DataFrame
    results = pd.DataFrame(results)
    return results