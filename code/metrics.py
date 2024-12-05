
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score


# GENERAL METRICS

def adjusted_rand_index(labels_true, labels_pred):
    return adjusted_rand_score(labels_true, labels_pred)

def purity_score(y_true, y_pred):
    # Compute contingency matrix
    contingency_matrix = pd.crosstab(y_true, y_pred)
    # Sum of maximum values in each column
    return np.sum(np.amax(contingency_matrix.values, axis=0)) / np.sum(contingency_matrix.values)

def davies_bouldin_index(data, labels):
    return davies_bouldin_score(data, labels)

def silhouette_coefficient(data, labels):
    return silhouette_score(data, labels)

def f_measure(labels_true, labels_pred):
    contingency_matrix = pd.crosstab(labels_true, labels_pred)
    precision = contingency_matrix.max(axis=0).sum() / len(labels_pred)
    recall = contingency_matrix.max(axis=1).sum() / len(labels_true)
    return 2 * (precision * recall) / (precision + recall)

def calinski_harabasz_index(data, labels):
    return calinski_harabasz_score(data, labels)


# SPECIFIC METRICS

def xie_beni(U, X, C, m):
    """
    Compute the Xie-Beni index for fuzzy clustering.

    Parameters:
    U (numpy array): The membership matrix of shape (n_clusters, n_samples).
    X (numpy array): The data points of shape (n_samples, n_features).
    C (numpy array): Centers of the clusters (n_clusters, n_features).
    m (int): Fuzzy clustering parameter.
    """
    c, n = U.shape

    # Numerator (sum of squared distances to the centers)
    numerator = 0
    for i in range(n):
        for k in range(c):
            dist = np.linalg.norm(X[i] - C[k]) ** 2
            numerator += (U[k,i] ** m) * dist

    # Denominator (minimum squared distance between cluster centers)
    min_distance = float('inf')
    for k in range(c):
        for j in range(k + 1, c):
            dist = np.linalg.norm(C[k] - C[j]) ** 2
            min_distance = min(min_distance, dist)

    if min_distance==0: # Avoid division by 0
        min_distance = 0.001

    xie_beni_index = numerator / (n * min_distance)

    return xie_beni_index

# FUNCTIONS TO GET DICTIONARIES WITH ALL METRICS

def get_metrics_fuzzy(X, labels_true, labels_pred, method, time, n_iterations,u,c,m):
    """
    Calculate performance metrics for FC.

    X (numpy.ndarray): The dataset of shape (n_samples, n_features).
    labels_true (numpy.ndarray): Ground truth labels of shape (n_samples,).
    labels_pred (numpy.ndarray): Predicted labels of shape (n_samples,).
    method (str)
    time (float)
    n_iterations (int)
    u (numpy array): The membership matrix of shape (n_clusters, n_samples).
    c (numpy array): Centers of the clusters (n_clusters, n_features).
    m (float): The fuzzification parameter.
    """
    # Compute metrics
    dbi = davies_bouldin_index(X, labels_pred)
    silhouette = silhouette_coefficient(X, labels_pred)
    calinski = calinski_harabasz_index(X, labels_pred)

    ari = adjusted_rand_index(labels_true, labels_pred)
    purity = purity_score(labels_true, labels_pred)
    fmeasure = f_measure(labels_true, labels_pred)

    # Append results
    results = {
        "Method": method,
        "ARI": ari,
        "Purity": purity,
        "F-Measure": fmeasure,
        "Davies-Bouldin Index": dbi,
        "Silhouette Coefficient": silhouette,
        "Calinski": calinski,
        "Xie-Beni": xie_beni(u,np.array(X),c,m),
        "Solving Time": time,
        "Iterations": n_iterations
    }
    return results


def get_metrics_general(X, labels_true, labels_pred, method, time, n_iterations = None):
    # Compute metrics
    dbi = davies_bouldin_index(X, labels_pred)
    silhouette = silhouette_coefficient(X, labels_pred)
    calinski = calinski_harabasz_index(X, labels_pred)

    ari = adjusted_rand_index(labels_true, labels_pred)
    purity = purity_score(labels_true, labels_pred)
    fmeasure = f_measure(labels_true, labels_pred)

    # Append results
    results = {
        "Method": method,
        "ARI": ari,
        "Purity": purity,
        "F-Measure": fmeasure,
        "Davies-Bouldin Index": dbi,
        "Silhouette Coefficient": silhouette,
        "Calinski": calinski,
        "Solving Time": time,
        "Iterations": n_iterations if n_iterations else np.nan
    }
    return results

def get_metrics(X, y, labels_pred, k, dist):

    # Compute metrics
    dbi = davies_bouldin_index(X, labels_pred)
    silhouette = silhouette_coefficient(X, labels_pred)

    ari = adjusted_rand_index(y, labels_pred)
    purity = purity_score(y, labels_pred)
    fmeasure = f_measure(y, labels_pred)
    
    # Append results
    results = {
        "k": k,
        "distance": dist,
        "ARI": ari,
        "Purity": purity,
        "F-Measure": fmeasure,
        "Davies-Bouldin Index": dbi,
        "Silhouette Coefficient": silhouette
    }
    return results


def get_metrics_optics(X, labels_true, labels_pred, method, t, n_iterations=None):
    if len(np.unique(labels_pred)) > 1:
        # Compute metrics
        dbi = davies_bouldin_index(X, labels_pred)
        silhouette = silhouette_coefficient(X, labels_pred)
        calinski = calinski_harabasz_index(X, labels_pred)

        ari = adjusted_rand_index(labels_true, labels_pred)
        purity = purity_score(labels_true, labels_pred)
        fmeasure = f_measure(labels_true, labels_pred)

        # Append results
        results = {
            "Method": method,
            "ARI": ari,
            "Purity": purity,
            "F-Measure": fmeasure,
            "Davies-Bouldin Index": dbi,
            "Silhouette Coefficient": silhouette,
            "Calinski": calinski,
            "Solving Time": t,
            "Iterations": n_iterations if n_iterations else np.NaN
        }
    else:
        results = {
            "Method": method,
            "ARI": np.nan,
            "Purity": np.nan,
            "F-Measure": np.nan,
            "Davies-Bouldin Index": np.nan,
            "Silhouette Coefficient": np.nan,
            "Calinski": np.nan,
            "Solving Time": t,
            "Iterations": n_iterations if n_iterations else np.NaN}
    return results