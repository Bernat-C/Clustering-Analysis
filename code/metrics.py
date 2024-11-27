
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics import davies_bouldin_score

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

def get_metrics(results, X, labels_pred, k, dist):

    
    # Compute metrics
    dbi = davies_bouldin_index(X, labels_pred)
    silhouette = silhouette_coefficient(X, labels_pred)

    ari = adjusted_rand_index(y, labels_pred)
    purity = purity_score(y, labels_pred)
    fmeasure = f_measure(y, labels_pred)
    
    # Append results
    results.append({
        "k": k,
        "distance": dist,
        "ARI": ari,
        "Purity": purity,
        "F-Measure": fmeasure,
        "Davies-Bouldin Index": dbi,
        "Silhouette Coefficient": silhouette
    })
    return results