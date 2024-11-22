import pandas as pd
import sklearn
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np


def apply_optics(data, metric, algorithm, xi=0.05, min_samples=5, plot=False):  # Changed data_file to data
    """
    Applies OPTICS clustering to the provided data.

    Args:
        data (DataFrame): The input data for clustering. # updated description to reflect parameter change
        metric (str): The distance metric to use.
        algorithm (str): The algorithm to use for nearest neighbors search.

    Returns:
        tuple: A tuple containing the cluster labels, the silhouette score, and the input data.
    """
    # Apply OPTICS clustering to already preprocessed data
    optics = OPTICS(metric=metric, algorithm=algorithm, min_samples=min_samples, xi=xi)  # Adjust parameters

    # Convert DataFrame to NumPy array before fitting
    data_array = data.values

    labels = optics.fit_predict(data_array)
    # Evaluate clustering performance
    # Check if labels contain more than one unique value
    if len(np.unique(labels)) > 1:  # Check for more than one cluster
        score = silhouette_score(data_array, labels)
        reachability = optics.reachability_[optics.ordering_]
        space = np.arange(len(data))

        if (plot == True):
            # Plot reachability
            plt.figure(figsize=(10, 7))
            plt.plot(space, reachability)
            plt.title(f"Optics reachability plot Metric: {metric}, Algorithm: {algorithm}, Silhouette Score: {score}")
            plt.xlabel('Data Point Index (Ordered)')
            plt.ylabel('Reachability Distance')
            plt.show()
        else:
            print(f"Metric : {metric}, algorithm {algorithm}, Silhouette Score: {score}. ")
    else:
        score = np.nan  # Assign NaN if only one cluster is found
        print(f"Metric : {metric}, algorithm {algorithm}, Optics only found one cluser. ")
    return labels, score, data