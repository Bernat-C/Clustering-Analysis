import pandas as pd
import sklearn
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np


def apply_optics(data, metric, algorithm, xi=0.05, min_samples=15):  # Changed data_file to data
    """
    Applies OPTICS clustering to the provided data.

    Args:
        data (DataFrame): The input data for clustering. # updated description to reflect parameter change
        metric (str): The distance metric to use.
        algorithm (str): The algorithm to use for nearest neighbors search.

    Returns:
        tuple: A tuple containing the cluster labels
    """
    # Apply OPTICS clustering to already preprocessed data
    optics = OPTICS(metric=metric, algorithm=algorithm, min_samples=min_samples, xi=xi)  # Adjust parameters

    # Convert DataFrame to NumPy array before fitting
    data_array = data

    labels = optics.fit_predict(data_array)
    return labels