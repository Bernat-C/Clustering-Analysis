import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import umap.umap_ as umap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from gmeans import run_gmeans

def reduce_and_plot_with_umap(data, labels, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
    """
    Reduces dimensionality of the dataset with UMAP and plots the clusters.
    :param data: Original high-dimensional data.
    :param labels: Cluster labels corresponding to each data point.
    :param n_neighbors: UMAP parameter for balancing local and global structure.
    :param min_dist: UMAP parameter controlling the tightness of embedding.
    :param n_components: Number of dimensions to reduce to (default is 2 for visualization).
    :param metric: Distance metric for UMAP.
    """
    # Reduce dimensionality using UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
    reduced_data = reducer.fit_transform(data)

    # Plot clusters in reduced space
    plt.figure(figsize=(8, 6))
    for cluster_id in np.unique(labels):
        cluster_points = reduced_data[labels == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}', alpha=0.6, s=30)
    
    # Labels and title
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('Clusters Visualized in 2D Space (UMAP)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_confusion(targets, labels):
    # Compute confusion matrix
    cm = confusion_matrix(targets, labels)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.show()

path = './datasets_processed'
file = os.path.join(path,'grid.csv')
df = pd.read_csv(file)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
k = 4
distance = 'euclidean'
clusters = run_gmeans(np.array(X), max_clusters=k, distance=distance)
labels = np.concatenate([np.full(len(c['points']), cluster_id) for cluster_id, c in clusters.items()])
data = np.vstack([c['points'] for c in clusters.values()])

reduce_and_plot_with_umap(data, labels)
plot_confusion(y, labels)
# Integrate SOM into your pipeline
feature_names = X.columns.tolist()


path = './datasets_processed'
file = os.path.join(path,'sick.csv')
df = pd.read_csv(file)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
k = 2
distance = 'euclidean'
clusters = run_gmeans(np.array(X), max_clusters=k, distance=distance)
labels = np.concatenate([np.full(len(c['points']), cluster_id) for cluster_id, c in clusters.items()])
data = np.vstack([c['points'] for c in clusters.values()])

reduce_and_plot_with_umap(data, labels)
plot_confusion(y, labels)

path = './datasets_processed'
file = os.path.join(path,'vowel.csv')
df = pd.read_csv(file)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
k = 10
distance = 'euclidean'
clusters = run_gmeans(np.array(X), max_clusters=k, distance=distance)
labels = np.concatenate([np.full(len(c['points']), cluster_id) for cluster_id, c in clusters.items()])
data = np.vstack([c['points'] for c in clusters.values()])

reduce_and_plot_with_umap(data, labels)
plot_confusion(y, labels)

