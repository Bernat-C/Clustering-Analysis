
import pandas as pd
import numpy as np
import os

from kmeans import run_kmeans
from fuzzyclustering import gs_fcm, defuzzyfy, get_cluster_list
from spectralclustering import spectral_clustering
from gmeans import run_gmeans
from utils import reduce_and_plot_with_umap, reduce_and_plot_with_pca, plot_confusion, plot_clusters

path = './datasets_processed'
file = os.path.join(path,'sick.csv')
df = pd.read_csv(file)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
k = 7
affinity = 'nearest_neighbors'
n_neighbors = 100
assign_labels = 'discretize'
eigen_solver = 'arpack'
distance = 'euclidean'
labels = spectral_clustering(X, n_clusters=k, affinity=affinity, n_neighbors=n_neighbors, assign_labels=assign_labels,  eigen_solver
= 'arpack')
reduce_and_plot_with_umap(X, labels)
reduce_and_plot_with_pca(X, labels)
plot_confusion(y, labels)


path = './datasets_processed'
file = os.path.join(path,'vowel.csv')
df = pd.read_csv(file)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
k = 12
distance = 'manhattan'
clusters = run_kmeans(np.array(X), n_clusters=k, distance=distance)
labels = np.concatenate([np.full(len(c['points']), cluster_id) for cluster_id, c in clusters.items()])
data = np.vstack([c['points'] for c in clusters.values()])

reduce_and_plot_with_umap(data, labels)
reduce_and_plot_with_pca(data, labels)
plot_confusion(y, labels)


path = './datasets_processed'
file = os.path.join(path,'grid.csv')
df = pd.read_csv(file)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
k = 7
m = 2
eta = 0.01
u, n_iterations, centers = gs_fcm(X, k, m=m, suppress=True, generalized=True, eta=eta)
clusters = defuzzyfy(u) # Get final clusters
clusters = get_cluster_list(X,centers,clusters,k)
data = np.vstack([c['points'] for c in clusters.values()])

plot_clusters(clusters)
#plot_confusion(y, labels)
# Integrate SOM into your pipeline
feature_names = X.columns.tolist()





