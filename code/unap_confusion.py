
import pandas as pd
import numpy as np
import os

from gmeans import run_gmeans
from utils import reduce_and_plot_with_umap, reduce_and_plot_with_pca, plot_confusion, plot_clusters



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

plot_clusters(clusters)
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
reduce_and_plot_with_pca(data, labels)
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
reduce_and_plot_with_pca(data, labels)
plot_confusion(y, labels)

