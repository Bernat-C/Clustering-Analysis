import os
import time

import numpy as np

os.environ["OMP_NUM_THREADS"] = "4"

import pandas as pd

from metrics import get_metrics_general, get_metrics_fuzzy, xie_beni, get_metrics_optics
from optics import apply_optics
from preprocessing import preprocess_vowel, preprocess_sick, preprocess_grid
from utils import get_user_choice, plot_clusters, plot_spectral
from fuzzyclustering import gs_fcm, run_all_gs_fcm, get_cluster_list, defuzzyfy
from spectralclustering import spectral_clustering
from kmeans import run_kmeans
from global_kmeans import run_global_kmeans
from gmeans import run_gmeans


def preprocess_datasets():
    df_sick_X, df_sick_y = preprocess_sick()
    df_grid_X, df_grid_y = preprocess_grid()
    df_vowel_X, df_vowel_y = preprocess_vowel()

def load_ds(name):

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets_processed")
    file = os.path.join(base_dir,f'{name}.csv')
    df = pd.read_csv(file)

    # We remove the class of the dataset as we will not be using it
    return df.iloc[:,:-1], df.iloc[:,-1]

def runAllFuzzyClustering(datasets):
    for dataset in datasets:
        df_X, y = load_ds(dataset)
        results = run_all_gs_fcm(df_X, y, dataset=dataset)
        results.to_csv(f"./output/fuzzyclustering_noov_{dataset}.csv")

def main():
    print("Welcome to our Clustering application.")

    while True:
        dataset = get_user_choice("Please, select the dataset you would like to use:", ["sick", "grid","vowel"])
        method = get_user_choice("Please, select the algorithm to use:", ["OPTICS", "Spectral Clustering", "K-Means", "Global-K-Means", "G-Means", "GS Fuzzy Clustering"])

        df_X, df_y = load_ds(dataset)

        if method=="OPTICS":
            algorithm = get_user_choice("Select algorithm:", ["ball_tree", "brute"])
            metric = get_user_choice("Select the distance to use:", ["euclidean", "manhattan", "hamming"])
            min_samples = get_user_choice("Select the minimum number of samples:", [5, 15], is_numeric=True)
            start_time = time.time()
            clusters = apply_optics(df_X, metric=metric, algorithm=algorithm, min_samples=min_samples)
            end_time = time.time()
            elapsed_time = end_time - start_time

            k = len(np.unique(clusters))
            methodused = f"optics_k_{k}_distance_{metric}_min_samples{min_samples}_algorithm_{algorithm}"
            metrics = get_metrics_optics(df_X, df_y, clusters, methodused, elapsed_time, False)

            print("---------------------------------------------------------------------------------------")
            print("Metrics Summary: ")
            for key, value in metrics.items():
                print(f"{key}: {value}")
            print("---------------------------------------------------------------------------------------")

            if dataset == "grid":
                plot_spectral(df_X, clusters)

        elif method=="Spectral Clustering":
            n_clusters = get_user_choice("Select the number of clusters:", [2,3,4,5,6,7,8,9,10,11,12], is_numeric=True)
            affinity = get_user_choice("Select the affinity type:", ["rbf", "nearest_neighbors"])
            n_neighbors = None
            if affinity == "nearest_neighbors":
                n_neighbors = get_user_choice("Select the number of neighbors:", [5, 10, 15, 20], is_numeric=True)
            assign_labels = get_user_choice("Select the label assignment method:", ["kmeans", "discretize"])
            eigen_solver = get_user_choice("Select the eigen solver method:", ['arpack', 'lobpcg'])

            start_time = time.time()
            labels = spectral_clustering(df_X, n_clusters, affinity, n_neighbors, assign_labels,  eigen_solver)
            end_time = time.time()
            elapsed_time = end_time - start_time
            metrics = get_metrics_general(df_X, df_y, labels, f"spectral_n_{n_clusters}_affinity_{affinity}_assign_labels_{assign_labels}_eigen_solver_{eigen_solver}", elapsed_time, False)

            print("---------------------------------------------------------------------------------------")
            print("Metrics Summary: ")
            for key, value in metrics.items():
                print(f"{key}: {value}")
            print("---------------------------------------------------------------------------------------")

            if dataset == "grid":
                plot_spectral(df_X, labels)

        elif method=="K-Means":
            n_clusters = get_user_choice("Select the number of clusters:", [2,3,4,5,6,7,8,9,10,11,12], is_numeric=True)
            distance = get_user_choice("Select the distance to use:", ["euclidean", "manhattan", "cosine"])
            clusters = run_kmeans(df_X, n_clusters=n_clusters, init=None, distance=distance)
            if dataset == "grid":
                plot_clusters(clusters)
        elif method=="Global-K-Means":
            max_clusters = get_user_choice("Select the maximum number of clusters:", [2,3,4,5,6,7,8,9,10,11,12], is_numeric=True)
            distance = get_user_choice("Select the distance to use:", ["euclidean", "manhattan", "cosine"])
            clusters = run_global_kmeans(df_X, max_clusters=max_clusters, distance=distance)
            if dataset == "grid":
                plot_clusters(clusters)
        elif method=="G-Means":
            max_clusters = get_user_choice("Select the maximum number of clusters:", [2,3,4,5,6,7,8,9,10,11,12], is_numeric=True)
            distance = get_user_choice("Select the distance to use:", ["euclidean", "manhattan", "cosine"])
            clusters = run_gmeans(df_X, max_clusters=max_clusters, distance=distance)
            if dataset == "grid":
                plot_clusters(clusters)
        elif method=="GS Fuzzy Clustering":
            c = get_user_choice("How many centroids would you like to use:", [2,3,4,5,6,7,8,9,10,11,12], is_numeric=True)
            m = get_user_choice("What m (fuzzification parameter) do you want to use:", [1.05, 1.2, 1.5, 1.75, 2], is_numeric=True, is_float=True)
            eta = get_user_choice("What eta (generalized suppression factor) do you want to use:", [0.1,0.3,0.5,0.7,0.9], is_numeric=True, is_float=True)
            start_time = time.time()
            u, iters, centers = gs_fcm(df_X,c,m,suppress=True,generalized=True,eta=eta)
            end_time = time.time()
            clusters = defuzzyfy(u)
            elapsed_time = end_time - start_time


            # Showing the results
            print("---------------------------------------------------------------------------------------")
            print("RESULTS: ")
            print("---------------------------------------------------------------------------------------")
            print(f"GS_FCM converged in {iters} iterations.")
            assignments = get_cluster_list(df_X,centers,clusters,c)
            metrics = get_metrics_fuzzy(df_X, np.array(df_y), clusters, f"GS-FCM_k{c}_m{m}_eta{eta}", elapsed_time, iters,u,centers,m)
            print("---------------------------------------------------------------------------------------")
            print("Metrics Summary: ")
            for key, value in metrics.items():
                print(f"{key}: {value}")
            print("---------------------------------------------------------------------------------------")
            if dataset == "grid":
                plot_clusters(assignments)

        x = get_user_choice("Do you want to exit?",["y","n"])
        if x == "y":
            exit()



if __name__ == "__main__":
    main()