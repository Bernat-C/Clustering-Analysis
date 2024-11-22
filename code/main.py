import os
import pandas as pd

from preprocessing import preprocess_vehicle, preprocess_sick, preprocess_grid
from utils import get_user_choice, plot_clusters
from fuzzyclustering import gs_fcm
from spectralclustering import spectral_clustering

def preprocess_datasets():
    df_sick_X, df_sick_y = preprocess_sick()
    df_grid_X, df_grid_y = preprocess_grid()
    df_vehicle_X, df_vehicle_y = preprocess_vehicle()

def load_ds(name):

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets_processed")
    file = os.path.join(base_dir,f'{name}.csv')
    df = pd.read_csv(file)

    ## TODO: DE MOMENT NO UTILITZEM LA CLASSE, VAN DIR Q EN UN FUTUR IGUAL SI
    return df.iloc[:,:-1]

def main():
    print("Welcome to our Clustering application.")

    dataset = get_user_choice("Please, select the dataset you would like to use:", ["sick", "grid","vehicle"])
    if dataset == "sick": ###### TODO: TREURE QUAN ES PUGUI
        raise EnvironmentError('The dataset "SICK" is not implemented because we are waiting for the answer from the teacher.')
    method = get_user_choice("Please, select the algorithm to use:", ["OPTICS", "Spectral Clustering", "K-Means", "Fuzzy Clustering"])

    dataset = load_ds(dataset)

    if method=="OPTICS":
        pass
    elif method=="Spectral Clustering":
        n_clusters = get_user_choice("Select the number of clusters:", [2,3,4,5,6,7,8,9,10], is_numeric=True)
        affinity = get_user_choice("Select the affinity type:", ["rbf", "nearest_neighbors"])
        assign_labels = get_user_choice("Select the label assignment method:", ["kmeans", "discretize"])
        n_neighbors = None
        if affinity == "nearest_neighbors":
            n_neighbors = get_user_choice("Select the number of neighbors:", [5, 10, 15, 20], is_numeric=True)

        labels, silhouette = spectral_clustering(dataset, n_clusters, affinity, assign_labels, n_neighbors)
        print(f"Clustering completed with {n_clusters} clusters. Silhouette Score: {silhouette:.4f}")

    elif method=="K-Means":
        pass
    elif method=="Fuzzy Clustering":
        c = get_user_choice("How many centroids would you like to use:", [2,3,4,5,6,7,8,9,10], is_numeric=True)
        m = get_user_choice("What m do you want to use:", [1,2,5,10,15,20,50,75,100], is_numeric=True)
        clusters = gs_fcm(dataset,c,m)
        plot_clusters(clusters)



if __name__ == "__main__":
    main()