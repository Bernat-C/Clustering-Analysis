import sys

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import umap.umap_ as umap


import pandas as pd
import numpy as np

### PREPROCESSING

def drop_columns(df, column_names):
    df = df.drop(columns = column_names)
    return df

def drop_rows_sick(df, column_names):
    df = df.dropna(subset=column_names)
    df = df[df["age"] <= 120] # Drop ages over 120
    df = df.reset_index(drop=True)

    return df

"""
Applies a minmaxscaler to all numerical columns.
If it finds a nan in a numerical column it removes the instance.
"""
def min_max_scaler(df, numerical_cols=slice(None)):

    scaler = MinMaxScaler()

    # Scaler Training with all the train and test information.
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

def one_hot_encoding(df):
    categorical_features = df.select_dtypes(include=['object']).nunique()[lambda x: x > 2].index.tolist()

    ohe = OneHotEncoder(handle_unknown='ignore')

    encoded_array = ohe.fit_transform(df[categorical_features]).toarray()

    # Create new column names for the encoded features
    new_cols = [f'{col}_{cat}' for col in categorical_features for cat in
                ohe.categories_[categorical_features.index(col)]]

    # Create a DataFrame for the encoded features
    df_encoded = pd.DataFrame(encoded_array, columns=new_cols, index=df.index)

    # Substitute the original categorical features with the new numeric ones
    df = df.drop(categorical_features, axis=1)
    df = df.join(df_encoded)

    return df

def binary_encoding(df):

    binary_features = df.select_dtypes(include=['object']).nunique()[lambda x: x <= 2].index.tolist()

    # Encode only the binary features
    for feature in binary_features:

        label_encoder = LabelEncoder()
        df[feature] = label_encoder.fit_transform(df[feature])

    return df

def label_encoder(df):
    le = LabelEncoder()
    for column in df.columns:
        df[column] = le.fit_transform(df[column])
    return df



def fill_nans(df, columns_predict):

    model = LinearRegression()

    # Train with all columns except the ones to predict
    cols = [col for col in df.columns if col not in columns_predict]

    for col in columns_predict:
        df_model = df.dropna(subset=[col])
        df_nans = df[df[col].isna()]

        if not df_model.empty:
            x = df_model[cols]
            y = df_model[col]

            model.fit(x, y)

            if not df_nans.empty:
                df.loc[df_nans.index, col] = model.predict(df_nans[cols])

    return df

#### UTILS REGARDING INPUT

def get_user_choice(prompt, options, is_numeric = False, is_float = False):
    while True:
        print(prompt)
        for i, option in enumerate(options, 1):
            if is_numeric:
                print(f"  {option}")
            else:
                print(f" {i}. {option}")
        choice = input("Please enter the number of your choice: ")

        if is_numeric and is_float and float(choice) in options:
            return float(choice)
        if not is_float and is_numeric and int(choice) in options:
            return int(choice)
        if choice in options:
            return choice
        if not is_numeric and choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        else:
            print("Invalid choice. Try again.\n")

def loading_bar(iteration, total, length=40):
    percent = (iteration / total)
    bar_length = int(length * percent)
    bar = '#' * bar_length + '-' * (length - bar_length)
    sys.stdout.write(f'\r[{bar}] {percent:.2%} Complete')
    sys.stdout.flush()

### UTILS REGARDING PLOTTING

def plot_clusters(clusters):
    """
        Plots the clusters
    :param clusters: Containing a list of clusters, and each cluster containing points (a list of pairs x,y for every point) and center, containing a pair x,y indicating the center of the cluster.
    """
    plt.figure(figsize=(8, 6))

    print("Cluster point distribution: ")

    # Loop through the clusters
    for cluster_id, cluster_data in clusters.items():
        points = np.array(cluster_data['points'])
        center = cluster_data['center']

        # Debugging: Print the shape of points to check if it's 2D
        print(f"Cluster {cluster_id} - Points shape: {points.shape}")

        plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {cluster_id} Points', alpha=0.6, s=30)

        # Plot the center with a different marker (e.g., red 'X')
        plt.scatter(center[0], center[1], color='red', marker='X', s=100, label=f'Cluster {cluster_id} Center')

    # Labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Cluster Points and Centers')
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.show()

def compute_final_clusters(data, labels, centers):
    clusters = {i: {'points': data[labels==i], 'center': centers[i]} for i in range(len(centers))}
    return clusters

def plot_spectral(data, labels):
    data = np.array(data)

    # plot results
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Cluster Points')

    legend_labels = [f"Cluster {i}" for i in np.unique(labels)]
    handles, _ = scatter.legend_elements()
    plt.legend(handles, legend_labels, title="Clusters")

    plt.show()

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

def rank_and_sort(df,metrics=["Davies-Bouldin Index","Calinski","Silhouette Coefficient"],n=3):
    """
    Rank a solution df by its columns, getting the first n of every metric
    :param df:
    :return:
    """

    # Get the top 3 rows for each metric
    top_rows = pd.DataFrame()
    for metric in metrics:
        if metric == "Davies-Bouldin Index":
            # For Davies-Bouldin Index, lower is better, so sort ascending
            top_rows = pd.concat([top_rows, df.nsmallest(n, metric)])
        else:
            # For all other metrics, higher is better
            top_rows = pd.concat([top_rows, df.nlargest(n, metric)])

    # Drop duplicate rows
    top_rows = top_rows.drop_duplicates()

    rankings = top_rows.copy()

    # Rank the rows based on all metrics
    for metric in metrics:
        if metric == "Davies-Bouldin Index":
            # Rank ascending for Davies-Bouldin Index
            rankings[f"{metric}_Rank"] = rankings[metric].rank(ascending=True)
        if metric == "Xie-Beni":
            rankings[f"{metric}_Rank"] = rankings[metric].rank(ascending=True)
        else:
            # Rank descending for other metrics
            rankings[f"{metric}_Rank"] = rankings[metric].rank(ascending=False)

    # Calculate the mean rank across all metrics
    ranking_columns = [f"{metric}_Rank" for metric in metrics]
    rankings["Mean_Rank"] = rankings[ranking_columns].mean(axis=1)

    # Sort by mean rank
    rankings = rankings.sort_values("Mean_Rank").reset_index(drop=True)

    return rankings