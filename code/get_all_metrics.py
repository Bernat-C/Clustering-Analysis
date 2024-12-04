import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from kmeans import run_all_kmeans
from gmeans import run_all_gmeans
from global_fastkmeans import run_all_global_kmeans


def get_all_metrics():

    datasets = ['grid', 'vowel', 'sick']

    for dataset in datasets:
        file = f'./datasets_processed/{dataset}.csv'
        df = pd.read_csv(file)
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]

        r = run_all_kmeans(X,y)
        r.to_csv(f'./output/kmeans_{dataset}.csv')
        print('i')
        r = run_all_gmeans(X,y)
        r.to_csv(f'./output/gmeans_{dataset}.csv')
        
        """print('i')
        r = run_all_global_kmeans(X,y)
        r.to_csv(f'./output/GlobalFastKmeans_{dataset}.csv')"""


# Define datasets, models, and path
datasets = ['grid', 'sick', 'vowel']
models = ['kmeans', 'gmeans']
path = './output/'

# Metrics to analyze
metrics = ["Davies-Bouldin Index", "Silhouette Coefficient", "Calinski"]
distances = ['euclidean', 'manhattan', 'cosine']
colors = ['#FF6347', '#1f77b4', '#2ca02c']
markers = ['o', 's', '^']

# Function to preprocess results
def preprocess_results(filepath):
    results = pd.read_csv(filepath)
    results['k'] = results['Method'].str.split('_').str[1].str.split('k').str[1].astype(int)
    results['distance'] = results['Method'].str.split('_').str[2].str.split('distance-').str[1]
    return results

# Function to plot a single row (1x3) for KMeans
def plot_kmeans(datasets):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)  # 1x3 plot with shared y-axis
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)

    for ax, dataset in zip(axes, datasets):
        results = preprocess_results(f'./output/kmeans_{dataset}.csv')

        for distance, color in zip(distances, colors):
            results_dist = results[results['distance'] == distance]
            sns.violinplot(
                data=results_dist,
                x="k", y="Silhouette Coefficient",
                inner=None, color=color, ax=ax, 
            )

        for violin in ax.collections:
            violin.set_edgecolor(violin._facecolors[0])
            violin.set_facecolor('none')  # Remove the fill

        # Set the title to the dataset name
        ax.set_title(f"Dataset {dataset}", fontsize=14)
        ax.set_xlabel("k (Number of Clusters)", fontsize=12)
        if dataset == 'grid':
            ax.set_ylabel("Silhouette Coefficient", fontsize=12)
        else:
            ax.set_ylabel("", fontsize=12)


    # Adjust layout to prevent overlap and ensure proper spacing
    plt.tight_layout()
    plt.show()

# Function to plot a grid (3x3) for GMeans
def plot_3x3_kmeans(model, datasets, metrics):
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True)

    for col_idx, dataset in enumerate(datasets):  # Change row_idx to col_idx for datasets
        dataset_results = preprocess_results(f'{path}/{model}_{dataset}.csv')

        for row_idx, metric in enumerate(metrics):  # Change col_idx to row_idx for metrics
            ax = axes[row_idx, col_idx]  # Adjust the indexing to switch rows and columns
            for distance, color, marker in zip(distances, colors, markers):
                subset = dataset_results[dataset_results['distance'] == distance]

                # Calculate mean rank
                subset['mean_rank'] = subset[metrics].apply(
                    lambda row: row.rank(ascending=False).mean() if row.name != "Davies-Bouldin Index" else row.rank().mean(), axis=1
                )
                # For each k, find the row with the highest mean rank
                subset = subset.loc[subset.groupby('k')['mean_rank'].idxmax()]

                ax.scatter(subset['k'], subset[metric], color=color, marker=marker, label=distance, alpha=0.7)
                ax.plot(subset['k'], subset[metric], color=color, alpha=0.7)

            # Format the axis values to one decimal place for Davies-Bouldin and Silhouette Coefficient
            if metric in ["Davies-Bouldin Index", "Silhouette Coefficient"]:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}"))  # One decimal for y-axis

            # No decimals for cluster numbers (k) on the x-axis
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

            # Display all k values on x-axis
            ax.set_xticks(sorted(subset['k'].unique()))  # Show all k values
            ax.tick_params(axis='x', labelsize=10)  # Set x-axis label size smaller

            # Add grid
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            # Set titles, axis names, and labels
            if row_idx == 0:
                ax.set_title(f"{dataset}", fontsize=14)  # Dataset name in the first row
            if col_idx == 0:
                ax.set_ylabel(f"{metric}", fontsize=12)  # Metric name for the Y-axis
            if row_idx == 2:
                ax.set_xlabel("k (Number of Clusters)", fontsize=12)  # X-axis label for the last row

            # Show legend only on the first row and last column
            if row_idx == 0 and col_idx == 2:
                ax.legend(title="Distance", fontsize=10)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def ploat_all():
    # Main execution loop
    for model in models:
        if model == 'kmeans':
            plot_kmeans(datasets)
            plot_3x3_kmeans(model, datasets, metrics)
        elif model == 'gmeans':
            plot_3x3_kmeans(model, datasets, metrics)
        elif model == 'global_fastkmeans':
            plot_3x3_kmeans(model, datasets, metrics)


ploat_all()


