import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils import rank_and_sort

# Define datasets, models, and path
datasets = ['grid', 'sick', 'vowel']
models = ['kmeans', 'gmeans']
path = './output'

# Metrics to analyze
metrics = ["Davies-Bouldin Index", "Silhouette Coefficient", "Calinski"]
distances = ['euclidean', 'manhattan', 'cosine']
ms = [1.05, 1.5, 1.75, 2]
etas = [0.1, 0.5, 0.9]
affinity = ['rbf', 'nearest-neighbors']
colors = ['#FF6347', '#1f77b4', '#2ca02c', "#FFF200"]
markers = ['o', 's', ".", '^']

# Function to preprocess results
def preprocess_results(filepath, model=None):
    results = pd.read_csv(filepath)
    if model == 'kmeans' or model == 'gmeans':
        results['k'] = results['Method'].str.split('_').str[1].str.split('k').str[1].astype(int)
        results['distance'] = results['Method'].str.split('_').str[2].str.split('distance-').str[1]
    elif model == 'spectral':
        results['k'] = results['Method'].str.split('_').str[1].str.split('k').str[1].astype(int)
        results['affinity'] = results['Method'].str.split('_').str[2].str.split('affinity').str[1]
        results['n_neighbors'] = results['Method'].str.split('_').str[3].str.split('n-neighbors').str[1].astype(int)
        results['assign_labels'] = results['Method'].str.split('_').str[4].str.split('assign-labels').str[1]
        results['eigen_solver'] = results['Method'].str.split('_').str[5].str.split('eigen-solver').str[1]
    elif model == 'GlobalFastKmeans':
        results['k'] = results['Method'].str.split('_').str[1].str.split('k').str[1].astype(int)
        results['distance'] = results['Method'].str.split('_').str[2].str.split('distance-').str[1]
        results['kfound'] = results['Method'].str.split('_').str[2].str.split('kfound-').str[1]



    return results

# Function to plot a single row (1x3) for KMeans
def plot_kmeans(datasets, model):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)  # 1x3 plot with shared y-axis
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)

    for ax, dataset in zip(axes, datasets):
        results = preprocess_results(f'./output/kmeans_{dataset}.csv', model)

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

        # Add grid
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.grid(axis="x", linestyle="--", alpha=0.5)


    # Adjust layout to prevent overlap and ensure proper spacing
    plt.tight_layout()
    plt.show()

# Function to plot a grid (3x3) for GMeans
def plot_3x3_fuzzy(model, datasets, metrics):
    fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=True)

    for col_idx, dataset in enumerate(datasets):  # Change row_idx to col_idx for datasets
        dataset_results = preprocess_results(f'{path}/{model}_{dataset}.csv',"kmeans")

        for row_idx, metric in enumerate(metrics):  # Change col_idx to row_idx for metrics
            ax = axes[row_idx, col_idx]  # Adjust the indexing to switch rows and columns
            for m, color, marker in zip(ms, colors, markers):
                dataset_results[['Model', 'k', 'm', 'eta']] = dataset_results['Method'].str.split('_', expand=True)

                # Further process to clean up the extracted columns
                dataset_results['k'] = dataset_results['k'].str.extract('(\d+)', expand=False).astype(int)  # Extract just the number for k
                dataset_results['m'] = dataset_results['m'].str.extract('(\d+\.\d+|\d+)', expand=False).astype(float)  # Extract number for m
                dataset_results['eta'] = dataset_results['eta'].str.extract('(\d+\.\d+|\d+)', expand=False).astype(float)  # Extract number for eta
                subset = dataset_results[dataset_results['m'] == m]

                subset = subset.copy()  # Create a full copy to ensure safe modifications

                subset['mean_rank'] = subset[metrics].apply(
                    lambda row: row.rank(ascending=False).mean()
                    if row.name != "Davies-Bouldin Index" and row.name != "Xie-Beni"
                    else row.rank().mean(), axis=1
                )
                # For each k, find the row with the highest mean rank
                subset = subset.loc[subset.groupby('k')['mean_rank'].idxmax()]

                ax.scatter(subset['k'], subset[metric], color=color, marker=marker, label=m, alpha=0.7)
                ax.plot(subset['k'], subset[metric], color=color, alpha=0.7)

            # Format the axis values to one decimal place for Davies-Bouldin and Silhouette Coefficient
            if metric in ["Davies-Bouldin Index", "Silhouette Coefficient"]:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}"))  # One decimal for y-axis

            if metric in ['Xie-Beni'] and dataset == 'vowel':
                # No decimals for cluster numbers (k) on the x-axis
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y/1e10:.1f}e10"))  # Scale by 1e4 and add e4


            # Display all k values on x-axis
            ax.set_xticks(sorted(subset['k'].unique()))  # Show all k values
            ax.tick_params(axis='x', labelsize=10)  # Set x-axis label size smaller

            # Add grid
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.grid(axis="x", linestyle="--", alpha=0.5)

            # Set titles, axis names, and labels
            if row_idx == 0:
                ax.set_title(f"Dataset {dataset}", fontsize=14)  # Dataset name in the first row
            if col_idx == 0:
                if metric == 'Davies-Bouldin Index':
                    ax.set_ylabel(f"DBI", fontsize=12)  # Metric name for the Y-axis
                else:
                    ax.set_ylabel(f"{metric}", fontsize=12)  # Metric name for the Y-axis
            if row_idx == 3:
                ax.set_xlabel("k (Number of Clusters)", fontsize=12)  # X-axis label for the last row

            # Show legend only on the first row and last column
            if row_idx == 0 and col_idx == 2:
                ax.legend(title="m", fontsize=10)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def plot_3x3_fuzzy2(model, datasets, metrics):
    fig, axes = plt.subplots(4, 3, figsize=(12, 8), sharex=True)

    for col_idx, dataset in enumerate(datasets):  # Change row_idx to col_idx for datasets
        dataset_results = preprocess_results(f'{path}/{model}_{dataset}.csv',"kmeans")

        for row_idx, metric in enumerate(metrics):  # Change col_idx to row_idx for metrics
            ax = axes[row_idx, col_idx]  # Adjust the indexing to switch rows and columns
            for eta, color, marker in zip(etas, colors, markers):
                dataset_results[['Model', 'k', 'm', 'eta']] = dataset_results['Method'].str.split('_', expand=True)

                # Further process to clean up the extracted columns
                dataset_results['k'] = dataset_results['k'].str.extract('(\d+)', expand=False).astype(int)  # Extract just the number for k
                dataset_results['m'] = dataset_results['m'].str.extract('(\d+\.\d+|\d+)', expand=False).astype(float)  # Extract number for m
                dataset_results['eta'] = dataset_results['eta'].str.extract('(\d+\.\d+|\d+)', expand=False).astype(float)  # Extract number for eta
                subset = dataset_results[dataset_results['eta'] == eta]

                subset = subset.copy()  # Create a full copy to ensure safe modifications

                subset['mean_rank'] = subset[metrics].apply(
                    lambda row: row.rank(ascending=False).mean()
                    if row.name != "Davies-Bouldin Index" and row.name != "Xie-Beni"
                    else row.rank().mean(), axis=1
                )
                # For each k, find the row with the highest mean rank
                subset = subset.loc[subset.groupby('m')['mean_rank'].idxmax()]

                ax.scatter(subset['m'], subset[metric], color=color, marker=marker, label=eta, alpha=0.7)
                ax.plot(subset['m'], subset[metric], color=color, alpha=0.7)

            # Format the axis values to one decimal place for Davies-Bouldin and Silhouette Coefficient
            if metric in ["Davies-Bouldin Index", "Silhouette Coefficient","Xie-Beni"]:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}"))  # One decimal for y-axis

            # No decimals for cluster numbers (k) on the x-axis
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

            # Display all k values on x-axis
            ax.set_xticks(sorted(subset['m'].unique()))  # Show all k values
            ax.tick_params(axis='x', labelsize=10)  # Set x-axis label size smaller

            # Add grid
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            # Set titles, axis names, and labels
            if row_idx == 0:
                ax.set_title(f"{dataset}", fontsize=14)  # Dataset name in the first row
            if col_idx == 0:
                ax.set_ylabel(f"{metric}", fontsize=12)  # Metric name for the Y-axis
            if row_idx == 2:
                ax.set_xlabel("m (Fuzzy Parameter)", fontsize=12)  # X-axis label for the last row

            # Show legend only on the first row and last column
            if row_idx == 0 and col_idx == 2:
                ax.legend(title="eta", fontsize=10)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

# Function to plot a grid (3x3) for GMeans
def plot_3x3_spectral(model, datasets, metrics):
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True)

    for col_idx, dataset in enumerate(datasets):  # Change row_idx to col_idx for datasets
        dataset_results = preprocess_results(f'{path}/{model}_{dataset}.csv', model)

        for row_idx, metric in enumerate(metrics):  # Change col_idx to row_idx for metrics
            ax = axes[row_idx, col_idx]  # Adjust the indexing to switch rows and columns
            for aff in affinity:
                subset = dataset_results[dataset_results['affinity'] == aff]
                if aff == 'nearest-neighbors':
                    for n_value, color, marker in zip([25, 50, 100, 200], colors, markers):
                        subset2 = subset[subset['n_neighbors'] == n_value]
                        ranking = rank_and_sort(subset2,metrics=["Davies-Bouldin Index","Calinski","Silhouette Coefficient"],n=3)
                        subset2 = subset2[subset2[["affinity", "assign_labels", "n_neighbors", "eigen_solver"]]
                        .eq(ranking.loc[0, ["affinity", "assign_labels", "n_neighbors", "eigen_solver"]])
                        .all(axis=1)]
                        ax.scatter(subset2['k'], subset2[metric], color=color, marker=marker, label=f'nn {n_value}', alpha=0.7)
                        ax.plot(subset2['k'], subset2[metric], color=color, alpha=0.7)
                else:
                    ranking = rank_and_sort(subset,metrics=["Davies-Bouldin Index","Calinski","Silhouette Coefficient"],n=3)
                    subset = subset[subset[["affinity", "assign_labels", "n_neighbors", "eigen_solver"]]
                    .eq(ranking.loc[0, ["affinity", "assign_labels", "n_neighbors", "eigen_solver"]])
                    .all(axis=1)]

                    ax.scatter(subset['k'], subset[metric], color='black', marker='*', label=aff, alpha=0.7)
                    ax.plot(subset['k'], subset[metric], color='black', alpha=0.7)

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
            ax.grid(axis="x", linestyle="--", alpha=0.5)

            # Set titles, axis names, and labels
            if row_idx == 0:
                ax.set_title(f"Dataset {dataset}", fontsize=14)  # Dataset name in the first row
            if col_idx == 0:
                if metric == 'Davies-Bouldin Index':
                    ax.set_ylabel(f"DBI", fontsize=12)  # Metric name for the Y-axis
                else:
                    ax.set_ylabel(f"{metric}", fontsize=12)  # Metric name for the Y-axis
            if row_idx == 2:
                ax.set_xlabel("k (Number of Clusters)", fontsize=12)  # X-axis label for the last row

            # Show legend only on the first row and last column
            if row_idx == 0 and col_idx == 2:
                ax.legend(title="Affinity", fontsize=10)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def plot_3x3_kmeans(model, datasets, metrics):
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)

    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True)

    for col_idx, dataset in enumerate(datasets):  # Change row_idx to col_idx for datasets
        dataset_results = preprocess_results(f'{path}/{model}_{dataset}.csv', model)

        for row_idx, metric in enumerate(metrics):  # Change col_idx to row_idx for metrics
            ax = axes[row_idx, col_idx]  # Adjust the indexing to switch rows and columns
            for distance, color, marker in zip(distances, colors, markers):
                result = dataset_results[dataset_results['distance'] == distance]
                subset = pd.DataFrame()
                for k in result['k'].unique():
                    sub = rank_and_sort(result[result['k']==k],metrics=["Davies-Bouldin Index","Calinski","Silhouette Coefficient"],n=3)
                    subset = pd.concat((subset, sub.head(1)))

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
            ax.grid(axis="x", linestyle="--", alpha=0.5)

            # Set titles, axis names, and labels
            if row_idx == 0:
                ax.set_title(f"Dataset {dataset}", fontsize=14)  # Dataset name in the first row
            if col_idx == 0:
                if metric == 'Davies-Bouldin Index':
                    ax.set_ylabel(f"DBI", fontsize=12)  # Metric name for the Y-axis
                else:
                    ax.set_ylabel(f"{metric}", fontsize=12)  # Metric name for the Y-axis
            if row_idx == 2:
                ax.set_xlabel("k (Number of Clusters)", fontsize=12)  # X-axis label for the last row

            # Show legend only on the first row and last column
            if row_idx == 0 and col_idx == 2:
                ax.legend(title="Distance", fontsize=10)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

models = ['spectral', 'fuzzyclustering','GlobalFastKmeans', 'kmeans', 'gmeans', 'spectral']

def plot_all():
    # Main execution loop
    for model in models:
        if model == 'kmeans':
            plot_kmeans(datasets, 'kmeans')
            plot_3x3_kmeans(model, datasets, metrics)
        elif model == 'gmeans':
            plot_3x3_kmeans(model, datasets, metrics)
        elif model == 'GlobalFastKmeans':
            plot_3x3_kmeans(model, datasets, metrics)
        elif model == 'spectral':
            plot_3x3_spectral(model, datasets, metrics)
        elif model == 'fuzzyclustering':
            plot_3x3_fuzzy(model, datasets, ["Davies-Bouldin Index", "Silhouette Coefficient", "Calinski", "Xie-Beni"])

plot_all()
