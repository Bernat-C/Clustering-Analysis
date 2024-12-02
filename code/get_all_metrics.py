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

        """r = run_all_kmeans(X,y)
        r.to_csv(f'./output/kmeans_{dataset}.csv')
        print('i')
        r = run_all_gmeans(X,y)
        r.to_csv(f'./output/gmeans_{dataset}.csv')
        """
        print('i')
        r = run_all_global_kmeans(X,y)
        r.to_csv(f'./output/GlobalFastKmeans_{dataset}.csv')

def plot_metrics():
    
    datasets = ['grid', 'sick', 'vowel']
    models = ['gmeans', 'kmeans', 'fuzzyclustering']

    path = './output/'

    markers = ["X", "v", "H", "*", "^", "<", ">", "1", "2", "3", "4", "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_", "P", "X"]

    for dataset in datasets:
        for model in models:
            results = pd.read_csv(f'{path}/{model}_{dataset}.csv')
            if model == 'kmeans':
                ### PLOT COMPARING KMEANS

                # Set Seaborn style for aesthetics
                sns.set(style="whitegrid", palette="muted", font_scale=1.2)

                results['k'] = results['Method'].str.split('_').str[1].str.split('k').str[1]
                results['distance'] = results['Method'].str.split('_').str[2].str.split('distance-').str[1]
                for distance, color in zip(['euclidean', 'manhattan', 'cosine'], ['#FF6347', '#1f77b4', '#2ca02c']):
                    results_dataset_model_dist = results[results['distance'] == distance]
                    ax = sns.violinplot(data=results_dataset_model_dist, x="k", y="Silhouette Coefficient", inner=None, color=color, label=f'{distance.capitalize()}')
                    i = 0
                for violin in ax.collections:
                    violin.set_edgecolor(violin._facecolors[0])
                    violin.set_facecolor('none')  # Remove the fill
                # Set title, labels, and customize them for clarity
                plt.title(f"Silhouette Coefficient for {dataset} dataset", fontsize=14)
                plt.xlabel("k (Number of Clusters)", fontsize=14)
                plt.ylabel("Silhouette Coefficient", fontsize=14)

                handles = [
                    plt.Line2D([0], [0], color='#FF6347', lw=4, label='Euclidean'),
                    plt.Line2D([0], [0], color='#1f77b4', lw=4, label='Manhattan'),
                    plt.Line2D([0], [0], color='#2ca02c', lw=4, label='Cosine'),
                ]
                if dataset == 'vowel':
                    plt.legend(handles=handles, title="Distance Metric", title_fontsize='13', loc='lower right', fontsize='12')
                plt.grid(axis="y", linestyle="--", alpha=0.5)
                plt.tight_layout()  # Ensures elements fit neatly
                plt.show()
                plt.clf()

            if model == 'gmeans':

                results['k'] = results['Method'].str.split('_').str[1].str.split('k').str[1]
                results['distance'] = results['Method'].str.split('_').str[2].str.split('distance-').str[1]

                for distance, color, marker in zip(['euclidean', 'manhattan', 'cosine'], ['#FF6347', '#1f77b4', '#2ca02c'], ['X', 'H', 'v']):
                    results_dataset_model_dist = results[results['distance'] == distance]
                    plt.scatter(
                        results_dataset_model_dist["k"], results_dataset_model_dist["Silhouette Coefficient"], 
                        marker = marker, color = color, label=distance, zorder=3, alpha = 0.5)
                    plt.plot(results_dataset_model_dist["k"], results_dataset_model_dist["Silhouette Coefficient"], color = color, alpha = 0.5)

                # Styling
                plt.title(f"Silhouette Coefficient for {dataset} dataset", fontsize=14)
                plt.xlabel("k (Number of Clusters)", fontsize=14)
                plt.ylabel("Silhouette Coefficient", fontsize=14)
                plt.xticks(sorted(results["k"].unique()))  # Ensure k starts at 2
                plt.legend()
                plt.grid(axis="y", linestyle="--", alpha=0.6)

                plt.tight_layout()
                plt.show()
                plt.clf()


get_all_metrics()


