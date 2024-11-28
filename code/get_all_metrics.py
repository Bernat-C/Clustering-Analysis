import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from kmeans import run_all_kmeans
from gmeans import run_all_gmeans
from global_kmeans import run_all_global_kmeans


def get_all_metrics():

    datasets = ['grid', 'vowel', 'sick']

    for dataset in datasets:
        file = f'./datasets_processed/{dataset}.csv'
        df = pd.read_csv(file)
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]

        r = run_all_kmeans(X,y)
        r['model'] = 'kmeans'
        r['dataset'] = dataset
        r.to_csv(f'./output/results_kmeans_{dataset}.csv')
        print('i')
        r = run_all_gmeans(X,y)
        r['model'] = 'gmeans'
        r['dataset'] = dataset
        r.to_csv(f'./output/results_gmeans_{dataset}.csv')
        """
        print('i')
        run_all_global_kmeans(X,y)
        r['model'] = 'global_kmeans'
        r['dataset'] = dataset
        r.to_csv(f'./output/results_global_kmeans_{dataset}.csv')"""

def plot_metrics():
    
    datasets = ['grid', 'sick', 'vowel']

    path = './output/'
    files = os.listdir(path)

    results = pd.DataFrame()
    for file in files:
        results = pd.concat((results, pd.read_csv(path+file)))

    for dataset in datasets:
        results_dataset = results[results['dataset'] == dataset]

        markers = ["X", "v", "H", "*", "^", "<", ">", "1", "2", "3", "4", "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_", "P", "X"]
        
        ### PLOT COMPARING KMEANS
        results_dataset_model = results_dataset[results_dataset['model'] == 'kmeans']
        for distance, color in zip(['euclidean', 'manhattan', 'cosine'], ['red', 'blue', 'green']):
            results_dataset_model_dist = results_dataset_model[results_dataset_model['distance'] == distance]
            sns.violinplot(data=results_dataset_model_dist, x="k", y="Silhouette Coefficient", inner=None, color=color, alpha=0.4, label='KMeans (Mean ± STD)')
        plt.title(f"Silhouette Coefficient for KMeans", fontsize=14)
        plt.ylabel("Silhouette Coefficient")
        plt.xlabel("k")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        plt.show()
        plt.clf()

        ### PLOT COMPARING BEST KMANS WITH OTHER MODELS
        for model in results_dataset['model'].unique():
            i_m = 0
            results_dataset_model = results_dataset[results_dataset['model'] == model]
            for distance in results_dataset_model['distance'].unique():
                results_dataset_model_dist = results_dataset_model[results_dataset_model['distance'] == distance]
                label = model + ' ' + distance if len(results_dataset_model['distance'].unique())>1 else model
                # Overlay the single-sample dataset
                plt.scatter(
                    results_dataset_model_dist["k"], results_dataset_model_dist["Silhouette Coefficient"], 
                    marker = markers[i_m], color = 'black', label=label, zorder=3, alpha = 0.3)
                i_m += 1
                plt.plot(results_dataset_model_dist["k"], results_dataset_model_dist["Silhouette Coefficient"], color = 'black', alpha = 0.3)

            # Styling
            plt.title(f"Silhouette Coefficient for {dataset} dataset", fontsize=14)
            plt.ylabel("Silhouette Coefficient")
            plt.xlabel("k")
            plt.xticks(sorted(results["k"].unique()))  # Ensure k starts at 2
            plt.legend()
            plt.grid(axis="y", linestyle="--", alpha=0.6)

            plt.tight_layout()
            plt.show()
            plt.clf()


plot_metrics()


