from autorank import autorank, plot_stats, create_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from skcriteria.pipeline import mkpipe
from skcriteria.preprocessing.invert_objectives import (
    InvertMinimize,
    NegateMinimize,
)
from skcriteria.cmp import mkrank_cmp
from skcriteria.preprocessing.filters import FilterNonDominated
from skcriteria.preprocessing.scalers import SumScaler, VectorScaler
from skcriteria.agg.simple import WeightedProductModel, WeightedSumModel
from skcriteria.agg.similarity import TOPSIS


def preprocess_results(filepath, method):
    results = pd.read_csv(filepath)
    if method == 'kmeans':
        results['k'] = results['Method'].str.split('_').str[1].str.split('k').str[1].astype(int)
        results['distance'] = results['Method'].str.split('_').str[2].str.split('distance-').str[1]
    elif method == 'spectral':
        results['k'] = results['Method'].str.split('_').str[1].str.split('k').str[1].astype(int)
        results['affinity'] = results['Method'].str.split('_').str[2].str.split('affinity').str[1]
        results['n_neighbors'] = results['Method'].str.split('_').str[3].str.split('n-neighbors').str[1].astype(int)
        results['assign_labels'] = results['Method'].str.split('_').str[4].str.split('assign-labels').str[1]
        results['eigen_solver'] = results['Method'].str.split('_').str[5].str.split('eigen-solver').str[1]
    return results

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

    print(rankings)

    # Rank the rows based on all metrics
    for metric in metrics:
        if metric == "Davies-Bouldin Index":
            # Rank ascending for Davies-Bouldin Index
            rankings[f"{metric}_Rank"] = rankings[metric].rank(ascending=True)
        elif metric == "Xie-Beni":
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

distances = ['euclidean', 'manhattan', 'cosine']

model = 'kmeans'
dataset = 'sick'


metrics = ["Davies-Bouldin Index", "Silhouette Coefficient", "Calinski"]

for distance in distances:
    result = preprocess_results(f'./output/{model}_{dataset}.csv', 'kmeans')
    
    result = result[result['distance'] == distance]
    subset = pd.DataFrame()
    for k in result['k'].unique():
        
        sub = rank_and_sort(result[result['k']==k],metrics=["Davies-Bouldin Index","Calinski","Silhouette Coefficient"],n=3)
        subset = pd.concat((subset, sub.head(1)))

    # For each k, find the row with the highest mean rank
    
    """result = rank_and_sort(subset, metrics=["Davies-Bouldin Index","Silhouette Coefficient", "Calinski"],n=3)
    print(result)
    # Calculate mean rank
    subset['mean_rank'] = subset[metrics].apply(
        lambda row: row.rank(ascending=False).mean() if row.name != "Davies-Bouldin Index" else row.rank().mean(), axis=1
    )
    # For each k, find the row with the highest mean rank
    subset = subset.loc[subset.groupby('k')['mean_rank'].idxmax()]"""
    break