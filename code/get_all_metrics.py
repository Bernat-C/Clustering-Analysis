import pandas as pd


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
        print('i')"""
        r = run_all_gmeans(X,y)
        r.to_csv(f'./output/gmeans_{dataset}.csv')
        """
        print('i')
        r = run_all_global_kmeans(X,y)
        r.to_csv(f'./output/GlobalFastKmeans_{dataset}.csv')"""

get_all_metrics()


