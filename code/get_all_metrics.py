import pandas as pd

from kmeans import run_all_kmeans
from gmeans import run_all_gmeans
from global_kmeans import run_all_global_kmeans


datasets = ['grid', 'sick', 'vowel']

for dataset in datasets:
    file = f'./datasets_processed/{dataset}.csv'
    df = pd.read_csv(file)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    r = run_all_kmeans(X,y)
    r['model'] = 'kmeans'
    r.to_csv(f'./output/results_kmeans_{dataset}.csv')

    r = run_all_gmeans(X,y)
    r['model'] = 'gmeans'
    r.to_csv(f'./output/results_gmeans_{dataset}.csv')

    run_all_global_kmeans(X,y)
    r['model'] = 'global_kmeans'
    r.to_csv(f'./output/results_global_kmeans_{dataset}.csv')

