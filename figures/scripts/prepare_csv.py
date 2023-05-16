import pandas as pd
import pickle

def set_cluster_from_fold(folds, idx):
    for fold in folds:
        if int(idx) in folds[fold]['test']:
            return fold
    return -1

def main(csv_path='../../data/dataset.csv', folds_path='../../data/dhs_incountry_folds.pkl'):
    with open(folds_path, 'rb') as f:
        folds = pickle.load(f)
    original_csv=pd.read_csv(csv_path)
    clustered_csv = original_csv.drop(columns=['households', 'cluster'],axis=1)
    for i in range(len(original_csv)):
        clustered_csv.at[i,'cluster'] = set_cluster_from_fold(folds,i)
    clustered_csv.to_csv('../../data/dataset_with_clusters.csv')
    return clustered_csv

if __name__ == "__main__":
    main()
