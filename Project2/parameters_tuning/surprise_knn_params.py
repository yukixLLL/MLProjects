from surprise import *
from surprise.model_selection import KFold, PredefinedKFold
from surprise import accuracy
from helpers import *
import pandas as pd
import numpy as np

surprise_train_path = "../datas/surprise_train_knn.csv"
train_dataset = "../datas/data_train.csv"

def prepare_surprise_data(train):
    """Save as a (User, Movie, Rating) pandas dataframe without column names"""
    # save to csv for later use
    print("[prepare_surprise_data] Saving to {}...".format(surprise_train_path))
    train.to_csv(surprise_train_path, index=False, header=False)

def surprise_cv_algo(data, algo, k_fold=5, verbose=True):
    # Split into folds
    kf = KFold(n_splits=k_fold)
    rmse_ = 0
        
    for trainset, testset in kf.split(data):
        # train and test algorithm.
        model = algo.fit(trainset)
        predictions = algo.test(testset)

        # Compute and print RMSE
        rmse_ += accuracy.rmse(predictions, verbose=verbose)
    
    rmse_mean = rmse_/k_fold
    return rmse_mean
    
    
def surprise_knn_best_params(train_path=train_dataset, verbose=True, t = Timer()):
    train_df = load_dataset(train_path)
    prepare_surprise_data(train_df)
    
    # reader with rating scale
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5))
    # load data from df
    data = Dataset.load_from_file(surprise_train_path, reader)
    
    #knn parameters
    ks = np.linspace(40, 200, 9, dtype=np.int64)
    names = ['pearson_baseline', 'pearson', 'msd', 'cosine']
    user_baseds = [True, False]
    params = dict()
    rmses = dict()
    
    for k in ks:
        params['k'] = k
        for name in names:
            params['name'] = name
            for user_based in user_baseds:
                params['user_based'] = user_based
                algo = KNNBaseline(k=k, sim_options={'name': name, 'user_based': user_based}, verbose=True)
                rmse = surprise_cv_algo(data, algo)
                print("------Time:{}, rmse: {}, k: {}, name: {}, user_based: {}------\n\n".format(t.now(), rmse, k, name, user_based))
                rmses[rmse] = params
    
    # Find the model with least RMSE
    lowest_rmse = min(rmses.keys())
    best_params = rmses[lowest_rmse]
    
    print("Best knn rmse: {}. Params: k: {}, name: {}, user_based: {}".format(lowest_rmse, best_params['k'], best_params['name'], best_params['user_based']))

surprise_knn_best_params()
