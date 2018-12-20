from surprise import *
from surprise.model_selection import KFold, PredefinedKFold
from surprise import accuracy
from src.helpers import Timer
import pandas as pd
import numpy as np

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
    
def surprise_svd_best_params(train_path="datas/train.csv", test_path="datas/test.csv", verbose=True, t = Timer()):
    # reader with rating scale
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5))
    # load data from df
    data = Dataset.load_from_file(train_path, reader)
    
    #svd parameters
    n_factors = [50, 100, 200]
    n_epochss = np.linspace(200, 40, 9, dtype=np.int32)
    n_epochss = [200, 500, 800]
    reg_alls = np.logspace(-2, -5, 4)
    lr_bus = np.logspace(-10, -2, 9)
    lr_qis = np.logspace(-10, -2, 9)
    params = dict()
    rmses = dict()
    
    t.start()
    
    ## ------rmse: 1.0665431544988566, n_factor:50, n_epoch: 200, reg_all: 0.01, lr_bu: 1e-09, lr_qi: 1e-05------
    params['lr_bu'] = 1e-09
    params['lr_qi'] = 1e-05
    params['n_factor'] = 50
    lr_bu = 1e-09
    lr_qi = 1e-05
    n_factor = 50
    for n_epoch in n_epochss:
        params['n_epoch'] = n_epoch
        for reg_all in reg_alls:
            params['reg_all'] = reg_all
            for lr_bu in lr_bus:
                params['lr_bu'] = lr_bu
                for lr_qi in lr_qis:
                    params['lr_qi'] = lr_qi
                    for n_factor in n_factors:
                        params['n_factor'] = n_factor
            
                        
            algo = SVD(n_factors = n_factor, n_epochs = n_epoch, reg_all = reg_all, lr_bu = lr_bu, lr_qi = lr_qi, verbose=False)
            rmse = surprise_cv_algo(data, algo)
            print("------Time:{}, rmse: {}, n_factor:{}, n_epoch: {}, reg_all: {}, lr_bu: {}, lr_qi: {}------\n\n".format(t.now(), rmse, n_factor, n_epoch, reg_all, lr_bu, lr_qi))
            rmses[rmse] = params
    
    # Find the model with least RMSE
    lowest_rmse = min(rmses.keys())
    best_params = rmses[lowest_rmse]
    
    print("Best svd rmse: {}, n_epoch: {}, reg_all: {}, lr_bu: {}, lr_qi: {}".format(lowest_rmse, best_params['n_epoch'], best_params['reg_all'], best_params['lr_bu'], best_params['lr_qi']))
    

surprise_svd_best_params()