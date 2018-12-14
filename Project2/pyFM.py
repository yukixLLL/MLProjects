import pandas as pd
import numpy as np
from helpers import *
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold

def pyFM_cv_algo(algo, k_fold=5, verbose=True):
    
    kf = KFold(n_splits=k_fold)
    rmse_ = 0
    
    for trainset_ind, testset_ind in kf.split(train_df):
        
        trainset = train_df.iloc[trainset_ind]
        testset = train_df.iloc[testset_ind]
        
        (train_data, y_train, train_users, train_items) = toPyFMData(trainset)
        (test_data, y_test, test_users, test_items) = toPyFMData(testset)
        v = DictVectorizer()
        X_train = v.fit_transform(train_data)
        X_test = v.transform(test_data)
    
        algo.fit(X_train,y_train)
        preds = algo.predict(X_test)
        for i in range(len(preds)):
            if preds[i] > 5:
                preds[i] = 5
            elif preds[i] < 1:
                preds[i] = 1
        predictions = testset.copy()
        predictions['Rating'] = preds

        rmse_ += compute_rmse(predictions, testset)
        
    rmse_mean = rmse_/k_fold
    return rmse_mean
        
def pyFM_cv(verbose=True, t = Timer()): 
    #pyFM parameters
    factors = np.linspace(20, 200, 9, dtype=np.int64)
    learning_rates = np.logspace(-2, -5, 4)
    params = dict()
    rmses = dict()
    
    for k in factors:
        params['k'] = k
        for rate in learning_rates:
            params['rate'] = rate
            algo = pylibfm.FM(num_factors=k, num_iter=200, verbose=True, task="regression", initial_learning_rate=rate, learning_rate_schedule="optimal")
            rmse = pyFM_cv_algo(algo)
            print("------Time:{}, rmse: {}, factors: {}, learning_rates: {}------\n\n".format(t.now(), rmse, k, rate))
            rmses[rmse] = params
    
    # Find the model with least RMSE
    lowest_rmse = min(rmses.keys())
    best_params = rmses[lowest_rmse]
    
    print("Best pyFM rmse: {}. Params: factors: {}, learning_rates: {}".format(lowest_rmse, best_params['k'], best_params['rate']))
    
train_dataset = "./datas/data_train.csv"
train_df = load_dataset(train_dataset)

t = Timer()
t.start()
pyFM_cv()
t.stop(verbose=True)