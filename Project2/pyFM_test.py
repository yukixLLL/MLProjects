import pandas as pd
import numpy as np
from helpers import *
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold

def compute_rmse_test(pred, truth):
    """ compute RMSE for pandas dataframes """
    truth_sorted = truth.sort_values(['User', 'Movie']).reset_index(drop=True)
    pred_sorted = pred.sort_values(['User', 'Movie']).reset_index(drop=True)

    truth_sorted['square_error'] = np.square(truth_sorted['Rating'] - pred_sorted['Rating'])

    mse = truth_sorted['square_error'].mean()
    rmse = np.sqrt(mse)

    return rmse

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
        print("predictions['Rating']: ".format(predictions['Rating'].iloc[0]))
        rmse = compute_rmse_test(predictions, testset)
        print("rmse: {}".format(rmse))
        rmse_ += rmse
        
    rmse_mean = rmse_/k_fold
    return rmse_mean
    
train_dataset = "./datas/data_train.csv"
train_df = load_dataset(train_dataset)

t = Timer()
t.start()
algo = pylibfm.FM(num_factors=20, num_iter=200, verbose=True, task="regression", initial_learning_rate=0.001, learning_rate_schedule="optimal")
rmse = pyFM_cv_algo(algo)
print("------Time:{}, rmse: {}, factors: {}, learning_rates: {}------\n\n".format(t.now(), rmse,20, 0.001))
t.stop(verbose=True)