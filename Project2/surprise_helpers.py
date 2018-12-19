from surprise import *
from surprise.model_selection import KFold, PredefinedKFold
from surprise import accuracy
from itertools import islice
import pandas as pd
import numpy as np
import os 
from constants import *
from baseline_helpers import *

def prepare_surprise_data(train, test, folder="./datas/tmp/"):
    """Save as a (User, Movie, Rating) pandas dataframe without column names"""
    # save to csv for later use
    print("[prepare_surprise_data] Saving to {}, {}...".format(surprise_train_path, surprise_test_path))
    train.to_csv(surprise_train_path, index=False, header=False)
    test.to_csv(surprise_test_path, index=False, header=False)
    
    # reader with rating scale
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5))
    
    # Specify the training and test dataset
    folds_files = [(surprise_train_path, surprise_test_path)]

    data = Dataset.load_from_folds(folds_files, reader=reader)
    return data
    

def surprise_algo(train, test, algo, verbose=True, training=False):
    # prepare data
    data = prepare_surprise_data(train, test)
    
    pkf = PredefinedKFold()
    
    if verbose:
        print("Start prediction...")
    for trainset, testset in pkf.split(data):
        # train and predict algorithm.
        model = algo.fit(trainset)
        predictions = algo.test(testset)
    
    pred = pd.read_csv(surprise_test_path, names = ["User", "Movie", "Rating"])
    
    if verbose:
        print("Postprocessing predictions...")
    for index, row in pred.iterrows():
        rating = predictions[index].est
        row.Rating = rating
    
    return pred

def surprise_algo_rescaled(train, test, algo, verbose=True, training=False):
    train_rescaled = user_habit_standardize(train)
    pred = surprise_algo(train_rescaled, test, algo)
    # recover 
    pred_recovered = user_habit_standardize_recover(train, pred)
    
    return pred_recovered
