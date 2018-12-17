from surprise import *
from surprise.model_selection import KFold, PredefinedKFold
from surprise import accuracy
from itertools import islice
import pandas as pd
import numpy as np
import os 
from constants import *

def prepare_surprise_data(path, name):
    """Save as a (User, Movie, Rating) pandas dataframe without column names"""
    df = pd.read_csv(path)
    parsed_df = pd.DataFrame()
    # Get all pairs of (r44_c1) -> (44, 1) (user, movie)
    user_movie_indices = df.Id.apply(lambda x: x.split('_'))
    parsed_df['User'] =  [int(i[0][1:]) for i in user_movie_indices]
    parsed_df['Movie'] = [int(i[1][1:]) for i in user_movie_indices]
    parsed_df['Rating'] = df['Prediction']
    num_items = parsed_df.Movie.nunique()
    num_users = parsed_df.User.nunique()
    
    # save to csv for later use
    parsed_df.to_csv(name, index=False, header=False)
    
    print("Saved {}; USERS: {} ITEMS: {}".format(name, num_users, num_items))

def surprise_algo(train, test, algo, verbose=True, training=False):
    """"Note: Train and test are not used"""
    # Check if file exists
    train_exists = os.path.isfile(surprise_train_path)
    if not train_exists:
        prepare_surprise_data(train_dataset, surprise_train_path)
    
    test_exists = os.path.isfile(surprise_test_path)
    if not test_exists:
        prepare_surprise_data(test_dataset, surprise_test_path)
    
    # reader with rating scale
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5))
    
    # Specify the training and test dataset
    folds_files = [(surprise_train_path, surprise_test_path)]

    data = Dataset.load_from_folds(folds_files, reader=reader)
    pkf = PredefinedKFold()
    
    if verbose:
        print("Start prediction...")
    for trainset, testset in pkf.split(data):
        # train and predict algorithm.
        model = algo.fit(trainset)
        predictions = algo.test(testset)
    
    pred = pd.read_csv(test_path, names = ["User", "Movie", "Rating"])
    
    if verbose:
        print("Postprocessing predictions...")
    for index, row in pred.iterrows():
        rating = predictions[index].est
        row.Rating = rating
    
    return pred
