from helpers import create_csv_submission, load_dataset 
import pandas as pd
import numpy as np
from constants import *
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
from baseline_helpers import user_habit_standardize, user_habit_standardize_recover

def toPyFMData(df):
    """Transform pandas dataframe into the dataformat PyFM needs"""
    data = []
    users = set(df.User.unique())
    movies = set(df.Movie.unique())
    ratings = df.Rating.astype(float).tolist()
    for row in df.iterrows():
        data.append({"user_id": str(row[1].User), "movie_id": str(row[1].Movie)})
    return (data, np.array(ratings), users, movies)

def pyfm_algo(train_df, test_df, model):
    (train_data, y_train, train_users, train_items) = toPyFMData(train_df)
    (test_data, y_test, test_users, test_items) = toPyFMData(test_df)

    v = DictVectorizer()
    X_train = v.fit_transform(train_data)
    X_test = v.transform(test_data)
    
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    
    predictions = test_df.copy()
    predictions.Rating = preds
    
    return predictions
    
def pyfm_algo_user_std(train_df, test_df, model):
    train_user_std = user_habit_standardize(train_df)
    pred = pyfm_algo(train_user_std, test_df, model)
    # recover 
    pred_recovered = user_habit_standardize_recover(train_df, pred)
    
    return pred_recovered