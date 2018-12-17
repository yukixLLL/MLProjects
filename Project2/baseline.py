import pandas as pd
import numpy as np
from helpers import compute_rmse

def baseline_global_mean(train, test, training=False):
    mean = train.Rating.mean()
    pred_test = test.copy()
    pred_test.Rating = mean
    
    if training:
        rmse = compute_rmse(pred_test, test)
        print("baseline_global_mean: {}".format(rmse))
    return pred_test

def baseline_global_median(train, test, training=False):
    median = train.Rating.median()
    pred_test = test.copy()
    pred_test.Rating = median

    if training:
        rmse = compute_rmse(pred_test, test)
        print("baseline_global_median: {}".format(rmse))
          
    return pred_test

def baseline_user_mean(train, test, training=False):
    mean_per_user = train.groupby('User').mean().Rating

    pred_test = test.copy()

    def predict(sub_df):
        sub_df['Rating'] = mean_per_user[sub_df.iloc[0,0]]
        return sub_df

    pred_test = pred_test.groupby('User').apply(predict)
    
    if training:
        rmse = compute_rmse(pred_test, test)
        print("baseline_user_mean: {}".format(rmse))

    return pred_test

def baseline_user_median(train, test, training=False):
    median_per_user = train.groupby('User').median().Rating

    pred_test = test.copy()

    def predict(sub_df):
        sub_df['Rating'] = median_per_user[sub_df.iloc[0,0]]
        return sub_df

    pred_test = pred_test.groupby('User').apply(predict)
    if training:
        rmse = compute_rmse(pred_test, test)
        print("baseline_user_median: {}".format(rmse))

    return pred_test 

def baseline_movie_mean(train, test, training=False):
    mean_per_movie = train.groupby('Movie').mean().Rating

    pred_test = test.copy()

    def predict(sub_df):
        sub_df['Rating'] = mean_per_movie[sub_df.iloc[0,1]]
        return sub_df

    pred_test = pred_test.groupby('Movie').apply(predict)
    if training:
        rmse = compute_rmse(pred_test, test)
        print("baseline_movie_mean: {}".format(rmse))

    return pred_test

def baseline_movie_median(train, test, training=False):
    median_per_movie = train.groupby('Movie').median().Rating

    pred_test = test.copy()

    def predict(sub_df):
        sub_df['Rating'] = median_per_movie[sub_df.iloc[0,1]]
        return sub_df

    pred_test = pred_test.groupby('Movie').apply(predict)
    if training:
        rmse = compute_rmse(pred_test, test)
        print("baseline_movie_mean: {}".format(rmse))

    return pred_test

def movie_mean_user_standardize(train, test, training=False):
    stand_train = user_standardize(train)
    stand_pred_test, rmse = baseline_movie_mean(stand_train, test)

    #recover from the standardized predicted test rating
    pred_test = user_standardize_recover(train, stand_pred_test)

    #compute the rmse
    if training:
        rmse = compute_rmse(pred_test, test)
        print("movie_mean_user_std: {}".format(rmse))
    
    return pred_test

def movie_median_user_standardize(train, test, training=False):
    #standardize the rating according to per user mean and variance
    stand_train = user_standardize(train)

    #predict the standardized test rating
    stand_pred_test, rmse = baseline_movie_median(stand_train, test)

    #recover from the standardized predicted test rating
    pred_test = user_standardize_recover(train, stand_pred_test)

    #compute the rmse
    if training:
        rmse = compute_rmse(pred_test, test)
        print("movie_median_user_std: {}".format(rmse))

    return pred_test

def movie_mean_user_habit_standardize(train, test, training=False):
    #standardize the rating according to per user habit
    pred_test = test.copy()
    pred_test.Rating = pred_test.Rating.apply(lambda x: float(x))
    stand_train = user_habit_standardize(train)

    #predict the standardized test rating
    stand_pred_test, rmse = baseline_movie_mean(stand_train, test)

    #recover from the standardized predicted test rating
    pred_test = user_habit_standardize_recover(train, stand_pred_test)

    #compute the rmse
    if training:
        rmse = compute_rmse(pred_test, test)
        print("movie_mean_user_habit_std: {}".format(rmse))

    return pred_test


def movie_median_user_habit_standardize(train, test, training=False):
    #standardize the rating according to per user mean and variance
    stand_train = user_habit_standardize(train)

    #predict the standardized test rating
    stand_pred_test, rmse = baseline_movie_median(stand_train, test)

    #recover from the standardized predicted test rating
    pred_test = user_habit_standardize_recover(train, stand_pred_test)

    #compute the rmse
    if training:
        rmse = compute_rmse(pred_test, test)
        print("movie_median_user_habit_std: {}".format(rmse))

    return pred_test

def movie_mean_user_habit(train, test, training=False):
    habit = user_habit(train)
    mean_per_movie = train.groupby('Movie').mean().Rating

    pred_test = test.copy()
    pred_test['Rating'] = pred_test['Rating'].apply(lambda x: float(x))

    def predict(x):
        x['Rating'] = mean_per_movie[x['Movie']] + habit[x['User']]
        return x

    pred_test = pred_test.apply(predict, axis=1)
    pred_test['User'] = pred_test['User'].apply(lambda x: int(x))
    pred_test['Movie'] = pred_test['Movie'].apply(lambda x: int(x))

    if training:
        rmse = compute_rmse(pred_test, test)
        print("movie_mean_user_habit: {}".format(rmse))

    return pred_test

def movie_median_user_habit(train, test, training=False):
    habit = user_habit(train)
    median_per_movie = train.groupby('Movie').median().Rating

    pred_test = test.copy()
    pred_test['Rating'] = pred_test['Rating'].apply(lambda x: float(x))

    def predict(x):
        x['Rating'] = median_per_movie[x['Movie']] + habit[x['User']]
        return x

    pred_test = pred_test.apply(predict, axis=1)
    pred_test['User'] = pred_test['User'].apply(lambda x: int(x))
    pred_test['Movie'] = pred_test['Movie'].apply(lambda x: int(x))
    
    if training:
        rmse = compute_rmse(pred_test, test)
        print("movie_median_user_habit: {}".format(rmse))

    return pred_test

