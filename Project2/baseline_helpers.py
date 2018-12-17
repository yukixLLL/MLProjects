import pandas as pd
import numpy as np

def user_standardize(df):
    mean_per_user = df.groupby('User').mean().Rating
    var_per_user = df.groupby('User').var().Rating
    stand_df = df.copy()
    stand_df['Rating'] = df.apply(lambda x: (x['Rating'] - mean_per_user[x['User']]) / var_per_user[x['User']], axis=1)
    return stand_df

def user_standardize_recover(df, stand_pred_test):
    mean_per_user = df.groupby('User').mean().Rating
    var_per_user = df.groupby('User').var().Rating
    pred_test = stand_pred_test.copy()
    pred_test['Rating'] = stand_pred_test.apply(lambda x: (x['Rating'] * var_per_user[x['User']] + mean_per_user[x['User']]), axis=1)
    return pred_test

def user_habit(df):
    global_mean = df.Rating.mean()
    mean_per_user = df.groupby('User').mean().Rating
    habit =mean_per_user - global_mean
    return habit

def user_habit_standardize(df):
    habit = user_habit(df)
    stand_df = df.copy()
    stand_df['Rating'] = df.apply(lambda x: x['Rating'] - habit[x['User']], axis=1)
    return stand_df

def user_habit_standardize_recover(df, stand_pred_test):
    habit = user_habit(df)
    pred_test = stand_pred_test.copy()
    pred_test['Rating'] = stand_pred_test.apply(lambda x: x['Rating'] + habit[x['User']], axis=1)
    return pred_test

def baseline_algo(train, test, model, training=False):
    return model(train, test, training=training)