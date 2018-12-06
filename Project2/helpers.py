import pandas as pd
import numpy as np

def load_dataset(path):
    """Load dataset as a (User, Movie, Rating) pandas dataframe"""
    df = pd.read_csv(path)
    parsed_df = pd.DataFrame()
    # Get all pairs of (r44_c1) -> (44, 1) (user, movie)
    user_movie_indices = df.Id.apply(lambda x: x.split('_'))
    parsed_df['User'] =  [int(i[0][1:]) for i in user_movie_indices]
    parsed_df['Movie'] = [int(i[1][1:]) for i in user_movie_indices]
    parsed_df['Rating'] = df['Prediction']
    return parsed_df

def split_dataset(df, p_test=0.2, min_num_ratings = 0):
    """ split dataframe into train and test set """
    # select user and item based on the condition.
    user_counts = df.User.value_counts()
    valid_users = user_counts[user_counts > min_num_ratings].index.values
    movie_counts = df.Movie.value_counts()
    valid_movies = movie_counts[movie_counts > min_num_ratings].index.values

    valid_ratings = df[df.User.isin(valid_users) & df.Movie.isin(valid_movies)].reset_index(drop=True)

    # Split data
    size = df.shape[0]
    indexes = list(range(size))
    np.random.shuffle(indexes)
    
    test_ind = indexes[:int(size*p_test)]
    train_ind = indexes[int(size*p_test):]
    
    test = valid_ratings.loc[test_ind]
    train = valid_ratings.loc[train_ind]

    print("Train: {}, Test: {}".format(test.shape, train.shape))
    
    # Test that the sum of nb rows of splitted dataframes = nb rows of original
    if (train.shape[0] + test.shape[0] == df.shape[0]):
        return train.reset_index(drop=True), test.reset_index(drop=True)
    else:
        raise Exception("[Error] Train: {} + Test {} != Original: {} !!".format(train_tr.shape[0], test_tr.shape[0], df.shape[0]))

def compute_rmse(pred, truth):
    """ compute RMSE for pandas dataframes """
    truth_sorted = truth.sort_values(['User', 'Movie']).reset_index(drop=True)
    pred_sorted = pred.sort_values(['User', 'Movie']).reset_index(drop=True)

    truth_sorted['square_error'] = np.square(truth_sorted['Rating'] - prediction_sorted['Rating'])

    mse = truth_sorted['square_error'].mean()
    rmse = np.sqrt(mse)

    return rmse