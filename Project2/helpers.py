import pandas as pd
import numpy as np
import time
import datetime

class Timer:
    import time
    import datetime 
    
    def __init__(self):
        self.t = 0
        
    def start(self):
        self.t = time.time()
        
    def stop(self, verbose = False):
        time_taken = datetime.timedelta(seconds=time.time() - self.t).__str__()
        if verbose:
            print("Time taken: {}".format(time_taken))
        self.t = 0
        return time_taken
    
    def now(self):
        time_taken = datetime.timedelta(seconds=time.time() - self.t).__str__()
        return time_taken

def load_dataset(path, min_num_ratings = 0):
    """Load dataset as a (User, Movie, Rating) pandas dataframe"""
    df = pd.read_csv(path)
    parsed_df = pd.DataFrame()
    # Get all pairs of (r44_c1) -> (44, 1) (user, movie)
    user_movie_indices = df.Id.apply(lambda x: x.split('_'))
    parsed_df['User'] =  [int(i[0][1:]) for i in user_movie_indices]
    parsed_df['Movie'] = [int(i[1][1:]) for i in user_movie_indices]
    parsed_df['Rating'] = df['Prediction']
    
    # select user and item based on the condition.
    user_counts = parsed_df.User.value_counts()
    valid_users = user_counts[user_counts > min_num_ratings].index.values
    movie_counts = parsed_df.Movie.value_counts()
    valid_movies = movie_counts[movie_counts > min_num_ratings].index.values

    valid_ratings = parsed_df[(parsed_df.User.isin(valid_users)) & (parsed_df.Movie.isin(valid_movies))].reset_index(drop=True)
    print("[load_dataset] Valid: {}".format(valid_ratings.shape))
    
    return valid_ratings

def split_dataset(df, p_test=0.2, min_num_ratings = 0, verbose=False):
    """ split dataframe into train and test set """
    # select user and item based on the condition.
    user_counts = df.User.value_counts()
    valid_users = user_counts[user_counts > min_num_ratings].index.values
    movie_counts = df.Movie.value_counts()
    valid_movies = movie_counts[movie_counts > min_num_ratings].index.values

    valid_ratings = df[(df.User.isin(valid_users)) & (df.Movie.isin(valid_movies))].reset_index(drop=True)
    print("[split_dataset] Valid: {}".format(valid_ratings.shape))

    # Split data
    size = valid_ratings.shape[0]
    indexes = list(range(size))
    np.random.shuffle(indexes)
    
    test_ind = indexes[:int(size*p_test)]
    train_ind = indexes[int(size*p_test):]
    
    test = valid_ratings.loc[test_ind]
    train = valid_ratings.loc[train_ind]
    
    if verbose:
        print("Train: {}, Test: {}".format(train.shape, test.shape))
    
    # Test that the sum of nb rows of splitted dataframes = nb rows of original
    if (train.shape[0] + test.shape[0] == valid_ratings.shape[0]):
        return train.reset_index(drop=True), test.reset_index(drop=True)
    else:
        raise Exception("[Error] Train: {} + Test {} != Original: {} !!".format(train_tr.shape[0], test_tr.shape[0], df.shape[0]))

def compute_rmse(pred, real):
    pred_sorted = pred.sort_values(['Movie', 'User']).reset_index(drop=True)
    real_sorted = real.sort_values(['Movie', 'User']).reset_index(drop=True)

    mse = np.square(pred_sorted.Rating - real_sorted.Rating).mean()
    rmse = np.sqrt(mse)

    return rmse

def create_csv_submission(predictions):
    """Create submission file """
    print("Creating submission file...")
    def round_(x):
        if x > 5:
            return 5
        elif x < 1:
            return 1
        else:
            return round(x)
     
    predictions['Id'] = predictions.apply(lambda x: 'r{}_c{}'.format(int(x.User), int(x.Movie)), axis=1)
    predictions['Prediction'] = predictions.Rating.apply(lambda x: round_(x))
    return predictions[['Id', 'Prediction']]