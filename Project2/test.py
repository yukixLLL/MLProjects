import pandas as pd
import numpy as np
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer

train_dataset = "./datas/data_train.csv"
pred_dataset = "./datas/sampleSubmission.csv"

def load_dataset(path):
    """Load dataset as a (User, Movie, Rating) pandas dataframe"""
    df = pd.read_csv(path)
    parsed_df = pd.DataFrame()
    # Get all pairs of (r44_c1) -> (44, 1) (user, movie)
    user_movie_indices = df.Id.apply(lambda x: x.split('_'))
    parsed_df['User'] =  [int(i[0][1:]) for i in user_movie_indices]
    parsed_df['Movie'] = [int(i[1][1:]) for i in user_movie_indices]
    parsed_df['Rating'] = df['Prediction']
    
    num_items = parsed_df.Movie.nunique()
    num_users = parsed_df.User.nunique()
    print("USERS: {} ITEMS: {}".format(num_users, num_items))
    return parsed_df

train_df = load_dataset(train_dataset)

def split_dataset(parsed_df, p_test=0.1, min_num_ratings=0):
    movies_per_user = parsed_df.User.value_counts()
    users_per_movie = parsed_df.Movie.value_counts()

    valid_users = movies_per_user[movies_per_user > min_num_ratings].index.values
    valid_movies = users_per_movie[users_per_movie > min_num_ratings].index.values
    valid_parsed_df = parsed_df[parsed_df.User.isin(valid_users) & parsed_df.Movie.isin(valid_movies)].reset_index(drop=True)
    
    print("movies per user: min[{a}], max[{b}], users per movie: min[{c}], max[{d}].".
          format(a=movies_per_user.min(), b=movies_per_user.max(), c=users_per_movie.min(), d=users_per_movie.max()))

    size = valid_parsed_df.shape[0]
    indexes = list(range(size))
    np.random.shuffle(indexes)

    test_ind = indexes[:int(size*p_test)]
    train_ind = indexes[int(size*p_test):]

    test = valid_parsed_df.loc[test_ind].reset_index(drop=True)
    train = valid_parsed_df.loc[train_ind].reset_index(drop=True)
    print("The shape of test_dataset: {test}, train_dataset: {train}".format(test=test.shape, train=train.shape))
    
    return train, test

train, test = split_dataset(train_df)

def toPyFMData(df):
    """Transform pandas dataframe into the dataformat PyFM needs"""
    data = []
    users = set(df.User.unique())
    movies = set(df.Movie.unique())
    ratings = df.Rating.astype(float).tolist()
    for row in df.iterrows():
        data.append({"user_id": str(row[1].User), "movie_id": str(row[1].Movie)})
    return (data, np.array(ratings), users, movies)

(train_data, y_train, train_users, train_items) = toPyFMData(train)
(test_data, y_test, test_users, test_items) = toPyFMData(test)
v = DictVectorizer()
X_train = v.fit_transform(train_data)
X_test = v.transform(test_data)

fm = pylibfm.FM(num_factors=20, num_iter=200, verbose=True, task="regression", initial_learning_rate=0.001, learning_rate_schedule="optimal")
fm.fit(X_train,y_train)

preds = fm.predict(X_test)
from sklearn.metrics import mean_squared_error
print("FM MSE: %.4f" % mean_squared_error(y_test,preds))