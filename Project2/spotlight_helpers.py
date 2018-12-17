from helpers import *
from spotlight.interactions import Interactions
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score
from spotlight.factorization.explicit import ExplicitFactorizationModel
import torch
from constants import *

# -----------Time: 0:11:05.077740, Loss: regression, n_iter: 50, l2: 1e-05, batch_size: 256, learning_rate: 0.0001, embedding_dim: 150, rmse: 0.9847699999809265-------------


def spotlight_algo(train, test, model, verbose=True):
    # Explicitly convert into datatypes needed by spotlight models
    user_tr = np.array(train.User, dtype=np.int32)
    movie_tr = np.array(train.Movie, dtype=np.int32)
    rating_tr = np.array(train.Rating, dtype=np.float32)
    user_te = np.array(test.User, dtype=np.int32)
    movie_te = np.array(test.Movie, dtype=np.int32)
    
    train_data = Interactions(user_ids=user_tr, item_ids=movie_tr, ratings=rating_tr)
    test_data = Interactions(user_ids=user_tr, item_ids=movie_te)
    
    model.fit(train_data, verbose=verbose)
    # predict
    predictions = model.predict(user_te, movie_te)
    
    predictions_df = pd.DataFrame()
    predictions_df['User'] = user_te
    predictions_df['Movie'] = movie_te
    predictions_df['Rating'] = predictions
    
    return predictions_df