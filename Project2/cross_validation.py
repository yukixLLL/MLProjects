from helpers import *
import pandas as pd
from spotlight.interactions import Interactions
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score
from spotlight.factorization.explicit import ExplicitFactorizationModel
import torch
 

train_dataset = "./datas/data_train.csv"
test_dataset = "./datas/sampleSubmission.csv"

train = load_dataset(train_dataset)
test = load_dataset(test_dataset)

# Explicitly convert into datatypes needed by spotlight models
user_tr = np.array(train.User, dtype=np.int32)
movie_tr = np.array(train.Movie, dtype=np.int32)
rating_tr = np.array(train.Rating, dtype=np.float32)
user_te = np.array(train.User, dtype=np.int32)
movie_te = np.array(train.Movie, dtype=np.int32)

# Transform into Spotlight interactions
train_data = Interactions(user_ids=user_tr, item_ids=movie_tr, ratings=rating_tr)
test_data = Interactions(user_ids=user_tr, item_ids=movie_te)

loss = ['regression', 'logistic', 'poisson']
n_iter = np.linspace(20, 100, 9)
batch_size = [256, 512, 1024, 2048, 4096]
l2 = np.logspace(-15, -1, 100)
learning_rate = np.logspace(-15, -3, 100)
embedding_dim = [20, 50, 100, 150, 200]

def best_params_spotlight(losses, n_iters, batch_sizes, l2s, learning_rates, embedding_dims, train_data, t = Timer()):
    rmses = dict()
    params = dict()
    t.start()
    for loss in losses:
        params['loss'] = loss
        for n_iter in n_iters:
            params['n_iter'] = n_iter
            for batch_size in batch_sizes:
                params['batch_size'] = batch_size
                for l2 in l2s:
                    params['l2'] = l2
                    for learning_rate in learning_rates:
                        params['learning_rate'] = learning_rate
                        for embedding_dim in embedding_dims:
                            params['embedding_dim'] = embedding_dim
                            model = ExplicitFactorizationModel(loss='regression',
                               embedding_dim=embedding_dim,  # latent dimensionality
                               n_iter=n_iter,  # number of epochs of training
                               batch_size=batch_size,  # minibatch size
                               l2=l2,  # strength of L2 regularization
                               learning_rate=learning_rate,
                               use_cuda=torch.cuda.is_available())
                            
                            params['model'] = model
                                
                            train_tr_data, test_tr_data = random_train_test_split(train_data, random_state=np.random.RandomState(42))

                            model.fit(train_tr_data, verbose=True)

                            rmse = rmse_score(model, test_tr_data)
                            
                            rmses[rmse] = params  
                            print("-----------Time: {}, Loss: {}, n_iter: {}, l2: {}, batch_size: {}, learning_rate: {}, embedding_dim: {}, rmse: {}-------------\n\n".format(t.stop(), loss, n_iter, l2, batch_size, learning_rate, embedding_dim, rmse))
                            # restart timer
                            t.start()
    return rmses

timer = Timer()
timer.start()
best_params_spotlight(loss, n_iter, batch_size, l2, learning_rate, embedding_dim, train_data)
print("Finished in : {}".format(timer.stop()))