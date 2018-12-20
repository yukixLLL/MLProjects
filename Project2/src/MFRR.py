import pandas as pd
import numpy as np
from helpers import *
from sklearn.linear_model import Ridge
from baseline_helpers import user_habit_standardize, user_habit_standardize_recover

def compute_rmse_rr(pred, rating):
    """compute RMSE"""
    err = rating - pred
    mse = err**2
    return np.sqrt(np.mean(mse.mean(skipna=True)))

def update_user_features(W,Z,train_df,lambda_,num_users):
    """update user feature matrix"""
    # join movie feature matrix to expand features for training in ridge regression
    user_ft_data = train_df.set_index('Movie').join(Z).sort_values('User').set_index('User').fillna(0)
    # initial ridge regression model
    model = Ridge(fit_intercept=False,alpha = lambda_)
    
    # update user feature matrix 
    for i in range(1,num_users+1):
        df = user_ft_data.loc[i,:]
        try:
            X = df.loc[:,df.columns!='Rating']
            y = df.loc[:,['Rating']]
    #         print(y.shape)
            model.fit(X,y)
            W.loc[i,:] = model.coef_
        except:
            # for just 1 rating
            W.loc[i,:] = df
    return W


def update_movie_features(W,Z,train_df,lambda_,num_movies):
    """update user feature matrix"""  
    # join user feature matrix to expand features for training in ridge regression
    movie_ft_data = train_df.set_index('User').join(W).sort_values('Movie').set_index('Movie')
    model = Ridge(fit_intercept=False,alpha = lambda_)
    
    # update movie feature matrix 
    for i in range(1,num_movies+1):
        df = movie_ft_data.loc[i,:]
        X = df.loc[:,df.columns!='Rating']
        y = df.loc[:,['Rating']]
        model.fit(X,y)
        Z.loc[i,:] = model.coef_
    return Z

def MF_RR(train_tr, rating,num_features,lambda_,iterations=20):
    """Ridge Regression (RR) algorithm."""
    # define parameters
    stop_criterion = 1e-5
    change = 1
    error_list = [0, 0]
    it = 0
 
    num_users, num_movies = len(rating),len(rating.columns)
    # init matrix using dataframe
    user_features = pd.DataFrame(np.random.rand(num_users,num_features),index=range(1,num_users+1),columns=range(1,num_features+1))
    movie_features = pd.DataFrame(np.random.rand(num_movies,num_features),index=range(1,num_movies+1),columns=range(1,num_features+1))
    
    train_rmse = 0
    # deep copy
    W = user_features.copy()
    Z = movie_features.copy()
    # start updating feature matrix
    while(it < iterations):
        W = update_user_features(W,Z,train_tr,lambda_,num_users)
        
        Z = update_movie_features(W,Z,train_tr,lambda_,num_movies)
        # predict
        pred = W.dot(Z.T)
        pred[pred > 5] = 5
        pred[pred < 1] = 1
        train_rmse = compute_rmse_rr(pred,rating)
        print("MF-RR training RMSE : {err}".format(err=train_rmse))
        error_list.append(train_rmse)
        change = np.fabs(error_list[-1] - error_list[-2])
        # stop when change is small enough
        if (change < stop_criterion):
            print("Converge!")
            break;
            
        it += 1
        
#     print("MF-RR Final training RMSE : {err}".format(err=train_rmse))
    return W,Z

def mf_rr_algo(train_df,test_df, model):
    """MFRR for blending"""  
    # initial parameters
    num_features = 20
    lambda_ = 19
    # convert into matrix
    rating = train_df.pivot(index="User",columns="Movie",values="Rating")
    # RR
    user_features,movie_features = MF_RR(train_df,rating,num_features,lambda_,iterations=50)
    # predict
    pred = user_features.dot(movie_features.T)
    pred[pred > 5] = 5
    pred[pred < 1] = 1
    
    # match test_df
    test_user = test_df['User'].values
    test_movie = test_df['Movie'].values
    
    pred_test =[]
    for user,movie in zip(test_user,test_movie):
        pred_rating = pred.loc[user,movie]
        pred_test.append(pred_rating)
        
    pred_test = np.asarray(pred_test)

    test_ret = test_df.copy()
    test_ret.drop(columns='Rating',inplace=True)
    test_ret['Rating'] = pred_test
    prediction = test_ret.copy()
#     prediction = create_csv_submission(test_ret)
    
    return prediction

def mf_rr_algo_user_std(train_df,test_df, model):
    """ run MFRR with user habit standardization"""
    # standradize
    train_user_std = user_habit_standardize(train_df)
    pred = mf_rr_algo(train_user_std, test_df, model)
    # recover 
    pred_recovered = user_habit_standardize_recover(train_df, pred)
    
    return pred_recovered