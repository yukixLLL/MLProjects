from helpers import *
from baseline import *
from baseline_helpers import *
from surprise_helpers import *
from spotlight_helpers import *
from pyfm_helpers import *
import scipy.optimize as sco
from MFRR import *
from als import *
from stack import load_algos

from os import listdir
from os.path import isfile, join
import shutil
import sys

def load_baseline_models():
    print("Loading baseline models...")
    models_dict = dict(
        # Baseline parameters: (train, test)
        baseline = dict(
            global_mean = baseline_global_mean,
            global_median = baseline_global_median,
            user_mean = baseline_user_mean,
            user_median = baseline_user_median,
            movie_mean = baseline_movie_mean,
            movie_median = baseline_movie_median,
            movie_mean_user_std = movie_mean_user_standardize,
            movie_median_user_std = movie_median_user_standardize,
            movie_mean_user_habit_std = movie_mean_user_habit_standardize,
            movie_median_user_habit_std = movie_median_user_habit_standardize,
            movie_mean_user_habit = movie_mean_user_habit,
            movie_median_user_habit = movie_median_user_habit,
        ),
    )
    
    model_msg = "{} model families loaded:\n ".format(len(list(models_dict.keys())))
    for i, model in models_dict.items():
        model_msg = model_msg + "{}: ".format(i)
        for key, value in model.items():
            model_msg = model_msg + "{}, ".format(key)
    model_msg = model_msg + "; \n"
    print(model_msg)
    return models_dict

def load_surprise1_models():    
    print("Loading load_surprise1_models models...")
    models_dict = dict(
#         surprise
        surprise = dict(
            surprise_svd = SVD(n_factors=50, n_epochs=200, lr_bu=1e-9 , lr_qi=1e-5, reg_all=0.01),           
            surprise_knn = KNNBaseline(k=100, sim_options={'name': 'pearson_baseline', 'user_based': False}),
#             surprise_svd_pp = SVDpp(n_factors=50, n_epochs=200, lr_bu=1e-9 , lr_qi=1e-5, reg_all=0.01),
        )
    )
    
    model_msg = "{} model families loaded:\n ".format(len(list(models_dict.keys())))
    for i, model in models_dict.items():
        model_msg = model_msg + "{}: ".format(i)
        for key, value in model.items():
            model_msg = model_msg + "{}, ".format(key)
    model_msg = model_msg + "; \n"
    print(model_msg)
    return models_dict

def load_surprise1_user_std_models():    
    print("Loading surprise1_user_std models...")
    models_dict = dict(
#         surprise
        surprise_user_std = dict(
            surprise_svd_user_std = SVD(n_factors=50, n_epochs=200, lr_bu=1e-9 , lr_qi=1e-5, reg_all=0.01),           
            surprise_knn_user_std = KNNBaseline(k=100, sim_options={'name': 'pearson_baseline', 'user_based': False}),
#             surprise_svd_pp = SVDpp(n_factors=50, n_epochs=200, lr_bu=1e-9 , lr_qi=1e-5, reg_all=0.01),
        )
    )
    
    model_msg = "{} model families loaded:\n ".format(len(list(models_dict.keys())))
    for i, model in models_dict.items():
        model_msg = model_msg + "{}: ".format(i)
        for key, value in model.items():
            model_msg = model_msg + "{}, ".format(key)
    model_msg = model_msg + "; \n"
    print(model_msg)
    return models_dict

def load_surprise2_models():    
    print("Loading surprise2 models...")
    models_dict = dict(
#         surprise
        surprise = dict(
#             surprise_svd = SVD(n_factors=50, n_epochs=200, lr_bu=1e-9 , lr_qi=1e-5, reg_all=0.01),           
#             surprise_knn = KNNBaseline(k=100, sim_options={'name': 'pearson_baseline', 'user_based': False}),
            surprise_svd_pp = SVDpp(n_factors=50, n_epochs=200, lr_bu=1e-9 , lr_qi=1e-5, reg_all=0.01),
        )
    )
    
    model_msg = "{} model families loaded:\n ".format(len(list(models_dict.keys())))
    for i, model in models_dict.items():
        model_msg = model_msg + "{}: ".format(i)
        for key, value in model.items():
            model_msg = model_msg + "{}, ".format(key)
    model_msg = model_msg + "; \n"
    print(model_msg)
    return models_dict

def load_spotlight_models():    
    print("Loading spotlight models...")
    models_dict = dict(
# #         spotlight
        spotlight = dict(
            spotlight=ExplicitFactorizationModel(loss='regression',
                                   embedding_dim=150,  # latent dimensionality
                                   n_iter=50,  # number of epochs of training
                                   batch_size=256,  # minibatch size
                                   l2=1e-5,  # strength of L2 regularization
                                   learning_rate=0.0001,
                                   use_cuda=torch.cuda.is_available()),
        ),
    )
    
    model_msg = "{} model families loaded:\n ".format(len(list(models_dict.keys())))
    for i, model in models_dict.items():
        model_msg = model_msg + "{}: ".format(i)
        for key, value in model.items():
            model_msg = model_msg + "{}, ".format(key)
    model_msg = model_msg + "; \n"
    print(model_msg)
    return models_dict

def load_spotlight_user_std_models():    
    print("Loading spotlight_user_std models...")
    models_dict = dict(
# #         spotlight
        spotlight_user_std = dict(
            spotlight_user_std=ExplicitFactorizationModel(loss='regression',
                                   embedding_dim=150,  # latent dimensionality
                                   n_iter=50,  # number of epochs of training
                                   batch_size=256,  # minibatch size
                                   l2=1e-5,  # strength of L2 regularization
                                   learning_rate=0.0001,
                                   use_cuda=torch.cuda.is_available()),
        ),
    )
    
    model_msg = "{} model families loaded:\n ".format(len(list(models_dict.keys())))
    for i, model in models_dict.items():
        model_msg = model_msg + "{}: ".format(i)
        for key, value in model.items():
            model_msg = model_msg + "{}, ".format(key)
    model_msg = model_msg + "; \n"
    print(model_msg)
    return models_dict

def load_pyfm_models():    
    print("Loading pyfm models...")
    models_dict = dict(
#         # pyfm
        pyfm = dict(
            pyfm=pylibfm.FM(num_factors=20, num_iter=200, verbose=True, 
                          task="regression", initial_learning_rate=0.001, 
                          learning_rate_schedule="optimal")
        ),
#         # keras
#         # MF
    )
    
    model_msg = "{} model families loaded:\n ".format(len(list(models_dict.keys())))
    for i, model in models_dict.items():
        model_msg = model_msg + "{}: ".format(i)
        for key, value in model.items():
            model_msg = model_msg + "{}, ".format(key)
    model_msg = model_msg + "; \n"
    print(model_msg)
    return models_dict

def load_pyfm_user_std_models():    
    print("Loading pyfm_user_std models...")
    models_dict = dict(
#         # pyfm
        pyfm_user_std = dict(
            pyfm_user_std=pylibfm.FM(num_factors=20, num_iter=200, verbose=True, 
                          task="regression", initial_learning_rate=0.001, 
                          learning_rate_schedule="optimal")
        ),
    )
    
    model_msg = "{} model families loaded:\n ".format(len(list(models_dict.keys())))
    for i, model in models_dict.items():
        model_msg = model_msg + "{}: ".format(i)
        for key, value in model.items():
            model_msg = model_msg + "{}, ".format(key)
    model_msg = model_msg + "; \n"
    print(model_msg)
    return models_dict
   
def load_mfrr_models():    
    print("Loading baseline models...")
    models_dict = dict(
#         # mrff
        mfrr = dict(
            mfrr= None
        ),

    )
    
    model_msg = "{} model families loaded:\n ".format(len(list(models_dict.keys())))
    for i, model in models_dict.items():
        model_msg = model_msg + "{}: ".format(i)
        for key, value in model.items():
            model_msg = model_msg + "{}, ".format(key)
    model_msg = model_msg + "; \n"
    print(model_msg)
    return models_dict

def load_mfrr_user_std_models():    
    print("Loading baseline models...")
    models_dict = dict(
#         # mrff
        mfrr_user_std = dict(
            mfrr_user_std = None
        ),

    )
    
    model_msg = "{} model families loaded:\n ".format(len(list(models_dict.keys())))
    for i, model in models_dict.items():
        model_msg = model_msg + "{}: ".format(i)
        for key, value in model.items():
            model_msg = model_msg + "{}, ".format(key)
    model_msg = model_msg + "; \n"
    print(model_msg)
    return models_dict

def load_als_models():    
    print("Loading baseline models...")
    models_dict = dict(
        als = dict(
            als = None
        ),

    )
    
    model_msg = "{} model families loaded:\n ".format(len(list(models_dict.keys())))
    for i, model in models_dict.items():
        model_msg = model_msg + "{}: ".format(i)
        for key, value in model.items():
            model_msg = model_msg + "{}, ".format(key)
    model_msg = model_msg + "; \n"
    print(model_msg)
    return models_dict

def load_als_user_std_models():    
    print("Loading baseline models...")
    models_dict = dict(
#         # mrff
        als_user_std = dict(
            als_user_std = None
        ),

    )
    
    model_msg = "{} model families loaded:\n ".format(len(list(models_dict.keys())))
    for i, model in models_dict.items():
        model_msg = model_msg + "{}: ".format(i)
        for key, value in model.items():
            model_msg = model_msg + "{}, ".format(key)
    model_msg = model_msg + "; \n"
    print(model_msg)
    return models_dict


def predict_and_save(saving_folder, models, training = True):
    # create folder 
    if not os.path.exists(saving_folder):
#         shutil.rmtree(saving_folder)
        os.makedirs(saving_folder)
    
    # load csv
    train_df = load_dataset(train_dataset, min_num_ratings = 0)
    test_df = load_dataset(test_dataset, min_num_ratings = 0)
    
    # Split training to blend
    if training:
        print("Splitting data for training...")
        train = train_df.copy()
        train_df, test_df = split_dataset(train_df, p_test=0.5, min_num_ratings = 0)
        print("Results of split: {}; \n {}".format(train_df.head(), test_df.head()))
        # folds_dict = define_folds(train_df, 5) - FOR FOLDS?
    
    # dictionary of the predictions
    predictions = dict()
        
    # load models
#     models_dict = load_models()
    models_dict = models
    # load algos
    algo_dict = load_algos()
    t = Timer()
    t.start()
    for model_family_name, model_family in models_dict.items():
        algo = algo_dict[model_family_name]
        print("Predicting using algo: {}, model: {}...".format(algo, model_family_name))

        for model_name, model in model_family.items():
            print("Time: {}, predicting with model: {}".format(t.now(), model_name))
            if model_family == 'baseline':
                if training:
                    prediction = algo(train, test_df, model)
                else: # predicting
                    prediction = algo(train_df.copy(), test_df.copy(), model)
            else:
                prediction = algo(train_df, test_df, model)
            print("Time: {}, Saving results of {}...\n".format(t.now(), model_name))
            prediction.to_csv("{}{}_predictions_{}.csv".format(saving_folder, model_name, t.now()))
            predictions[model_name] = prediction
    if training:
        gt_path = saving_folder + "ground_truth_{}.csv".format(t.now())
        print("Saving ground_truth to {}".format(gt_path))
        test_df.to_csv(gt_path)
        
    t.stop()
    return predictions, test_df


def load_predictions(reading_folder):
    def get_model_name(file_name):
        results = file_name.split('_predictions')
        return results[0]
        
    pred_array = [f for f in listdir(reading_folder) if isfile(join(reading_folder, f))]
    # Set user, col indices
    predictions = pd.read_csv(join(reading_folder, pred_array[0]), index_col=0).copy().reset_index(drop=True)
    predictions = predictions.drop(['Rating'], axis=1)
    predictions = predictions.sort_values(by=['User', 'Movie'])

    for i, pred in enumerate(pred_array):
        print("Reading {}/{} : {}...".format(i + 1, len(pred_array), pred))
        p = pd.read_csv(join(reading_folder, pred), index_col=0).sort_values(by=['User', 'Movie']).reset_index(drop=True)
        p = p.rename(index=str, columns={'Rating': get_model_name(pred)})
        predictions = pd.merge(predictions, p, how='outer', on=['User', 'Movie']).reset_index(drop=True)
    
    return predictions


if __name__ == '__main__':
    std = sys.argv[1]
    model_chosen = sys.argv[2] 
    
    if std == 'std':
        folder = "./user_std_predict_save/"
        folder_predict = "./user_std_train_predictions/"
    elif std == 'none':
        folder = "./predict_save/"
        folder_predict = "./train_predictions/"
        
    if model_chosen == 'pyfm':
        models = load_pyfm_models()
    elif model_chosen == 'baseline':
        models = load_baseline_models()
    elif model_chosen == 'surprise1':
        models = load_surprise1_models()
    elif model_chosen == 'surprise2':
        models = load_surprise2_models()
        _, _ = predict_and_save(folder_predict, models, training=False)
        exit()
    elif model_chosen == 'spotlight':
        models = load_spotlight_models()
    elif model_chosen == 'mfrr':
        models = load_mfrr_models()
    elif model_chosen == 'als':
        models = load_als_models()
    elif model_chosen == 'surprise1_user_std':
        models = load_surprise1_user_std_models()
    elif model_chosen == 'spotlight_user_std':
        models = load_spotlight_user_std_models()
    elif model_chosen == 'mfrr_user_std':
        models = load_mfrr_user_std_models()
    elif model_chosen == 'als_user_std':
        models = load_als_user_std_models()
    elif model_chosen == 'pyfm_user_std':
        models = load_pyfm_user_std_models()
 
        
    predictions, ground_truth = predict_and_save(folder, models)
    _, _ = predict_and_save(folder_predict, models, training=False)
#     res, predictions_tr = optimize(models, ground_truth, folder)
#     best_dict, rmse = get_best_weights(res, models, predictions_tr, ground_truth)
#     predictions = predict(best_dict)
#     submission = create_csv_submission(predictions)
#     submission.to_csv("./predictions_tr/blended_baseline.csv")