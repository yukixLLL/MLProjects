from helpers import *
from baseline import *
from baseline_helpers import *
from surprise_helpers import *
from spotlight_helpers import *
from pyfm_helpers import *
import scipy.optimize as sco

from os import listdir
from os.path import isfile, join

def load_models():
    print("Loading models...")
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
        
#         surprise
#         surprise = dict(
#             surprise_svd = SVD(n_factors=50, n_epochs=200, lr_bu=1e-9 , lr_qi=1e-5, reg_all=0.01),
#             surprise_svd_pp = SVDpp(n_factors=50, n_epochs=200, lr_bu=1e-9 , lr_qi=1e-5, reg_all=0.01),
#             surprise_knn = KNNBaseline(k=100, sim_options={'name': 'pearson_baseline', 'user_based': False}),
#         ),
#         spotlight
#         spotlight = dict(
#             spotlight=ExplicitFactorizationModel(loss='regression',
#                                    embedding_dim=150,  # latent dimensionality
#                                    n_iter=50,  # number of epochs of training
#                                    batch_size=256,  # minibatch size
#                                    l2=1e-5,  # strength of L2 regularization
#                                    learning_rate=0.0001,
#                                    use_cuda=torch.cuda.is_available()),
#         ),
#         # als
        
#         # pyfm
#         pyfm = dict(
#             pyfm=pylibfm.FM(num_factors=20, num_iter=200, verbose=True, 
#                           task="regression", initial_learning_rate=0.001, 
#                           learning_rate_schedule="optimal")
#         ),
        # keras
        # MF
    )
    
    model_msg = "{} model families loaded:\n ".format(len(list(models_dict.keys())))
    for i in list(models_dict.keys()):
        model_msg = model_msg + "{}; ".format(i)
    print(model_msg)
    return models_dict
   
    
def load_algos():
    algo_dict = dict(
        baseline = baseline_algo, # baseline_algo(train, test, model)
        surprise = surprise_algo, # surprise_algo(train, test, algo, verbose=True, training=False)
        spotlight = spotlight_algo, # spotlight_algo(train, test, model, verbose=True)
        pyfm = pyfm_algo,
    )
    return algo_dict
algos = load_algos()


def predict_and_save(folder = "./predictions/", training = True):
    # create folder if not existent
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # load csv
    train_df = load_dataset(train_dataset, min_num_ratings = 0)
    test_df = load_dataset(test_dataset, min_num_ratings = 0)
    
    # Split training to blend
    if training:
        print("Splitting data for training...")
        train = train_df.copy()
        train_df, test_df = split_dataset(train_df, p_test=0.5)
        # folds_dict = define_folds(train_df, 5) - FOR FOLDS?
    
    # dictionary of the predictions
    predictions = dict()
        
    # load models
    models_dict = load_models()
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
            prediction.to_csv("{}{}_predictions({}).csv".format(folder, model_name, t.now()))
            predictions[model_name] = prediction
        
    return predictions, test_df


def load_predictions(folder="./predictions"):
    def get_model_name(file_name):
        results = file_name.split('_predictions')
        return results[0]
        
    pred_array = [f for f in listdir(folder) if isfile(join(folder, f))]
    # Set user, col indices
    predictions = pd.read_csv(join(folder, pred_array[0]), index_col=0).copy().reset_index(drop=True)
    predictions = predictions.drop(['Rating'], axis=1)
    predictions = predictions.sort_values(by=['User', 'Movie'])

    for i, pred in enumerate(pred_array):
        print("Reading {}/{} : {}...".format(i + 1, len(pred_array), pred))
        p = pd.read_csv(join(folder, pred), index_col=0).sort_values(by=['User', 'Movie']).reset_index(drop=True)
        p = p.rename(index=str, columns={'Rating': get_model_name(pred)})
        predictions = pd.merge(predictions, p, how='outer', on=['User', 'Movie']).reset_index(drop=True)
    
    return predictions

def optimize(models, ground_truth, folder="./predictions_train/"):
    t = Timer()
    t.start()
    print("Loading predictions....")
    predictions = load_predictions(folder=folder)
    print("Time: {}, Finished loading.".format(t.now()))
    t.stop(verbose= False)
    
    # Initialize first weights (- nb columns for User, Movie)
    w0 = [1/(len(predictions.columns) - 2) for i in range(len(predictions.columns) - 2)]
    
    print("Optimizing...")
    t.start()
    res = sco.minimize(evaluate_stacking, w0, method='SLSQP', args=(models, predictions, ground_truth), options={'maxiter': 1000, 'disp': True})
    print("Time: {}. Optimization done.".format(t.now()))
    t.stop()
    
    return res

def get_best_weights(res, models, predictions, ground_truth):
    # Create best dictionnary
    best_dict = {}
    idx = 0
    for key, model_family in models.items():
        best_dict[key] = dict()
        for name in model_family.keys():
            best_dict[key][name] = res.x[idx]
            idx = idx + 1
    
    print("Best weights: \n {}".format(best_dict))
    # test
    rmse = evaluate_stacking(res.x, models, predictions, ground_truth)
    print("Best weights rmse: {}".format(rmse))
    return best_dict, rmse


def predict(weight_dict):
    print("Predicting....")
    predictions, _ = predict_and_save(folder="./pred_tmp/", training=False)
    predictions = load_predictions(folder="./pred_tmp")
    print("Finished loading.")
    
    stacked = np.zeros(predictions.shape[0])
    for key, model_fam in models.items():
        weights = weight_dict[key]
        for name in model_fam.keys():
            weight = weights[name]
            print("Stacking {} * {}...".format(weight, name))
            stacked = stacked + weight * predictions[name]
    
    pred = predictions[['User', 'Movie']].copy()
    pred['Rating'] = stacked
    return pred


if __name__ == '__main__':
    models = load_models()
    predictions, ground_truth = predict_and_save("./predictions_tr/")
    res = optimize(models, ground_truth, folder="./predictions_tr/")
    best_dict, rmse = get_best_weights(res, models, predictions_tr, ground_truth)
    predictions = predict(best_dict)
    submission = create_csv_submission(predictions)
    submission.to_csv("./predictions_tr/blended_baseline.csv")