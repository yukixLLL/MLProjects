# -*- coding: utf-8 -*-

from implementation import *
from cross_validation import *

TRAIN_PATH = "../data/train.csv"
TEST_PATH = "../data/test.csv"

def main():
    # without_data_preprocessing()
    # with_correlation_feature_cross_product()
     split_into_8_subset()
#    logistic_without_data_preprocessing()
#     logistic_with_correlation_feature_cross_product()

def without_data_preprocessing():
    """optimize ridge regression without data pre-processing"""
    y, x, ids = load_csv_data(TRAIN_PATH, sub_sample=False)
    x, mean_x, std_x = standardize(x)
    print("shape of x {x} shape of y {y}".format(x=x.shape, y=y.shape))
    y_test, x_test, ids_test = load_csv_data(TEST_PATH, sub_sample=False)
    x_test, _, _ = standardize(x_test, mean_x, std_x)

    degrees = np.arange(2, 3)
    k_fold = 4
    lambdas = np.logspace(-10, 0, 60)
    best_degree, best_lambda_, _ = best_param_selection(y, x, degrees, k_fold, lambdas)

    # use best degree and lambda_ to get best weight
    degree = best_degree
    x_train = build_poly(x, degree)
    weight_star, loss_train = ridge_regression(y, x_train, best_lambda_)

    # use the trained weight to get the prediction result
    x_test = build_poly(x_test, degree)
    y_pred = predict_labels(weight_star, x_test)
    create_csv_submission(ids_test, y_pred, "ridge_predit.csv")


def with_correlation_feature_cross_product():
    y, x, ids = load_csv_data(TRAIN_PATH, sub_sample=False)
    print("shape of x {x} shape of y {y}".format(x=x.shape, y=y.shape))
    y_test, x_test, ids_test = load_csv_data(TEST_PATH, sub_sample=False)

    x_train, x_valid_means = handle_missing(x)
    # Get indexes of the features that have a correlation with labels bigger that the threshold
    corr_ind = correlated(y, x_train, 0.015)
    x_train_prod = feature_cross_prod(x_train, corr_ind)
    x_train = x_train[:, corr_ind]
    # Concatenate with combinated features
    x_train = np.concatenate((x_train, x_train_prod), axis=1)
    print("The shape of feature after data pre-processing: ", x_train.shape)
    x_train, mean_x_train, std_x_train = standardize(x_train)

    degrees = np.arange(2, 12)
    k_fold = 4
    lambdas = np.logspace(-10, 0, 40)
    best_degree, best_lambda_, _ = best_param_selection(y, x_train, degrees, k_fold, lambdas)

    # use best degree and lambda_ to get best weight
    degree = best_degree
    x_train = build_poly(x, degree)
    weight_star, loss_train = ridge_regression(y, x_train, best_lambda_)

    x_test, _ = handle_missing(x_test, x_valid_means)
    x_test_prod = feature_cross_prod(x_test, corr_ind)
    x_test = x_test[:, corr_ind]
    x_test = np.concatenate((x_test, x_test_prod), axis=1)
    x_test, _, _ = standardize(x_test, mean_x_train, std_x_train)

    x_test = build_poly(x_test, best_degree)
    y_pred = predict_labels(weight_star, x_test)
    create_csv_submission(ids_test, y_pred, "ridge_predit.csv")


def split_into_8_subset():
    # data pre-processing
    # y_tr, x_tr, ids_tr, y_te, x_te, ids_te = load_data(train_path, test_path)
    # headers = load_headers(train_path)
    # split_data_according_to_jet_and_mass(y_tr, x_tr, ids_tr, y_te, x_te, ids_te, headers)

    # start training
    # first load data
    file_names_train = generate_processed_filenames(True)
    print(file_names_train)
    ys_train, xs_train, ids_train = load_processed_data(file_names_train)

    # Standandize xs
    x_means = []
    x_stds = []
    x_standardized = []
    for x_ in xs_train:
        x, mean_x, std_x = standardize(x_)
        x_standardized.append(x)
        x_means.append(mean_x)
        x_stds.append(std_x)

    # Initialize a degrees and lambdas dictionary for each jet
    degrees_jet = dict.fromkeys(file_names_train)
    lambdas_jet = dict.fromkeys(file_names_train)

    # build w using ridge regression
    k_fold = 4
    degrees = np.arange(5, 6)
    lambdas = np.logspace(-6, -3, 10)
    seed = 12

    for tx, y, f in zip(x_standardized, ys_train, file_names_train):
        print("Training for {}".format(f))
        best_degree, best_lambda_, _ = best_param_selection(y, tx, degrees, k_fold, lambdas, seed)
        degrees_jet[f] = best_degree
        lambdas_jet[f] = best_lambda_

    # degrees_final = {k: v for k, v in degrees_jet.items() if v is not None}
    # lambdas_final = {k: v for k, v in lambdas_jet.items() if v is not None}

    weights = []
    for x, y, f in zip(x_standardized, ys_train, file_names_train):
        x_poly = build_poly(x, degrees_jet[f])
        w, _ = ridge_regression(y, x_poly, lambdas_jet[f])
        print(len(w), degrees_jet[f])
        weights.append(w)

    file_names_test = generate_processed_filenames(False)
    print(file_names_test)
    ys_test, xs_test, ids_test = load_processed_data(file_names_test)

    degrees = list(degrees_jet.values())
    idds, yys = predict(xs_test, ids_test, x_means, x_stds, degrees, weights)
    create_csv_submission(idds, yys, "swimming.csv")


def predict(xs_test, ids_test, x_means, x_stds, degrees, weights):
    ids = []
    y_preds = []
    for x, id_, mean, std, degree, weight in zip(xs_test, ids_test, x_means, x_stds, degrees, weights):
        x_std, _, _ = standardize(x, mean, std)
        x_expanded = build_poly(x_std, degree)
        y_pred = predict_labels(weight, x_expanded)
        ids = np.append(ids, id_)
        y_preds = np.append(y_preds, y_pred)
    return ids, y_preds

def logistic_without_data_preprocessing():
    """logistic regression without data pre-processing"""
    y, x, ids = load_csv_data(TRAIN_PATH, sub_sample=False)
    x, mean_x, std_x = standardize(x)
    print("shape of x {x} shape of y {y}".format(x=x.shape, y=y.shape))
    y_test, x_test, ids_test = load_csv_data(TEST_PATH, sub_sample=False)
    x_test, _, _ = standardize(x_test, mean_x, std_x)

    #pre-process data
    y_train,tx_train=build_model_data(y,x)
    y_test,tx_test=build_model_data(y_test,x_test)
    y_train,_,_,_=data_preprocess_logsitic(y_train,tx_train,tx_test,y_test)
    #initial parameters
    max_iters_log = 2000
    gamma_log = 1e-6
    initial_w_log = np.zeros((tx_train.shape[1]))
    
    #get the weight and the loss of training data
    log_w,log_loss = logistic_regression(y_train,tx_train,initial_w_log,max_iters_log,gamma_log)

    # use the trained weight to get the prediction result
    y_pred = predict_labels_logistic(log_w, tx_test)
    create_csv_submission(ids_test, y_pred, "logistic_regression_predit.csv")
    
def logistic_with_correlation_feature_cross_product():
    """logistic regression with feature cross product"""
    y, x, ids = load_csv_data(TRAIN_PATH, sub_sample=False)
    print("shape of x {x} shape of y {y}".format(x=x.shape, y=y.shape))
    y_test, x_test, ids_test = load_csv_data(TEST_PATH, sub_sample=False)
    
    x_train, x_valid_means = handle_missing(x)
    # Get indexes of the features that have a correlation with labels bigger that the threshold
    corr_ind = correlated(y, x_train, 0.015)
    x_train_prod = feature_cross_prod(x_train, corr_ind)
    x_train = x_train[:, corr_ind]
    # Concatenate with combinated features
    x_train = np.concatenate((x_train, x_train_prod), axis=1)
    print("The shape of feature after data pre-processing: ", x_train.shape)
    x_train, mean_x_train, std_x_train = standardize(x_train)
    
    x_test, _ = handle_missing(x_test, x_valid_means)
    x_test_prod = feature_cross_prod(x_test, corr_ind)
    x_test = x_test[:, corr_ind]
    x_test = np.concatenate((x_test, x_test_prod), axis=1)
    x_test, _, _ = standardize(x_test, mean_x_train, std_x_train)

    #pre-process data
    y_train,tx_train=build_model_data(y,x_train)
    y_test,tx_test=build_model_data(y_test,x_test)
    y_train,_,_,_=data_preprocess_logsitic(y_train,tx_train,tx_test,y_test)
    #initial parameters
    max_iters_log = 2000
    gamma_log = 1e-6
    initial_w_log = np.zeros((tx_train.shape[1]))
    
    #get the weight and the loss of training data
    log_w,log_loss = logistic_regression(y_train,tx_train,initial_w_log,max_iters_log,gamma_log)

    # use the trained weight to get the prediction result
    y_pred = predict_labels_logistic(log_w, tx_test)
    create_csv_submission(ids_test, y_pred, "logistic_regression_predit.csv")

if __name__ == "__main__":
    main()
