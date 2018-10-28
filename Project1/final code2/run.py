# -*- coding: utf-8 -*-

from implementation import *
from cross_validation import *

TRAIN_PATH = "../data/train.csv"
TEST_PATH = "../data/test.csv"
HEADER_PATH = "../data/all_headers.csv"

def main():
     split_into_8_subset()


def split_into_8_subset():
    # data pre-processing
    y_tr, x_tr, ids_tr, y_te, x_te, ids_te = load_data(TRAIN_PATH, TEST_PATH)
    headers = load_headers(TRAIN_PATH)
    all_headers = split_data_according_to_jet_and_mass(y_tr, x_tr, ids_tr, y_te, x_te, ids_te, headers)

    # start training
    # first load data
    file_names_train = generate_processed_filenames(isTrain=True)
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
    degrees = np.arange(4, 13)
    lambdas = np.logspace(-20, -3, 100)
    seed = 12

    for tx, y, f in zip(x_standardized, ys_train, file_names_train):
        print("Training for {}".format(f))
        best_degree, best_lambda_, _ = best_param_selection(y, tx, degrees, k_fold, lambdas, seed)
        degrees_jet[f] = best_degree
        lambdas_jet[f] = best_lambda_

    # Build weights using the best selected accuracies, and log before standardizing
    # Reload data to log before standardizing
    ys_train, xs_train, ids_train = load_processed_data(file_names_train)
    log_left_skewed(all_headers, file_names_train, xs_train)
    x_means = []
    x_stds = []
    x_standardized = []

    for x_ in xs_train:
        x, mean_x, std_x = standardize(x_)
        x_standardized.append(x)
        x_means.append(mean_x)
        x_stds.append(std_x)

    weights = []
    for x, y, f in zip(x_standardized, ys_train, file_names_train):
        x_poly = build_poly(x, degrees_jet[f])
        w, _ = ridge_regression(y, x_poly, lambdas_jet[f])
        print(len(w), degrees_jet[f])
        weights.append(w)

    # Process the test the same way
    file_names_test = generate_processed_filenames(isTrain=False)
    ys_test, xs_test, ids_test = load_processed_data(file_names_test)



    degrees = list(degrees_jet.values())
    idds, yys = predict(xs_test, ids_test, x_means, x_stds, degrees, weights, all_headers)
    create_csv_submission(idds, yys, "swimming.csv")
    print("File creation success!")


def predict(xs_test, ids_test, x_means, x_stds, degrees, weights, all_headers):
    # Log all left_skewed data
    print("Predicting...")
    test_filenames = generate_processed_filenames(False)
    log_left_skewed(all_headers, test_filenames, xs_test)
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


if __name__ == "__main__":
    main()
