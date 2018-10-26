# -*- coding: utf-8 -*-

from implementation import *
from cross_validation import *


def main():
    # without_data_preprocessing()
    with_correlation_feature_cross_product()


def without_data_preprocessing():
    """optimize ridge regression without data pre-processing"""
    y, x, ids = load_csv_data("train.csv", sub_sample=False)
    x, mean_x, std_x = standardize(x)
    print("shape of x {x} shape of y {y}".format(x=x.shape, y=y.shape))
    y_test, x_test, ids_test = load_csv_data("test.csv", sub_sample=False)
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
    y, x, ids = load_csv_data("train.csv", sub_sample=False)
    print("shape of x {x} shape of y {y}".format(x=x.shape, y=y.shape))
    y_test, x_test, ids_test = load_csv_data("test.csv", sub_sample=False)

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


if __name__ == "__main__":
    main()
