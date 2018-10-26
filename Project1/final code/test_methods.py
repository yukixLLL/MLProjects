

from implementation import *
from cross_validation import *


def main():
    """
    test the correctness of 6 methods.
    Without any data pre-processing except standardize here.
    Split train data set into 2 subset, one is used for train
    the other is used for test to see the accuracy of the methods
    if you want to see the best model, you need to run run.py
    """
    y, x, ids = load_csv_data("train.csv", sub_sample=False)
    x, mean_x, std_x = standardize(x)

    ratio = 0.5
    y_tr, y_te, x_tr, x_te = split_data(y, x, ratio, seed=1)
    print("the shape of y_train: {yi}, the shape of x_train: {xi}, "
          "the shape of y_test: {yii}, the shape of x_test: {xii}"
          .format(yi=y_tr.shape, xi=x_tr.shape, yii=y_te.shape, xii=x_te.shape))

    # least_squares_gd_test(y_tr, y_te, x_tr, x_te)

    # least_squares_sgd_test(y_tr, y_te, x_tr, x_te)

    # least_squares_test(y_tr, y_te, x_tr, x_te)

    ridge_regression_test(y_tr, y_te, x_tr, x_te)


def least_squares_gd_test(y_tr, y_te, x_tr, x_te):
    y_train, x_train = build_model_data(y_tr, x_tr)
    initial_w = np.zeros(x_train.shape[1])
    max_iters = 500
    gamma = 0.1

    weight_star, loss_train = least_squares_gd(y_train, x_train, initial_w, max_iters, gamma)

    y_test, x_test = build_model_data(y_te, x_te)
    loss_test = compute_mse(y_test, x_test, weight_star)
    accuracy = degree_of_accuracy(y_test, x_test, weight_star)

    print("Train loss: {l_tr:.4f}, test_loss: {l_te:.4f}, accuracy of GD is: {accu: .4f}"
          .format(l_tr=loss_train, l_te=loss_test, accu=accuracy))


def least_squares_sgd_test(y_tr, y_te, x_tr, x_te):
    y_train, x_train = build_model_data(y_tr, x_tr)
    initial_w = np.zeros(x_train.shape[1])
    batch_size = 1
    max_iters = 1000
    gamma = 0.01
    weight_star, loss_train = least_squares_sgd(y_train, x_train, initial_w, batch_size, max_iters, gamma)

    y_test, x_test = build_model_data(y_te, x_te)
    loss_test = compute_mse(y_test, x_test, weight_star)
    accuracy = degree_of_accuracy(y_test, x_test, weight_star)

    print("Train loss: {l_tr:.4f}, test_loss: {l_te:.4f}, accuracy of SGD is: {accu: .4f}".format(
        l_tr=loss_train, l_te=loss_test, accu=accuracy))


def least_squares_test(y_tr, y_te, x_tr, x_te):
    y_train, x_train = build_model_data(y_tr, x_tr)
    weight_star, loss_train = least_squares(y_train, x_train)

    y_test, x_test = build_model_data(y_te, x_te)
    loss_test = compute_mse(y_test, x_test, weight_star)
    accuracy = degree_of_accuracy(y_test, x_test, weight_star)

    print("Train loss: {l_tr:.4f}, test_loss: {l_te:.4f}, accuracy of Least Squares is: {accu: .4f}".format(
        l_tr=loss_train, l_te=loss_test, accu=accuracy))


def ridge_regression_test(y_tr, y_te, x_tr, x_te):
    # use cross-validation to find best degree and lambda_
    degrees = np.arange(2, 6)
    k_fold = 4
    lambdas = np.logspace(-10, 0, 40)
    best_degree, best_lambda_, _ = best_param_selection(y_tr, x_tr, degrees, k_fold, lambdas)

    # use best degree and lambda_ to get best weight
    degree = best_degree
    x_train = build_poly(x_tr, degree)
    weight_star, loss_train = ridge_regression(y_tr, x_train, best_lambda_)

    # use the trained weight to get the prediction result
    x_test = build_poly(x_te, degree)
    loss_test = np.sqrt(2 * compute_mse(y_te, x_test, weight_star))
    accuracy = degree_of_accuracy(y_te, x_test, weight_star)

    print("Train loss: {l_tr:.4f}, test_loss: {l_te:.4f}, accuracy of Ridge Regression is: {accu: .4f}".format(
        l_tr=loss_train, l_te=loss_test, accu=accuracy))


if __name__ == "__main__":
    main()
