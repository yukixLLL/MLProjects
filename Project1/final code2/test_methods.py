

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
    y, x, ids = load_csv_data("train.csv", sub_sample=False,cut_values=False)
    x, mean_x, std_x = standardize(x)

    ratio = 0.5
    y_tr, y_te, x_tr, x_te = split_data(y, x, ratio, seed=1)
    print("the shape of y_train: {yi}, the shape of x_train: {xi}, "
          "the shape of y_test: {yii}, the shape of x_test: {xii}"
          .format(yi=y_tr.shape, xi=x_tr.shape, yii=y_te.shape, xii=x_te.shape))

    # least_squares_gd_test(y_tr, y_te, x_tr, x_te)

    # least_squares_sgd_test(y_tr, y_te, x_tr, x_te)

    # least_squares_test(y_tr, y_te, x_tr, x_te)

#     ridge_regression_test(y_tr, y_te, x_tr, x_te)
    
#     logistic_regression_test(y_tr,y_te,x_tr,x_te)

    reg_logistic_regression_test(y_tr,y_te,x_tr,x_te)

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
def logistic_regression_test(y_tr,y_te,x_tr,x_te):    
    #pre-process data
    y_tr,tx_tr,y_te,tx_te=data_preprocess_logsitic(y_tr,x_tr,x_te,y_te,'test')
    
    #initial parameters
    max_iters_log = 20000
    gamma_log = 1e-7
    initial_w_log = np.zeros((tx_tr.shape[1]))
    
    #get the weight and the loss of training data
    log_w,log_loss = logistic_regression(y_tr,tx_tr,initial_w_log,max_iters_log,gamma_log)
    #get the loss of testing data
    loss_test = calculate_logistic_loss(y_te, tx_te, log_w)
    #test the accuracy of losgistic regression
    accuracy =degree_of_accuracy_logitstic(y_te, tx_te, log_w)
    print("Train loss: {l_tr:.4f}, test_loss: {l_te:.4f}, accuracy of Logistic Regression is: {accu: .4f}".format(
        l_tr=log_loss, l_te=loss_test, accu=accuracy))
    
def reg_logistic_regression_test(y_tr,y_te,x_tr,x_te):
    #pre-process data
    y_tr,tx_tr,y_te,tx_te=data_preprocess_logsitic(y_tr,x_tr,x_te,y_te,'test')
    #initial parameters
    max_iters_reg=2000
    lambda_reg=1
    gamma_reg=1e-6
    initial_w_reg=np.zeros((tx_tr.shape[1]))
    
    #get the weight and the loss of training data
    reg_w,reg_loss=reg_logistic_regression(y_tr,tx_tr,lambda_reg,initial_w_reg,max_iters_reg,gamma_reg)
    #get the loss of testing data
    loss_test=calculate_logistic_loss(y_te, tx_te, reg_w)+0.5*lambda_reg*np.sum(reg_w**2)
    accuracy =degree_of_accuracy_logitstic(y_te, tx_te, reg_w)
    print("Train loss: {l_tr:.4f}, test_loss: {l_te:.4f}, accuracy of Regularized Logistic Regression is: {accu: .4f}".format(
        l_tr=reg_loss, l_te=loss_test, accu=accuracy))

if __name__ == "__main__":
    main()
