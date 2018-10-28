

from implementation import ridge_regression
from proj1_helpers import *


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    # form data with polynomial degree
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)
    # ridge regression
    w, loss = ridge_regression(y_tr, tx_tr, lambda_)
    # calculate the loss for train and test data
    loss_tr = np.sqrt(2 * compute_mse(y_tr, tx_tr, w))
    loss_te = np.sqrt(2 * compute_mse(y_te, tx_te, w))
    accu = degree_of_accuracy(y_te, tx_te, w)
    return loss_tr, loss_te, w, accu


def best_param_selection(y, x, degrees, k_fold, lambdas, seed=4):
    """use k-fold cross-validation to find the best degree and lambda_"""
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    # for each degree, we compute the best lambdas and the associated rmse and accuracy
    best_lambdas = []
    best_rmses = []
    best_accuracy = []
    # vary degree
    for degree in degrees:
        # cross validation
        rmse_te = []
        accuracy_te = []
        for lambda_ in lambdas:
            rmse_te_tmp = []
            accuracy_te_tmp = []
            for k in range(k_fold):
                loss_tr, loss_te, _, accu = cross_validation(y, x, k_indices, k, lambda_, degree)
                rmse_te_tmp.append(loss_te)
                accuracy_te_tmp.append(accu)
                print("Degree: {d}, k-th train: {k}, train loss: {l_tr:.4f}, test_loss: {l_te:.4f}, lambda: {lam}, "
                      "accuracy: {a}".format(d=degree, k=k, l_tr=loss_tr, l_te=loss_te, lam=lambda_, a=accu))
            rmse_te.append(np.mean(rmse_te_tmp))
            accuracy_te.append(np.mean(accuracy_te_tmp))

        # We aim to maximize the accuracy of the test prediciton
        ind_accuracy_opt = np.argmax(accuracy_te)
        best_lambdas.append(lambdas[ind_accuracy_opt])
        best_rmses.append(rmse_te[ind_accuracy_opt])
        best_accuracy.append(accuracy_te[ind_accuracy_opt])
        print("-------------Degree: {d}, test_loss: {l_te:.4f}, lambda: {lam}, best_accuracy: {accu}-----------------".
              format(d=degree, l_te=rmse_te[ind_accuracy_opt],
                     lam=lambdas[ind_accuracy_opt], accu=accuracy_te[ind_accuracy_opt]))

    ind_best = np.argmax(best_accuracy)
    print("*********Best Degree: {d}, best_test_loss: {l_te:.4f}, Best_lambda: {lam}, best accuracy: {accu}***********".
          format(d=degrees[ind_best], l_te=best_rmses[ind_best], lam=best_lambdas[ind_best], accu=best_accuracy[ind_best]))

    return degrees[ind_best], best_lambdas[ind_best], best_rmses[ind_best]
