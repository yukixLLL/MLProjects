from costs import *
from implementation_helpers import *
from implementation import *


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


# def cross_validation(args, y, x, k_indices, k, degree, model):
#     """return the loss of ridge regression."""
#     indices = np.arange(k_indices.shape[0])
#     k_ind = k_indices[k]
#     x_te = x[k_ind]
#     x_tr = x[k_indices[indices != k, :].reshape(-1)]
#
#     y_te = y[k_indices[k]]
#     y_tr = y[k_indices[indices != k, :].reshape(-1)]
#
#     #x_tr_extended = build_poly(x_tr, degree)
#     #x_te_extended = build_poly(x_te, degree)
#
#     args.append(x_tr)
#     args.append(y_tr)
#     #w = model(args)
#
#     w = ridge_regression(args)
#
#     #loss_tr = np.sqrt(2 * compute_loss(y_tr, x_tr_extended, w))
#     loss_tr = np.sqrt(2 * compute_loss(y_tr, x_tr, w))
#     #loss_te = np.sqrt(2 * compute_loss(y_te, x_te_extended, w))
#     loss_te = np.sqrt(2 * compute_loss(y_te, x_te, w))
#     print("Cross-validation: ")
#     return loss_tr, loss_te

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
    w = ridge_regression(y_tr, tx_tr, lambda_)
    # calculate the loss for train and test data
    loss_tr = np.sqrt(2 * compute_loss(y_tr, tx_tr, w))
    loss_te = np.sqrt(2 * compute_loss(y_te, tx_te, w))
    return loss_tr, loss_te, w
