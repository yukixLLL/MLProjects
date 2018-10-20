# -*- coding: utf-8 -*-
from costs import *
from gradient import *
from implementation_helpers import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    loss = np.inf
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
    return loss, w


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
        print("SGD({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
    return loss, w


def least_squares(y, tx):
    """The least squares using Normal"""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)


def ridge_regression(y, tx, lambda_):
#def ridge_regression(args):
    # y = args[2]
    # tx = args[1]
    # lambda_ = args[0]
    """implement ridge regression."""
    a = tx.T.dot(tx) + lambda_ * 2 * len(y) * np.identity(tx.shape[1])
    b = tx.T.dot(y)
    w_ridge = np.linalg.solve(a, b)
    return w_ridge

