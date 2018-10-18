import numpy as np

import numpy as np
import matplotlib.pyplot as plt

from proj1_helpers import *
from data_preprocessing import *
from costs import *

""""""
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # dL/d(w) = -1/N * Xe
    e = y - tx.dot(w)
    gradient = -1/y.shape[0]*np.transpose(tx).dot(e)
    return gradient

# VERIFY!!!
def compute_stoch_gradient(y, tx, w):
    e = y - tx * w
    return -1 * tx * e

def compute_loss(y, tx, w):
    """Compute loss using MSE"""
    e = y - np.dot(tx, w)
    return 0.5*y.shape[0] * e.dot(e)


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        loss = compute_loss(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        # update w by gradient
        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}, gradient={gradient}".format(
            #bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1], gradient=gradient))
    return losses, ws

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    losses = []
    ws = [initial_w]
    w = initial_w
    for n_iter in range(max_iters):
        # choose random number
        n = np.random.randint(0, y.shape[0])
        loss = compute_loss(y, tx, w)
        gradient = compute_gradient(y[n], tx[n], w[n])
        losses.append(loss)
        gradient = gradient
        # compute w
        w = w - gamma * gradient
        ws.append(w)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}, gradient={gradient}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1], gradient=gradient))
    return losses, ws

