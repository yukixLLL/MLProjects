# -*- coding: utf-8 -*-


from proj1_helpers import *


"""
least_squares_GD
"""


def least_squares_gd(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        # gradient w by descent update
        w = w - gamma * grad
        print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
    return w, loss


"""
least_squares_SGD
"""


def least_squares_sgd(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad = compute_gradient(y_batch, tx_batch, w)
            loss = compute_mse(y, tx, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
        print("SGD({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
    return w, loss


"""
least_squares
"""


def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss


"""
ridge regression
"""


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    weight = np.linalg.solve(a, b)
    rmse = np.sqrt(2 * compute_mse(y, tx, weight))
    return weight, rmse

"""
logistic regression
"""
def logistic_regression(y,tx,initial_w,max_iters,gamma):
   # init parameters
    threshold = 1e-8
    losses = []

    w = initial_w
    loss=0

    # start the logistic regression
    for iter in range(max_iters):
        # compute the cost
        loss=calculate_logistic_loss(y,tx,w)
        # compute the gradient
        gradient=calculate_logistic_gradient(y,tx,w)
        # update w
        w=w-gamma*gradient
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    print("loss={l}".format(l=loss))
    return w,loss

"""
regularized logistic regression
"""
def reg_logistic_regression(y, tx,lambda_,initial_w,max_iters,gamma):
    # init parameters
    threshold = 1e-8
    losses = []
    
    w = initial_w
    loss=0

    # start the logistic regression
    for iter in range(max_iters):
        
        #calculate the penalty lambda_*||w||^2
        penalty_w=np.sum(w**2)
        #calculate the loss
        loss=calculate_logistic_loss(y,tx,w)+0.5*lambda_*penalty_w/tx.shape[0]
        #calculate the gradient
        gradient=calculate_logistic_gradient(y,tx,w)+lambda_*w
        # update w
        w=w-gamma*gradient
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    print("weight={w},loss={l}".format(w=w,l=loss))
    return w,loss