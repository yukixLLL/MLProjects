# -*- coding: utf-8 -*-
"""
a function used to compute the loss.
"""

import numpy as np


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w, mse=True):
    """Calculate the loss.
    """
    e = y - tx.dot(w)
    if mse:
        return calculate_mse(e)
    else:
        return calculate_mae(e)
