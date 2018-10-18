# -*- coding: utf-8 -*-
from implementation import *
#from implementations_yw import *

y, x, ids = load_csv_data("./data/train.csv", sub_sample=False)
x, mean_x, std_x = standardize(x)
y, tx = build_model_data(x, y)

"""
initial_w = np.zeros(tx.shape[1])
max_iters = 1000
gamma = 0.05
loss, weight = least_squares_GD(y, tx, initial_w, max_iters, gamma)

y_test, x_test, ids_test = load_csv_data("test.csv", sub_sample=False)
x_test, mean_x_test, std_x_test = standardize(x_test)
y_test, tx_test = build_model_data(x_test, y_test)
y_pred = predict_labels(weight, tx_test)
create_csv_submission(ids_test, y_pred, "gd_predit.csv")
"""

if __name__ == "__main__":
    initial_w = np.zeros(tx.shape[1])
    batch_size = 5
    max_iters = 50
    gamma = 0.1
    loss, weight = least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma)

    y_test, x_test, ids_test = load_csv_data("./data/test.csv", sub_sample=False)
    x_test, mean_x_test, std_x_test = standardize(x_test)
    y_test, tx_test = build_model_data(x_test, y_test)
    y_pred = predict_labels(weight, tx_test)
    create_csv_submission(ids_test, y_pred, "sgd_predit.csv")