# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
from implementation import *
import csv
import numpy as np



def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def get_headers(data_path):
    """
        Get the headers from the file given in parameter
    """

    f = open(data_path, 'r')
    reader = csv.DictReader(f)
    headers = reader.fieldnames

    return headers


def data_analysis_splitting(TRAIN, TEST, TRAINING_DATA, TESTING_DATA):
    """
        This long function is used for the data analysis and splitting.
        This function will split the data according as the description made in the report.

        We first split in 4 by the number of jets, then we split by the remaining NaNs in the
        first column. Then, we can write the new files.
    """

    print('START THE DATA ANALYSIS / SPLITTING FOR DATA-SETS')
    print('  Load the data. It may take a few seconds.')

    # First we load the data
    headers = get_headers(TRAIN)
    y_train, tx_train, ids_train = load_csv_data(TRAIN)
    y_test, tx_test, ids_test = load_csv_data(TEST)

    # Start the loop for the four kind of jets
    for jet in range(4):
        print("  Cleaning for Jet {0:d}".format(jet))

        # Get the new matrix with only the same jets
        # The information about the number of jets is in column 22
        tx_jet_train = tx_train[tx_train[:, 22] == jet]
        tx_jet_test = tx_test[tx_test[:, 22] == jet]

        # Cut the predictions for the same jet
        y_jet_train = y_train[tx_train[:, 22] == jet]
        y_jet_test = y_test[tx_test[:, 22] == jet]

        # Cut the ids for the same jet
        ids_jet_train = ids_train[tx_train[:, 22] == jet]
        ids_jet_test = ids_test[tx_test[:, 22] == jet]

        # Delete column 22 in Sample matrix
        tx_jet_train = np.delete(tx_jet_train, 22, 1)
        tx_jet_test = np.delete(tx_jet_test, 22, 1)

        # Delete column 24 (column 1 is ids, column 2 is pred) in headers
        headers_jet = np.delete(headers, 24)

        # Get all the columns with only NaNs
        nan_jet = np.ones(tx_jet_train.shape[1], dtype=bool)
        header_nan_jet = np.ones(tx_jet_train.shape[1] + 2, dtype=bool)
        for i in range(tx_jet_train.shape[1]):
            array = tx_jet_train[:, i]
            nbr_nan = len(array[array == -999])
            if nbr_nan == len(array):
                nan_jet[i] = False
                header_nan_jet[i + 2] = False

        # For Jet 0, there is a really big outlier in the column 3. So, we will remove it
        if jet == 0:
            to_remove = (tx_jet_train[:, 3] < 200)

        """ Start removing values """

        if jet == 0:
            tx_jet_train = tx_jet_train[to_remove, :]
            y_jet_train = y_jet_train[to_remove]
            ids_jet_train = ids_jet_train[to_remove]

            # We also remove the last column which is full of 0
            nan_jet[-1] = False
            header_nan_jet[-1] = False

        # Delete the columns in tX and headers
        tx_jet_train = tx_jet_train[:, nan_jet]
        tx_jet_test = tx_jet_test[:, nan_jet]

        headers_jet = headers_jet[header_nan_jet]

        # Get the NaNs in the mass
        nan_mass_jet_train = (tx_jet_train[:, 0] == -999)
        nan_mass_jet_test = (tx_jet_test[:, 0] == -999)
        header_nan_mass_jet = np.ones(len(headers_jet), dtype=bool)
        header_nan_mass_jet[2] = False

        # Write the files
        write_data(TRAINING_DATA[2 * jet], y_jet_train[nan_mass_jet_train], tx_jet_train[nan_mass_jet_train, :][:, 1:],
                   ids_jet_train[nan_mass_jet_train], headers_jet[header_nan_mass_jet], 'train')

        write_data(TRAINING_DATA[2 * jet + 1], y_jet_train[~nan_mass_jet_train], tx_jet_train[~nan_mass_jet_train, :],
                   ids_jet_train[~nan_mass_jet_train], headers_jet, 'train')

        write_data(TESTING_DATA[2 * jet], y_jet_test[nan_mass_jet_test], tx_jet_test[nan_mass_jet_test, :][:, 1:],
                   ids_jet_test[nan_mass_jet_test], headers_jet[header_nan_mass_jet], 'test')

        write_data(TESTING_DATA[2 * jet + 1], y_jet_test[~nan_mass_jet_test], tx_jet_test[~nan_mass_jet_test, :],
                   ids_jet_test[~nan_mass_jet_test], headers_jet, 'test')

    print("FINISHED SPLITTING THE DATA-SETS")


def write_data(output, y, tx, ids, headers, type_):
    """
        Write the data into a CSV file
    """
    with open(output, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=headers)
        writer.writeheader()
        if type_ == 'train':
            for r1, r2, r3 in zip(ids, y, tx):
                if r2 == 1:
                    pred = 's'
                elif r2 == -1:
                    pred = 'b'
                else:
                    pred = r2
                dic = {'Id': int(r1), 'Prediction': pred}
                for i in range(len(r3)):
                    dic[headers[i + 2]] = float(r3[i])
                writer.writerow(dic)
        elif type_ == 'test':
            for r1, r3 in zip(ids, tx):
                dic = {'Id': int(r1), 'Prediction': '?'}
                for i in range(len(r3)):
                    dic[headers[i + 2]] = float(r3[i])
                writer.writerow(dic)


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x
# def standardize(x, mean_x=None, std_x=None):
#     """Standardize the original data set."""
#     if mean_x is None:
#         mean_x = np.mean(x, axis=0)
#     x = x - mean_x
#     if std_x is None:
#         std_x = np.std(x, axis=0)
#     x = x / std_x
#     return x, mean_x, std_x


# handle missing values (-999)
def handle_missing(data):
    means = []
    for i in range(data.shape[1]):
        missing = (data[:,i] == -999)
        no_missing = (data[:,i] != -999)
        mean_i = np.mean(data[no_missing,i])
        data[missing,i] = mean_i
        means.append(mean_i)
    return data, means


def handle_missing_test(data, means):
    for i in range(data.shape[1]):
        missing = (data[:,i] == -999)
        data[missing,i] = means[i]
    return data


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_model_data(y, x):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def split_data(y, x, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return y_tr, y_te, x_tr, x_te


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)





def degree_of_accurancy(y, x, w):
    """ return the percentage of right prediction"""
    y_pred = np.dot(x, w)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    right = np.sum(y_pred == y)
    accurancy = float(right) / float(y.shape[0])

    return accurancy





def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse





