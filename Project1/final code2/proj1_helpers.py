# -*- coding: utf-8 -*-


import csv
import numpy as np


"""---------------DATA PRE-PROCESSING FUNCTION---------------"""

DATA_PATH = "../data/"
USE_COLS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 21, 23, 24, 25, 26, 28, 29, 31)
LEFT_SKEWED = ['DER_mass_vis', 'DER_mass_jet_jet', 'DER_sum_pt', 'DER_pt_ratio_lep_tau',
              'PRI_tau_pt', 'PRI_lep_pt', 'PRI_met', 'PRI_met_sumet', 'PRI_jet_subleading_pt']

def load_data(train_path, test_path):
    print('Split_data_according_to_jet')
    print('Loading files...')
    y_tr, x_tr, ids_tr = load_csv_data(train_path)
    y_te, x_te, ids_te = load_csv_data(test_path)
    return y_tr, x_tr, ids_tr, y_te, x_te, ids_te


def load_headers(train_path):
    """Load all the headers from the training file and drop the unnecessary ones"""
    with open(train_path) as train_file:
        reader = csv.reader(train_file)
        headers = next(reader)

    # Only use the columns in USE_COLS
    headers = [headers[i] for i in USE_COLS]
    # drop ID and Predictions cols
    headers = headers[2:]

    return headers


def load_csv_data(data_path, sub_sample=False, cut_values=True):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    if cut_values:
        print('load_csv_data : dropping uniform distribution values')
        # drop Pri_tau_phi(17), Pri_lep_phi(20), Pri_met_phi(22), Pri_jet_leading_Phi(27), Pri_jet_subleading_phi(30)
        # because of uniform distribution
        x = np.genfromtxt(data_path, delimiter=",", skip_header=1, usecols=USE_COLS)
    else:
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


def output_to_csv(x, y, ids, headers, jet, isTrain, isMassValid):
    """
    Write data into new csv file
    """
    # Add 'Id' & 'Prediction' to headers
    headers = np.insert(headers, 0, ['Id', 'Prediction'])

    # Remove 'DER_mass_MMC' if mass is not valid
    if not isMassValid:
        headers = np.delete(headers, np.where(headers == 'DER_mass_MMC'))

    # Generate file name
    base = DATA_PATH + 'train_' if isTrain else DATA_PATH + 'test_'
    valid = '_valid_mass' if isMassValid else '_invalid_mass'
    file_name = base + 'jet_' + str(jet) + valid + '.csv'

    print("Outputing {}".format(file_name))

    with open(file_name, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        data_ = dict.fromkeys(headers)
        # Transform -1 and 1 into 's' and 'b'
        for id_, y_, x_ in zip(ids, y, x):
            data_['Id'] = int(id_)
            if y_ != -1 and y_ != 1:
                raise Exception('Prediction not -1 and 1!!!')
            data_['Prediction'] = 's' if y_ == 1 else 'b'

            for idx, x_value in enumerate(x_):
                data_[headers[idx + 2]] = float(x_value)
            writer.writerow(data_)

    return file_name, headers[2:]


def split_data_according_to_jet_and_mass(y_tr, x_tr, ids_tr, y_te, x_te, ids_te, headers):
    print("The shape of x_tr: ", x_tr.shape)
    print("The shape of x_te: ", x_te.shape)
    # Save the headers of each csv files for later use
    all_headers = dict()
    for jet in range(4):
        print("\n\nSplitting both train and test data for jet {}, remove the col PRI_jet_num".format(jet))
        # PRI_jet_num (24 -> 24 - 3 cols dropped before 24 - 2 cols (id, label) = col 19)
        col_jet = 19
        
        # TRAIN - Get all the rows having Pri_jet_num = jet for TRAINING set and delete PRI_jet_num col
        x_tr_jet = x_tr[x_tr[:, col_jet] == jet]
        x_tr_jet = np.delete(x_tr_jet, col_jet, axis=1)
        # Delete PRI_jet_num in headers
        headers_jet = np.delete(headers, col_jet)

        # Using the row found in x_tr to select the rows in y and ids
        y_tr_jet = y_tr[x_tr[:, col_jet] == jet]
        ids_tr_jet = ids_tr[x_tr[:, col_jet] == jet]

        # TEST - Get all the rows having Pri_jet_num = jet for TEST set and delete PRI_jet_num col
        x_te_jet = x_te[x_te[:, col_jet] == jet]
        x_te_jet = np.delete(x_te_jet, col_jet, axis=1)
                
        # Using the row found in x_tr to select the rows in y and ids
        y_te_jet = y_te[x_te[:, col_jet] == jet]
        ids_te_jet = ids_te[x_te[:, col_jet] == jet]
        print("The shape of x_tr_jet{j}: {shape}".format(j=jet, shape=x_tr_jet.shape))
        print("The shape of x_te_jet{j}: {shape}\n".format(j=jet, shape=x_te_jet.shape))
            
        # Remove outliers
        print("Removing outliers in Der_pt_h of jet{} for train data".format(jet))
        y_tr_jet, x_tr_jet, ids_tr_jet = remove_outlier_in_DER_pt_h(y_tr_jet, x_tr_jet, ids_tr_jet, jet)
        print("remove outlier in train set, The shape of x_tr_jet{j} after remove outliers: {shape}\n".
              format(j=jet, shape=x_tr_jet.shape))

        # Remove col PRI_jet_all_pt from x because it only contains 0 values
        if jet == 0:
            print("remove col Pri_jet_all_pt both for train and test set in jet{}".format(jet))
            x_tr_jet = np.delete(x_tr_jet, -1, axis=1)
            x_te_jet = np.delete(x_te_jet, -1, axis=1)
            print("The shape of x_tr_jet{j} after delete col Pri_jet_all_pt: {shape}".format(j=jet, shape=x_tr_jet.shape))
            print("The shape of x_te_jet{j} after delete col Pri_jet_all_pt: {shape}\n".format(j=jet, shape=x_te_jet.shape))
            headers_jet = np.delete(headers_jet, -1)
        
        # Remove all the columns with only NaN values
        print("Remove all the columns with only NaN values for both train and test data.")
        x_tr_jet, x_te_jet, headers_jet = remove_all_NAN_columns(x_tr_jet, x_te_jet, headers_jet)
        print("The shape of x_tr_jet{j} after remove nan col: {shape}".format(j=jet, shape=x_tr_jet.shape))
        print("The shape of x_te_jet{j} after remove nan col: {shape}\n".format(j=jet, shape=x_te_jet.shape))

        # Split the dataset again into valid/invalid values of DER_mass_MMC

        print("split the dataset again according valid/invalid values of DER_mass_MMC both for train and test data")
        # TRAIN
        x_tr_jet_invalid_mass, x_tr_jet_valid_mass, y_tr_jet_invalid_mass, y_tr_jet_valid_mass, ids_tr_jet_invalid_mass, ids_tr_jet_valid_mass = split_data_according_to_mass(x_tr_jet, y_tr_jet, ids_tr_jet)
        # TEST
        x_te_jet_invalid_mass, x_te_jet_valid_mass, y_te_jet_invalid_mass, y_te_jet_valid_mass, ids_te_jet_invalid_mass, ids_te_jet_valid_mass = split_data_according_to_mass(x_te_jet, y_te_jet, ids_te_jet)
        print("The shape of x_tr_jet{j}_valid: {shape}".format(j=jet, shape=x_tr_jet_valid_mass.shape))
        print("The shape of x_tr_jet{j}_invalid: {shape}".format(j=jet, shape=x_tr_jet_invalid_mass.shape))
        print("The shape of x_te_jet{j}_valid: {shape}".format(j=jet, shape=x_te_jet_valid_mass.shape))
        print("The shape of x_te_jet{j}_invalid: {shape}\n".format(j=jet, shape=x_te_jet_invalid_mass.shape))

        # Remove 'DER_mass_MMC' (col 0) if the mass is not valid
        print("Remove 'DER_mass_MMC' (col 0) if the mass is not valid")
        x_tr_jet_invalid_mass = np.delete(x_tr_jet_invalid_mass, 0, axis=1)
        x_te_jet_invalid_mass = np.delete(x_te_jet_invalid_mass, 0, axis=1)
        print("The shape of x_tr_jet{j}_valid: {shape}".format(j=jet, shape=x_tr_jet_valid_mass.shape))
        print("The shape of x_tr_jet{j}_invalid: {shape}".format(j=jet, shape=x_tr_jet_invalid_mass.shape))
        print("The shape of x_te_jet{j}_valid: {shape}".format(j=jet, shape=x_te_jet_valid_mass.shape))
        print("The shape of x_te_jet{j}_invalid: {shape}\n".format(j=jet, shape=x_te_jet_invalid_mass.shape))

        # Save into CSV
        #x, y, ids, headers, jet, isTrain, isMassValid
        # TRAIN
        file_name, header = output_to_csv(x_tr_jet_invalid_mass, y_tr_jet_invalid_mass, ids_tr_jet_invalid_mass, headers_jet, jet, True, False)
        all_headers[file_name] = header
        file_name, header = output_to_csv(x_tr_jet_valid_mass, y_tr_jet_valid_mass, ids_tr_jet_valid_mass, headers_jet, jet, True, True)
        all_headers[file_name] = header

        # TEST
        file_name, header = output_to_csv(x_te_jet_invalid_mass, y_te_jet_invalid_mass, ids_te_jet_invalid_mass, headers_jet, jet, False, False)
        all_headers[file_name] = header
        file_name, header = output_to_csv(x_te_jet_valid_mass, y_te_jet_valid_mass, ids_te_jet_valid_mass, headers_jet, jet, False, True)
        all_headers[file_name] = header

    return all_headers


def remove_outlier_in_DER_pt_h(y_tr, x_tr, ids_tr, jet):
    # Remove the outliers in DER_pt_h (col 3):
    #  JET 0: 2834.999 when the max value is 117.707 outside of outlier - threshold to 120
    #  JET 2: 1053.807 when max value is 734 outside of outlier- Threshold to 800
    OUTLIERS = [120, 999, 800, 999]
    outlier = OUTLIERS[jet]
    tr_smaller_than_outlier = (x_tr[:, 3] < outlier)
    x_tr = x_tr[tr_smaller_than_outlier]
    y_tr = y_tr[tr_smaller_than_outlier]
    ids_tr = ids_tr[tr_smaller_than_outlier]
    return y_tr, x_tr, ids_tr


def remove_all_NAN_columns(x_tr, x_te, headers_jet):
    nan_cols = []
    # Find all columns with -999
    for col_idx in range(x_tr.shape[1]):
        col = x_tr[:, col_idx]
        nb_nan_in_col = len(x_tr[col == -999])
        # A column has all NaN if len of col = nb NaN values in col
        if nb_nan_in_col == len(col):
            nan_cols.append(col_idx)

    # Remove all nan columns
    x_tr_updated = np.delete(x_tr, nan_cols, axis=1)
    x_te_updated = np.delete(x_te, nan_cols, axis=1)
    headers_jet_updated = np.delete(headers_jet, nan_cols)

    return x_tr_updated, x_te_updated, headers_jet_updated


def split_data_according_to_mass(x, y, ids):
    # Get all the rows idx with invalid mass (i.e. DER_mass_MMC = -999)
    invalid_mass_row_idx = x[:, 0] == -999
    valid_mass_row_idx = x[:, 0] > 0
    # Process for each data table
    x_invalid_mass = x[invalid_mass_row_idx]
    x_valid_mass = x[valid_mass_row_idx]
    y_invalid_mass = y[invalid_mass_row_idx]
    y_valid_mass = y[valid_mass_row_idx]
    ids_invalid_mass = ids[invalid_mass_row_idx]
    ids_valid_mass = ids[valid_mass_row_idx]

    return x_invalid_mass, x_valid_mass, y_invalid_mass, y_valid_mass, ids_invalid_mass, ids_valid_mass


def load_processed_data(file_names):
    """Load all Train/Test processed data"""
    ys = []
    xs = []
    ids = []

    for f in file_names:
        y, x, id_ = load_csv_data(f, cut_values=False)
        ys.append(y)
        xs.append(x)
        ids.append(id_)

    return ys, xs, ids


# Read the real files
def generate_processed_filenames(isTrain):
    file_names = []
    isMassValids = [True, False]
    jets = range(4)

    for isMassValid in isMassValids:
        for jet in jets:
            # Generate file name
            base = DATA_PATH + 'train_' if isTrain else DATA_PATH + 'test_'
            valid = '_valid_mass' if isMassValid else '_invalid_mass'
            file_name = base + 'jet_' + str(jet) + valid + '.csv'
            file_names.append(file_name)

    return file_names


def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def log_left_skewed(all_headers, file_names, xs_train):
    """Log all columns that are left_skewed"""
    for f, x in zip(file_names, xs_train):
        header = all_headers[f]
        col_mask = np.isin(header, LEFT_SKEWED)
        _ = np.log(x, out=x, where=col_mask)


def correlated(y, x, threshold=0):
    """
    compute the correlation between with each feature
    return the index of the feature with correlation bigger than threshold
    """
    print('y shape', y.shape)
    cor = np.corrcoef(y.T, x.T)
    corr_degree = cor[0, 1:]
    print("All correlation\n", corr_degree)
    select_corr = corr_degree[np.abs(corr_degree) >= threshold]
    print("select_correlation\n", select_corr)
    sorted_index = np.argsort(np.abs(corr_degree))[::-1]
    print("sorted_index\n", sorted_index[:len(select_corr)])
    return sorted_index[:len(select_corr)]


def feature_cross_prod(x, corr_ind):
    prod_x = []
    for i in range(len(corr_ind)):
        for j in range(i+1, len(corr_ind)):
            m = x[:, corr_ind[i]] * x[:, corr_ind[j]]
            prod_x.append(m)
    prod_x = np.asarray(prod_x)
    return prod_x.T


"""---------------HELPER FUNCTIONS FOR LEAST SQUARES GD---------------"""


def build_model_data(y, x):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad


"""---------------HELPER FUNCTIONS FOR LEAST SQUARES SGD---------------"""


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


"""---------------HELPER FUNCTIONS FOR RIDGE REGRESSION---------------"""


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

"""---------------HELPER FUNCTIONS FOR LOGSITC REGRESSION---------------"""

def data_preprocess_logsitic(y_tr,x_tr,x_te=None,y_te=None,state='pred'):
    """
    pre-process the data for the logistic regression.
    because we use logistic function sigmoid() to train data,so the label of y={0,1}
      
    """
    if(state=='test'):
        y_tr,x_tr=build_model_data(y_tr,x_tr)
        y_te,x_te=build_model_data(y_te,x_te)
        y_te[np.where(y_te == -1)]=0
    
    y_tr[np.where(y_tr == -1)] = 0
    
    
    return y_tr,x_tr,y_te,x_te

def data_postprocess_logistic(y_pred):
    """change 0 label into 1 to fit the prediction"""
    y_pred[np.where(y_pred == 0)]=-1
    
    return y_pred

def sigmoid(t):
    """apply sigmoid function on t."""
    return (1.0/(np.exp(-t)+1))


def calculate_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    
    loss = np.log(1 + np.exp(np.dot(tx,w))) - (y * (np.dot(tx,w)))
    return (np.sum(loss)) 


def calculate_logistic_gradient(y, tx, w):
    """compute the logistic gradient of loss."""
    gradient=tx.T.dot(sigmoid(tx.dot(w))-y)
    return gradient

"""---------------HELPER FUNCTIONS FOR REGUALRIZED LOGSITC REGRESSION---------------"""

def reg_logistic_grad_loss(y, tx, w, lambda_):
    """return the loss, gradient"""
    penalty_w=np.sum(w**2)
    loss=calculate_logistic_loss(y,tx,w)+0.5*lambda_*penalty_w
    gradient=calculate_logistic_gradient(y,tx,w)+lambda_*w
    return loss,gradient

"""---------------COMMON HELPER FUNCTIONS FOR 6 METHODS---------------"""


def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


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


# def load_csv_data(data_path, sub_sample=False):
#     """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
#     y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
#     x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
#     ids = x[:, 0].astype(np.int)
#     input_data = x[:, 2:]
#
#     # convert class labels from strings to binary (-1,1)
#     yb = np.ones(len(y))
#     yb[np.where(y == 'b')] = -1
#
#     # sub-sample
#     if sub_sample:
#         yb = yb[::50]
#         input_data = input_data[::50]
#         ids = ids[::50]
#
#     return yb, input_data, ids


def degree_of_accuracy(y, x, w):
    """ return the percentage of right prediction"""
    y_pred = np.dot(x, w)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    right = np.sum(y_pred == y)
    accuracy = float(right) / float(y.shape[0])

    return accuracy

def degree_of_accuracy_logitstic(y, x, w):
    """ return the percentage of right prediction of logistic regression"""
    y_pred = sigmoid(x.dot(w))
    
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    right = np.sum(y_pred == y)
    accuracy = float(right) / float(y.shape[0])

    return accuracy

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred

def predict_labels_logistic(weights, data):
    """Generates class predictions given weights, and a test data matrix using logistic regression"""
    y_pred = sigmoid(data.dot(weights))
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    # change {0,1} label into {-1,1}
    y_pred=data_postprocess_logistic(y_pred)
    return y_pred

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    print("Creating csv submission file {}".format(name))
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})