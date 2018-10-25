from implementation import *
from proj1_helpers import *

"""
def main():
    # y, tX, ids = load_csv_data("train.csv", sub_sample=False)
    # print("the shape of y: {yi}, the shape of x: {xi}".format(yi=y.shape, xi=x.shape))
    # print("Loading train set...")
    DATA_TRAIN_PATH = 'train.csv'  # TODO: download train data and supply path here
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    #y_te, tX_test, ids_test = load_csv_data("test.csv", sub_sample=False)

    ratio = 0.5
    y, y_te, tX, tX_test = split_data(y, tX, ratio, seed=1)

    # replace missing values by mean
    tX, means_missing = handle_missing(tX)
    # Get indexes of the features that have a correlation with labels bigger that the threshold
    corr_idx = correlated(y, tX, 0.01)
    # Keep only correlated features
    tX_comb = feature_combinations(tX, corr_idx)

    # Filter uncorrelated features and concatenate with combinations
    tX = tX[:, corr_idx]

    # Concatenate with combinated features
    tX = np.concatenate((tX, tX_comb), axis=1)

    # Compute polynomial basis
    tX = build_poly(tX, 1)

    # standardization
    print('Standardization')
    #tX, means_tX, std_tX = standardize(tX)

    tX_test = handle_missing_test(tX_test, means_missing)
    tX_test_comb = feature_combinations(tX_test, corr_idx)
    tX_test = tX_test[:, corr_idx]
    tX_test = np.concatenate((tX_test, tX_test_comb), axis=1)
    tX_test = build_poly(tX_test, 1)

    #tX_test, m, s = standardize(tX_test, means_tX, std_tX)
    weight_star, loss_train = least_squares(y, tX)

    loss_test = compute_mse(y_te, tX_test, weight_star)
    accurancy = degree_of_accurancy(y_te, tX_test, weight_star)
    print("Train loss: {l_tr:.4f}, test_loss: {l_te:.4f}, accurancy of Least Squares is: {accu: .4f}".format(
        l_tr=loss_train, l_te=loss_test, accu=accurancy))
    # y_pred = predict_labels(weight_star, tX_test)
    # create_csv_submission(ids_test, y_pred, "ridge_predit.csv")
    # x, mean_x, std_x = standardize(x)
    #
    # ratio = 0.5
    # y_tr, y_te, x_tr, x_te = split_data(y, x, ratio, seed=1)
    # print("the shape of y_tr: {yi}, the shape of x_tr: {xi}, the shape of y_te: {yii}, the shape of x_te: {xii}"
    #       .format(yi=y_tr.shape, xi=x_tr.shape, yii=y_te.shape, xii=x_te.shape))
    #
    # least_squares_gd_test(y_tr, y_te, x_tr, x_te)
    #
    # #least_squares_sgd_test(y_tr, y_te, x_tr, x_te)
    #
    #least_squares_test(y, y_te, tX, tX_test)
    #
    # ridge_regression_test1(y_tr, y_te, x_tr, x_te)
    #
    # #ridge_regression_test2()
"""


def main():

    y, x, ids = load_csv_data("train.csv", sub_sample=False)
    x, mean_x, std_x = standardize(x)

    ratio = 0.5
    y_tr, y_te, x_tr, x_te = split_data(y, x, ratio, seed=1)
    print("the shape of y_tr: {yi}, the shape of x_tr: {xi}, the shape of y_te: {yii}, the shape of x_te: {xii}"
          .format(yi=y_tr.shape, xi=x_tr.shape, yii=y_te.shape, xii=x_te.shape))

    least_squares_gd_test(y_tr, y_te, x_tr, x_te)

    # least_squares_sgd_test(y_tr, y_te, x_tr, x_te)

    least_squares_test(y_tr, y_te, x_tr, x_te)

    ridge_regression_test1(y_tr, y_te, x_tr, x_te)

    #ridge_regression_test2()


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


def correlated(y, tx, threshold = 0):
    """ compute the correlation between the label y and each features of tx
    return the array of arg of the nth most correlated feature
    """
    print('y shape', y.shape)
    cor = np.corrcoef(y.T, tx.T)
    y_xs_cor = cor[0, 1:]
    y_xs_threshold = y_xs_cor[np.abs(y_xs_cor) >=threshold]
    arg_sorted = np.argsort(np.abs(y_xs_cor))[::-1]
    print('All agr', arg_sorted)
    print('Arg with threshol',arg_sorted[:len(y_xs_threshold)])
    print('Corr',np.sort(np.abs(y_xs_cor))[::-1])
    return arg_sorted[:len(y_xs_threshold)]


def feature_combinations(tX, c_index):
    comb_tX = []
    for l in range(len(c_index)):
        for k in range(l+1, len(c_index)):
            m = 0
            m=tX[:, c_index[l]]*tX[:, c_index[k]]
            comb_tX.append(m)
    comb_tX = np.asarray(comb_tX)
    return comb_tX.T



def least_squares_gd_test(y_tr, y_te, x_tr, x_te):
    y_train, x_train = build_model_data(y_tr, x_tr)
    initial_w = np.zeros(x_train.shape[1])
    max_iters = 500
    gamma = 0.1

    weight_star, loss_train = least_squares_gd(y_train, x_train, initial_w, max_iters, gamma)

    y_test, x_test = build_model_data(y_te, x_te)
    loss_test = compute_mse(y_test, x_test, weight_star)
    accurancy = degree_of_accurancy(y_test, x_test, weight_star)

    print("Train loss: {l_tr:.4f}, test_loss: {l_te:.4f}, accurancy of GD is: {accu: .4f}".format(
        l_tr=loss_train, l_te=loss_test, accu=accurancy))


def least_squares_sgd_test(y_tr, y_te, x_tr, x_te):
    """此函数仍需做进一步的调试，关于stocastic gradient的问题"""
    y_train, x_train = build_model_data(y_tr, x_tr)
    initial_w = np.zeros(x_train.shape[1])
    batch_size = 1
    max_iters = 500
    gamma = 0.001
    weight_star, loss_train = least_squares_sgd(y_train, x_train, initial_w, batch_size, max_iters, gamma)

    y_test, x_test = build_model_data(y_te, x_te)
    loss_test = compute_mse(y_test, x_test, weight_star)
    accurancy = degree_of_accurancy(y_test, x_test, weight_star)

    print("Train loss: {l_tr:.4f}, test_loss: {l_te:.4f}, accurancy of SGD is: {accu: .4f}".format(
        l_tr=loss_train, l_te=loss_test, accu=accurancy))


def least_squares_test(y_tr, y_te, x_tr, x_te):
    y_train, x_train = build_model_data(y_tr, x_tr)
    weight_star, loss_train = least_squares(y_train, x_train)

    y_test, x_test = build_model_data(y_te, x_te)
    loss_test = compute_mse(y_test, x_test, weight_star)
    accurancy = degree_of_accurancy(y_test, x_test, weight_star)

    print("Train loss: {l_tr:.4f}, test_loss: {l_te:.4f}, accurancy of Least Squares is: {accu: .4f}".format(
        l_tr=loss_train, l_te=loss_test, accu=accurancy))


def ridge_regression_test1(y_tr, y_te, x_tr, x_te):
    best_degree, best_lambdas, best_rmses = best_degree_selection(y_tr, x_tr, np.arange(2, 6), 4, np.logspace(-4, 0, 40))
    print(best_degree, best_lambdas, best_rmses)
    degree = best_degree
    x_train = build_poly(x_tr, degree)
    weight_star, loss_train = ridge_regression(y_tr, x_train, best_lambdas)
    x_test = build_poly(x_te, degree)
    loss_test = compute_mse(y_te, x_test, weight_star)
    accurancy = degree_of_accurancy(y_te, x_test, weight_star)

    print("Train loss: {l_tr:.4f}, test_loss: {l_te:.4f}, accurancy of Ridge Regression is: {accu: .4f}".format(
        l_tr=loss_train, l_te=loss_test, accu=accurancy))


def ridge_regression_test2():
    TRAIN = 'train.csv'
    TEST = 'test.csv'
    TRAINING_DATA = ['train_jet_0_wout_mass.csv', 'train_jet_0_with_mass.csv',
                     'train_jet_1_wout_mass.csv', 'train_jet_1_with_mass.csv',
                     'train_jet_2_wout_mass.csv', 'train_jet_2_with_mass.csv',
                     'train_jet_3_wout_mass.csv', 'train_jet_3_with_mass.csv']

    # Name of the test data
    TESTING_DATA = ['test_jet_0_wout_mass.csv', 'test_jet_0_with_mass.csv',
                    'test_jet_1_wout_mass.csv', 'test_jet_1_with_mass.csv',
                    'test_jet_2_wout_mass.csv', 'test_jet_2_with_mass.csv',
                    'test_jet_3_wout_mass.csv', 'test_jet_3_with_mass.csv']
    data_analysis_splitting(TRAIN, TEST, TRAINING_DATA, TESTING_DATA)


if __name__ == "__main__":
    main()