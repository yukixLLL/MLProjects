from implementation import *
from proj1_helpers import *
from cross_validation import *
from data_preprocessing import *

# load csv file
y, x, ids = load_csv_data("./data/train.csv", sub_sample=True)


def make_predictions(weight, degree):
    y_test, x_test, ids_test = load_csv_data("./data/test.csv", sub_sample=False)
    x_test, mean_x_test, std_x_test = standardize(x_test)
    tx_poly = build_poly(x_test, degree)
    y_pred = predict_labels(weight, tx_poly)
    create_csv_submission(ids_test, y_pred, "sgd_predit_35.csv")


if __name__ == "__main__":
    # put all -999 to 0

    #x = replace_999_by_value(x, 0)
    # standardize data
    x_test, mean_x_test, std_x_test = standardize(x)
    # build the matrix x (parameters) and vector y (true labels)
    #y, tx = build_model_data(x_test, y)
    tx = x_test

    # build w using ridge regression
    k_fold = 35
    degrees = np.arange(1, 6)
    lambdas = np.logspace(-4, 0, 60)
    #lambdas = np.logspace(-4, 0, 5)
    seed = 12

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    # for each degree, we compute the best lambdas and the associated rmse
    best_lambdas = []
    best_rmses = []
    best_w = []
    # vary degree
    for degree in degrees:
        # cross validation
        rmse_te = []
        w_te = []
        for lambda_ in lambdas:
            rmse_te_tmp = []
            w_te_temp = []
            for k in range(k_fold):
                #args = [lambda_]
                #_, loss_te, w = cross_validation(args, y=y, x=tx, k_indices=k_indices, k=k, degree=degree, model=ridge_regression)
                _, loss_te, w = cross_validation(y, tx, k_indices, k, lambda_, degree)
                rmse_te_tmp.append(loss_te)
                w_te_temp.append(w)

            rmse_te.append(np.mean(rmse_te_tmp))
            # mean of all the ws
            w_te.append(np.mean(w_te_temp, axis=0))

        ind_lambda_opt = np.argmin(rmse_te)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_rmses.append(rmse_te[ind_lambda_opt])
        best_w.append(w_te[ind_lambda_opt])
        print("For degree {deg}, te_loss={te_loss}".format(deg=degree, te_loss=rmse_te[ind_lambda_opt]))

    # find the one having the least test error
    ind_best_degree = np.argmin(best_rmses)

    best_weights = best_w[ind_best_degree]
    print("\n\n Best weights shape:", best_weights.shape)
    best_degree = degrees[ind_best_degree]

    make_predictions(best_weights, best_degree)








