def surprise_cv_algo(data, algo, k_fold=5, verbose=True):
    # Split into folds
    kf = KFold(n_splits=k_fold)
    rmse_ = 0
        
    for trainset, testset in kf.split(data):
        # train and test algorithm.
        model = algo.fit(trainset)
        predictions = algo.test(testset)

        # Compute and print RMSE
        rmse_ += accuracy.rmse(predictions, verbose=verbose)
    
    rmse_mean = rmse_/k_fold
    return rmse_mean
    
def surprise_svd_best_params(train_path="datas/train.csv", test_path="datas/test.csv", verbose=True, t = Timer()):
    # reader with rating scale
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5))
    # load data from df
    data = Dataset.load_from_file(train_path, reader)
    
    #svd parameters
    n_epochss = np.linspace(200, 40, 9, dtype=np.int32)
    reg_alls = np.logspace(-2, -5, 4)
    lr_bus = np.logspace(-10, -2, 9)
    lr_qis = np.logspace(-10, -2, 9)
    params = dict()
    rmses = dict()
    
    t.start()
    
    for n_epoch in n_epochss:
        params['n_epoch'] = k
        for reg_all in reg_alls:
            params['reg_all'] = reg_all
            for lr_bu in lr_bus:
                params['lr_bu'] = lr_bu
                for lr_qi in lr_qis:
                    params['lr_qi'] = lr_qi
                    algo = SVD(n_epoch = n_epoch, reg_all = reg_all, lr_bu = lr_bu, lr_qi = lr_qi)
                    rmse = surprise_cv_algo(data, algo)
                    print("------Time:{}, rmse: {}, n_epoch: {}, reg_all: {}, lr_bu: {}, lr_qi: {}------\n\n".format(t.now(), rmse, n_epoch, reg_all, lr_bu, lr_qi))
                    rmses[rmse] = params
    
    # Find the model with least RMSE
    lowest_rmse = min(rmses.keys())
    best_params = rmses[lowest_rmse]
    
    print("Best svd rmse: {}, n_epoch: {}, reg_all: {}, lr_bu: {}, lr_qi: {}".format(lowest_rmse, best_params['n_epch'], best_params['reg_all'], best_params['lr_bu'], best_params['lr_qi']))
    

surprise_svd_best_params()