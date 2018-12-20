from surprise import *
from surprise.model_selection import KFold, PredefinedKFold
from surprise import accuracy
from itertools import islice
from helpers import *

def surprise_algo(algo, train_path="datas/train.csv", test_path="datas/test.csv", verbose=True):
    # reader with rating scale
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5))
    
    # Specify the training and test dataset
    folds_files = [(train_path, test_path)]

    data = Dataset.load_from_folds(folds_files, reader=reader)
    pkf = PredefinedKFold()
    
    print("Start prediction...")
    for trainset, testset in pkf.split(data):
        # train and predict algorithm.
        model = algo.fit(trainset)
        predictions = algo.test(testset)
    
    pred = pd.read_csv(test_path, names = ["User", "Movie", "Rating"])
    
    print("Postprocessing predictions...")
    for index, row in pred.iterrows():
        rating = round(predictions[index].est)
        if rating > 5:
            rating = 5
        elif rating < 1:
            rating = 1
        row.Rating = rating
    
    return pred

t = Timer()

# ------rmse: 1.0665431544988566, n_factor:50, n_epoch: 200, reg_all: 0.01, lr_bu: 1e-09, lr_qi: 1e-05------
# svd++
t.start()
algo = SVDpp(n_factors=50, n_epochs=200, lr_bu=1e-9 , lr_qi= 1e-5, reg_all=0.01)
predictions_2 = surprise_algo(algo)
t.stop()
t.start()
submission_pp = create_csv_submission(predictions_2)
submission_pp.to_csv("surprise_svd_pp.csv")
t.now()
t.stop()
