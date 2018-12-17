import pandas as pd
import numpy as np
from helpers import *
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
    
train_dataset = "./datas/data_train.csv"
test_dataset = "./datas/sampleSubmission.csv"
train_df = load_dataset(train_dataset)
test_df = load_dataset(test_dataset)

t = Timer()
t.start()

(train_data, y_train, train_users, train_items) = toPyFMData(train_df)
(test_data, y_test, test_users, test_items) = toPyFMData(test_df)
v = DictVectorizer()
X_train = v.fit_transform(train_data)
X_test = v.transform(test_data)

algo = pylibfm.FM(num_factors=42, num_iter=200, verbose=True, task="regression", initial_learning_rate=0.01, learning_rate_schedule="optimal")

algo.fit(X_train,y_train)
preds = algo.predict(X_test)
predictions = test_df.copy()
predictions['Rating'] = preds

submission = create_csv_submission(predictions)
submission.to_csv("./datas/pyfm.csv")

t.stop(verbose=True)