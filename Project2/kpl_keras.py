from sklearn.model_selection import train_test_split
import pandas as pd
from helpers import *
from cf import CollaborativeFilteringV1

data_dir_path = "./datas/data_train.csv"
output_dir_path = './models'
records = load_dataset(data_dir_path)

user_id_train = ratings_train["User"]
item_id_train = ratings_train["Movie"]
rating_train = ratings_train["Rating"]

user_id_test = ratings_test["User"]
item_id_test = ratings_test["Movie"]
rating_test = ratings_test["Rating"]

max_user_id = records["User"].max()
max_item_id = records["Movie"].max()

config = dict()
config['max_user_id'] = max_user_id
config['max_item_id'] = max_item_id

cf = CollaborativeFilteringV1()
history = cf.fit(config=config, user_id_train=user_id_train,
                 item_id_train=item_id_train,
                 rating_train=rating_train,
                 model_dir_path=output_dir_path)

metrics = cf.evaluate(user_id_test=user_id_test,
                      item_id_test=item_id_test,
                      rating_test=rating_test)