from sklearn.model_selection import train_test_split
import pandas as pd
from cf import CollaborativeFilteringV1
import numpy as np
from helpers import *

data_dir_path = "./datas/sampleSubmission.csv"
trained_model_dir_path = './models'

all_ratings = load_dataset(data_dir_path)

user_id_test = all_ratings['User']
item_id_test = all_ratings['Movie']
rating_test = all_ratings['Rating']

cf = CollaborativeFilteringV1()
cf.load_model(CollaborativeFilteringV1.get_config_file_path(trained_model_dir_path),
              CollaborativeFilteringV1.get_weight_file_path(trained_model_dir_path))
predicted_ratings = cf.predict(user_id_test, item_id_test)
print(predicted_ratings)

predictions = all_ratings.copy()
predictions['Rating'] = predicted_ratings

submission = create_csv_submission(predictions)
submission.to_csv("./datas/kerasV1.csv")