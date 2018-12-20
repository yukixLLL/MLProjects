from stack import *
from constants import *

# Please make sure that you have all the necessary files in predict_save and train_predictions before running this script!
""""
Best weights: 
 {'global_mean': 0.04349150446270624, 'global_median': 0.04314870861405616, 'user_mean': -0.05161249684542282, 'user_median': -0.004501407103595544, 'movie_mean': 0.008970185333798892, 'movie_median': 0.05045813082930219, 'movie_mean_user_std': 0.14102650168186012, 'movie_median_user_std': 0.1502732317267438, 'movie_mean_user_habit_std': -0.11833002139543332, 'movie_median_user_habit_std': -0.11828993985346828, 'movie_mean_user_habit': -0.08613414081218784, 'movie_median_user_habit': -0.04464648547851165, 'surprise_svd': 0.010565883176314931, 'surprise_knn': 0.029843085842787595, 'surprise_svd_pp': 0.008737405974155283, 'spotlight': 0.28877544038671826, 'als': 0.21455606179663697, 'pyfm': 0.28183782000328117, 'mfrr': 0.16722629378527226}
"""
# Load all models
models = load_models()
# Read ground_truth (1/2 of given train.csv)
ground_truth = pd.read_csv(folder + "ground_truth.csv", index_col=0)
# Trying to optimize the models predictions by minimizing the error
res, predictions = optimize(models, ground_truth, folder)
best_dict, rmse = get_best_weights(res, predictions, models, ground_truth)
# Predict using best weights
pred = predict(best_dict, models)
# Create submission
submission = create_csv_submission(pred)
submission.to_csv("../submission.csv")