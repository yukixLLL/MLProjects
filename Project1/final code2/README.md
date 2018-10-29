# Machine Learning Project 1 - Higgs Boson

## File Description:

`run.py` : Script used to produce the csv file having the best prediction accuracy. Includes all the processes for data preprocessing, cross-validation,etc.<br/>

`Implementation.py` : The implementation of six machine learning models including
Gradient Descent, Stochastic Gradient Descent, Least Sqaure, Ridge Regression, Logistic Regression and Regularized Logistic Regression.<br/>

`proj1_helpers.py` : This file includes helper functions used by run.py, such as load_data(), angle_processing(), etc.<br/>

`cross_validation.py` : cross-validation implementation including the customized function to choose the best hyperparameters.<br/>

`test_methods.py` : The initial test of six models without any data preprocessing.<br/>

## To Run the scripts

All the scripts are written with Python 3.6. Here are the instructions to run the scripts.

1. Create a folder  `data` in the current folder.
2. Place `train.csv` and `test.csv` in the data folder. If you want to place these data sheets elsewhere, you need to change the variables `TRAIN_PATH` and `TEST_PATH` in `projet1_helpers.py`, `test_methods.py`, `run.py`.
3. To test our 6 models run test_methods.py with `python test_methods.py`
4. To run the algorithm that produced our best accuracy, run  `python run.py`. This may take up to 12 hours depending on the quality of the machine.
5. If you want to import the function from `run.py` in a python script, you need to import the function `main()`.
6. The output `swimming.csv` will be produced in this current folder.

## Method General Description:
(Detailed description can be seen in the report)<br/>
### Model: Ridge Regression
### Data Preprocessing :
(All detailed function can be seen in `proj1_helpers.py`)
1. Angle feature augmentation using formula in the official document
2. Drop uniform distribution variables
3. Split data into 8 groups according to PRI_jet_num and DER_mass_MMC and drop undefined values(-999) in each group
4. Apply log operation on the Right-skewed distribution feature.
### Cross-validation : 
Used to estimate the best lambda and the best degree.
1. K-fold: 10
2. 100 lambas (`logspace(-20, -3, 100)`)
3. Error estimator: prediction accuracy (we maximize accuracy rather than minimize RMSE)
4. Degrees range for polynomial basis: 4 to 12

## Result:
83.365% on the Kaggle Leaderboard
