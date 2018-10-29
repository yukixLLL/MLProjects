# Machine Learning Project 1 - Higgs Boson

## File Description:

`run.py` : Prediction code including all the processes we use to reach the accuracy corresponding to Kaggle Leaderboard, including data pre-processing, cross-validation,etc.<br/>

`Implementation.py` : The implementation of six machine learning models which includes:
Gradient Descent, Stochastic Gradient Descent, Least Sqaure, Ridge Regression, Logistic Regression, Regularized Logistic Regression.<br/>

`proj1_helpers.py` : This file includes detailed functions related to run.py, such as load_data(), angle_processing(), etc.<br/>

`cross_validation.py` : cross-validation implementation including the function to choose the best hyperparameters.<br/>

`test_methods.py` : The initial test of six models without any data pre-processing.<br/>

## Method General Description:
(Detailed description can be seen in the report)<br/>
### Model: Ridge Regression
### Data Preprocessing :
(All detailed function can be seen in `proj1_helpers.py`)
1. Angle feature augmentation using formula in the official document
2. Drop uniform distribution variables
3. Split data into 8 groups according to PRI_jet_num and DER_mass_MMC and drop undefined values(-999) in each group
4. Apply log operation on the Right-skewed distribution feature.
### Cross-validation : Based on best accuracy

## Result:
83.365% on the Kaggle Leaderboard
