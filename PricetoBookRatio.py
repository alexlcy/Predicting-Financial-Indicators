#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 22:47:11 2018

@author: AlexLO
"""

#import the necessary library
import pandas as pd
import numpy as np

#import the dataset
df = pd.read_csv('FYP_data.csv')

def CalcPricetoBookRatio(CurrentSharePrice, BVperShare):
    Output = CurrentSharePrice/BVperShare;
    return Output;

PricetoBookRatio = CalcPricetoBookRatio(df['Stock Price'], df['BV Per Share'])

#Split df to be target and feature variable
X = df.drop(['Date', 'Stock Price', 'BV Per Share'], axis = 1).values
y = PricetoBookRatio


#Split df to be training and test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(
X, y, random_state=0)


#Import Machine Learning library
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV



# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('Ridge', Ridge(alpha = 0.95))]


# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

# Specify parameters and distributions to sample from
alpha_range = np.arange(0, 1, 0.05)
param_grid = {"Ridge__alpha": alpha_range}


# Run GridSearch
grid = GridSearchCV(pipeline, cv = 5, param_grid=param_grid, scoring = "r2")
grid.fit(X_train, y_train)
print (" \nGrid-Search with R2") 
print ("Best parameters:", grid.best_params_) 
print ("Best cross-validation score (r2): {:.3f}".format(grid.best_score_)) 
print ("Test set accuracy: {:.3f}".format(grid.score(X_test, y_test)))

#Fit the pipeline to the datset
pipeline.fit(X_train, y_train)

#Predict the dataset
y_pred = pipeline.predict(X_test)

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(pipeline, X_train, y_train, cv =5)

# Print the 5-fold cross-validation scores
print (" \nLearning Model Performance")

print("The Cross Vaildation R2 Score :\n {}".format(cv_scores))
print("Average 5-Fold CV R2 Score: {}".format(np.mean(cv_scores)))

# Examine the prediction by Mean Squared Error
print("The Mean Squared Error: {}".format(mean_squared_error(y_test, y_pred)))


# Examine the prediction by Mean Absolute Error
print("The Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))


