# XGBoost for python should be installed.

import pandas as pd
import os
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import sys
from sklearn.utils import shuffle

# Evaluation
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score


# FUNCTIONS

# Split the data into train and test
def train_test(data):  
    np.random.seed(100)
    data = shuffle(data)
    data.index = range(0,len(data))
    # ^ Instead of imputing - drop nans !
    data.dropna(inplace=True)
    # Do the split
    train = data.sample(frac=0.8,random_state=200)
    test = data.drop(train.index)
    return train, test

def set_params(n_classes,loss="binary:logistic",**kawrgs):
    # Defaults: 
    params ={} ;
    if loss == "binary:logistic":
        params = {"objective": loss,
              "booster" : "gbtree",
              "eta": 0.3,
              "max_depth": 10,
              "subsample": 0.9,
              "colsample_bytree": 0.7,
              "silent": 1,
              "seed": 1300 }
    if loss == "multi:softmax":
        print("Number of classes should be stated: give n_classes argument")
        params = {"objective": "multi:softmax",
              "booster" : "gbtree",
              "eta": 0.3,
              "max_depth": 10,
              "subsample": 0.9,
              "colsample_bytree": 0.7,
              "silent": 1,
              "seed": 1300,
              "num_class": n_classes}
    if loss == "reg:linear":
        params = {"objective": "reg:linear",
              "booster" : "gbtree",
              "eta": 0.3,
              "max_depth": 10,
              "subsample": 0.9,
              "colsample_bytree": 0.7,
              "silent": 1,
              "seed": 1300}
    
    return params

# Train test will be split into train and validation
# def gmb_xgb_binary(train,test, params=params,features=features,all_features=all_features,target=target):
def gmb_xgb_binary(train,test,params,features,all_features,target):
    num_boost_round = 300
    print("Train a XGBoost model")
    X_train, X_valid = train_test_split(train[all_features], test_size=0.20, random_state=10)
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)
    X_test = test[all_features]
    y_test = test[target]
    dtest = xgb.DMatrix(test[features],y_test)
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
      early_stopping_rounds=10, verbose_eval=True)
    
    if params['objective'] == "binary:logistic":
        _evaluate_binary(gbm=gbm,dvalid=dvalid,y_valid=y_valid,target=target,dtest=dtest,y_test=y_test) 
    elif params['objective'] == "multi:softmax":
        _evaluate_multi(gbm,dvalid=dvalid,y_valid=y_valid,y_test=y_test,dtest=dtest)
    if params['objective'] == "reg:linear":
        _evaluate_reg(gbm,dvalid=dvalid,y_valid=y_valid,dtest=dtest,y_test=y_test)
        
    return gbm

# def _evaluate_binary(gmb,X_valid=X_valid[features],target=target,dtest=dtest,y_test=test[target]):
def _evaluate_binary(gbm,dvalid,y_valid,target,dtest,y_test):
    # Do evaluation of the model
    print("Validating (Evaluation of validation split)")
    yhat = gbm.predict(dvalid)
    yhat = [1 if a>0.5 else 0 for a in yhat]
    pr = precision_score(y_valid, yhat)
    re = recall_score(y_valid, yhat)
    F1 = 2*(pr*re)/(pr+re)
    print("the precision is:", pr)
    print("the recall is:", re)
    print("the F-score is:", F1)
    print("Evaluation on test: make predictions on the test set")
    yhat_test = gbm.predict(dtest)
    yhat_test = [1 if a>0.5 else 0 for a in yhat_test]
    pr = precision_score(y_test, yhat_test)
    re = recall_score(y_test, yhat_test)
    F1 = 2*(pr*re)/(pr+re)
    print("the precision is:", pr)
    print("the recall is:", re)
    print("the F-score is:", F1)
    
    return 


def _evaluate_multi(gbm,dvalid,y_valid,y_test,dtest):
    # EVALUATE
    print("Validating (Evaluation of validation split)")
    yhat = gbm.predict(dvalid)
    F1 = f1_score(y_valid, yhat, average='weighted')
    re = recall_score(y_valid, yhat, average='weighted')
    pr = precision_score(y_valid, yhat, average='weighted')
    print("the precision is:", pr)
    print("the recall is:", re)
    print("the F-score is:", F1)
    print()
    print("Evaluation on test: make predictions on the test set")
    yhat_test = gbm.predict(dtest)
    F1 = f1_score(y_test, yhat_test, average='weighted')
    re = recall_score(y_test, yhat_test, average='weighted')
    pr = precision_score(y_test, yhat_test, average='weighted')
    print("the precision is:", pr)
    print("the recall is:", re)
    print("the F-score is:", F1)
    
    return


def _evaluate_reg(gbm,dvalid,y_valid,dtest,y_test):
    def rmspe(y, yhat):
        return np.sqrt(np.mean((yhat/y-1) ** 2))
    def squared_error(ys_orig,ys_line):
        return sum((ys_line - ys_orig) * (ys_line - ys_orig))
    def R2(ys_orig,ys_line):
        y_mean_line = [mean(ys_orig) for y in ys_orig]
        squared_error_regr = squared_error(ys_orig, ys_line)
        squared_error_y_mean = squared_error(ys_orig, y_mean_line)
        return 1 - (squared_error_regr/squared_error_y_mean)
    def R2_sklearn(ys_orig,ys_line):
        return r2_score(ys_orig,ys_line)
    # Do evaluation of the model
    print(" Validating (Evaluation of validation split) ")
    yhat = gbm.predict(dvalid)
    error = rmspe( y_valid.values, np.expm1(yhat) )
    print( 'RMSPE: {:.6f}'.format(error) )
    r2 = R2_sklearn( y_valid.values, yhat) 
    print( 'R2: {:.6f}\n'.format(r2) )
    print("Evaluation on test: Make predictions on the test set")
    yhat_test = gbm.predict(dtest)
    error = rmspe( y_test.values, yhat_test) 
    print( 'RMSPE: {:.6f}'.format(error) )
    r2 = R2_sklearn( y_test.values, yhat_test) 
    print( 'R2: {:.6f}'.format(r2) )
            
    return






