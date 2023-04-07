import os
import gc
import re
import sys
import json
import time
import shutil
import joblib
import random
import requests
import pickle
import arff
import warnings
warnings.filterwarnings('ignore')
from ast import literal_eval
import argparse

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import (StratifiedKFold, KFold, GroupKFold, StratifiedGroupKFold,)
import cvxpy as cp
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
import optuna

from scipy.io.arff import loadarff 

#https://lib-arts.hatenablog.com/entry/scipy_tutorial6
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint

from utils import *

def exponential_mixture(beta, pred_probs):
    """
    make prediction of exponential mixture model
    Args:
        beta:
            weight parameter (np.array)
        pred_probs:
            each models prediction probability list (list of np.array (n, n_classes))
    Returns:
        p_bar:
            prediction probability of exponential mixture model np.array (n, n_classes)
    """
    _n_models = len(pred_probs) # the number of component models
    _n_classes = len(pred_probs[0][0])
    
    # Take log of all elements -> element-wise weighted sum in beta
    p_bar = np.zeros_like(pred_probs[0]) # 指数型混合となるnd.array
    for i in range(_n_models):
        _log_pred_prob = np.log(pred_probs[i] + 1e-20)
        p_bar += beta[i] * _log_pred_prob
    
    # Take exp on each element of p_bar to form exponential mixture
    p_bar = np.exp(p_bar)

    # Create a vector by summing each row (creating a normalized term)
    normalization_term = np.sum(p_bar, axis=1)
    # https://tawara.hatenablog.com/entry/2020/05/03/223519
    # Match the size of the normalization term to p_bar so that the division is per-element
    normalization_term = np.repeat(normalization_term[None, :], _n_classes, axis=0).T 
    # normalization
    p_bar = p_bar / normalization_term

    return p_bar


# Perform supervised ensemble weight parameter tuning using the predicted probabilities of the individually trained classifiers, and 
# Output the results using the tuned parameters and the ensemble prediction probabilities for the evaluation data.
def supervised_ensemble(cfg, _labeled_train_probs, _df_ensemble_labeled_train,  _pred_eval_probs, show=False):
    """
    optimization of supervised ensemble & prediction of evaluation data
    Args:
        cfg
            Config instance
        _labeled_train_probs: list of np.array (n, n_classes)
           list of prediction probabilities for each model for supervised data used in weight parameter optimization
        _df_ensemble_labeled_train: pd.DataFrame
            supervised data used in weight parameter optimization
        _labeled_idx: list
            Index of supervised data used for weight parameter optimization
        _pred_eval_probs: list of np.array (n, n_classes)
            list of prediction probabilities for each model for evaluation data 
        show:
            boolean, True -> show sampled log
    Returns:
        _tuned_beta:
            optimized weight parameters
        _eval_exp_mix_prob:
            prediction probability of optimized exponential mixture model
    """

    _n_classes = len(_labeled_train_probs[0][0]) #the number of classes


    _labeled_df_train_target = _df_ensemble_labeled_train["target"].values # label of supervised data
    _labeled_df_train_target_ohe = np.identity(_n_classes)[_labeled_df_train_target] # one hot encoding of labels for supervised data

    # Objective function: negative log-likelihood (to minimize)
    def supervised_objective(beta):
        exp_mix_prob = exponential_mixture(beta, _labeled_train_probs) # calculate exponential mixture model
        log_likelihood = np.sum(_labeled_df_train_target_ohe * np.log(exp_mix_prob + 1e-20) , axis=None) # calculate log-likelihood

        return -1.0 * log_likelihood
    
    # Optimization
    beta_init = np.random.dirichlet(alpha= [1.0]*cfg.n_models, size = 1)[0] # initiate weight parameters
    res = minimize(fun = supervised_objective, 
                   x0 = beta_init,
                   method = "SLSQP",
                   bounds = Bounds(lb=1e-10, ub=1),#((0,1))*cfg.n_models, # The weight parameter is non-negative. if there is a completely 0, there is a part where the prediction result is nan, so we take measures to prevent this.
                   constraints = ({'type':'eq', "fun": lambda x: np.sum(x) - 1})#LinearConstraint([1.0]*cfg.n_models, 1.0, 1.0), # the sum of weight parameters is 1
                   #tol = 1e-6
                    )
    
    if show:
        print(res)

    _tuned_beta = res.x # optimized weighted parameters

    _eval_exp_mix_prob = exponential_mixture(_tuned_beta, _pred_eval_probs) # exponential mixture of prediction probabilities for evaluation data

    return _tuned_beta, _eval_exp_mix_prob

def unsupervised_ensemble(cfg, _unlabeled_train_probs, _df_ensemble_unlabeled_train, _pred_eval_probs, show=False):
    """
    optimization of unsupervised ensemble & prediction of evaluation data
    Args:
        cfg:
            Config instance
        _unlabeled_train_probs: list of np.array (n, n_classes)
           list of prediction probabilities for each model for unsupervised data used in weight parameter optimization
        _df_ensemble_unlabeled_train: pd.DataFrame
            Unsupervised data used for weight parameter optimization
        _unlabeled_idx: list
            index of unsupervised data to be used for weight parameter optimization
        _pred_eval_probs: list of np.array (n, n_classes)
            list of predicted probabilities for each model for the data for evaluation
        show:
            boolean, True -> show sampled log
    Returns:
        _tuned_beta:
            optimized weight parameters
        _eval_exp_mix_prob:
            prediction probability of optimized exponential mixture model
    """

    _n_classes = len(_unlabeled_train_probs[0][0]) # the number of classes 


    # Objective function
    def unsupervised_objective(beta):
        exp_mix_prob = exponential_mixture(beta, _unlabeled_train_probs) # calcuate exponential mixture model
        
        _objective = 0
        for i in range(cfg.n_models):
            _KL_div = np.sum( exp_mix_prob * np.log(exp_mix_prob + 1e-20) - exp_mix_prob * np.log(_unlabeled_train_probs[i] + 1e-20))
            _objective += beta[i] * _KL_div

        return -1.0 * _objective
    
    # 最適化
    beta_init = np.random.dirichlet(alpha= [1.0]*cfg.n_models, size = 1)[0] # initiate weight parameters
    res = minimize(fun = unsupervised_objective, 
                   x0 = beta_init,
                   method = "SLSQP",
                   bounds = Bounds(lb=1e-10, ub=1),#((0,1))*cfg.n_models, # weight paraemters are non-negative
                   constraints = ({'type':'eq', "fun": lambda x: np.sum(x) - 1})#LinearConstraint([1.0]*cfg.n_models, 1.0, 1.0), # the sum of weight parameters is 1
                   #tol = 1e-6
                    )
    
    if show:
        print(res)

    _tuned_beta = res.x # optimized weight parameter

    _eval_exp_mix_prob = exponential_mixture(_tuned_beta, _pred_eval_probs) # exponential mixture of prediction probability for evaluation data

    return _tuned_beta, _eval_exp_mix_prob

# proposed method
def proposed_supervised_ensemble(cfg, _labeled_train_probs, _df_ensemble_labeled_train, _pred_eval_probs, lam, show=False):
    """
    optimization of semi-supervised ensemble & prediction of evaluation data
    Args:
        cfg:
            Config instance
        _labeled_train_probs: list of np.array (n, n_classes)
           list of predicted probabilities for each model for the data used in the weight parameter optimization
        _df_ensemble_labeled_train: pd.DataFrame
            data used for weight parameter optimization
        _labeled_idx: list 
            index of supervised data used for weight parameter optimization
        _pred_eval_probs: list of np.array (n, n_classes)
            list of predicted probabilities for each model for the data for evaluation
        lam:
            hyperparameter for balance
        show:
            boolean, True -> show sampled log
    Returns:
        _tuned_beta:
            optimized weight parameters
        _eval_exp_mix_prob:
            prediction probability of optimized exponential mixture model
    """

    _n_classes = len(_labeled_train_probs[0][0]) # the number of classes
    
    # decide data ================================================================================================================
    _labeled_df_train_target = _df_ensemble_labeled_train["target"].values # labels of supervised data
    _labeled_df_train_target_ohe = np.identity(_n_classes)[_labeled_df_train_target] # one hot encoding of labels for supervised data
    
    ## Do the calculations we can in advance.
    _log_labeled_train_probs = []  # Logarithm of the individual model's probability of making a prediction on supervised data
    for i in range(cfg.n_models):
        _log_labeled_p_i = np.log(_labeled_train_probs[i] + 1e-20) # Take the logarithm of the probability value
        _log_labeled_train_probs.append(_log_labeled_p_i)

    _labeled_size = len(_df_ensemble_labeled_train)

    # 目的関数
    def proposed_supervised_objective(_beta):
        _p_beta_labeled = exponential_mixture(_beta, _labeled_train_probs) # calculate exponential mixture model 

        # 1st term
        _term1 = 0
        for i in range(cfg.n_models):
            _log_p_i = np.sum(_labeled_df_train_target_ohe * _log_labeled_train_probs[i] , axis=None)
            _term1 += _beta[i] * _log_p_i
        
        # 2nd term
        _term2 = 0
        for i in range(cfg.n_models):
            _KL_div = np.sum(_p_beta_labeled * np.log(_p_beta_labeled + 1e-20) - _p_beta_labeled * _log_labeled_train_probs[i] , axis=None)
            _term2 += _beta[i] * _KL_div
        

        _objective = lam * (1.0/_labeled_size) * _term1 +  (1 - lam) * (1.0/_labeled_size) * _term2

        return  -1.0 * _objective

        
    # 最適化
    _beta_init_random = np.random.dirichlet(alpha= [1.0]*cfg.n_models, size = 1)[0] # initiate weight parameter
    res = minimize(fun = proposed_supervised_objective, 
                x0 = _beta_init_random,
                method = "SLSQP",
                bounds = Bounds(lb=1e-10, ub=1),
                constraints = ({'type':'eq', "fun": lambda x: np.sum(x) - 1}) # the sum of weight parameter is 1
                #tol = 1e-6
                    )

    #print(res.fun)
    
    if show:
        print(res)

    _tuned_beta = res.x # optimized weight parameters

    _eval_exp_mix_prob = exponential_mixture(_tuned_beta, _pred_eval_probs) # exponential mixture of prediction probabilities for evaluation data

    return _tuned_beta, _eval_exp_mix_prob


# Function to assign fold numbers to training data
def get_stratified_kfold(train ,target_col, n_splits, seed):
    # Input:: train:training data(pd.DataFrame), target_col:target variable name, n_splits:the number of splits for CV, seed:random seed
    # return:: train: train with "fold" column containing fold numbers added
    kf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    train.reset_index(drop=True,inplace=True)
    train['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(kf.split(train, y=train[target_col])):
        train.loc[val_idx, 'fold'] = fold
    train.reset_index(drop=True, inplace=True)
    return train

def cross_validation(cfg, _labeled_train_probs, _df_ensemble_labeled_train, lam, criterion="accuracy_score"):
    
    # separate folds
    _df_ensemble_labeled_train = get_stratified_kfold(_df_ensemble_labeled_train.copy(), 'target', cfg.num_fold, cfg.seed)

    # CV
    valid_df_list = []
    valid_exp_mix_prob_list = []

    for fold in range(cfg.num_fold):
        
        # separate train/valid
        train_idx = _df_ensemble_labeled_train[_df_ensemble_labeled_train['fold']!=fold].index.tolist()
        valid_idx = _df_ensemble_labeled_train[_df_ensemble_labeled_train['fold']==fold].index.tolist()

        train_probs = [] # train part of _labeled_train_probs
        valid_probs = [] # validation part of _labeled_train_probs
        for model_idx in range(cfg.n_models):
            train_prob = _labeled_train_probs[model_idx][train_idx].copy()
            train_probs.append(train_prob)
            valid_prob = _labeled_train_probs[model_idx][valid_idx].copy() 
            valid_probs.append(valid_prob)

        train_df = _df_ensemble_labeled_train.iloc[train_idx].copy() # train part of _df_ensemble_labeled_train
        valid_df = _df_ensemble_labeled_train.iloc[valid_idx].copy() # validation part of _df_ensemble_labeled_train
        # training
        tuned_beta, valid_exp_mix_prob = proposed_supervised_ensemble(cfg, train_probs, train_df, valid_probs, lam, show=False)
    
        valid_df_list.append(valid_df)
        valid_exp_mix_prob_list.append(valid_exp_mix_prob)
    
    valid_all_df = pd.concat(valid_df_list, axis=0)
    valid_all_exp_mix_prob = np.concatenate(valid_exp_mix_prob_list, axis=0)
    
    # evaliation
    score = evaluation(cfg, valid_all_df, valid_all_exp_mix_prob)
    
    # return CV score
    return score[criterion]
    

def tuning_proposed_supervised(cfg, labeled_train_probs, df_ensemble_labeled_train, criterion, how, show_res=False):
    '''
    Tuning lambda of proposed supervised ensemble learning
    Args:
        cfg: Config Instance
        labeled_train_probs,: list of np.array (n, n_classes)
            list of prediction probabilities for each model for supervised data used in weight parameter optimization
        df_ensemble_labeled_train: pd.DataFrame
            Supervised data used for weight parameter optimization
        labeled_idx: list 
            index of supervised data used for weight parameter optimization
        criterion:
            such as accuracy_score
        how:
            "optuna", "grid" : tuning method for lam
    Returns:
        best_lam:
            best lambda value
    '''
    optuna.logging.disable_default_handler()
    
    if how == "optuna":
        class Objective:
            def __init__(self, cfg, labeled_train_probs, df_ensemble_labeled_train, criterion):
                self.cfg = cfg
                self.labeled_train_probs = labeled_train_probs.copy()
                self.df_ensemble_labeled_train = df_ensemble_labeled_train.copy()
                self.criterion = criterion
            
            def __call__(self, trial):
                
                lam = trial.suggest_uniform('lam', 0.0, 1.0)

                score = cross_validation(self.cfg, self.labeled_train_probs, self.df_ensemble_labeled_train, lam, self.criterion)

                return score

        objective = Objective(cfg, labeled_train_probs, df_ensemble_labeled_train, criterion)
        sampler = optuna.samplers.CmaEsSampler(seed=cfg.seed)
        study = optuna.create_study(sampler=sampler, direction='maximize', pruner=optuna.pruners.HyperbandPruner(reduction_factor=1))
        study.optimize(objective, n_trials=cfg.num_tuning, n_jobs=5, show_progress_bar=False)

        best_trial = study.best_trial

        if show_res:
            content = "--- Hyperparameter Tuning Result ---\n"
            content += "used optuna.\n"
            content += f"Number of finished_trials: {len(study.trials)}\n"
            content += f"Best trial: {best_trial.value}\n"
            content += f"   best_lam: {best_trial.params['lam']}\n"
            print(content)
            with open(cfg.EXP_LOG + f"/log_{cfg.MODEL_NAME}.txt", "a") as appender:
                appender.write(content + "\n")

        return best_trial.params['lam']

    elif how == "grid":

        explore_list = np.linspace(0, 1.0, cfg.num_tuning+1).tolist()
        best_score = 0
        best_lam = 0
        for lam in explore_list:
            score = cross_validation(cfg, labeled_train_probs,  df_ensemble_labeled_train, lam, criterion)
            if score > best_score:
                best_score = score
                best_lam = lam
        
        if show_res:
            content = "--- Hyperparameter Tuning Result ---\n"
            content += "used grid search.\n"
            content += f"Number of finished_trials: {cfg.num_tuning}\n"
            content += f"   best_lam: {best_lam}\n"
            print(content)
            with open(cfg.EXP_LOG + f"/log_{cfg.MODEL_NAME}.txt", "a") as appender:
                appender.write(content + "\n")

        return best_lam

    

    




