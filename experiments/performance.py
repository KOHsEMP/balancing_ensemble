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

import sys
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
import cvxpy as cp
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint

from scipy.io.arff import loadarff 

sys.path.append("./")
from utils import * 
from methods import *
from plots import *

parser = argparse.ArgumentParser(description='Experiment to compare performances of proposed and existing methods.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='bank', 
                    choices=['mnist','fashion-mnist','kuzushiji-mnist','bank','sensorless', 'volkert'])
parser.add_argument('--n_components', type=int, default=5, help="the number of component models")
parser.add_argument('--each_model_data', type=int, default=200, help="the number of data for each component models")
parser.add_argument('--ensemble_labeled_size', required=True, nargs="*", type=int, help="a list of ensemble labeled data size")
parser.add_argument('--eval_data', type=int, default=40000, help="evaluation data size")
parser.add_argument('--num_fold', type=int, default=5, help='the number of folds in CV')
parser.add_argument('--algorithms', required=True, nargs="*", type=str, help="a list of algorithms to create component models", 
                    choices=["logistic", "svm", "knn", "decision_tree", "QDA", "gaussian_process", "gaussian_naive_bayes", "adaboost", "random_forest"])
parser.add_argument('--num_tuning', type=int, default=100)
parser.add_argument('--seed', type=int, default=42, help="initial seed")
parser.add_argument('--n_experiments', type=int, default=5, help="the number of experiments")
parser.add_argument('--labeled_classes', required=True, nargs="*", type=int, help="classes region of labeled data")
parser.add_argument('--tuning', required=True, type=str, choices=['optuna', 'grid'])

args = parser.parse_args()

out_str = str(args)
print(out_str)

assert args.n_components == len(args.algorithms)

class Config:
    
    NAME = "exp-performance"
    MODEL_NAME = "various"
    DATA_NAME = ""

    n_models = 5 # the number of component models
    each_model_data_size = 200 # data size to train one component model
    ensemble_labeled_size = 20 
    ensemble_labeled_size_list = [50, 200, 500]
    eval_data_size = 40000 # data size of test data

    algorithm_list = ["logistic", "logistic", "svm", "svm", "knn"]
    #"logistic", "svm", "knn", "decision_tree", "QDA", "gaussian_process", "gaussian_naive_bayes", "adaboost", "random_forest"

    num_fold = 5
    num_tuning = 100

    MAIN_PATH = "../"
    DATA_PATH = "../data"
    
    seed = 42 # random seed
    num_experiment = 5 # the number of experiments

    DEBUG = False 

# Random Sampling
def sampling_data(cfg, df_data, show=False):
    """
    random sampling data 
    Args:
        cfg:
            Config instance
        df_data: pd.DataFrame
            all data 
        show: boolean
            True -> show sampled log
    Returns:
        _dataset: list
            list of datasets to train component models
        _df_ensemble_train: pd.DataFrame 
            data to optimize weight parameters of ensemble learning
        _df_eval:
            test data
    """

    set_seed(cfg.seed) # fix random seed
    # sampling data to train component models
    sampled_dataset = df_data.sample(n=cfg.each_model_data_size * cfg.n_models, replace=False)

    _datasets = [] # list of datasets to train component models
    for i in range(cfg.n_models):
        # separate DataFrames for each component model
        df_tmp = sampled_dataset.iloc[i*cfg.each_model_data_size : (i+1)*cfg.each_model_data_size] # From top to bottom
        df_tmp.reset_index(drop=True, inplace=True)
        _datasets.append(df_tmp) 

    # exclude sampling data from df_data
    ## _df_ensemble_train is rest of the data
    _df_ensemble_train = df_data.copy()
    _df_ensemble_train = _df_ensemble_train.merge(sampled_dataset, indicator=True, how="outer").query('_merge=="left_only"').drop('_merge', 1)
    _df_ensemble_train.reset_index(drop=True, inplace=True)

    # sampling test data
    _df_eval = _df_ensemble_train.sample(n=cfg.eval_data_size, replace=False)
    _df_ensemble_train = _df_ensemble_train.merge(_df_eval, indicator=True, how="outer").query('_merge=="left_only"').drop('_merge', 1)
    
    _df_ensemble_train.reset_index(drop=True, inplace=True) # for weight parameters optimization
    _df_eval.reset_index(drop=True, inplace=True) # test data

    # show sampling results 
    if show:
        print(f"original data shape: {df_data.shape}")
        print(f"sampled data shape: {sampled_dataset.shape}")
        print(f"df_ensemble_train shape: {_df_ensemble_train.shape}")
        print(f"df_eval shape: {_df_eval.shape}")
        for i in range(cfg.n_models):
            print(f"n: {i}, data_shape:{_datasets[i].shape}")
            print(f"    target ratio 0: {_datasets[i].target.value_counts()[0]/cfg.each_model_data_size}")
            print(f"{_datasets[i].target.value_counts()}")
    
    return _datasets, _df_ensemble_train, _df_eval

def make_models(cfg, datasets, df_ensemble_train_data, df_eval_data):
    """
    make each models and predict ensemble_train_data & eval_data
    Args:
        cfg:
            Config instance
        dataset: pd.DataFrame
            list of datasets to train component models
        df_ensemble_train_data: pd.DataFrame
            data to optimize weight parameters of ensemble learning 
        df_eval_data: pd.DataFrame
            test data
    Return:
        _pred_ensemble_train_labels: list of ndarray
            list of prediction labels of each component model for ensemble_train
        _pred_ensemble_train_probs: list of ndarray, ndarray is (n, n_classes)
            list of prediction probabilities of each component model for ensemble_train
        _pred_eval_labels: list of ndarray
            list of prediction labels of each component model for test data
        _pred_eval_probs: list of ndarray, ndarray is (n, n_classes)
            list of prediction probabilities of each component model for test data
    """

    _pred_ensemble_train_labels = [] # prediction labels of ensemble_train_data (each models output list)
    _pred_ensemble_train_probs = [] # prediction probabilitys of ensemble_train_data (each models output list)
    _pred_eval_labels = [] # prediction labels of eval_data (each models output list)
    _pred_eval_probs = [] # prediction probability of eval_data (each models output list)

    input_cols = df_ensemble_train_data.columns[df_ensemble_train_data.columns != "target"]
    for n, df_train in enumerate(datasets):
        # df_train: training data to be used in the nth model
        _df_ensemble_train = df_ensemble_train_data.copy()
        _df_eval = df_eval_data.copy()
        start_time = time.time()

        # Training and Inferring
        if cfg.algorithm_list[n] == "logistic":
            clf = LogisticRegression(random_state = cfg.seed, solver="lbfgs", multi_class='auto', n_jobs=1)
        elif cfg.algorithm_list[n] == "svm":
            clf = SVC(kernel='rbf', gamma='auto', random_state=cfg.seed, probability=True)
        elif cfg.algorithm_list[n] == "knn":
            clf = KNeighborsClassifier(n_neighbors=_df_ensemble_train["target"].nunique(), n_jobs=1)
        elif cfg.algorithm_list[n] == "decision_tree":
            clf = DecisionTreeClassifier(random_state=cfg.seed)
        elif cfg.algorithm_list[n] == "gaussian_naive_bayes":
            clf = GaussianNB()
        elif cfg.algorithm_list[n] == "QDA":
            clf = QuadraticDiscriminantAnalysis()
        elif cfg.algorithm_list[n] == "gaussian_process":
            _kernel = 1.0 * RBF(1.0)
            clf = GaussianProcessClassifier(kernel=_kernel, random_state=cfg.seed, n_jobs=1)
        elif cfg.algorithm_list[n] == "adaboost":
            clf = AdaBoostClassifier(random_state=cfg.seed,
                                     n_estimators=10,
                                     base_estimator= DecisionTreeClassifier(max_depth=10))
        elif cfg.algorithm_list[n] == "random_forest":
            clf = RandomForestClassifier(random_state=cfg.seed, n_jobs=1)
        else:
            print("model name error\n")
            exit(1)
        clf.fit(df_train[input_cols], df_train["target"])
        _pred_ensemble_train_label = clf.predict(_df_ensemble_train[input_cols])
        _pred_ensemble_train_prob  = clf.predict_proba(_df_ensemble_train[input_cols])
        _pred_test_label = clf.predict(_df_eval[input_cols]) # prediction label [0 0 0 1 1 1 ...]
        _pred_test_prob = clf.predict_proba(_df_eval[input_cols]) # prediction probability [[0 1], [0 1], ...]

        _pred_ensemble_train_labels.append(_pred_ensemble_train_label)
        _pred_ensemble_train_probs.append(_pred_ensemble_train_prob)
        _pred_eval_labels.append(_pred_test_label)
        _pred_eval_probs.append(_pred_test_prob)

        print(f"model:{n+1} done. --- time:{time.time() - start_time:.2f}[s]")
    
    return _pred_ensemble_train_labels, _pred_ensemble_train_probs, _pred_eval_labels, _pred_eval_probs

# Execution

# set config
cfg = Config()

cfg.DATA_NAME = args.dataset
cfg.dataset = args.dataset
cfg.NAME = cfg.NAME + "-" + args.dataset + "-various"
cfg.n_models = args.n_components
cfg.each_model_data_size = args.each_model_data
cfg.ensemble_labeled_size_list = args.ensemble_labeled_size

if len(cfg.ensemble_labeled_size_list) == 1:
    cfg.ensemble_labeled_size = args.ensemble_labeled_size[0]

cfg.eval_data_size = args.eval_data
cfg.algorithm_list = args.algorithms
cfg.seed = args.seed
cfg.orig_seed = args.seed
cfg.num_experiment = args.n_experiments
cfg.labeled_classes = args.labeled_classes
cfg.num_fold = args.num_fold
cfg.num_tuning = args.num_tuning
cfg.tuning = args.tuning

cfg = setup(cfg)

# make log name
_algorithm_list_str = ""
for i, _algorithm in enumerate(cfg.algorithm_list):
    _algorithm_list_str += f"{_algorithm}"
    if i != len(cfg.algorithm_list) - 1:
        _algorithm_list_str += "+"

log_name = _algorithm_list_str
log_name += "_each_" + str(cfg.each_model_data_size)
log_name += "_label_" + str(cfg.ensemble_labeled_size)
log_name += "_eval_" + str(cfg.eval_data_size)
log_name += "_seed_" + str(cfg.seed)
log_name += "_tri_" + str(cfg.num_experiment)
log_name += "_cv_" + str(cfg.num_fold)
log_name += "_turn_" + str(cfg.num_tuning) + "_" + str(cfg.tuning)

def experiment_info(cfg):
    exp_info = f"dataset:{cfg.dataset} \n"
    exp_info += f"n_components:{cfg.n_models}, n_experiments:{cfg.num_experiment}, seed:{cfg.seed}\n"
    exp_info += f"algorithms:{cfg.algorithm_list}\n"
    exp_info += f"each_model_data:{cfg.each_model_data_size}, eval_data:{cfg.eval_data_size}\n"
    exp_info += f"ensemble_labeled_size:{cfg.ensemble_labeled_size_list}\n"
    #exp_info += f"ensemble_unlabeled_size:{cfg.ensemble_unlabeled_size_list}\n"
    exp_info += f"labeled_classes:{cfg.labeled_classes}\n"
    #exp_info += f"unlabeled_classes:{cfg.unlabeled_classes}\n"
    exp_info += f'num_fold:{cfg.num_fold}\n'
    exp_info += f'num_tuning:{cfg.num_tuning}\n'
    exp_info += f'tuning:{cfg.tuning}'

    print(exp_info)

    return exp_info

experiment_info(cfg)

set_seed(cfg.seed) # fix random seed

# load data
print("loading data...")

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

if cfg.DATA_NAME == "mnist":
    df_data = pd.DataFrame(loadarff(os.path.join(cfg.DATA_PATH, cfg.DATA_NAME, "mnist_784.arff"))[0])
    df_data = df_data.rename(columns={"class":"target"})
    df_data = df_data.drop_duplicates().reset_index(drop=True) # delete dupulicated data
    df_data["target"] = df_data["target"].map(lambda x: int(x.decode())) # convert target to int type since target is a byte string

    cfg.num_cols = list(set(df_data.columns) - set(["target"]))

    ## normalization
    scaler = MinMaxScaler()
    df_data[cfg.num_cols] = scaler.fit_transform(df_data[cfg.num_cols])
elif cfg.DATA_NAME == "fashion-mnist":
    df_data = pd.DataFrame(loadarff(os.path.join(cfg.DATA_PATH, cfg.DATA_NAME, "fashion-mnist.arff"))[0])
    df_data = df_data.rename(columns={"class":"target"})
    df_data = df_data.drop_duplicates().reset_index(drop=True) # delete dupulicated data
    df_data["target"] = df_data["target"].map(lambda x: int(x.decode())) # convert target to int type since target is a byte string

    cfg.num_cols = list(set(df_data.columns) - set(["target"]))

    ## normalization
    scaler = MinMaxScaler()
    df_data[cfg.num_cols] = scaler.fit_transform(df_data[cfg.num_cols])

elif cfg.DATA_NAME == "kuzushiji-mnist":
    df_data = pd.DataFrame(loadarff(os.path.join(cfg.DATA_PATH, cfg.DATA_NAME, "kuzushiji-mnist.arff"))[0])
    df_data = df_data.rename(columns={"class":"target"})
    df_data = df_data.drop_duplicates().reset_index(drop=True) # delete dupulicated data
    df_data["target"] = df_data["target"].map(lambda x: int(x.decode())) # convert target to int type since target is a byte string

    cfg.num_cols = list(set(df_data.columns) - set(["target"]))

    ## normalization
    scaler = MinMaxScaler()
    df_data[cfg.num_cols] = scaler.fit_transform(df_data[cfg.num_cols])

elif cfg.DATA_NAME == "bank":
    df_data = pd.read_csv(os.path.join(cfg.DATA_PATH, cfg.DATA_NAME, "bank-full.csv"), sep=";")
    df_data = df_data.rename(columns={"y":"target"})
    df_data = df_data.drop_duplicates().reset_index(drop=True) # delete dupulicated data

    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    # one hot encoding
    ohe_cols = ["job","marital","education","default","housing", "loan", "contact","poutcome"]
    ohe = OneHotEncoder(sparse=False, categories='auto')
    ohe.fit(df_data[ohe_cols])

    tmp_columns = []
    for i, col in enumerate(ohe_cols):
        tmp_columns += [f'{col}_{v}' for v in ohe.categories_[i]]

    df_tmp = pd.DataFrame(ohe.transform(df_data[ohe_cols]), columns=tmp_columns)
    df_data = pd.concat([df_data.drop(ohe_cols, axis=1), df_tmp], axis=1)

    # label encoding
    le_cols = ["month", "target"]
    for le_col in le_cols:
        le = LabelEncoder()
        df_data[le_col] = le.fit_transform(df_data[le_col])

    cfg.num_cols = ["balance", "day", "duration", "campaign", "pdays", "previous"]
    cfg.n_data = len(df_data)

    ## standardization
    scaler = StandardScaler()
    df_data[cfg.num_cols] = scaler.fit_transform(df_data[cfg.num_cols])

elif cfg.DATA_NAME == "sensorless":
    df_data = pd.read_csv(os.path.join(cfg.DATA_PATH, cfg.DATA_NAME, "Sensorless_drive_diagnosis.txt"), sep=" ", header=None)
    df_data = df_data.rename(columns={48:"target"})
    df_data["target"] = df_data["target"] - 1 # 0, 1, 2, ...
    df_data = df_data.drop_duplicates().reset_index(drop=True)

    cfg.num_cols = list(set(df_data.columns) - set(["target"]))
    cfg.n_data = len(df_data)
    ## normalization
    scaler = MinMaxScaler()
    df_data[cfg.num_cols] = scaler.fit_transform(df_data[cfg.num_cols])

elif cfg.DATA_NAME == "volkert":
    df_data = pd.DataFrame(loadarff(os.path.join(cfg.DATA_PATH, cfg.DATA_NAME, "volkert.arff"))[0])
    df_data = df_data.rename(columns={"class":"target"})
    df_data = df_data.drop_duplicates().reset_index(drop=True) # delete dupulicated data
    df_data["target"] = df_data["target"].map(lambda x: int(x.decode())) # convert target to int type since target is a byte string

    cfg.num_cols = list(set(df_data.columns) - set(["target"]))

    ## standardization
    scaler = StandardScaler()
    df_data[cfg.num_cols] = scaler.fit_transform(df_data[cfg.num_cols])



cfg.n_data = len(df_data)
#cfg.seed = 42
experiment_results_supervised = []
experiment_results_unsupervised = []
experiment_results_proposed_supervised = []
experiment_results_component = []
experiment_results_per_component = {}
experiment_best_lams = []

for i in range(cfg.n_models):
    experiment_results_per_component[f"model-{i}"] = []

for exp in range(cfg.num_experiment):
    print(f"experiment {exp+1}: seed:{cfg.seed}")

    # make data and component models　===============================================================
    set_seed(cfg.seed) # fix random seed

    ## random sampling
    datasets, df_ensemble_train, df_eval = sampling_data(cfg, df_data, show=False)
    ## component models: training and inferring
    pred_ensemble_train_labels, pred_ensemble_train_probs, pred_eval_labels, pred_eval_probs = make_models(cfg, datasets, df_ensemble_train, df_eval)

    ## evaluate component models
    for i in range(cfg.n_models):
        scores = evaluation(cfg, df_eval, pred_eval_probs[i])
        #print(f"model{i+1}:{scores}")
        experiment_results_per_component[f"model-{i}"].append(scores) # scores of each component model
        experiment_results_component.append(scores) # for scores summarizing the entire component models

    ## random sampling data to be used weight parameter optimization
    labeled_index = list(df_ensemble_train[df_ensemble_train["target"].isin(cfg.labeled_classes)].sample(n=cfg.ensemble_labeled_size ,axis=0).index)

    # data to be used weight parameter optimization
    ## prediction probabilities of component models for this data 
    labeled_train_probs = []
    for i in range(cfg.n_models):
        labeled_train_prob = pred_ensemble_train_probs[i][labeled_index].copy()
        labeled_train_probs.append(labeled_train_prob)
    ## labeled train data
    df_ensemble_labeled_train = df_ensemble_train.iloc[labeled_index].copy()

    # for save results
    tuned_results = {}
    tuned_results['unsupervised_beta'] = []
    tuned_results['supervised_beta'] = []
    tuned_results['proposed_beta'] = []
    tuned_results['proposed_lambda'] = []
    tuned_results['unsupervised_scores'] = []
    tuned_results['supervised_scores'] = []
    tuned_results['proposed_scores'] = []

    # unsupervised ensemble learning ===========================================================================
    ## weight parameter optimization & inferring (used labeled data as unlabeled data)
    tuned_beta, eval_exp_mix_prob = unsupervised_ensemble(cfg, labeled_train_probs, df_ensemble_labeled_train, pred_eval_probs, show=False)
    ## evaluation
    scores = evaluation(cfg, df_eval, eval_exp_mix_prob)
    experiment_results_unsupervised.append(scores)
    tuned_results['unsupervised_beta'].append(tuned_beta)
    tuned_results['unsupervised_scores'].append(scores)
    print(scores)
    
    # Supervised ensemble learning ===========================================================================
    ## weight parameter optimization & inferring
    tuned_beta, eval_exp_mix_prob = supervised_ensemble(cfg, labeled_train_probs, df_ensemble_labeled_train, pred_eval_probs, show=False)
    ## evaluation
    scores = evaluation(cfg, df_eval, eval_exp_mix_prob)
    experiment_results_supervised.append(scores)
    tuned_results['supervised_beta'].append(tuned_beta)
    tuned_results['supervised_scores'].append(scores)
    print(scores)

    # Proposed supervised ensemble learning ====================================================================
    ## decide lambda by cross validation
    best_lam = tuning_proposed_supervised(cfg, labeled_train_probs, df_ensemble_labeled_train, criterion='accuracy_score',how=cfg.tuning, show_res=False)
    ## weight parameter optimization & inferring
    tuned_beta, eval_exp_mix_prob = proposed_supervised_ensemble(cfg, labeled_train_probs, df_ensemble_labeled_train, pred_eval_probs, best_lam, show=False)
    ## evaluation
    scores = evaluation(cfg, df_eval, eval_exp_mix_prob)
    experiment_results_proposed_supervised.append(scores)
    experiment_best_lams.append(best_lam)
    tuned_results['proposed_beta'].append(tuned_beta)
    tuned_results['proposed_lambda'].append(best_lam)
    tuned_results['proposed_scores'].append(scores)
    print(scores)

    cfg.seed += 1 # change seed for next experiment

# save log
pickle.dump(tuned_results, open(os.path.join(cfg.EXP_LOG, f'{log_name}_log.pkl'), 'wb'))

print(f"saving labeled size:{cfg.ensemble_labeled_size}")
save_result_csv(cfg, experiment_results_component, 
                    experiment_results_unsupervised,
                    experiment_results_supervised,
                    experiment_results_proposed_supervised,
                    )

for i in range(cfg.n_models):
    print(f"model{i}-{cfg.algorithm_list[i]}")
    save_component_csv(cfg, experiment_results_per_component[f"model-{i}"], f"model{i+1}-{cfg.algorithm_list[i]}")

print(f"saving labeled size:{cfg.ensemble_labeled_size}")

show_result(cfg, experiment_results_component, 
                experiment_results_unsupervised,
                experiment_results_supervised,
                experiment_results_proposed_supervised,
                )

# best lam
content = f"best_lams: {experiment_best_lams}\n"
with open(cfg.EXP_LOG + f"/log_{cfg.MODEL_NAME}.txt", "a") as appender:
    appender.write(content + "\n")
