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

from scipy.io.arff import loadarff 


# argument is Config class
def setup(cfg):
    cfg.EXP = cfg.NAME
    cfg.INPUT = cfg.DATA_PATH # path for source data
    cfg.OUTPUT = os.path.join(cfg.MAIN_PATH, "output") # path for output data

    cfg.OUTPUT_EXP = os.path.join(cfg.OUTPUT, cfg.EXP)
    cfg.EXP_MODEL = os.path.join(cfg.OUTPUT_EXP, "model")
    cfg.EXP_FIG = os.path.join(cfg.OUTPUT_EXP, "fig")
    cfg.EXP_LOG = os.path.join(cfg.OUTPUT_EXP, "log")
    cfg.EXP_PREDS = os.path.join(cfg.OUTPUT_EXP, "preds")

    for d in [cfg.EXP_MODEL, cfg.EXP_FIG, cfg.EXP_LOG,cfg.EXP_PREDS]:
        os.makedirs(d, exist_ok=True)
    return cfg

# fucntion to fix random seed
def set_seed(seed=42):
    random.seed(seed) # pythonのseed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) # numpyのseed


from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss)

# 予測結果を評価する関数
# function to evaluate prediction results
def evaluation(cfg, _df_eval_data, _eval_pred_prob):
    """
    evaluate model by some criterion for binary classification
    Args:
        cfg:
            Config instance
        _df_eval_data:
            evaluation data (pd.DataFrame)
        _eval_pred_prob:
            prediction probability of evaluatio data (np.array (n, n_classes))
    Returns:
        _scores:
            dictionary including some criterion result
    """

    _n_classes = len(_eval_pred_prob[0])
    _eval_pred_label = np.argmax(_eval_pred_prob, axis=1)

    label_scores = [
        accuracy_score,
        f1_score,
        precision_score,
        recall_score
    ]

    prob_scores = [
        roc_auc_score,
        log_loss
    ]

    _scores = {}
    for f in label_scores:
        if _n_classes == 2:
            _score = f(y_true=_df_eval_data["target"], y_pred=_eval_pred_label)
        else:
            if f != accuracy_score:
                _score = f(y_true=_df_eval_data["target"], y_pred=_eval_pred_label, average="macro")
            else:
                _score = f(y_true=_df_eval_data["target"], y_pred=_eval_pred_label)
        _scores[str(f.__name__)] = _score
    

    _normalization_term = np.sum(_eval_pred_prob, axis=1)
    _normalization_term = np.repeat(_normalization_term[None, :], _n_classes, axis=0).T
    _eval_pred_prob = _eval_pred_prob / _normalization_term

    for f in prob_scores:
        if _n_classes == 2:
            if f == roc_auc_score:
                _score = f(y_true=_df_eval_data["target"], y_score=_eval_pred_prob[:,1])
            elif f == log_loss:
                _score = f(y_true=_df_eval_data["target"], y_pred=_eval_pred_prob, labels=[i for i in range(_n_classes)])
        else:
            if f == roc_auc_score:
                _score = f(y_true=_df_eval_data["target"], y_score=_eval_pred_prob, average="macro", multi_class="ovr")
            elif f == log_loss:
                _score = f(y_true=_df_eval_data["target"], y_pred=_eval_pred_prob, labels=[i for i in range(_n_classes)])
        _scores[str(f.__name__)] = _score

    return _scores


# function to aggregate prediction results
def agg_scores(cfg, _experiment_results, show=True):
    '''
    aggregation experiment results
    Args:
        cfg:
            Config instance
        _experiment_results:
            list of scores dictionaries
    Returns:
        _means:
            means dictionary of metrics
        _stds:
            stds dictinary of metrics
        _content:
            strings of aggregation metrics
    '''
    _metrics = _experiment_results[0].keys()

    _means = _experiment_results[0].copy()
    _stds  = _experiment_results[0].copy()

    for _metric in _metrics:
        _tmp_list = []
        for i in range(len(_experiment_results)): 
            _tmp_list.append(_experiment_results[i][_metric])
        
        _means[_metric] = np.mean(_tmp_list)
        _stds[_metric]  = np.std(_tmp_list, ddof=1)

    if show:
        for _metric in _metrics:
            print(f"{_metric.replace('_score', '')}: {_means[_metric]:.4f} ±{_stds[_metric]:.4f}, ", end="")
    
    _content_4 = ""
    _content_8 = ""
    for _metric in _metrics:
        _content_4 += f"{_metric.replace('_score', '')}: {_means[_metric]:.4f} ±{_stds[_metric]:.4f}, "
        _content_8 += f"{_metric.replace('_score', '')}: {_means[_metric]:.8f} ±{_stds[_metric]:.8f}, "
        
    return _means, _stds, _content_4, _content_8

def show_result(cfg, _experiment_results_component, 
                     _experiment_results_unsupervised,
                     _experiment_results_supervised,
                     _experiment_results_proposed_supervised,
                     ):
    '''
    show experiment results
    Args:
        cfg:
            Config isntance
        _experiment_results_component:
            component predictor results
        _experiment_results_unsupervised:
            unsupervised ensemble results
        _experiment_results_supervised:
            supervised ensemble results
        _experiment_results_proposed_supervised:
            proposed supervised ensemble results
    Returns:
        None 
    '''
    
    content =  f"===== " + time.ctime() + " " + cfg.NAME +  " =====\n"
    content += f"Config:\n"
    content += f"    data: {cfg.DATA_NAME} (all_data_size=:{cfg.n_data}), model:{cfg.MODEL_NAME}, \n"
    content += f"    models: {cfg.algorithm_list}\n"
    content += f"    each_model_data_size: {cfg.each_model_data_size}, ensemble_labeled_size:{cfg.ensemble_labeled_size}, eval_data_size:{cfg.eval_data_size}\n"
    content += f"    n_models:{cfg.n_models}, num_experiments:{cfg.num_experiment}\n"
    content += f"   num_fold:{cfg.num_fold}, num_tuning:{cfg.num_tuning}, tuning:{cfg.tuning}\n"

    _component_means, _component_stds, _component_content_4, _component_content_8 = agg_scores(cfg, _experiment_results_component, show=False)
    _unsupervised_means, _unsupervised_stds, _unsupervised_content_4, _unsupervised_content_8 = agg_scores(cfg, _experiment_results_unsupervised, show=False)
    _supervised_means, _supervised_stds, _supervised_content_4, _supervised_content_8 = agg_scores(cfg, _experiment_results_supervised, show=False)
    _proposed_supervised_means, _proposed_supervised_stds, _proposed_supervised_content_4, _proposed_supervised_content_8 = agg_scores(cfg, _experiment_results_proposed_supervised, show=False)

    content += "# component predictor\n"
    content += _component_content_4 + "\n"
    content += "# unsupervised ensemble\n"
    content += _unsupervised_content_4 + "\n"
    content += "# supervised ensemble\n"
    content += _supervised_content_4 + "\n"
    content += "# proposed-supervised ensemble\n"
    content += _proposed_supervised_content_4 + "\n"

    content += "# component predictor\n"
    content += _component_content_8 + "\n"
    content += "# unsupervised ensemble\n"
    content += _unsupervised_content_8 + "\n"
    content += "# supervised ensemble\n"
    content += _supervised_content_8 + "\n"
    content += "# proposed-supervised ensemble\n"
    content += _proposed_supervised_content_8 + "\n"


    with open(cfg.EXP_LOG + f"/log_{cfg.MODEL_NAME}.txt", "a") as appender:
        appender.write(content + "\n")
    
    print(content)

def save_result_csv(cfg, _experiment_results_component, 
                     _experiment_results_unsupervised,
                     _experiment_results_supervised,
                     _experiment_results_proposed_supervised,
                    ):
    '''
    save experiment results to .csv file
    Args:
        cfg:
            Config isntance
        _experiment_results_component:
            component predictor results
        _experiment_results_unsupervised:
            unsupervised ensemble results
        _experiment_results_supervised:
            supervised ensemble results
        _experiment_results_proposed_supervised:
            proposed supervised ensemble results
    Returns:
        None 
    '''

    _save_file_name = cfg.EXP_LOG + f"/log_{cfg.DATA_NAME}_{cfg.MODEL_NAME}.csv"

    _is_file = os.path.isfile(_save_file_name)

    
    # when log file doesn't exist, make header
    if not _is_file: 
        _header = "data,model,n_models,model_names,all_data,component_data,ensemble_labeled_data,eval_data,n_experiments,labeled_classes,num_fold,num_tuning,tuning,"
        for _method in ["component", "unsupervised", "supervised", "proposed_supervised"]:
            for _metric in ["accuracy", "f1", "precision", "recall", "roc_auc", "log_loss"]:
                for _stat in ["mean", "std"]:
                    _header += f"{_method}_{_metric}_{_stat},"

        # delete an end comma
        _header = _header[:-1]

        with open(_save_file_name, "a") as appender:
            appender.write(_header + "\n")
        

    # change cfg.algorithm_list to str
    _algorithm_list_str = ""
    for i, _algorithm in enumerate(cfg.algorithm_list):
        _algorithm_list_str += f"{_algorithm}"
        if i != len(cfg.algorithm_list) - 1:
            _algorithm_list_str += "+"

    labeled_classes_str = ""
    for c in cfg.labeled_classes:
        labeled_classes_str += str(c) + "-"

    _log = f"{cfg.DATA_NAME},{cfg.MODEL_NAME},{cfg.n_models},{_algorithm_list_str},{cfg.n_data},{cfg.each_model_data_size},{cfg.ensemble_labeled_size},{cfg.eval_data_size},{cfg.num_experiment},{labeled_classes_str},{cfg.num_fold},{cfg.num_tuning},{cfg.tuning},"

    for _method_result in [_experiment_results_component, 
                            _experiment_results_unsupervised,
                            _experiment_results_supervised,
                            _experiment_results_proposed_supervised,
                            ]:

        _means, _stds, _, _ = agg_scores(cfg, _method_result, show=False)
        for _metric in ["accuracy", "f1", "precision", "recall", "roc_auc", "log_loss"]:
            
            _metric_name = ""
            if _metric == "log_loss":
                _metric_name = _metric
            else:
                _metric_name = _metric + "_score"
            
            # mean
            _log += f"{_means[_metric_name]},"
            #std
            _log += f"{_stds[_metric_name]},"

    # delete an end comma
    _log = _log[:-1]
    
    with open(_save_file_name, "a") as appender:
            appender.write(_log + "\n")

    print("saved log")

def save_component_csv(cfg, _experiment_results_component, _model_name):

    '''
    save experiment results to .csv file
    Args:
        cfg:
            Config isntance
        _experiment_results_component:
            component predictor results
        _model_name:
            model name (string)
    Returns:
        None 
    '''

    _save_file_name = cfg.EXP_LOG + f"/log_{cfg.DATA_NAME}_component.csv"

    _is_file = os.path.isfile(_save_file_name)

    
    # when log file doesn't exist, make header
    if not _is_file: 
        _header = "data,model,n_models,model_names,all_data,component_data,ensemble_labeled_data,eval_data,n_experiments,num_fold,num_tuning,tuning,"
        for _method in ["component", "unsupervised", "supervised", "proposed_supervised"]:
            for _metric in ["accuracy", "f1", "precision", "recall", "roc_auc", "log_loss"]:
                for _stat in ["mean", "std"]:
                    _header += f"{_method}_{_metric}_{_stat},"

        # delete an end comma
        _header = _header[:-1]

        with open(_save_file_name, "a") as appender:
            appender.write(_header + "\n")
        

    # change cfg.algorithm_list to str
    _algorithm_list_str = ""
    for i, _algorithm in enumerate(cfg.algorithm_list):
        _algorithm_list_str += f"{_algorithm}"
        if i != len(cfg.algorithm_list) - 1:
            _algorithm_list_str += "+"



    _log = f"{cfg.DATA_NAME},{_model_name},{cfg.n_models},{_algorithm_list_str},{cfg.n_data},{cfg.each_model_data_size},{cfg.ensemble_labeled_size},{cfg.eval_data_size},{cfg.num_experiment},{cfg.num_fold},{cfg.num_tuning},{cfg.tuning},"
    
    _means, _stds, _, _ = agg_scores(cfg, _experiment_results_component, show=False)
    
    for _metric in ["accuracy", "f1", "precision", "recall", "roc_auc", "log_loss"]:
        
        _metric_name = ""
        if _metric == "log_loss":
            _metric_name = _metric
        else:
            _metric_name = _metric + "_score"
        
        # mean
        _log += f"{_means[_metric_name]},"
        #std
        _log += f"{_stds[_metric_name]},"

    for nan_res in range(5):
        for _metric in ["accuracy", "f1", "precision", "recall", "roc_auc", "log_loss"]:
            
            # mean
            _log += f"-1,"
            #std
            _log += f"-1,"

    # delete an end comma
    _log = _log[:-1]
    
    with open(_save_file_name, "a") as appender:
            appender.write(_log + "\n")

    print("saved log")




def show_result_oversampling(cfg, _experiment_results_component, 
                                _experiment_results_unsupervised,
                                _experiment_results_supervised,
                                _experiment_results_proposed_supervised,
                                _experiment_results_proposed_oversampling,
                                ):
    '''
    show experiment results
    Args:
        cfg:
            Config isntance
        _experiment_results_component:
            component predictor results
        _experiment_results_unsupervised:
            unsupervised ensemble results
        _experiment_results_supervised:
            supervised ensemble results
        _experiment_results_proposed_supervised:
            proposed supervised ensemble results
    Returns:
        None 
    '''
    
    content =  f"===== " + time.ctime() + " " + cfg.NAME +  " =====\n"
    content += f"Config:\n"
    content += f"    data: {cfg.DATA_NAME} (all_data_size=:{cfg.n_data}), model:{cfg.MODEL_NAME}, \n"
    content += f"    models: {cfg.algorithm_list}\n"
    content += f"    each_model_data_size: {cfg.each_model_data_size}, ensemble_labeled_size:{cfg.ensemble_labeled_size}, eval_data_size:{cfg.eval_data_size}\n"
    content += f"    n_models:{cfg.n_models}, num_experiments:{cfg.num_experiment}\n"
    content += f"   num_fold:{cfg.num_fold}, num_tuning:{cfg.num_tuning}, tuning:{cfg.tuning}\n"
    content += f"   minority_rate:{cfg.minority_rate}\n"

    _component_means, _component_stds, _component_content_4, _component_content_8 = agg_scores(cfg, _experiment_results_component, show=False)
    _unsupervised_means, _unsupervised_stds, _unsupervised_content_4, _unsupervised_content_8 = agg_scores(cfg, _experiment_results_unsupervised, show=False)
    _supervised_means, _supervised_stds, _supervised_content_4, _supervised_content_8 = agg_scores(cfg, _experiment_results_supervised, show=False)
    _proposed_supervised_means, _proposed_supervised_stds, _proposed_supervised_content_4, _proposed_supervised_content_8 = agg_scores(cfg, _experiment_results_proposed_supervised, show=False)
    _proposed_oversampling_means, _proposed_oversampling_stds, _proposed_oversampling_content_4, _proposed_oversampling_content_8 = agg_scores(cfg, _experiment_results_proposed_oversampling, show=False)


    content += "# component predictor\n"
    content += _component_content_4 + "\n"
    content += "# unsupervised ensemble\n"
    content += _unsupervised_content_4 + "\n"
    content += "# supervised ensemble\n"
    content += _supervised_content_4 + "\n"
    content += "# proposed-supervised ensemble\n"
    content += _proposed_supervised_content_4 + "\n"
    content += "# proposed-oversampling ensemble\n"
    content += _proposed_oversampling_content_4 + "\n"

    content += "# component predictor\n"
    content += _component_content_8 + "\n"
    content += "# unsupervised ensemble\n"
    content += _unsupervised_content_8 + "\n"
    content += "# supervised ensemble\n"
    content += _supervised_content_8 + "\n"
    content += "# proposed-supervised ensemble\n"
    content += _proposed_supervised_content_8 + "\n"
    content += "# proposed-oversampling ensemble\n"
    content += _proposed_oversampling_content_8 + "\n"


    with open(cfg.EXP_LOG + f"/log_{cfg.MODEL_NAME}.txt", "a") as appender:
        appender.write(content + "\n")
    
    print(content)

def save_result_csv_oversampling(cfg, _experiment_results_component, 
                                    _experiment_results_unsupervised,
                                    _experiment_results_supervised,
                                    _experiment_results_proposed_supervised,
                                    _experiment_results_proposed_oversampling,
                                    ):
    '''
    save experiment results to .csv file
    Args:
        cfg:
            Config isntance
        _experiment_results_component:
            component predictor results
        _experiment_results_unsupervised:
            unsupervised ensemble results
        _experiment_results_supervised:
            supervised ensemble results
        _experiment_results_proposed_supervised:
            proposed supervised ensemble results
    Returns:
        None 
    '''

    _save_file_name = cfg.EXP_LOG + f"/log_{cfg.DATA_NAME}_{cfg.MODEL_NAME}.csv"

    _is_file = os.path.isfile(_save_file_name)

    
    # when log file doesn't exist, make header
    if not _is_file: 
        _header = "data,model,n_models,model_names,all_data,component_data,ensemble_labeled_data,eval_data,n_experiments,labeled_classes,num_fold,num_tuning,tuning,minority_rate,"
        for _method in ["component", "unsupervised", "supervised", "proposed_supervised", "proposed_oversampling"]:
            for _metric in ["accuracy", "f1", "precision", "recall", "roc_auc", "log_loss"]:
                for _stat in ["mean", "std"]:
                    _header += f"{_method}_{_metric}_{_stat},"

        # delete an end comma
        _header = _header[:-1]

        with open(_save_file_name, "a") as appender:
            appender.write(_header + "\n")
        

    # change cfg.algorithm_list to str
    _algorithm_list_str = ""
    for i, _algorithm in enumerate(cfg.algorithm_list):
        _algorithm_list_str += f"{_algorithm}"
        if i != len(cfg.algorithm_list) - 1:
            _algorithm_list_str += "+"

    labeled_classes_str = ""
    for c in cfg.labeled_classes:
        labeled_classes_str += str(c) + "-"
    
    _log = f"{cfg.DATA_NAME},{cfg.MODEL_NAME},{cfg.n_models},{_algorithm_list_str},{cfg.n_data},{cfg.each_model_data_size},{cfg.ensemble_labeled_size},{cfg.eval_data_size},{cfg.num_experiment},{labeled_classes_str},{cfg.num_fold},{cfg.num_tuning},{cfg.tuning},{cfg.minority_rate},"

    for _method_result in [_experiment_results_component, 
                            _experiment_results_unsupervised,
                            _experiment_results_supervised,
                            _experiment_results_proposed_supervised,
                            _experiment_results_proposed_oversampling,
                            ]:

        _means, _stds, _, _ = agg_scores(cfg, _method_result, show=False)
        for _metric in ["accuracy", "f1", "precision", "recall", "roc_auc", "log_loss"]:
            
            _metric_name = ""
            if _metric == "log_loss":
                _metric_name = _metric
            else:
                _metric_name = _metric + "_score"
            
            # mean
            _log += f"{_means[_metric_name]},"
            #std
            _log += f"{_stds[_metric_name]},"

    # delete an end comma
    _log = _log[:-1]
    
    with open(_save_file_name, "a") as appender:
            appender.write(_log + "\n")

    print("saved log")