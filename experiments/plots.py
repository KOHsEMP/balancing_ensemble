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

from utils import * 


# extract log data for ploting
def extract_log_data(_cfg):
    '''
    extract needed log (.csv) data
    Args:
        _cfg:
            Confing instance
    Return
        _df_log:
            extracted log data (pd.DataFrame)
    '''
    
    _df_log = pd.read_csv(os.path.join(_cfg.MAIN_PATH, "output", _cfg.NAME, "log", f"log_{_cfg.DATA_NAME}_various.csv"))

    # extract needed rows

    ## extract specified model_names
    extracted_model_names_list = []
    extracted_model_names = ""
    for i, algorithm_name in enumerate(_cfg.algorithm_list):
        extracted_model_names += algorithm_name
        if i != len(_cfg.algorithm_list) -1:
            extracted_model_names += "+"
        extracted_model_names_list.append(extracted_model_names)

    labeled_classes_str = ""
    for c in _cfg.labeled_classes:
        labeled_classes_str += str(c) + "-"
    
    unlabeled_classes_str = ""
    for c in _cfg.unlabeled_classes:
        unlabeled_classes_str += str(c) + "-"

    _df_log = _df_log[_df_log["model_names"].isin(extracted_model_names_list)]

    ## extract each_model_data_size
    _df_log = _df_log[_df_log["component_data"] == _cfg.each_model_data_size]
    ## extract ensemble_labeled_data_size
    _df_log = _df_log[_df_log["ensemble_labeled_data"].isin(_cfg.ensemble_labeled_size_list)]
    ## extract ensemble_labeled_data_size
    _df_log = _df_log[_df_log["ensemble_unlabeled_data"].isin(_cfg.ensemble_unlabeled_size_list)]
    ## extract eval_data_size
    _df_log = _df_log[_df_log["eval_data"] == _cfg.eval_data_size]
    ## extract num_experiments
    _df_log = _df_log[_df_log["n_experiments"] == _cfg.num_experiment]

    _df_log = _df_log[_df_log['num_fold'] == _cfg.num_fold]
    _df_log = _df_log[_df_log['num_tuning'] == _cfg.num_tuning]
    _df_log = _df_log[_df_log['tuning'] == _cfg.tuning]

    _df_log = _df_log[_df_log["labeled_classes"] == labeled_classes_str]
    _df_log = _df_log[_df_log["unlabeled_classes"] == unlabeled_classes_str]


    _df_log = _df_log.drop_duplicates().reset_index(drop=True)
    _df_log = _df_log.sort_values(["ensemble_labeled_data","ensemble_unlabeled_data"], ascending=True)

    return _df_log

def extract_component_log_data(_cfg):
    '''
    extract needed log (.csv) data
    Args:
        _cfg:
            Confing instance
    Return
        _df_log:
            extracted log data (pd.DataFrame)
    '''

    _df_log = pd.read_csv(os.path.join(_cfg.MAIN_PATH, "output", _cfg.NAME, "log", f"log_{_cfg.DATA_NAME}_component.csv"))

    # extract needed rows
    ## extract specified model_names
    extracted_model_names_list = []
    extracted_model_names = ""
    for i, algorithm_name in enumerate(_cfg.algorithm_list):
        extracted_model_names += algorithm_name
        if i != len(_cfg.algorithm_list) -1:
            extracted_model_names += "+"
    extracted_model_names_list.append(extracted_model_names)

    _df_log = _df_log[_df_log["model_names"].isin(extracted_model_names_list)]

    ## extract each_model_data_size
    _df_log = _df_log[_df_log["component_data"] == _cfg.each_model_data_size]
    _df_log = _df_log[_df_log["n_experiments"] == _cfg.num_experiment]

    _df_log = _df_log[_df_log['num_fold'] == _cfg.num_fold]
    _df_log = _df_log[_df_log['num_tuning'] == _cfg.num_tuning]
    _df_log = _df_log[_df_log['tuning'] == _cfg.tuning]

    _df_log = _df_log.drop_duplicates().reset_index(drop=True)
    _df_log = _df_log.sort_values(["ensemble_labeled_data","ensemble_unlabeled_data"], ascending=True)

    return _df_log

def plot_compare_size_score(_cfg, _metric="accuracy", _show_component=False, _show_std_bar=False, _save_fig=True):

    _df_log_all = extract_log_data(_cfg)
    _df_component_log_all = extract_component_log_data(_cfg)

    _markers = ["o", "v", "^", "s", "*", "X"]                          # https://matplotlib.org/stable/api/markers_api.html
    _lines   = ["solid", "dashed", "dashdot", "dotted"]
    _colors  = ["blue", "green", "red", "cyan", "black", "magenta"] # https://pythondatascience.plavox.info/matplotlib/%E8%89%B2%E3%81%AE%E5%90%8D%E5%89%8D
    #_colors = ["black", "tab:blue", "tab:cyan", "tab:green", "tab:red", "tab:olive","tab:purple", "tab:orange", ]

    comparing_method_list = ["component","unsupervised","supervised", "semi-supervised", "supervised-all", "unsupervised-EM"]

    # for paper
    # https://stackoverflow.com/questions/70819183/matplotlib-set-font-to-times-new-roman
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams['mathtext.fontset'] = 'stix'
    if _save_fig:
        plt.rcParams['figure.dpi'] = 500
    else:
        plt.rcParams['figure.dpi'] = 100

    fig = plt.figure(figsize=(4.3 * len(_cfg.ensemble_labeled_size_list), 3), constrained_layout=True)    

    fig.suptitle(f'dataset: {_cfg.DATA_NAME}',fontsize="x-large")

    _df_log = _df_log_all[_df_log_all["data"] == _cfg.DATA_NAME]
    _df_component_log = _df_component_log_all[_df_component_log_all["data"]==_cfg.DATA_NAME]

    # compare by changing labeled data size
    axes = fig.subplots(nrows=1, ncols=len(_cfg.ensemble_labeled_size_list), sharex="all", sharey="row")
    for col, _labeled_size in enumerate(_cfg.ensemble_labeled_size_list):
        _x_value = _df_log[_df_log["ensemble_labeled_data"] == _labeled_size]["ensemble_unlabeled_data"].values

        # compare methods
        for _method_idx, _method in enumerate(comparing_method_list):
            
            if _method == "component" and not _show_component: 
                continue

            _y_value_mean = _df_log[_df_log["ensemble_labeled_data"] == _labeled_size][f"{_method}_{_metric}_mean"].values
            _y_value_std  = _df_log[_df_log["ensemble_labeled_data"] == _labeled_size][f"{_method}_{_metric}_std"].values

            # plot
            axes[col].plot(_x_value, _y_value_mean,
                        marker=_markers[_method_idx % len(_markers)], linestyle=_lines[_method_idx % len(_lines)], color=_colors[_method_idx % len(_colors)], 
                        markeredgewidth=0.8, markeredgecolor='k', label=_method,)
            if _show_std_bar: # error bar (std)
                axes[col].errorbar(_x_value, _y_value_mean, yerr=_y_value_std, elinewidth=1)

        axes[col].set_title(f"labeled data size: {_labeled_size}")
        #axes[col].set_xticks(_x_value)
        axes[col].set_xlabel("unlabeled data size")
        if col == 0:
            axes[col].set_ylabel(_metric)
        
        #if (row == len(subfigs) -1) and (col == int(len(_cfg.ensemble_labeled_size_list)/2)) :
        #    axes[col].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol = len(_cfg.comparing_method_list))
        if col == len(_cfg.ensemble_labeled_size_list) -1:
            # legend
            axes[col].legend(loc="upper left", bbox_to_anchor=(1, 1), ncol = 1)
            # show component_score            
            component_score_str = ""
            for model_idx in range(len(_cfg.algorithm_list)):
                component_score_str += f"model{model_idx+1} acc: {_df_component_log[_df_component_log['model']==f'model{model_idx+1}-{_cfg.algorithm_list[model_idx]}'][f'component_{_metric}_mean'].values[0]:.4f}\n"
            axes[col].text(1.05, -0.05, component_score_str, transform=axes[col].transAxes)

    # labeled classes
    labeled_classes_str ="_labeledC"
    for c in _cfg.labeled_classes:
        labeled_classes_str += "-" + str(c)
    
    unlabeled_classes_str ="_unlabeledC"
    for c in _cfg.unlabeled_classes:
        unlabeled_classes_str += "-" + str(c)

    # save fig
    _fig_name = "data_"
    _fig_name += _cfg.DATA_NAME

    _fig_name += "_alg_"
    for model in _cfg.algorithm_list:
        _fig_name += "_" + model
    _fig_name += "_" + _metric
    _fig_name += labeled_classes_str
    _fig_name += unlabeled_classes_str
    _fig_name += f"_labeled_{_cfg.ensemble_labeled_size_list[0]}-{_cfg.ensemble_labeled_size_list[-1]}"
    _fig_name += f"_unlabeled_{_cfg.ensemble_unlabeled_size_list[0]}-{_cfg.ensemble_unlabeled_size_list[-1]}"
    _fig_name += f"_eval_{_cfg.eval_data_size}"
    _fig_name += f"_seed_{_cfg.seed}"
    _fig_name += f"_exp_{_cfg.num_experiment}"
    _fig_name += ".png"

    if _save_fig:
        fig.savefig(os.path.join(_cfg.MAIN_PATH, "output", _cfg.NAME, "fig", _fig_name), bbox_inches="tight", pad_inches=0.05)
