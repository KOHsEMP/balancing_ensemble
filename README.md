# Balancing Selection and Diversity in Ensemble Learning with Exponential Mixture Model

## Requirements

* python 3.9.14

```
numpy==1.23.5
scipy==1.9.3
pandas==1.5.1
imbalanced-learn==0.10.1
scikit-learn==1.0.2
matplotlib==3.6.2
seaborn==0.12.1
optuna==3.1.0
arff==0.9
```

## Algorithms setting to create component predictors
The hyperparameter settings of the component predictors used in this paper are shown below.
The values not shown are the default values in `scikit-learn 1.0.2`.

* Logistic Regression : `LogisticRegression`
  * solver="lbfgs", multi_class='auto'
* Support Vector Machine : `SVC`
  * kernel='rbf', gamma='auto', probability=True
* K Nearest Neighbours : `KNeighborsClassifier`
  * n_neighbors is the number of classes of each dataset
* Gaussian Naive Bayes : `GaussianNB` 
* Random Forest: `RandomForestClassifier`
  * the number of decision tree is 100. (It is default value.)


## Directory Structure

```
balancing_ensemble/
   ├ data/
   │   ├ bank/
   │   │  ├ bank-full.csv
   │   │  ├ bank-names.txt
   │   │  └ bank.csv
   │   ├ fashion-mnist/
   │   │  └ fashion-mnist.arff
   │   ├ kuzushiji-mnist/
   │   │  └ kuzushiji-mnist.arff
   │   ├ mnist/
   │   │  └ mnist_784.arff
   │   ├ sensorless/
   │   │  └ Sensorless_drive_diagnosis.txt
   │   └ volkert/
   │      └ volkert.arff
   │
   │
   ├ experiments/
       ├ performance.py
       ├ ...
```

* Once the program is executed, an output directory is created under `balancing_ensemble/output/` and the results are stored there.
  * if there are not directory `/balancing_ensemble/output`, it is created automatically
* Each data must be downloaded from [UCI repository](https://archive.ics.uci.edu/ml/index.php) or [OpenML](https://www.openml.org/).

## Datasets Download Links
* [MNIST](https://www.openml.org/search?type=data&status=active&id=554)
* [Fashion-MNIST](https://www.openml.org/search?type=data&status=active&id=40996)
* [Kuzushiji-MNIST](https://www.openml.org/search?type=data&status=active&id=41982)
* [Sensorless Drive Diagnosis](https://archive.ics.uci.edu/ml/datasets/dataset+for+sensorless+drive+diagnosis)
* [volkert](https://www.openml.org/search?type=data&status=active&id=41166&sort=runs)
* [Bank Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)


## Verification of Conjectures

This experiment corresponds to Figure 1 in this paper.
The corresponding program is `experiments/vary_lambda.py`.

**How to use**
```bash
python vary_lambda.py --dataset mnist --n_components 5 --n_experiments 5 --seed 42 \
                      --each_model_data 200 --ensemble_labeled_size 500 \
                      --labeled_classes 0 1 2 3 4 5 6 7 8 9
```

The explanation of the arguments is follow:
* `dataset`: dataset name  ('mnist','fashion-mnist','kuzushiji-mnist', 'bank','sensorless','volkert')
* `n_components`: the number of component predictors
* `n_experiments`: the number of experiments
* `seed`: random seed
* `each_model_data`: the number of data to train each component predictor
* `ensemble_labeled_size`: the number of data to optimize weight parameters of ensemble predictors
* `labeled_classes`: values of the objective variable to be used as supervised data
  * if dataset is 'mnist' that has 10 classes, labeled_classes is '0 1 2 3 4 5 6 7 8 9' 

**Execution results**
* The results will be saved in `/balancing_ensemble/output/exp-vary-lambda/`.
  * The figures of the results will be saved in `/balancing_ensemble/output/exp-vary-lambda/fig/`.
    * Please see these files.
  * The log files (.pkl) will be saved in `/balancing_ensemble/output/exp-vary-lambda/log/`.
    * These files is to make figure later.

## Weight Parameter Transition and Model Selection-Diversity Balance

This experiment corresponds to Figure 2 in this paper.
The corresponding program is `experiments/compare_lambda_beta.py`.

**How to use**
```bash
python compare_lambda_beta.py --dataset mnist --n_components 5 --n_experiments 1 --seed 42 \
                              --each_model_data 200 --ensemble_labeled_size 500 \
                              --algorithms logistic logistic svm svm knn \
                              --labeled_classes 0 1 2 3 4 5 6 7 8 9
```

The explanation of the arguments is follow:
* `dataset`: dataset name  ('mnist','fashion-mnist','kuzushiji-mnist', 'bank','sensorless','volkert')
* `n_components`: the number of component predictors
* `n_experiments`: the number of experiments
* `seed`: random seed
* `each_model_data`: the number of data to train each component predictor
* `ensemble_labeled_size`: the number of data to optimize weight parameters of ensemble predictors
* `labeled_classes`: values of the objective variable to be used as supervised data
  * if dataset is 'mnist' that has 10 classes, labeled_classes is '0 1 2 3 4 5 6 7 8 9' 

**Execution results**
* The results will be saved in `/balancing_ensemble/output/exp-compare-lambda-beta/`.
  * The figures of the results will be saved in `/balancing_ensemble/output/exp-compare-lambda-beta/fig/`.
    * Please see these files.
  * The log files (.pkl) will be saved in `/balancing_ensemble/output/exp-compare-lambda-beta/log/`.
    * These files is to make figure later.


## Comparison of Performance Experiment

This experiment corresponds to Figure 3 in this paper.
The corresponding program is `experiments/performance.py`.

**How to use**
```bash
python performance.py --dataset mnist --n_components 5 --each_model_data 200 \
                      --ensemble_labeled_size 500 \
                      --algorithms logistic logistic svm svm knn \
                      --n_experiments 5 \
                      --num_fold 5 --num_tuning 100 --tuning grid \
                      --labeled_classes 0 1 2 3 4 5 6 7 8 9                   
```

The explanation of the arguments is follow:
* `dataset`: dataset name  ('mnist','fashion-mnist','kuzushiji-mnist', 'bank','sensorless','volkert')
* `n_components`: the number of component predictors
* `each_model_data`: the number of data to train each component predictor
* `ensemble_labeled_size`: the number of data to optimize weight parameters of ensemble predictors
* `algorithms`: use algorithm names to make component predictor ("logistic", "svm", "knn", "decision_tree", "QDA", "gaussian_process", "gaussian_naive_bayes", "adaboost", "random_forest")
  * the number of agorithm names must be equal `n_components`
* `n_experiments`: the number of experiments
* `num_fold`: the number of folds of cross-validation
* `num_tuning`: hyperparameter lambda search count
* `tuning`: how to tune lambda ('optuna', 'grid')
* `labeled_classes`: values of the objective variable to be used as supervised data
  * If dataset is 'mnist' that has 10 classes, labeled_classes is '0 1 2 3 4 5 6 7 8 9' 

**Execution results**
* The results will be saved in `/balancing_ensemble/output/exp-performance-{datset}-various/`.
  * `dataset` is equal the argument `dataset` of `experiments/performance.py`.
  * The log files (.csv, .txt, .pkl) will be saved in `/balancing_ensemble/output/exp-performance-{datsetname}-various/log/`.
    * Settings and average results of the executed experiment will be saved in `log_{dataset}_various.csv`. 
    * Settings and average results of component predictors used in the experiment will be saved in `log_{dataset}_various.csv`. 
    * The results of the experiment, displayed in easy-to-read text, will be saved in `log_various.txt`.
    * The results of hyperparameter lambda tuning will be saved in `{experiment_settings...}_log.pkl`.


## Apply Oversampling

This experiment corresponds to Figure 4 in this paper.
The corresponding program is `experiments/oversampling.py`.

**How to use**
```bash
python oversampling.py --dataset volkert --n_components 5 --each_model_data 200 \
			--ensemble_labeled_size 500 \
			--algorithms logistic logistic svm svm knn \
			--n_experiments 5 \
			--num_fold 5 --num_tuning 100 --tuning grid \
			--minority_rate 0.3 \
			--labeled_classes 0 1 2 3 4 5 6 7 8 9                
```

The explanation of the arguments is follow:
* `dataset`: dataset name  ('mnist','fashion-mnist','kuzushiji-mnist', 'bank','sensorless','volkert')
* `n_components`: the number of component predictors
* `each_model_data`: the number of data to train each component predictor
* `ensemble_labeled_size`: the number of data to optimize weight parameters of ensemble predictors
* `algorithms`: use algorithm names to make component predictor ("logistic", "svm", "knn", "decision_tree", "QDA", "gaussian_process", "gaussian_naive_bayes", "adaboost", "random_forest")
  * the number of agorithm names must be equal `n_components`
* `n_experiments`: the number of experiments
* `num_fold`: the number of folds of cross-validation
* `num_tuning`: hyperparameter lambda search count
* `tuning`: how to tune lambda ('optuna', 'grid')
* `labeled_classes`: values of the objective variable to be used as supervised data
  * if dataset is 'mnist' that has 10 classes, labeled_classes is '0 1 2 3 4 5 6 7 8 9' 

**Execution results**
* The results will be saved in `/balancing_ensemble/output/exp-oversampling-{datset}-various/`.
  * `dataset` is equal the argument `dataset` of `experiments/oversampling.py`.
    * The log files (.csv, .txt, .pkl) will be saved in `/balancing_ensemble/output/exp-oversampling-{datsetname}-various/log/`.
      * Settings and average results of the executed experiment will be saved in `log_{dataset}_various.csv`. 
      * Settings and average results of component predictors used in the experiment will be saved in `log_{dataset}_various.csv`. 
      * The results of the experiment, displayed in easy-to-read text, will be saved in `log_various.txt`.
      * The results of hyperparameter lambda tuning will be saved in `{experiment_settings...}_log.pkl`.
