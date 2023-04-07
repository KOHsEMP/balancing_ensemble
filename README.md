# Balancing Selection adn Diversity in Ensemble Learning with Exponential Mixture Model

## Dependencies

```
numpy==1.23.5
scipy==1.9.3
pandas==1.5.1
imbalanced-learn==0.10.1
scikit-learn==1.0.2
```

## Algorithms setting to create component predictors
The hyperparameter settings of the component predictors used in the experiments are shown below.
The values not shown are the default values in `scikit-learn 1.0.2`.

* Logistic Regression : `LogisticRegression`
  * solver="lbfgs", multi_class='auto'
* Support Vector Machine : `SVC`
  * kernel='rbf', gamma='auto', probability=True
* K Nearest Neighbours : `KNeighborsClassifier`
  * n_neighbors is the number of classes of each dataset
* Gaussian Naive Bayes : `GaussianNB` 
* Random Forest: `RandomForestClassifier`


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
   ├ experiments/
       ├ performance.py
       ├ ...
```

* Once the program is executed, an output directory is created under `balancing_ensemble/output/` and the results are stored there.
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
  * if dataset is 'mnist' that has 10 classes, labeled_classes is '0 1 2 3 4 5 6 7 8 9' 


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
