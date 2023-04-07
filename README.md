# balancing_ensemble

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

* Once the program is executed, an output directory is created under balancing_ensemble/ and the results are stored there.
* Each data must be downloaded from [UCI repository](https://archive.ics.uci.edu/ml/index.php) or [OpenML](https://www.openml.org/).

## Datasets Download
* [MNIST](https://www.openml.org/search?type=data&status=active&id=554)
* [Fashion-MNIST](https://www.openml.org/search?type=data&status=active&id=40996)
* [Kuzushiji-MNIST](https://www.openml.org/search?type=data&status=active&id=41982)
* [Sensorless Drive Diagnosis](https://archive.ics.uci.edu/ml/datasets/dataset+for+sensorless+drive+diagnosis)
* [volkert](https://www.openml.org/search?type=data&status=active&id=41166&sort=runs)
* [Bank Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

## Comparison of Performance Experiment

```bash
python performance.py --dataset <> --n_components <> --each_model_data <> \
                      --ensemble_labeled_size <> --algorithms <> --n_experiments <> \
                      --num_fold <> --num_tuning <> --tuning <> \
                      --labeled_classes <>                   
```

The explanation of the arguments is follow:
* `dataset`: dataset name  ('mnist','fashion-mnist','kuzushiji-mnist', 'run-or-walk','bank','sensorless','volkert')
* `n_components`: the number of component predictors
* `each_model_data`: the number of data to train each component predictor
* `ensemble_labeled_size`: the number of data to optimize weight parameters of ensemble predictors
* `algorithms`: use algorithm names to make component predictor ("logistic", "svm", "knn", "decision_tree", "QDA", "gaussian_process", "gaussian_naive_bayes", "adaboost", "random_forest")
  * the number of agorithm names must be equal `n_components`
* `num_experiments`: the number of experiments
* `num_fold`: the number of folds of cross-validation
* `num_tuning`: hyperparameter lambda search count
* `tuning`: how to tune lambda ('optuna', 'grid')
* `labeled_classes`: values of the objective variable to be used as supervised data
  * if dataset is 'mnist' that has 10 classes, labeled_classes is '0 1 2 3 4 5 6 7 8 9' 

**Example**
```bash
python performance.py --dataset mnist --n_components 5 --each_model_data 200 \
			--ensemble_labeled_size 500 \
			--algorithms logistic logistic svm svm knn \
			--n_experiments 5 \
			--num_fold 5 --num_tuning 100 --tuning grid \
			--labeled_classes 0 1 2 3 4 5 6 7 8 9 
```
