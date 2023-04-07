#!/bin/bash

python compare_lambda_beta.py --dataset mnist --n_components 5 --n_experiments 1 --seed 42 --each_model_data 200 \
			--ensemble_labeled_size 500 \
			--algorithms logistic logistic svm svm knn \
			--labeled_classes 0 1 2 3 4 5 6 7 8 9