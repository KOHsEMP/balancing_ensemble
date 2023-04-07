#!/bin/bash

python oversampling.py --dataset volkert --n_components 5 --each_model_data 200 \
			--ensemble_labeled_size 500 \
			--algorithms logistic logistic svm svm knn \
			--n_experiments 5 \
			--num_fold 5 --num_tuning 100 --tuning grid \
			--minority_rate 0.3 \
			--labeled_classes 0 1 2 3 4 5 6 7 8 9 