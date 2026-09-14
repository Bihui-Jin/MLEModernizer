# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.12

# 3. Installed packages

catboost==1.2.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-multilearn==0.2.0
seaborn==0.12.2
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.9491

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install -U scikit-multilearn --quiet


## === cell 1
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from catboost.utils import eval_metric
from skmultilearn.model_selection import iterative_train_test_split

DATA_DIR = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/"
OUTPUT_DIR = "/kaggle/working/"
RANDOM_STATE = 42


## === cell 2
def unpack_zipfile(filename):
    """Unpacks zip-file by name from DATA_DIR to OUTPUT_DIR."""
    try:
        shutil.unpack_archive(
            filename=DATA_DIR + filename,
            extract_dir=OUTPUT_DIR,
            format="zip",
        )
    except Exception as e:
        print(e)
    else:
        print(f"Archive file '{filename}' has been unpacked successfully.")


## === cell 3
unpack_zipfile(filename="train.csv.zip")
unpack_zipfile(filename="test.csv.zip")
unpack_zipfile(filename="sample_submission.csv.zip")
unpack_zipfile(filename="test_labels.csv.zip")


## === cell 4
train_df = pd.read_csv(OUTPUT_DIR + "train.csv")
test_df = pd.read_csv(OUTPUT_DIR + "test.csv")


## === cell 5
corpus_train = train_df["comment_text"].values.astype("U")
corpus_train[:5]


## === cell 6
target_cols = list(train_df.columns[2:])
target_train = train_df[target_cols].values
target_train[:5]


## === cell 7
corpus_test = test_df["comment_text"].values.astype("U")
corpus_test[:5]


## === cell 8
row_ids = np.arange(len(target_train))

train_idx, y_train, val_idx, y_val = iterative_train_test_split(
    row_ids[:,np.newaxis], 
    target_train, 
    test_size=0.25,
)

x_train = [corpus_train[i] for i in train_idx.flatten()]
x_val = [corpus_train[i] for i in val_idx.flatten()]

x_train = np.array(x_train)
x_val = np.array(x_val)


## === cell 9
pool_train = Pool(
    data=x_train, 
    label=y_train, 
    text_features=[0],
)

pool_valid = Pool(
    data=x_val, 
    label=y_val, 
    text_features=[0],
)


## === cell 10
model = CatBoostClassifier(
    iterations=5000, 
    verbose=500,
    task_type="GPU",
    loss_function='MultiLogloss',
    class_names=list(train_df.columns[2:]),
)


## === cell 11
model.fit(
    pool_train, 
    eval_set=pool_valid,
    early_stopping_rounds=200,
)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
CatBoostError                             Traceback (most recent call last)
/tmp/ipykernel_11/2983380576.py in <cell line: 0>()
----> 1 model.fit(
      2     pool_train,
      3     eval_set=pool_valid,
      4     early_stopping_rounds=200,
      5 )

/usr/local/lib/python3.11/dist-packages/catboost/core.py in fit(self, X, y, cat_features, text_features, embedding_features, graph, sample_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)
   5243             CatBoostClassifier._check_is_compatible_loss(params['loss_function'])
   5244 
-> 5245         self._fit(X, y, cat_features, text_features, embedding_features, None, graph, sample_weight, None, None, None, None, baseline, use_best_model,
   5246                   eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period,
   5247                   silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _fit(self, X, y, cat_features, text_features, embedding_features, pairs, graph, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)
   2408 
   2409             with plot_wrapper(plot, plot_file, 'Training plots', [_get_train_dir(self.get_params())]):
-> 2410                 self._train(
   2411                     train_pool,
   2412                     train_params["eval_sets"],

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _train(self, train_pool, test_pool, params, allow_clear_pool, init_model)
   1788 
   1789     def _train(self, train_pool, test_pool, params, allow_clear_pool, init_model):
-> 1790         self._object._train(train_pool, test_pool, params, allow_clear_pool, init_model._object if init_model else None)
   1791         self._set_trained_model_attributes()
   1792 

_catboost.pyx in _catboost._CatBoost._train()

_catboost.pyx in _catboost._CatBoost._train()

CatBoostError: catboost/cuda/cuda_lib/cuda_base.h:281: CUDA error 35: CUDA driver version is insufficient for CUDA runtime version

## === cell 12
predictions_valid = model.predict(pool_valid)

for metric in ('Precision', 'Recall', 'F1'):
    print(metric)
    print(50 * "-")
    values = eval_metric(y_val, predictions_valid, metric)
    for cls, value in zip(model.classes_, values):
        print(f'class={cls}: {value:.4f}')
    print()


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
CatBoostError                             Traceback (most recent call last)
/tmp/ipykernel_11/3361144544.py in <cell line: 0>()
----> 1 predictions_valid = model.predict(pool_valid)
      2 
      3 for metric in ('Precision', 'Recall', 'F1'):
      4     print(metric)
      5     print(50 * "-")

/usr/local/lib/python3.11/dist-packages/catboost/core.py in predict(self, data, prediction_type, ntree_start, ntree_end, thread_count, verbose, task_type)
   5305                   with log probability for every class for each object.
   5306         """
-> 5307         return self._predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose, 'predict', task_type)
   5308 
   5309     def predict_proba(self, X, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type="CPU"):

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _predict(self, data, prediction_type, ntree_start, ntree_end, thread_count, verbose, parent_method_name, task_type)
   2618         if verbose is None:
   2619             verbose = False
-> 2620         data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
   2621         self._validate_prediction_type(prediction_type)
   2622 

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _process_predict_input_data(self, data, parent_method_name, thread_count, label)
   2594     def _process_predict_input_data(self, data, parent_method_name, thread_count, label=None):
   2595         if not self.is_fitted() or self.tree_count_ is None:
-> 2596             raise CatBoostError(("There is no trained model to use {}(). "
   2597                                  "Use fit() to train model. Then use this method.").format(parent_method_name))
   2598         is_single_object = _is_data_single_object(data)

CatBoostError: There is no trained model to use predict(). Use fit() to train model. Then use this method.

## === cell 13
pool_test = Pool(data=corpus_test, label=None, text_features=[0])


## === cell 14
proba_predictions_test = model.predict_proba(pool_test)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
CatBoostError                             Traceback (most recent call last)
/tmp/ipykernel_11/466415302.py in <cell line: 0>()
----> 1 proba_predictions_test = model.predict_proba(pool_test)

/usr/local/lib/python3.11/dist-packages/catboost/core.py in predict_proba(self, X, ntree_start, ntree_end, thread_count, verbose, task_type)
   5349                 with probability for every class for each object.
   5350         """
-> 5351         return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
   5352 
   5353     def predict_log_proba(self, data, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type="CPU"):

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _predict(self, data, prediction_type, ntree_start, ntree_end, thread_count, verbose, parent_method_name, task_type)
   2618         if verbose is None:
   2619             verbose = False
-> 2620         data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
   2621         self._validate_prediction_type(prediction_type)
   2622 

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _process_predict_input_data(self, data, parent_method_name, thread_count, label)
   2594     def _process_predict_input_data(self, data, parent_method_name, thread_count, label=None):
   2595         if not self.is_fitted() or self.tree_count_ is None:
-> 2596             raise CatBoostError(("There is no trained model to use {}(). "
   2597                                  "Use fit() to train model. Then use this method.").format(parent_method_name))
   2598         is_single_object = _is_data_single_object(data)

CatBoostError: There is no trained model to use predict_proba(). Use fit() to train model. Then use this method.

## === cell 15
submission = pd.DataFrame(
    proba_predictions_test, 
    columns=target_cols, 
    index=test_df.id
).reset_index()

submission.head()


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/578276366.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     proba_predictions_test,
      3     columns=target_cols,
      4     index=test_df.id
      5 ).reset_index()

NameError: name 'proba_predictions_test' is not defined

## === cell 16
submission.info()


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1198815387.py in <cell line: 0>()
----> 1 submission.info()

NameError: name 'submission' is not defined

## === cell 17
submission.to_csv('submission.csv', index=False)

print("The submission has been successfully saved.")


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3117316367.py in <cell line: 0>()
      1 # Saving the submission
----> 2 submission.to_csv('submission.csv', index=False)
      3 
      4 # Displaying the success message
      5 print("The submission has been successfully saved.")

NameError: name 'submission' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Submission is missing the following columns: {'severe_toxic', 'identity_hate', 'insult', 'threat', 'toxic', 'obscene'}
