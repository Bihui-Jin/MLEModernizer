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
import os
import shutil
import gc

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from catboost.utils import eval_metric
from skmultilearn.model_selection import iterative_train_test_split

DATA_DIR = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/"
OUTPUT_DIR = "/kaggle/working/"
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)




## === cell 1
def unpack_zipfile(filename):
    """Unpacks zip-file by name from DATA_DIR to OUTPUT_DIR."""
    zip_path = os.path.join(DATA_DIR, filename)
    try:
        shutil.unpack_archive(
            filename=zip_path,
            extract_dir=OUTPUT_DIR,
            format="zip",
        )
    except FileNotFoundError:
        print(f"Archive '{filename}' not found at {zip_path}. Skipping.")
    except Exception as e:
        print(f"Failed to unpack '{filename}': {e}")
    else:
        print(f"Archive file '{filename}' has been unpacked successfully.")




## === cell 2
def _ensure_extracted(csv_name: str, zip_name: str):
    out_csv = os.path.join(OUTPUT_DIR, csv_name)
    in_csv = os.path.join(DATA_DIR, csv_name)
    if os.path.exists(out_csv):
        print(f"Found extracted {csv_name} at {out_csv}; skipping unzip.")
        return
    if os.path.exists(in_csv):
        print(f"Found {csv_name} in input at {in_csv}; skipping unzip.")
        return
    unpack_zipfile(filename=zip_name)


_ensure_extracted("train.csv", "train.csv.zip")
_ensure_extracted("test.csv", "test.csv.zip")
_ensure_extracted("sample_submission.csv", "sample_submission.csv.zip")



## === cell 3
train_path = os.path.join(OUTPUT_DIR, "train.csv")
test_path = os.path.join(OUTPUT_DIR, "test.csv")
sample_path = os.path.join(OUTPUT_DIR, "sample_submission.csv")

if not os.path.exists(train_path):
    train_path = os.path.join(DATA_DIR, "train.csv")
if not os.path.exists(test_path):
    test_path = os.path.join(DATA_DIR, "test.csv")
if not os.path.exists(sample_path):
    sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

target_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train_df = pd.read_csv(
    train_path,
    usecols=["id", "comment_text"] + target_cols,
    dtype={
        "id": "object",
        "comment_text": "object",
        **{c: "int8" for c in target_cols},
    },
    low_memory=False,
)
test_df = pd.read_csv(
    test_path,
    usecols=["id", "comment_text"],
    dtype={"id": "object", "comment_text": "object"},
    low_memory=False,
)
sample_sub = pd.read_csv(
    sample_path,
    usecols=["id"] + target_cols,
    dtype={"id": "object", **{c: "float32" for c in target_cols}},
    low_memory=False,
)

train_df.head()



## === cell 4
target_cols = list(train_df.columns[2:])
assert target_cols == [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
], f"Unexpected target cols: {target_cols}"

corpus_train = train_df["comment_text"].fillna("").to_numpy(copy=False)
corpus_test = test_df["comment_text"].fillna("").to_numpy(copy=False)

target_train = np.ascontiguousarray(
    train_df[target_cols].to_numpy(dtype=np.int8, copy=False)
)

len(corpus_train), len(corpus_test), target_train.shape



## === cell 5
row_ids = np.arange(len(target_train), dtype=np.int32)

train_idx, y_train, val_idx, y_val = iterative_train_test_split(
    row_ids[:, np.newaxis],
    target_train,
    test_size=0.25,
)

train_idx_1d = np.ascontiguousarray(train_idx.ravel())
val_idx_1d = np.ascontiguousarray(val_idx.ravel())

x_train_1d = np.take(corpus_train, train_idx_1d, axis=0)
x_val_1d = np.take(corpus_train, val_idx_1d, axis=0)

y_train = np.ascontiguousarray(y_train)
y_val = np.ascontiguousarray(y_val)

y_train.shape, y_val.shape, x_train_1d.shape, x_val_1d.shape



## === cell 6
x_train_df = pd.DataFrame({"comment_text": x_train_1d})
x_val_df = pd.DataFrame({"comment_text": x_val_1d})

pool_train = Pool(
    data=x_train_df,
    label=y_train,
    text_features=["comment_text"],
)

pool_valid = Pool(
    data=x_val_df,
    label=y_val,
    text_features=["comment_text"],
)

del x_train_1d, x_val_1d, train_idx, val_idx, train_idx_1d, val_idx_1d
gc.collect()



## === cell 7
model = CatBoostClassifier(
    iterations=5000,
    verbose=0,  # keep
    task_type="CPU",
    loss_function="MultiLogloss",
    class_names=target_cols,
    random_seed=RANDOM_STATE,
    thread_count=-1,
    use_best_model=True,
    allow_writing_files=False,
    pinned_memory_size=2 * 1024**3,  # ignored if unsupported; safe to keep
)



## === cell 8
pool_train_q = pool_train.quantize()
pool_valid_q = pool_valid.quantize()
del pool_train, pool_valid
gc.collect()

model.fit(
    pool_train_q,
    eval_set=pool_valid_q,
    early_stopping_rounds=200,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
CatBoostError                             Traceback (most recent call last)
/tmp/ipykernel_11/2121361267.py in <cell line: 0>()
      6 gc.collect()
      7 
----> 8 model.fit(
      9     pool_train_q,
     10     eval_set=pool_valid_q,

/usr/local/lib/python3.11/dist-packages/catboost/core.py in fit(self, X, y, cat_features, text_features, embedding_features, graph, sample_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)
   5243             CatBoostClassifier._check_is_compatible_loss(params['loss_function'])
   5244 
-> 5245         self._fit(X, y, cat_features, text_features, embedding_features, None, graph, sample_weight, None, None, None, None, baseline, use_best_model,
   5246                   eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period,
   5247                   silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _fit(self, X, y, cat_features, text_features, embedding_features, pairs, graph, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)
   2388         with log_fixup(log_cout, log_cerr):
   2389             if X is None:
-> 2390                 raise CatBoostError("X must not be None")
   2391 
   2392             if y is None and not isinstance(X, PATH_TYPES + (Pool,)):

CatBoostError: X must not be None

## === cell 9
proba_valid_raw = model.predict_proba(pool_valid_q)

if isinstance(proba_valid_raw, list):
    proba_valid = np.column_stack([p[:, 1] for p in proba_valid_raw])
else:
    proba_valid = np.asarray(proba_valid_raw)

for metric in ("Precision", "Recall", "F1"):
    print(metric)
    print(50 * "-")
    values = eval_metric(y_val, proba_valid, metric)
    for cls, value in zip(model.classes_, values):
        print(f"class={cls}: {value:.4f}")
    print()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
CatBoostError                             Traceback (most recent call last)
/tmp/ipykernel_11/2999452654.py in <cell line: 0>()
----> 1 proba_valid_raw = model.predict_proba(pool_valid_q)
      2 
      3 if isinstance(proba_valid_raw, list):
      4     proba_valid = np.column_stack([p[:, 1] for p in proba_valid_raw])
      5 else:

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

## === cell 10
x_test_df = pd.DataFrame({"comment_text": corpus_test})
pool_test = Pool(data=x_test_df, label=None, text_features=["comment_text"])
pool_test_q = pool_test.quantize()
del pool_test, x_test_df
gc.collect()



## === cell 11
proba_raw = model.predict_proba(pool_test_q)

if isinstance(proba_raw, list):
    proba_predictions_test = np.column_stack([np.asarray(p)[:, 1] for p in proba_raw])
else:
    proba_predictions_test = np.asarray(proba_raw)

if proba_predictions_test.ndim == 3 and proba_predictions_test.shape[1] == 2:
    proba_predictions_test = proba_predictions_test[:, 1, :]
elif proba_predictions_test.ndim == 3 and proba_predictions_test.shape[2] == 2:
    proba_predictions_test = proba_predictions_test[:, :, 1]

proba_predictions_test = proba_predictions_test.astype(np.float32, copy=False)

assert proba_predictions_test.shape[0] == len(test_df), (
    proba_predictions_test.shape,
    len(test_df),
)
assert proba_predictions_test.shape[1] == len(target_cols), (
    proba_predictions_test.shape,
    len(target_cols),
)

proba_predictions_test.shape



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
CatBoostError                             Traceback (most recent call last)
/tmp/ipykernel_11/3835963360.py in <cell line: 0>()
----> 1 proba_raw = model.predict_proba(pool_test_q)
      2 
      3 if isinstance(proba_raw, list):
      4     proba_predictions_test = np.column_stack([np.asarray(p)[:, 1] for p in proba_raw])
      5 else:

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

## === cell 12
submission = pd.DataFrame({"id": test_df["id"].to_numpy(copy=False)})
submission[target_cols] = proba_predictions_test
submission = submission[["id"] + target_cols]
submission.head()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1978184817.py in <cell line: 0>()
      1 submission = pd.DataFrame({"id": test_df["id"].to_numpy(copy=False)})
----> 2 submission[target_cols] = proba_predictions_test
      3 submission = submission[["id"] + target_cols]
      4 submission.head()
      5 

NameError: name 'proba_predictions_test' is not defined

## === cell 13
assert list(submission.columns) == ["id"] + target_cols
assert len(submission) == len(test_df)
assert submission["id"].iloc[0] == test_df["id"].iloc[0]
submission.info()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/557355336.py in <cell line: 0>()
----> 1 assert list(submission.columns) == ["id"] + target_cols
      2 assert len(submission) == len(test_df)
      3 assert submission["id"].iloc[0] == test_df["id"].iloc[0]
      4 submission.info()
      5 

AssertionError: 

## === cell 14
out_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print(f"The submission has been successfully saved to: {out_path}")
print(submission.head())

## --- ERROR in outputing the csv:
Invalid submission: Submission is missing the following columns: {'threat', 'identity_hate', 'obscene', 'severe_toxic', 'toxic', 'insult'}
