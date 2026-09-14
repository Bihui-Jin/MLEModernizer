# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))
CPU_COUNT = os.cpu_count() or 4
THREADS = min(8, CPU_COUNT)
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(THREADS))

pd.options.mode.copy_on_write = True




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

y_train = np.ascontiguousarray(y_train)
y_val = np.ascontiguousarray(y_val)

y_train.shape, y_val.shape, train_idx_1d.shape, val_idx_1d.shape




## === cell 6
cb_cache_dir = os.path.join(OUTPUT_DIR, "catboost_info_cache")
os.makedirs(cb_cache_dir, exist_ok=True)

pool_train = Pool(
    data=corpus_train[train_idx_1d],
    label=y_train,
    text_features=[0],
)

pool_valid = Pool(
    data=corpus_train[val_idx_1d],
    label=y_val,
    text_features=[0],
)

del train_idx, val_idx, train_idx_1d, val_idx_1d
gc.collect()




## === cell 7
model = CatBoostClassifier(
    iterations=5000,
    verbose=0,  # keep
    task_type="CPU",
    loss_function="MultiLogloss",
    class_names=target_cols,
    random_seed=RANDOM_STATE,
    thread_count=THREADS,
    use_best_model=True,
    allow_writing_files=True,
    train_dir=cb_cache_dir,
    save_snapshot=True,
    snapshot_file=os.path.join(cb_cache_dir, "snapshot"),
    snapshot_interval=60,
    od_type="Iter",
    od_wait=200,
)




## === cell 8
model.fit(
    pool_train,
    eval_set=pool_valid,
)




## === cell 9
proba_valid_raw = model.predict_proba(pool_valid)

if isinstance(proba_valid_raw, list):
    proba_valid = np.asarray([p[:, 1] for p in proba_valid_raw], dtype=np.float32).T
else:
    proba_valid = np.asarray(proba_valid_raw)

if proba_valid.ndim == 3 and proba_valid.shape[1] == 2:
    proba_valid = proba_valid[:, 1, :]
elif proba_valid.ndim == 3 and proba_valid.shape[2] == 2:
    proba_valid = proba_valid[:, :, 1]

assert proba_valid.shape[0] == y_val.shape[0], (proba_valid.shape, y_val.shape)
assert proba_valid.shape[1] == len(target_cols), (proba_valid.shape, len(target_cols))

RUN_VALID_METRICS = False
if RUN_VALID_METRICS:
    for metric in ("Precision", "Recall", "F1"):
        print(metric)
        print(50 * "-")
        values = eval_metric(y_val, proba_valid, metric)
        for cls, value in zip(model.classes_, values):
            print(f"class={cls}: {value:.4f}")
        print()




## === cell 10
pool_test = Pool(data=corpus_test, label=None, text_features=[0])
gc.collect()




## === cell 11
proba_raw = model.predict_proba(pool_test)

if isinstance(proba_raw, list):
    proba_predictions_test = np.asarray(
        [p[:, 1] for p in proba_raw], dtype=np.float32
    ).T
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




## === cell 12
submission = sample_sub.copy()
submission["id"] = test_df["id"].to_numpy(copy=False)
submission[target_cols] = proba_predictions_test
submission = submission[["id"] + target_cols]
submission.head()




## === cell 13
assert list(submission.columns) == ["id"] + target_cols
assert len(submission) == len(test_df)
assert submission["id"].iloc[0] == test_df["id"].iloc[0]
submission.info()




## === cell 14
out_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print(f"The submission has been successfully saved to: {out_path}")
print(submission.head())
