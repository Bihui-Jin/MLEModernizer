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

os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 4))




## === cell 1
def unpack_zipfile(filename):
    """Unpacks zip-file by name from DATA_DIR to OUTPUT_DIR.

    Runtime optimization (correctness-preserving):
    - Skip unpacking if the target file already exists in OUTPUT_DIR to avoid repeated I/O.
    """
    out_name = filename[:-4] if filename.lower().endswith(".zip") else filename
    out_path = os.path.join(OUTPUT_DIR, out_name)
    if os.path.exists(out_path):
        print(f"Archive '{filename}' skipped (already unpacked).")
        return

    try:
        shutil.unpack_archive(
            filename=os.path.join(DATA_DIR, filename),
            extract_dir=OUTPUT_DIR,
            format="zip",
        )
    except Exception as e:
        print(e)
    else:
        print(f"Archive file '{filename}' has been unpacked successfully.")




## === cell 2
unpack_zipfile(filename="train.csv.zip")
unpack_zipfile(filename="test.csv.zip")
unpack_zipfile(filename="sample_submission.csv.zip")




## === cell 3
train_path = os.path.join(OUTPUT_DIR, "train.csv")
test_path = os.path.join(OUTPUT_DIR, "test.csv")

train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
target_cols = train_cols[2:]

usecols_train = ["id", "comment_text"] + target_cols
dtype_train = {c: "int8" for c in target_cols}

train_df = pd.read_csv(train_path, usecols=usecols_train, dtype=dtype_train)
test_df = pd.read_csv(test_path, usecols=["id", "comment_text"])




## === cell 4
corpus_train = train_df["comment_text"].fillna("").to_numpy(dtype=str)
corpus_train[:5]




## === cell 5
target_train = train_df[target_cols].to_numpy()
target_train[:5]




## === cell 6
corpus_test = test_df["comment_text"].fillna("").to_numpy(dtype=str)
corpus_test[:5]




## === cell 7
row_ids = np.arange(len(target_train), dtype=np.int32)

train_idx, y_train, val_idx, y_val = iterative_train_test_split(
    row_ids[:, np.newaxis],
    target_train,
    test_size=0.25,
)

train_idx_f = train_idx.ravel()
val_idx_f = val_idx.ravel()

x_train = corpus_train[train_idx_f]
x_val = corpus_train[val_idx_f]




## === cell 8
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




## === cell 9
model = CatBoostClassifier(
    iterations=5000,
    verbose=500,
    task_type="GPU",
    devices="0",
    loss_function="MultiLogloss",
    class_names=target_cols,
    random_seed=RANDOM_STATE,
    thread_count=(os.cpu_count() or 4),
)




## === cell 10
try:
    model.fit(
        pool_train,
        eval_set=pool_valid,
        early_stopping_rounds=200,
    )
except Exception as e:
    msg = str(e)
    if ("CUDA error 35" in msg) or ("CUDA driver version is insufficient" in msg):
        model.set_params(task_type="CPU", thread_count=(os.cpu_count() or 4))
        model.fit(
            pool_train,
            eval_set=pool_valid,
            early_stopping_rounds=200,
        )
    else:
        raise




## === cell 11
predictions_valid = model.predict(pool_valid)

for metric in ("Precision", "Recall", "F1"):
    print(metric)
    print(50 * "-")
    values = eval_metric(y_val, predictions_valid, metric)
    for cls, value in zip(model.classes_, values):
        print(f"class={cls}: {value:.4f}")
    print()




## === cell 12
pool_test = Pool(data=corpus_test, label=None, text_features=[0])




## === cell 13
proba_predictions_test = model.predict_proba(pool_test)




## === cell 14
submission = pd.DataFrame(
    proba_predictions_test,
    columns=target_cols,
    index=test_df.id,
).reset_index()

submission.head()




## === cell 15
submission.info()




## === cell 16
submission.to_csv("submission.csv", index=False)

print("The submission has been successfully saved.")

## --- ERROR in outputing the csv:
Invalid submission: Submission is missing the following columns: {'toxic', 'threat', 'obscene', 'insult', 'severe_toxic', 'identity_hate'}
