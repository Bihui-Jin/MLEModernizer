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
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split  # fast split
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool as MP_Pool, cpu_count

DATA_DIR = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/"
OUTPUT_DIR = "/kaggle/working/"
RANDOM_STATE = 42




## === cell 1
def unpack_zipfile(filename):
    """Unpacks a zip file from DATA_DIR into OUTPUT_DIR only if not already extracted."""
    target_path = os.path.join(OUTPUT_DIR, filename.replace(".zip", ""))
    if os.path.isdir(target_path) or os.path.isfile(target_path):
        print(f"Archive '{filename}' already unpacked; skipping.")
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
train_df = pd.read_csv(os.path.join(OUTPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(OUTPUT_DIR, "test.csv"))




## === cell 4
corpus_train = train_df["comment_text"]  # Series of strings
corpus_test = test_df["comment_text"]  # Series of strings
target_cols = list(train_df.columns[2:])  # ['toxic', 'severe_toxic', ...]
target_train = train_df[target_cols].values




## === cell 5
train_idx, val_idx = train_test_split(
    np.arange(len(target_train)),
    test_size=0.25,
    random_state=RANDOM_STATE,
    shuffle=True,
)

x_train_df = pd.DataFrame({"comment_text": corpus_train.iloc[train_idx]})
x_val_df = pd.DataFrame({"comment_text": corpus_train.iloc[val_idx]})
y_train = target_train[train_idx]
y_val = target_train[val_idx]




## === cell 6
text_feat = [0]  # column index for text feature
pool_test = Pool(
    data=pd.DataFrame({"comment_text": corpus_test}), text_features=text_feat
)
test_pred_proba = np.zeros((len(corpus_test), len(target_cols)))




## === cell 7
label_args = [(i, col) for i, col in enumerate(target_cols)]
threads_per_model = max(1, cpu_count() // len(label_args))


def train_one_label(args):
    """Fit CatBoost for a single label; returns (col_index, auc, test_pred_column)."""
    i, col_name = args
    y_train_i = y_train[:, i]
    y_val_i = y_val[:, i]

    model = CatBoostClassifier(
        iterations=1500,
        verbose=0,
        loss_function="Logloss",
        thread_count=threads_per_model,  # limited threads per model
        random_seed=RANDOM_STATE,
        early_stopping_rounds=100,
    )

    model.fit(
        x_train_df,
        y_train_i,
        text_features=text_feat,
        eval_set=[(x_val_df, y_val_i)],
        verbose=False,
    )

    val_pred = model.predict_proba(x_val_df)[:, 1]
    auc = roc_auc_score(y_val_i, val_pred)
    print(f"{col_name} validation AUC: {auc:.4f}")

    test_pred = model.predict_proba(pool_test)[:, 1]
    return i, auc, test_pred


with MP_Pool(processes=len(label_args)) as pool:
    results = pool.map(train_one_label, label_args)

for i, auc, test_pred in results:
    test_pred_proba[:, i] = test_pred




## === cell 8
required = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
submission = pd.DataFrame(test_pred_proba, columns=required)
submission.insert(0, "id", test_df["id"].values)
submission = submission[["id"] + required]




## === cell 9
submission.to_csv("submission.csv", index=False)
print("The submission has been successfully saved as 'submission.csv'.")

## --- ERROR in outputing the csv:
Invalid submission: Submission is missing the following columns: {'threat', 'identity_hate', 'insult', 'severe_toxic', 'obscene', 'toxic'}
