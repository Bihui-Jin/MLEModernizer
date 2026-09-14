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

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The script is streamlined by removing the unnecessary scikit‑multilearn install, using sklearn’s fast `train_test_split`, suppressing verbose output during CatBoost training, and constructing the submission DataFrame explicitly to guarantee all required columns are present in the correct order. These changes keep the model architecture, loss, and training logic unchanged while eliminating extra overhead and fixing the missing‑column error, allowing the whole pipeline to complete well within the 600‑second limit.'
- What this solution (achieved 0.5) has done: 'Implemented pool reuse to avoid rebuilding CatBoost pools for every label, which eliminates costly repeated tokenization and data handling. Added a safety check to guarantee all required submission columns exist, preserving correctness while keeping the original modeling logic unchanged. The changes are confined to cells handling data preparation and model training/prediction.'

# 9. Code solution

## === cell 0
import os
import shutil

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split  # fast split

DATA_DIR = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/"
OUTPUT_DIR = "/kaggle/working/"
RANDOM_STATE = 42




## === cell 1
def unpack_zipfile(filename):
    """Unpacks a zip file from DATA_DIR into OUTPUT_DIR."""
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




## === cell 2
unpack_zipfile(filename="train.csv.zip")
unpack_zipfile(filename="test.csv.zip")
unpack_zipfile(filename="sample_submission.csv.zip")




## === cell 3
train_df = pd.read_csv(OUTPUT_DIR + "train.csv")
test_df = pd.read_csv(OUTPUT_DIR + "test.csv")




## === cell 4
corpus_train = train_df["comment_text"]  # Series of strings




## === cell 5
target_cols = list(train_df.columns[2:])  # ['toxic', 'severe_toxic', ...]
target_train = train_df[target_cols].values




## === cell 6
corpus_test = test_df["comment_text"]  # Series of strings




## === cell 7
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




## === cell 8
from sklearn.metrics import roc_auc_score

train_pool = Pool(data=x_train_df, text_features=[0])
val_pool = Pool(data=x_val_df, text_features=[0])
pool_test = Pool(data=pd.DataFrame({"comment_text": corpus_test}), text_features=[0])

test_pred_proba = np.zeros((len(corpus_test), len(target_cols)))




## === cell 9
for i, col in enumerate(target_cols):
    y_train_i = y_train[:, i]
    y_val_i = y_val[:, i]

    model = CatBoostClassifier(
        iterations=1500,
        verbose=0,
        loss_function="Logloss",
        thread_count=-1,
        random_seed=RANDOM_STATE,
        early_stopping_rounds=100,
    )

    model.fit(
        train_pool,
        y=y_train_i,
        eval_set=(val_pool, y_val_i),
        verbose=False,
    )

    val_pred = model.predict_proba(val_pool)[:, 1]
    auc = roc_auc_score(y_val_i, val_pred)
    print(f"{col} validation AUC: {auc:.4f}")

    test_pred_proba[:, i] = model.predict_proba(pool_test)[:, 1]




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
CatBoostError                             Traceback (most recent call last)
/tmp/ipykernel_11/1673479796.py in <cell line: 0>()
     13 
     14     # Fit using the pre‑built pools; label arrays are passed separately.
---> 15     model.fit(
     16         train_pool,
     17         y=y_train_i,

/usr/local/lib/python3.11/dist-packages/catboost/core.py in fit(self, X, y, cat_features, text_features, embedding_features, graph, sample_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)
   5243             CatBoostClassifier._check_is_compatible_loss(params['loss_function'])
   5244 
-> 5245         self._fit(X, y, cat_features, text_features, embedding_features, None, graph, sample_weight, None, None, None, None, baseline, use_best_model,
   5246                   eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period,
   5247                   silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _fit(self, X, y, cat_features, text_features, embedding_features, pairs, graph, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)
   2393                 raise CatBoostError("y may be None only when X is an instance of catboost.Pool or string")
   2394 
-> 2395             train_params = self._prepare_train_params(
   2396                 X=X, y=y, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features,
   2397                 pairs=pairs, graph=graph, sample_weight=sample_weight, group_id=group_id, group_weight=group_weight,

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _prepare_train_params(self, X, y, cat_features, text_features, embedding_features, pairs, graph, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks)
   2273         embedding_features = _process_feature_indices(embedding_features, X, params, 'embedding_features')
   2274 
-> 2275         train_pool = _build_train_pool(X, y, cat_features, text_features, embedding_features, pairs, graph,
   2276                                        sample_weight, group_id, group_weight, subgroup_id, pairs_weight,
   2277                                        baseline, column_description)

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _build_train_pool(X, y, cat_features, text_features, embedding_features, pairs, graph, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, column_description)
   1503             )
   1504         if (not X.has_label()) and X.num_pairs() == 0:
-> 1505             raise CatBoostError("Label in X has not been initialized.")
   1506         if y is not None:
   1507             raise CatBoostError("Incorrect value of y: X is catboost.Pool object, y must be initialized inside catboost.Pool.")

CatBoostError: Label in X has not been initialized.

## === cell 10
submission = pd.DataFrame(test_pred_proba, columns=target_cols)
submission.insert(0, "id", test_df["id"].values)

required = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
for col in required:
    if col not in submission.columns:
        submission[col] = 0.0

submission = submission[["id"] + required]




## === cell 11
submission.to_csv("submission.csv", index=False)
print("The submission has been successfully saved as 'submission.csv'.")
