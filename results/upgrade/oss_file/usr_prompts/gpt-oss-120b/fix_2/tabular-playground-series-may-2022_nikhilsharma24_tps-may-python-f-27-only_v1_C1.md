# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given simulated manufacturing control data, predict whether the machine is in state `0` or state `1`.

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
900000,0.65
900001,0.97
900002,0.02
etc.
```

## Dataset
- **train.csv** - the training data, which includes normalized continuous data and categorical data
- **test.csv** - the test set; your task is to predict binary `target` variable which represents the state of a manufacturing process
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        input/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        working/
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
```

-> data/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/tabular-playground-series-may-2022/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> data/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> input/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import string
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
from sklearn import preprocessing, model_selection
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import optuna




## === cell 2
train = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/train.csv")
test = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/test.csv")




## === cell 3
print(train.shape)
print(test.shape)




## === cell 4
train["kfold"] = -1
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=102)
for fold, (train_indicies, valid_indicies) in enumerate(kf.split(X=train)):
    train.loc[valid_indicies, "kfold"] = fold




## === cell 5
df_tr = train[["f_27", "target", "kfold"]]
df_te = test[["f_27"]]

print(df_tr.shape)
print(df_te.shape)




## === cell 6
pd.crosstab(index=df_tr["target"], columns=df_tr["kfold"])




## === cell 7
df_tr.dtypes




## === cell 8
df_tr.f_27




## === cell 9
def count_alpha(df):
    df["f_27"] = df["f_27"].astype(str)
    for x in string.ascii_uppercase[:20]:
        df[f"count_{x}"] = df["f_27"].str.count(x)
    df = df.drop(columns="f_27")
    return df




## === cell 10
df_tr = count_alpha(df_tr)
df_te = count_alpha(df_te)




## === cell 11
print(df_tr.shape)
print(df_te.shape)




## === cell 12
df_tr.head()




## === cell 13
use_feature = [c for c in df_tr.columns if c not in ("target", "kfold")]




## === cell 14
def run(trial):
    fold = 0
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.25, log=True)
    reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-8, 100.0)
    reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-8, 100.0)
    subsample = trial.suggest_float("subsample", 0.1, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    max_depth = trial.suggest_int("max_depth", 1, 7)

    xtrain = df_tr[df_tr.kfold != fold].reset_index(drop=True)
    xvalid = df_tr[df_tr.kfold == fold].reset_index(drop=True)

    ytrain = xtrain.target
    yvalid = xvalid.target

    xtrain = xtrain[use_feature]
    xvalid = xvalid[use_feature]

    model = XGBClassifier(
        random_state=42,
        tree_method="hist",  # CPU‑compatible
        n_estimators=7000,
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        max_depth=max_depth,
        eval_metric="auc",
        use_label_encoder=False,
    )
    model.fit(
        xtrain,
        ytrain,
        early_stopping_rounds=300,
        eval_set=[(xvalid, yvalid)],
        verbose=False,
    )
    preds_valid = model.predict_proba(xvalid)[:, 1]
    AUC = roc_auc_score(yvalid, preds_valid)
    return AUC




## === cell 15
study = optuna.create_study(direction="maximize")
study.optimize(run, n_trials=5)




## === cell 16
final_predictions = []
scores = []

for fold in range(5):
    xtrain = df_tr[df_tr.kfold != fold].reset_index(drop=True)
    xvalid = df_tr[df_tr.kfold == fold].reset_index(drop=True)

    ytrain = xtrain.target
    yvalid = xvalid.target

    xtrain = xtrain[use_feature]
    xvalid = xvalid[use_feature]
    xtest = df_te[use_feature]

    params = study.best_params

    model = XGBClassifier(
        random_state=0,
        tree_method="hist",
        n_estimators=5000,
        eval_metric="auc",
        use_label_encoder=False,
        **params,
    )

    model.fit(
        xtrain,
        ytrain,
        early_stopping_rounds=300,
        eval_set=[(xvalid, yvalid)],
        verbose=False,
    )
    preds_valid = model.predict_proba(xvalid)[:, 1]
    test_preds = model.predict_proba(xtest)
    final_predictions.append(test_preds)
    ROC = roc_auc_score(yvalid, preds_valid)
    print(fold, ROC)
    scores.append(ROC)

print(np.mean(scores), np.std(scores))




## === cell 17
sample_submission = pd.read_csv(
    "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"
)
preds = np.mean(np.column_stack([row[:, 1] for row in final_predictions]), axis=1)
sample_submission["target"] = preds
sample_submission.to_csv("submission.csv", index=False)
