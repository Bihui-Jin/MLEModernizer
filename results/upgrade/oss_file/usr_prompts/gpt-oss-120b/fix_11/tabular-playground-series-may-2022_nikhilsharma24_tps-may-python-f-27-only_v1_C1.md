# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.63094

# 6. Current score

0.929

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50241) has done: 'The changes parallelize the expensive parts of the pipeline: Optuna trials now run on all CPU cores, and the five fold‑training loops are executed concurrently with a thread pool. This reduces wall‑clock time without altering any model architecture, hyper‑parameter search space, or evaluation logic, so the predictions remain identical (aside from negligible floating‑point ordering differences).'
- What this solution (achieved 0.929) has done: 'The changes lower the maximum number of trees from 4000 to 1500 in every XGBoost model (both the Optuna trial and the final fold training). This keeps the same algorithm, early‑stopping logic, and hyper‑parameter search space, but reduces the amount of work per model, allowing all folds to finish well within the 600‑second limit while preserving prediction accuracy. No other logic or I/O paths are altered.'

# 9. Code solution

## === cell 0
from sklearn import preprocessing, model_selection
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import optuna
import numpy as np
import pandas as pd
import string
import concurrent.futures  # retained for possible future use

best_model_global = None
best_valid_preds = None
best_test_preds = None
best_auc = None




## === cell 1
train = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/train.csv")
test = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/test.csv")




## === cell 2
print(train.shape)
print(test.shape)




## === cell 3
train["kfold"] = -1
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=102)
for fold, (train_indicies, valid_indicies) in enumerate(kf.split(X=train)):
    train.loc[valid_indicies, "kfold"] = fold




## === cell 4
df_tr = train.drop(columns=["id"])
df_te = test.drop(columns=["id"])

print(df_tr.shape)
print(df_te.shape)




## === cell 5
pd.crosstab(index=df_tr["target"], columns=df_tr["kfold"])




## === cell 6
df_tr.dtypes




## === cell 7
df_tr.f_27




## === cell 8
def count_alpha(df):
    df["f_27"] = df["f_27"].astype(str)
    for x in string.ascii_uppercase[:20]:
        df[f"count_{x}"] = df["f_27"].str.count(x)
    df = df.drop(columns="f_27")
    return df




## === cell 9
df_tr = count_alpha(df_tr)
df_te = count_alpha(df_te)




## === cell 10
print(df_tr.shape)
print(df_te.shape)




## === cell 11
df_tr.head()




## === cell 12
use_feature = [c for c in df_tr.columns if c not in ("target", "kfold")]




## === cell 13
X_all = df_tr[use_feature].astype(np.float32).values
y_all = df_tr["target"].values
folds = df_tr["kfold"].values
X_test = df_te[use_feature].astype(np.float32).values


def run(trial):
    fold = 0
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.25, log=True)
    reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-8, 100.0)
    reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-8, 100.0)
    subsample = trial.suggest_float("subsample", 0.1, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    max_depth = trial.suggest_int("max_depth", 1, 7)

    train_mask = folds != fold
    valid_mask = folds == fold

    xtrain = X_all[train_mask]
    ytrain = y_all[train_mask]
    xvalid = X_all[valid_mask]
    yvalid = y_all[valid_mask]

    model = XGBClassifier(
        random_state=42,
        tree_method="hist",
        n_estimators=1500,
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        max_depth=max_depth,
        eval_metric="auc",
        use_label_encoder=False,
        n_jobs=1,  # avoid oversubscription when folds run in parallel
    )
    model.fit(
        xtrain,
        ytrain,
        early_stopping_rounds=300,
        eval_set=[(xvalid, yvalid)],
        verbose=False,
    )
    global best_model_global, best_valid_preds, best_test_preds, best_auc
    if best_model_global is None or trial.number == 0:
        best_model_global = model
        best_valid_preds = model.predict_proba(xvalid)[:, 1]
        best_test_preds = model.predict_proba(X_test)[:, 1]
        best_auc = roc_auc_score(yvalid, best_valid_preds)

    preds_valid = model.predict_proba(xvalid)[:, 1]
    AUC = roc_auc_score(yvalid, preds_valid)
    return AUC




## === cell 14
study = optuna.create_study(direction="maximize")
study.optimize(run, n_trials=1, n_jobs=-1)  # parallelize trials




## === cell 15
final_predictions = []
scores = []


def train_fold(fold):
    """Train one fold and return its test predictions and validation AUC."""
    train_mask = folds != fold
    valid_mask = folds == fold

    xtrain = X_all[train_mask]
    ytrain = y_all[train_mask]
    xvalid = X_all[valid_mask]
    yvalid = y_all[valid_mask]

    params = study.best_params

    if fold == 0 and best_model_global is not None:
        preds_valid = best_valid_preds
        test_preds = best_test_preds
        ROC = best_auc
        return test_preds, ROC, ROC

    model = XGBClassifier(
        random_state=0,
        tree_method="hist",
        n_estimators=1500,
        eval_metric="auc",
        use_label_encoder=False,
        n_jobs=1,  # keep consistency with the trial model
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
    test_preds = model.predict_proba(X_test)
    ROC = roc_auc_score(yvalid, preds_valid)
    return test_preds, ROC, ROC  # duplicate ROC for easy unpacking


with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    future_to_fold = {executor.submit(train_fold, f): f for f in range(5)}
    fold_results = {}
    for future in concurrent.futures.as_completed(future_to_fold):
        f = future_to_fold[future]
        fold_results[f] = future.result()

for fold in range(5):
    test_preds, ROC, _ = fold_results[fold]
    final_predictions.append(test_preds)
    print("fold ROC:", ROC)
    scores.append(ROC)

print(np.mean(scores), np.std(scores))




## === cell 16
sample_submission = pd.read_csv(
    "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"
)
preds = np.mean(np.column_stack(final_predictions), axis=1)
sample_submission["target"] = preds
sample_submission.to_csv("submission.csv", index=False)
