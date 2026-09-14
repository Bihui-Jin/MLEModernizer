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

catboost==1.2.8
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
lightgbm==4.6.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
xgboost==2.0.3

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
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "/kaggle/input/tabular-playground-series-may-2022"
train_path = f"{DATA_DIR}/train.csv"
test_path = f"{DATA_DIR}/test.csv"
sample_path = f"{DATA_DIR}/sample_submission.csv"



## === cell 1
read_kwargs = dict(low_memory=False)
try:
    df_train = pd.read_csv(train_path, engine="pyarrow", **read_kwargs)
    df_test = pd.read_csv(test_path, engine="pyarrow", **read_kwargs)
    df_sampleSubmission = pd.read_csv(sample_path, engine="pyarrow", **read_kwargs)
except Exception:
    df_train = pd.read_csv(train_path, engine="c", **read_kwargs)
    df_test = pd.read_csv(test_path, engine="c", **read_kwargs)
    df_sampleSubmission = pd.read_csv(sample_path, engine="c", **read_kwargs)

if "kfolds" not in df_train.columns:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    df_train["kfolds"] = -1
    y_all = df_train["target"].to_numpy()
    for fold, (_, val_idx) in enumerate(skf.split(df_train, y_all)):
        df_train.loc[val_idx, "kfolds"] = fold

assert "target" in df_train.columns
assert "id" in df_train.columns and "id" in df_test.columns
assert df_train["kfolds"].min() == 0 and df_train["kfolds"].max() == 4



## === cell 2
fold_no = int(df_train["kfolds"].max() + 1)
fold_no



## === cell 3
feature_cols = [c for c in df_train.columns if c not in ["id", "target", "kfolds"]]
cat_cols = [c for c in feature_cols if df_train[c].dtype == "object"]
num_cols = [c for c in feature_cols if c not in cat_cols]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]
)
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop",
)

oof_preds = {
    1: np.zeros(len(df_train), dtype=np.float64),
    2: np.zeros(len(df_train), dtype=np.float64),
    3: np.zeros(len(df_train), dtype=np.float64),
}
test_preds = {
    1: np.zeros(len(df_test), dtype=np.float64),
    2: np.zeros(len(df_test), dtype=np.float64),
    3: np.zeros(len(df_test), dtype=np.float64),
}

lgb_params = dict(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
xgb_params = dict(
    n_estimators=900,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist",
)
logreg_params = dict(
    C=1.0,
    max_iter=200,
    n_jobs=-1,
    solver="lbfgs",
)



## === cell 4
from sklearn.base import clone
from scipy import sparse

X_all_raw = df_train[feature_cols]
X_te_raw = df_test[feature_cols]
y_all = df_train["target"].to_numpy()

preprocess_full = clone(preprocess)
preprocess_full.fit(X_all_raw, y_all)

X_te_full = preprocess_full.transform(X_te_raw)
if sparse.issparse(X_te_full) and not sparse.isspmatrix_csr(X_te_full):
    X_te_full = X_te_full.tocsr()

skf = StratifiedKFold(n_splits=fold_no, shuffle=True, random_state=RANDOM_STATE)
fold_indices = list(skf.split(X_all_raw, y_all))

for fold, (trn_idx, val_idx) in enumerate(fold_indices):
    X_tr_raw = X_all_raw.take(trn_idx)
    X_va_raw = X_all_raw.take(val_idx)
    y_tr = y_all[trn_idx]
    y_va = y_all[val_idx]

    X_tr = preprocess_full.transform(X_tr_raw)
    X_va = preprocess_full.transform(X_va_raw)
    X_te = X_te_full

    if sparse.issparse(X_tr) and not sparse.isspmatrix_csr(X_tr):
        X_tr = X_tr.tocsr()
    if sparse.issparse(X_va) and not sparse.isspmatrix_csr(X_va):
        X_va = X_va.tocsr()

    lgb_clf = lgb.LGBMClassifier(**lgb_params)
    lgb_clf.fit(X_tr, y_tr)
    p_va = lgb_clf.predict_proba(X_va)[:, 1]
    p_te = lgb_clf.predict_proba(X_te)[:, 1]
    oof_preds[1][val_idx] = p_va
    test_preds[1] += p_te / fold_no

    xgb_clf = XGBClassifier(
        **xgb_params,
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
    )
    xgb_clf.fit(X_tr, y_tr)
    p_va = xgb_clf.predict_proba(X_va)[:, 1]
    p_te = xgb_clf.predict_proba(X_te)[:, 1]
    oof_preds[2][val_idx] = np.clip(p_va, 0.0, 1.0)
    test_preds[2] += np.clip(p_te, 0.0, 1.0) / fold_no

    lr_clf = LogisticRegression(**logreg_params)
    lr_clf.fit(X_tr, y_tr)
    p_va = lr_clf.predict_proba(X_va)[:, 1]
    p_te = lr_clf.predict_proba(X_te)[:, 1]
    oof_preds[3][val_idx] = p_va
    test_preds[3] += p_te / fold_no

df_train["pred_1"] = oof_preds[1]
df_train["pred_2"] = oof_preds[2]
df_train["pred_3"] = oof_preds[3]

df_test["pred_1"] = test_preds[1]
df_test["pred_2"] = test_preds[2]
df_test["pred_3"] = test_preds[3]

for i in [1, 2, 3]:
    auc = roc_auc_score(df_train["target"], df_train[f"pred_{i}"])
    print(f"Base pred_{i} OOF AUC: {auc:.6f}")



## === cell 5
df_train.head()



## === cell 6
useful_features = ["pred_1", "pred_2", "pred_3"]
df_test = df_test[useful_features]



## === cell 7
final_predictions = []
final_valid_predictions = {}
scores = []

ids_all = df_train["id"].to_numpy()
y_all = df_train["target"].to_numpy()
folds_all = df_train["kfolds"].to_numpy()
X_meta_all = df_train[useful_features].to_numpy(dtype=np.float64, copy=False)
X_meta_test = df_test.to_numpy(dtype=np.float64, copy=False)

for fold in range(fold_no):
    tr_mask = folds_all != fold
    va_mask = ~tr_mask

    X_train = X_meta_all[tr_mask]
    y_train = y_all[tr_mask]
    X_valid = X_meta_all[va_mask]
    y_valid = y_all[va_mask]

    valid_ids = ids_all[va_mask]

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds_valid = model.predict(X_valid)
    test_preds_fold = model.predict(X_meta_test)

    final_predictions.append(test_preds_fold)
    final_valid_predictions.update(dict(zip(valid_ids.tolist(), preds_valid.tolist())))

    roc = roc_auc_score(y_valid, preds_valid)
    print(fold, roc)
    scores.append(roc)

print(np.mean(scores), np.std(scores))



## === cell 8
test_pred_mean = np.mean(np.column_stack(final_predictions), axis=1)
test_pred_mean = np.clip(test_pred_mean, 0.0, 1.0)

df_sampleSubmission["target"] = test_pred_mean
df_sampleSubmission.to_csv("submission.csv", index=False)

print(df_sampleSubmission.head())
print("Wrote submission.csv with shape:", df_sampleSubmission.shape)
