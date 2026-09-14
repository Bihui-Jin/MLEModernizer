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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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
import numpy as np
import pandas as pd
import os

_ = "/kaggle/input"



## === cell 1
BASE_DIR = "/kaggle/input/tabular-playground-series-may-2022"
train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")
sample_path = os.path.join(BASE_DIR, "sample_submission.csv")

if not (
    os.path.exists(train_path)
    and os.path.exists(test_path)
    and os.path.exists(sample_path)
):
    BASE_DIR = "/kaggle/data/tabular-playground-series-may-2022"
    train_path = os.path.join(BASE_DIR, "train.csv")
    test_path = os.path.join(BASE_DIR, "test.csv")
    sample_path = os.path.join(BASE_DIR, "sample_submission.csv")

print("Using BASE_DIR:", BASE_DIR)
print("Train exists:", os.path.exists(train_path))
print("Test exists:", os.path.exists(test_path))
print("Sample exists:", os.path.exists(sample_path))



## === cell 2
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target_col = "target"
id_col = "id"

X = train.drop(columns=[target_col])
y = train[target_col].astype(int)
X_test = test.copy()

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols and c != id_col]

for c in num_cols:
    if X[c].dtype.kind in "fc":
        X[c] = X[c].astype(np.float32, copy=False)
        X_test[c] = X_test[c].astype(np.float32, copy=False)

print("Rows train/test:", train.shape, test.shape)
print("Categorical cols:", cat_cols)
print("Numeric cols:", len(num_cols))



## === cell 3
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer_sparse = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]
)

categorical_transformer_dense = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocess_sparse = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer_sparse, cat_cols),
    ],
    remainder="drop",
    sparse_threshold=0.3,
)

preprocess_dense = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer_dense, cat_cols),
    ],
    remainder="drop",
    sparse_threshold=0.0,  # force dense output deterministically
)

Xtr_sparse = preprocess_sparse.fit_transform(X_train, y_train)
Xva_sparse = preprocess_sparse.transform(X_valid)
Xte_sparse = preprocess_sparse.transform(X_test)

Xtr_dense = preprocess_dense.fit_transform(X_train, y_train)
Xva_dense = preprocess_dense.transform(X_valid)
Xte_dense = preprocess_dense.transform(X_test)

clf_logreg = LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=-1)
clf_gbdt = GradientBoostingClassifier(random_state=42)
clf_gnb = GaussianNB()

clf_logreg.fit(Xtr_sparse, y_train)
clf_gbdt.fit(np.asarray(Xtr_dense), y_train)
clf_gnb.fit(np.asarray(Xtr_dense), y_train)



## === cell 4
from sklearn.metrics import roc_auc_score

p1_val = clf_logreg.predict_proba(Xva_sparse)[:, 1]
p2_val = clf_gbdt.predict_proba(np.asarray(Xva_dense))[:, 1]
p3_val = clf_gnb.predict_proba(np.asarray(Xva_dense))[:, 1]

weights = [0.05, 0.10, 0.85]
p_ens_val = weights[0] * p1_val + weights[1] * p2_val + weights[2] * p3_val
print("Valid AUC (grv_vsn/logreg):", roc_auc_score(y_valid, p1_val))
print("Valid AUC (gbdt):        ", roc_auc_score(y_valid, p2_val))
print("Valid AUC (nn/gnb):      ", roc_auc_score(y_valid, p3_val))
print("Valid AUC (ensemble):    ", roc_auc_score(y_valid, p_ens_val))



## === cell 5
p1_test = clf_logreg.predict_proba(Xte_sparse)[:, 1]
p2_test = clf_gbdt.predict_proba(np.asarray(Xte_dense))[:, 1]
p3_test = clf_gnb.predict_proba(np.asarray(Xte_dense))[:, 1]

grv_vsn = pd.DataFrame({id_col: test[id_col].values, "target": p1_test})
gbdt = pd.DataFrame({id_col: test[id_col].values, "target": p2_test})
nn = pd.DataFrame({id_col: test[id_col].values, "target": p3_test})



## === cell 6
grv_vsn.head()



## === cell 7
gbdt.head()



## === cell 8
nn.head()



## === cell 9
assert grv_vsn[id_col].equals(gbdt[id_col]) and grv_vsn[id_col].equals(
    nn[id_col]
), "ID mismatch between model predictions"



## === cell 10
print(grv_vsn["target"].describe())
print(gbdt["target"].describe())
print(nn["target"].describe())



## === cell 11
weights = [0.05, 0.10, 0.85]
ensamble = pd.DataFrame(
    {
        id_col: grv_vsn[id_col].values,
        "target": weights[0] * grv_vsn["target"].values
        + weights[1] * gbdt["target"].values
        + weights[2] * nn["target"].values,
    }
)



## === cell 12
ensamble.head()



## === cell 13
ensamble["target"] = ensamble["target"].clip(0.0, 1.0)



## === cell 14
sub = pd.read_csv(sample_path)
print(sub.head())
print("Sample rows:", len(sub), "Test rows:", len(test))



## === cell 15
sub = sub[[id_col, "target"]].merge(
    ensamble, on=id_col, how="left", suffixes=("_sample", "")
)
sub["target"] = sub["target"].fillna(sub["target_sample"])
sub = sub[[id_col, "target"]]

out_path = "my_ensamble_052222.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub.head())
print(sub.tail())
