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

# 5. Target score

0.9980200053055768

# 6. Current score

0.74032

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.74924) has done: 'Main timeout drivers are (1) fitting two separate preprocessors (sparse+dense) and re-transforming large matrices multiple times, and (2) slow single-threaded `GradientBoostingClassifier` on 800k rows. The changes below keep the exact same models, splits, and feature semantics, but eliminate redundant work by fitting imputers only once, reusing their fitted state in both pipelines, and avoiding repeated dense casts/copies. It also speeds up GBDT training without changing its algorithm by enabling histogram-based splitting (`HistGradientBoostingClassifier`), which is still Gradient Boosting for classification with the same loss semantics and is designed for large tabular data within tight time limits. All file paths, weights, and output format remain unchanged.'
- What this solution (achieved 0.74032) has done: 'Your current score (0.74924 AUC) is far below the target (0.99802), so we should increase performance while keeping the same overall approach (train/valid split, preprocessing, and the same three model families + weighted averaging). The biggest issue is that the `GaussianNB` branch is being trained on **ordinal-encoded categorical features**, which badly breaks the intended “naive Bayes over categorical” signal and tends to collapse AUC; switching that branch to a `CategoricalNB` trained on ordinal-encoded integer categories is a minimal, metric-aligned fix that preserves the ensemble design. Additionally, the numeric features in TPS May 2022 benefit a lot from using the provided `f_27` string as engineered numerical features; we add a small, standard feature extraction for `f_27` (split into 10 digits and 2 letters) without changing the modeling approach. These two changes should move AUC substantially toward the target while keeping runtime under the limit and still producing a valid submission CSV.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

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



## === cell 1
target_col = "target"
id_col = "id"

_tmp = pd.read_csv(train_path, nrows=5)
all_cols = _tmp.columns.tolist()
feature_cols = [c for c in all_cols if c != target_col]

cat_guess = [c for c in feature_cols if _tmp[c].dtype == "object"]
cat_cols = cat_guess

dtype_train = {c: np.float32 for c in feature_cols if c not in cat_cols and c != id_col}
dtype_train[id_col] = np.int32
dtype_train[target_col] = np.int8
for c in cat_cols:
    dtype_train[c] = "category"

dtype_test = {c: dtype_train.get(c, np.float32) for c in feature_cols}
dtype_test[id_col] = np.int32

usecols_train = feature_cols + [target_col]
usecols_test = feature_cols

train = pd.read_csv(train_path, dtype=dtype_train, usecols=usecols_train)
test = pd.read_csv(test_path, dtype=dtype_test, usecols=usecols_test)

X = train.drop(columns=[target_col])
y = train[target_col].astype(np.int32, copy=False)
X_test = test  # already a separate object

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols and c != id_col]

print("Rows train/test:", train.shape, test.shape)
print("Categorical cols:", cat_cols)
print("Numeric cols:", len(num_cols))




## === cell 2
def add_f27_features(df: pd.DataFrame) -> pd.DataFrame:
    if "f_27" not in df.columns:
        return df

    s = df["f_27"].astype(str)
    for i in range(10):
        df[f"f27_d{i}"] = pd.to_numeric(s.str[i], errors="coerce").astype(np.float32)
    df["f27_c0"] = s.str[10].astype("category")
    df["f27_c1"] = s.str[11].astype("category")
    return df


X = add_f27_features(X)
X_test = add_f27_features(X_test)

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols and c != id_col]

print("After f_27 FE -> categorical cols:", cat_cols)
print("After f_27 FE -> numeric cols:", len(num_cols))



## === cell 3
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.naive_bayes import CategoricalNB

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

num_imputer.fit(X_train[num_cols])
if len(cat_cols) > 0:
    cat_imputer.fit(X_train[cat_cols])

numeric_transformer = Pipeline(steps=[("imputer", num_imputer)])

categorical_transformer_sparse = Pipeline(
    steps=[
        ("imputer", cat_imputer),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]
)

preprocess_sparse = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer_sparse, cat_cols),
    ],
    remainder="drop",
    sparse_threshold=0.3,
    n_jobs=-1,
)

categorical_transformer_dense_ord = Pipeline(
    steps=[
        ("imputer", cat_imputer),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]
)

preprocess_dense_ord = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer_dense_ord, cat_cols),
    ],
    remainder="drop",
    sparse_threshold=0.0,
    n_jobs=-1,
)

Xtr_sparse = preprocess_sparse.fit_transform(X_train, y_train)
Xva_sparse = preprocess_sparse.transform(X_valid)

Xtr_dense_ord = preprocess_dense_ord.fit_transform(X_train, y_train)
Xva_dense_ord = preprocess_dense_ord.transform(X_valid)

Xtr_dense_gbdt = Xtr_dense_ord.astype(np.float32, copy=False)
Xva_dense_gbdt = Xva_dense_ord.astype(np.float32, copy=False)

from sklearn.preprocessing import KBinsDiscretizer

n_num = len(num_cols)
n_cat = len(cat_cols)

if n_num > 0:
    kbd = KBinsDiscretizer(n_bins=32, encode="ordinal", strategy="quantile")
    Xtr_num_b = kbd.fit_transform(Xtr_dense_ord[:, :n_num]).astype(np.int32, copy=False)
    Xva_num_b = kbd.transform(Xva_dense_ord[:, :n_num]).astype(np.int32, copy=False)
else:
    kbd = None
    Xtr_num_b = np.empty((Xtr_dense_ord.shape[0], 0), dtype=np.int32)
    Xva_num_b = np.empty((Xva_dense_ord.shape[0], 0), dtype=np.int32)

if n_cat > 0:
    Xtr_cat_i = Xtr_dense_ord[:, n_num:].astype(np.int32, copy=False) + 1
    Xva_cat_i = Xva_dense_ord[:, n_num:].astype(np.int32, copy=False) + 1
else:
    Xtr_cat_i = np.empty((Xtr_dense_ord.shape[0], 0), dtype=np.int32)
    Xva_cat_i = np.empty((Xva_dense_ord.shape[0], 0), dtype=np.int32)

Xtr_nb = np.hstack([Xtr_num_b, Xtr_cat_i]).astype(np.int32, copy=False)
Xva_nb = np.hstack([Xva_num_b, Xva_cat_i]).astype(np.int32, copy=False)

clf_logreg = LogisticRegression(
    max_iter=200, solver="lbfgs", n_jobs=-1, warm_start=True
)
clf_gbdt = HistGradientBoostingClassifier(random_state=42)
clf_nb = CategoricalNB()

clf_logreg.fit(Xtr_sparse, y_train)
clf_gbdt.fit(Xtr_dense_gbdt, y_train)
clf_nb.fit(Xtr_nb, y_train)



## === cell 4
from sklearn.metrics import roc_auc_score

p1_val = clf_logreg.predict_proba(Xva_sparse)[:, 1]
p2_val = clf_gbdt.predict_proba(Xva_dense_gbdt)[:, 1]
p3_val = clf_nb.predict_proba(Xva_nb)[:, 1]

weights = [0.05, 0.10, 0.85]
p_ens_val = weights[0] * p1_val + weights[1] * p2_val + weights[2] * p3_val
print("Valid AUC (logreg):      ", roc_auc_score(y_valid, p1_val))
print("Valid AUC (gbdt):        ", roc_auc_score(y_valid, p2_val))
print("Valid AUC (catnb):       ", roc_auc_score(y_valid, p3_val))
print("Valid AUC (ensemble):    ", roc_auc_score(y_valid, p_ens_val))



## === cell 5
Xte_sparse = preprocess_sparse.transform(X_test)

Xte_dense_ord = preprocess_dense_ord.transform(X_test)
Xte_dense_gbdt = Xte_dense_ord.astype(np.float32, copy=False)

if n_num > 0:
    Xte_num_b = kbd.transform(Xte_dense_ord[:, :n_num]).astype(np.int32, copy=False)
else:
    Xte_num_b = np.empty((Xte_dense_ord.shape[0], 0), dtype=np.int32)

if n_cat > 0:
    Xte_cat_i = Xte_dense_ord[:, n_num:].astype(np.int32, copy=False) + 1
else:
    Xte_cat_i = np.empty((Xte_dense_ord.shape[0], 0), dtype=np.int32)

Xte_nb = np.hstack([Xte_num_b, Xte_cat_i]).astype(np.int32, copy=False)

p1_test = clf_logreg.predict_proba(Xte_sparse)[:, 1]
p2_test = clf_gbdt.predict_proba(Xte_dense_gbdt)[:, 1]
p3_test = clf_nb.predict_proba(Xte_nb)[:, 1]

test_ids = test[id_col].to_numpy(copy=False)
grv_vsn = pd.DataFrame({id_col: test_ids, "target": p1_test})
gbdt = pd.DataFrame({id_col: test_ids, "target": p2_test})
nn = pd.DataFrame({id_col: test_ids, "target": p3_test})



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
        id_col: test_ids,
        "target": weights[0] * grv_vsn["target"].to_numpy(copy=False)
        + weights[1] * gbdt["target"].to_numpy(copy=False)
        + weights[2] * nn["target"].to_numpy(copy=False),
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
ens_idx = ensamble.set_index(id_col)["target"]
sub = sub[[id_col, "target"]]
sub_targets = ens_idx.reindex(sub[id_col].to_numpy(copy=False)).to_numpy()
if np.isnan(sub_targets).any():
    fill_value = float(np.nanmean(sub_targets))
    sub_targets = pd.Series(sub_targets).fillna(fill_value).to_numpy()
sub["target"] = sub_targets

out_path = "my_ensamble_052222.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub.head())
print(sub.tail())
