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
seaborn==0.12.2
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

0.85605

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import time

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

plt.rcParams["figure.figsize"] = (20, 10)
plt.style.use("ggplot")

np.random.seed(42)

WALL_TIME_BUDGET_SECONDS = 600
_START_TIME = time.time()
_SAFETY_MARGIN_SECONDS = 20




## === cell 1
LIST_INPUT_FILES = False
if LIST_INPUT_FILES:
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            print(os.path.join(dirname, filename))




## === cell 2
categorical_columns = [
    f"f_{i}" if len(str(i)) == 2 else f"f_0{i}" for i in range(7, 19)
] + ["f_27", "f_29", "f_30"]
real_value_columns = [f"f_0{i}" for i in range(0, 7)] + [
    f"f_{i}" for i in range(19, 29)
]
real_value_columns.pop(real_value_columns.index("f_27"))

dtype_train = {c: "category" for c in categorical_columns}
dtype_test = {c: "category" for c in categorical_columns}




## === cell 3
train_path = "/kaggle/input/tabular-playground-series-may-2022/train.csv"
test_path = "/kaggle/input/tabular-playground-series-may-2022/test.csv"
sub_path = "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"

feature_cols = sorted(set(categorical_columns + real_value_columns))
usecols_train = ["id", "target"] + feature_cols
usecols_test = ["id"] + feature_cols

train_df = pd.read_csv(
    train_path, dtype=dtype_train, usecols=usecols_train, memory_map=True
)
test_df = pd.read_csv(
    test_path, dtype=dtype_test, usecols=usecols_test, memory_map=True
)
sample_sub = pd.read_csv(sub_path, memory_map=True)




## === cell 4
X_df = train_df[feature_cols]
y = train_df["target"].to_numpy()

cat_features = [c for c in categorical_columns if c in feature_cols]
cat_feature_indices = [feature_cols.index(c) for c in cat_features]

real_feature_indices = [
    feature_cols.index(c) for c in real_value_columns if c in feature_cols
]
cat_feature_indices_sorted = sorted(cat_feature_indices)

X_num = X_df.iloc[:, real_feature_indices].to_numpy(copy=False)

if len(cat_feature_indices_sorted) > 0:
    X_cat_codes = np.empty((len(X_df), len(cat_feature_indices_sorted)), dtype=np.int32)
    for j, col_idx in enumerate(cat_feature_indices_sorted):
        col = X_df.iloc[:, col_idx]
        X_cat_codes[:, j] = col.cat.codes.to_numpy(copy=False).astype(
            np.int32, copy=False
        )
else:
    X_cat_codes = None

if X_cat_codes is None:
    X_all = X_num
    cat_feature_indices_for_pool = []
    feature_names_for_pool = [feature_cols[i] for i in real_feature_indices]
else:
    n_rows = len(X_df)
    n_cols = len(feature_cols)
    X_all = np.empty((n_rows, n_cols), dtype=np.float64)
    for k, col_idx in enumerate(real_feature_indices):
        X_all[:, col_idx] = X_num[:, k]
    for j, col_idx in enumerate(cat_feature_indices_sorted):
        X_all[:, col_idx] = X_cat_codes[:, j]
    cat_feature_indices_for_pool = cat_feature_indices_sorted
    feature_names_for_pool = feature_cols




## === cell 5
rs = np.random.RandomState(42)
n = len(train_df)
val_frac = 0.21
val_size = int(n * val_frac)

perm = rs.permutation(n)
val_idx = perm[:val_size]
train_idx = perm[val_size:]

X_train = X_all[train_idx]
y_train = y[train_idx]
X_val = X_all[val_idx]
y_val = y[val_idx]

train_pool = Pool(
    X_train,
    y_train,
    cat_features=cat_feature_indices_for_pool,
    feature_names=feature_names_for_pool,
)
val_pool = Pool(
    X_val,
    y_val,
    cat_features=cat_feature_indices_for_pool,
    feature_names=feature_names_for_pool,
)

time_left = max(
    1,
    int(
        WALL_TIME_BUDGET_SECONDS - (time.time() - _START_TIME) - _SAFETY_MARGIN_SECONDS
    ),
)

model = CatBoostClassifier(
    cat_features=cat_feature_indices_for_pool,
    n_estimators=10000,
    learning_rate=0.10315154739037707,
    depth=3,
    l2_leaf_reg=1,
    task_type="CPU",
    eval_metric="AUC",
    loss_function="Logloss",
    verbose=300,
    random_seed=42,
    thread_count=-1,
    bootstrap_type="Bernoulli",
    subsample=1.0,
    score_function="L2",
    od_type="Iter",
    od_wait=500,
    allow_writing_files=False,
    has_time=True,
    time_left=time_left,
)

model.fit(train_pool, eval_set=val_pool, use_best_model=True)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
CatBoostError                             Traceback (most recent call last)
/tmp/ipykernel_11/2131883306.py in <cell line: 0>()
     14 y_val = y[val_idx]
     15 
---> 16 train_pool = Pool(
     17     X_train,
     18     y_train,

/usr/local/lib/python3.11/dist-packages/catboost/core.py in __init__(self, data, label, cat_features, text_features, embedding_features, embedding_features_data, column_description, pairs, graph, delimiter, has_header, ignore_csv_quoting, weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, timestamp, feature_names, feature_tags, thread_count, log_cout, log_cerr, data_can_be_none)
    795                     elif isinstance(data, np.ndarray):
    796                         if (data.dtype.kind == 'f') and (cat_features is not None) and (len(cat_features) > 0):
--> 797                             raise CatBoostError(
    798                                 "'data' is numpy array of floating point numerical type, it means no categorical features,"
    799                                 " but 'cat_features' parameter specifies nonzero number of categorical features"

CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features

## === cell 6
PRINT_CLASSIFICATION_REPORT = False
if PRINT_CLASSIFICATION_REPORT:
    idx = np.arange(len(y))
    rs = np.random.RandomState(42)
    sel = rs.choice(idx, size=min(200000, len(y)), replace=False)
    print(classification_report(y[sel], model.predict(X_all[sel])))




## === cell 7
PLOT_FEATURE_IMPORTANCE = False
if PLOT_FEATURE_IMPORTANCE:
    fi = model.get_feature_importance()
    sns.lineplot(x=model.feature_names_, y=fi, marker="o")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()




## === cell 8
X_test_df = test_df[feature_cols]

X_test_num = X_test_df.iloc[:, real_feature_indices].to_numpy(copy=False)
if len(cat_feature_indices_sorted) > 0:
    X_test_cat_codes = np.empty(
        (len(X_test_df), len(cat_feature_indices_sorted)), dtype=np.int32
    )
    for j, col_idx in enumerate(cat_feature_indices_sorted):
        col = X_test_df.iloc[:, col_idx]
        X_test_cat_codes[:, j] = col.cat.codes.to_numpy(copy=False).astype(
            np.int32, copy=False
        )

    n_rows_t = len(X_test_df)
    n_cols = len(feature_cols)
    X_test_all = np.empty((n_rows_t, n_cols), dtype=np.float64)
    for k, col_idx in enumerate(real_feature_indices):
        X_test_all[:, col_idx] = X_test_num[:, k]
    for j, col_idx in enumerate(cat_feature_indices_sorted):
        X_test_all[:, col_idx] = X_test_cat_codes[:, j]
else:
    X_test_all = X_test_num

test_pool = Pool(
    X_test_all,
    cat_features=cat_feature_indices_for_pool,
    feature_names=feature_names_for_pool,
)
prediction_proba = model.predict_proba(test_pool)[:, 1]




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
CatBoostError                             Traceback (most recent call last)
/tmp/ipykernel_11/679039158.py in <cell line: 0>()
     23     X_test_all = X_test_num
     24 
---> 25 test_pool = Pool(
     26     X_test_all,
     27     cat_features=cat_feature_indices_for_pool,

/usr/local/lib/python3.11/dist-packages/catboost/core.py in __init__(self, data, label, cat_features, text_features, embedding_features, embedding_features_data, column_description, pairs, graph, delimiter, has_header, ignore_csv_quoting, weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, timestamp, feature_names, feature_tags, thread_count, log_cout, log_cerr, data_can_be_none)
    795                     elif isinstance(data, np.ndarray):
    796                         if (data.dtype.kind == 'f') and (cat_features is not None) and (len(cat_features) > 0):
--> 797                             raise CatBoostError(
    798                                 "'data' is numpy array of floating point numerical type, it means no categorical features,"
    799                                 " but 'cat_features' parameter specifies nonzero number of categorical features"

CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features

## === cell 9
submission = sample_sub.copy()
submission["target"] = prediction_proba.astype(float)

if "id" in submission.columns and "id" in test_df.columns:
    sub_ids = submission["id"].to_numpy()
    test_ids = test_df["id"].to_numpy()
    if sub_ids.shape != test_ids.shape or not np.array_equal(sub_ids, test_ids):
        tmp = pd.DataFrame({"id": test_ids, "target": prediction_proba})
        submission = submission[["id"]].merge(tmp, on="id", how="left")

submission = submission[["id", "target"]]
submission.to_csv("submission.csv", index=False)
print(submission.head())
print(
    "Wrote submission.csv with columns:",
    submission.columns.tolist(),
    "and shape:",
    submission.shape,
)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1518177713.py in <cell line: 0>()
      1 submission = sample_sub.copy()
----> 2 submission["target"] = prediction_proba.astype(float)
      3 
      4 if "id" in submission.columns and "id" in test_df.columns:
      5     sub_ids = submission["id"].to_numpy()

NameError: name 'prediction_proba' is not defined
