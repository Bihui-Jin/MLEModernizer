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

0.91948

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the correlation heatmap crash by computing correlations on numeric columns only (the dataset contains a string categorical feature). I fix the CatBoost GPU crash by switching training to CPU while keeping the same model/core training loop, so it runs in this Kaggle environment. I also ensure we generate test predictions by averaging the 5 fold models (same CV approach, just making inference consistent) and then write a valid `submission.csv` with `id,target`. These changes are necessary for end-to-end execution and should yield a reasonable AUC without altering the intended modeling approach.'
- What this solution (achieved 0.5) has done: 'Most of the timeout comes from training 5 CatBoost models with 1500 trees each and doing extra work per fold (Pool slicing, repeated evaluation plumbing, and suboptimal CPU threading). I keep the exact same model, CV, and prediction logic, but reduce overhead by (1) using CatBoost’s built-in CV split assignment so we train directly on the full Pool without per-fold Pool slicing, and (2) enabling Intel-accelerated scikit-learn where available for the ROC AUC computations. I also set CatBoost parameters that remove non-essential runtime costs (no best-model shrinking overhead, deterministic-friendly settings, and explicit CPU thread control) without changing training semantics. File paths, folds, iterations, early stopping, and prediction averaging remain identical.'
- What this solution (achieved 0.5) has done: 'The timeout is dominated by 5× CatBoost training plus repeated Pool construction and pandas slicing overhead on 800k rows. I keep the exact same model, folds, and evaluation, but reduce overhead by (1) using numpy arrays for splits and creating Pools via row indices (no per-fold dataframe materialization), (2) converting feature matrices once to contiguous numpy (float32) while keeping `f_27` categorical intact, and (3) ensuring CatBoost uses the already-built full Pools with `set_feature_names` implicitly handled. This preserves the same learning algorithm and metrics while cutting CPU and memory churn significantly. I also avoid unnecessary `.head()` calls and extra copies that cost time but don’t affect results.'

# 9. Code solution

## === cell 0
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

N_THREADS = int(os.environ.get("OMP_NUM_THREADS", "8"))

try:
    from sklearnex import patch_sklearn  # type: ignore

    patch_sklearn()
except Exception:
    pass

np.random.seed(RANDOM_STATE)



## === cell 1
TRAIN_PATH = r"../input/tabular-playground-series-may-2022/train.csv"

train_dtypes = {"id": "int32", "target": "int8", "f_27": "category"}
train_cols = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
for c in train_cols:
    if c not in train_dtypes and c != "target":
        train_dtypes[c] = "float32"

train = pd.read_csv(
    TRAIN_PATH,
    dtype=train_dtypes,
    engine="c",
    low_memory=False,
)



## === cell 2
TEST_PATH = r"../input/tabular-playground-series-may-2022/test.csv"
test_dtypes = {"id": "int32", "f_27": "category"}
test_cols = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()
for c in test_cols:
    if c not in test_dtypes:
        test_dtypes[c] = "float32"

test = pd.read_csv(
    TEST_PATH,
    dtype=test_dtypes,
    engine="c",
    low_memory=False,
)



## === cell 3
sub = pd.read_csv(r"../input/tabular-playground-series-may-2022/sample_submission.csv")



## === cell 4
train_id = train["id"].copy()
test_id = test["id"].copy()

train.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)



## === cell 5
print(f"train set have {train.shape[0]} rows and {train.shape[1]} columns.")
print(f"test set have {test.shape[0]} rows and {test.shape[1]} columns.")
print(f"sample_submission set have {sub.shape[0]} rows and {sub.shape[1]} columns.")



## === cell 6
pass



## === cell 7
pass



## === cell 8
pass



## === cell 9
cat = ["f_27"]
X = train.drop("target", axis=1)
y = train["target"]

for c in cat:
    if c in X.columns and c in test.columns:
        all_cats = pd.Categorical(pd.concat([X[c], test[c]], axis=0)).categories
        X[c] = pd.Categorical(X[c], categories=all_cats)
        test[c] = pd.Categorical(test[c], categories=all_cats)
    elif c in X.columns:
        X[c] = X[c].astype("category")
    elif c in test.columns:
        test[c] = test[c].astype("category")



## === cell 10
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

cat_features_idx = [X.columns.get_loc(c) for c in cat]

folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof = np.zeros(len(X), dtype=np.float64)
test_pred = np.zeros(len(test), dtype=np.float64)

y_np = y.to_numpy(copy=False)

num_cols = [c for c in X.columns if c not in cat]
X_num_np = np.ascontiguousarray(X[num_cols].to_numpy(dtype=np.float32, copy=False))
test_num_np = np.ascontiguousarray(
    test[num_cols].to_numpy(dtype=np.float32, copy=False)
)

if cat:
    X_cat = X[cat[0]]
    test_cat = test[cat[0]]
    X_all = pd.DataFrame(X_num_np, columns=num_cols)
    X_all[cat[0]] = X_cat
    test_all = pd.DataFrame(test_num_np, columns=num_cols)
    test_all[cat[0]] = test_cat
else:
    X_all = pd.DataFrame(X_num_np, columns=num_cols)
    test_all = pd.DataFrame(test_num_np, columns=num_cols)

full_train_pool = Pool(X_all, y_np, cat_features=cat_features_idx)
full_test_pool = Pool(test_all, cat_features=cat_features_idx)

for fold, (tr_idx, val_idx) in enumerate(folds.split(X_all)):
    print(f"Fold: {fold}")

    train_pool = full_train_pool.slice(tr_idx)
    valid_pool = full_train_pool.slice(val_idx)

    model = CatBoostClassifier(
        n_estimators=1500,
        task_type="CPU",
        bootstrap_type="Bayesian",
        random_seed=RANDOM_STATE,
        loss_function="Logloss",
        eval_metric="AUC",
        thread_count=N_THREADS,
        allow_writing_files=False,
        use_best_model=True,
        grow_policy="SymmetricTree",
        sampling_frequency="PerTree",
    )

    model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=400,
        verbose=False,
    )

    y_val = y_np[val_idx]
    y_pred = model.predict_proba(valid_pool)[:, 1]
    oof[val_idx] = y_pred
    roc = roc_auc_score(y_val, y_pred)
    print(f" roc_auc_score: {roc}")
    print("-" * 50)

    test_pred += model.predict_proba(full_test_pool)[:, 1] / folds.n_splits

print(f"Overall OOF ROC AUC: {roc_auc_score(y_np, oof)}")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
CatBoostError                             Traceback (most recent call last)
_catboost.pyx in _catboost.get_cat_factor_bytes_representation()

_catboost.pyx in _catboost.get_id_object_bytes_string_representation()

CatBoostError: bad object for id: 158.8207244873047

During handling of the above exception, another exception occurred:

CatBoostError                             Traceback (most recent call last)
/tmp/ipykernel_11/1845952123.py in <cell line: 0>()
     33     test_all = pd.DataFrame(test_num_np, columns=num_cols)
     34 
---> 35 full_train_pool = Pool(X_all, y_np, cat_features=cat_features_idx)
     36 full_test_pool = Pool(test_all, cat_features=cat_features_idx)
     37 

/usr/local/lib/python3.11/dist-packages/catboost/core.py in __init__(self, data, label, cat_features, text_features, embedding_features, embedding_features_data, column_description, pairs, graph, delimiter, has_header, ignore_csv_quoting, weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, timestamp, feature_names, feature_tags, thread_count, log_cout, log_cerr, data_can_be_none)
    853                         )
    854 
--> 855                     self._init(data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight,
    856                                group_id, group_weight, subgroup_id, pairs_weight, baseline, timestamp, feature_names, feature_tags, thread_count)
    857             elif not data_can_be_none:

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _init(self, data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, timestamp, feature_names, feature_tags, thread_count)
   1489         if feature_tags is not None:
   1490             feature_tags = self._check_transform_tags(feature_tags, feature_names)
-> 1491         self._init_pool(data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight,
   1492                         group_id, group_weight, subgroup_id, pairs_weight, baseline, timestamp, feature_names, feature_tags, thread_count)
   1493 

_catboost.pyx in _catboost._PoolBase._init_pool()

_catboost.pyx in _catboost._PoolBase._init_pool()

_catboost.pyx in _catboost._PoolBase._init_features_order_layout_pool()

_catboost.pyx in _catboost._set_features_order_data_pd_data_frame()

_catboost.pyx in _catboost.get_cat_factor_bytes_representation()

CatBoostError: Invalid type for cat_feature[non-default value idx=0,feature_idx=27]=158.8207244873047 : cat_features must be integer or string, real number values and NaN values should be converted to string.

## === cell 11
pred = test_pred



## === cell 12
sub = sub.copy()
sub["id"] = test_id.values  # keep exact ordering from test read
sub["target"] = pred.astype(float)
sub = sub[["id", "target"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
