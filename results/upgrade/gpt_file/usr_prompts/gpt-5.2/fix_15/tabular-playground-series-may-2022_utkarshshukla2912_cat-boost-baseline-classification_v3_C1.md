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
base_dir = "/kaggle/input/tabular-playground-series-may-2022"
if not os.path.exists(os.path.join(base_dir, "train.csv")):
    base_dir = "/kaggle/input"

train_path = os.path.join(base_dir, "train.csv")
test_path = os.path.join(base_dir, "test.csv")
sub_path = os.path.join(base_dir, "sample_submission.csv")

feature_cols = sorted(set(categorical_columns + real_value_columns))
usecols_train = ["id", "target"] + feature_cols
usecols_test = ["id"] + feature_cols

dtype_train_full = dict(dtype_train)
dtype_test_full = dict(dtype_test)
for c in real_value_columns:
    if c in feature_cols:
        dtype_train_full[c] = np.float32
        dtype_test_full[c] = np.float32

train_df = pd.read_csv(
    train_path, dtype=dtype_train_full, usecols=usecols_train, memory_map=True
)
test_df = pd.read_csv(
    test_path, dtype=dtype_test_full, usecols=usecols_test, memory_map=True
)
sample_sub = pd.read_csv(sub_path, memory_map=True)




## === cell 4
X_df = train_df[feature_cols]
y = train_df["target"].to_numpy()

cat_features = [c for c in categorical_columns if c in feature_cols]
cat_feature_indices = [feature_cols.index(c) for c in cat_features]

rs = np.random.RandomState(42)
n = len(train_df)
val_frac = 0.21
val_size = int(n * val_frac)

perm = rs.permutation(n)
val_idx = perm[:val_size]
train_idx = perm[val_size:]

train_pool = Pool(
    data=X_df,
    label=y,
    cat_features=cat_feature_indices,
    feature_names=feature_cols,
    row_indices=train_idx,
)
val_pool = Pool(
    data=X_df,
    label=y,
    cat_features=cat_feature_indices,
    feature_names=feature_cols,
    row_indices=val_idx,
)

time_left = max(
    1,
    int(
        WALL_TIME_BUDGET_SECONDS - (time.time() - _START_TIME) - _SAFETY_MARGIN_SECONDS
    ),
)

max_iterations = 10000

iter_cap = int(min(max_iterations, max(500, time_left * 12)))

model = CatBoostClassifier(
    cat_features=cat_features,
    iterations=iter_cap,
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
    subsample=0.66,  # was 1.0 (ineffective for Bernoulli); enables intended bootstrap
    score_function="L2",
    od_type="Iter",
    od_wait=500,
    allow_writing_files=False,
)

model.fit(train_pool, eval_set=val_pool, use_best_model=True)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/169290993.py in <cell line: 0>()
     17 train_idx = perm[val_size:]
     18 
---> 19 train_pool = Pool(
     20     data=X_df,
     21     label=y,

TypeError: Pool.__init__() got an unexpected keyword argument 'row_indices'

## === cell 5
PRINT_CLASSIFICATION_REPORT = False
if PRINT_CLASSIFICATION_REPORT:
    idx = np.arange(len(y))
    rs = np.random.RandomState(42)
    sel = rs.choice(idx, size=min(200000, len(y)), replace=False)
    X_sel = X_df.iloc[sel]
    print(classification_report(y[sel], model.predict(X_sel)))




## === cell 6
PLOT_FEATURE_IMPORTANCE = False
if PLOT_FEATURE_IMPORTANCE:
    fi = model.get_feature_importance()
    sns.lineplot(x=model.feature_names_, y=fi, marker="o")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()




## === cell 7
X_test_df = test_df[feature_cols]

test_pool = Pool(
    X_test_df,
    cat_features=cat_feature_indices,
    feature_names=feature_cols,
)
prediction_proba = model.predict_proba(test_pool)[:, 1]




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2690778190.py in <cell line: 0>()
      7     feature_names=feature_cols,
      8 )
----> 9 prediction_proba = model.predict_proba(test_pool)[:, 1]
     10 
     11 

NameError: name 'model' is not defined

## === cell 8
submission = sample_sub.copy()
submission["target"] = prediction_proba.astype(float)

if "id" in submission.columns and "id" in test_df.columns:
    sub_ids = submission["id"].to_numpy()
    test_ids = test_df["id"].to_numpy()
    if sub_ids.shape != test_ids.shape or not np.array_equal(sub_ids, test_ids):
        tmp = pd.DataFrame({"id": test_ids, "target": prediction_proba.astype(float)})
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

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/774965271.py in <cell line: 0>()
      1 submission = sample_sub.copy()
----> 2 submission["target"] = prediction_proba.astype(float)
      3 
      4 if "id" in submission.columns and "id" in test_df.columns:
      5     sub_ids = submission["id"].to_numpy()

NameError: name 'prediction_proba' is not defined
