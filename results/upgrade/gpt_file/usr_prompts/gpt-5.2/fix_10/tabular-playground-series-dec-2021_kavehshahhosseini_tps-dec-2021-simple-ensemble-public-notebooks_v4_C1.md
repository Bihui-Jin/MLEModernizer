# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the class of a given image from a synthetic dataset.

## MetricMulti-class classification accuracy.

## Submission FormatFor each `Id` in the test set, you must predict the `Cover_Type` class. The file should contain a header and have the following format:
```
Id,Cover_Type
4000000,2
4000001,1
4000001,3
etc.
```

## Dataset 
- train.csv - the training data with the target `Cover_Type` column
- test.csv - the test set; you will be predicting the `Cover_Type` for each row in this file (the target integer class)
- sample_submission.csv - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

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
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        input/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        working/
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
```

-> data/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/tabular-playground-series-dec-2021/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> data/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> input/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

try:
    from sklearn import set_config

    set_config(
        enable_categorical=False, enable_hist_gradient_boosting=False, enable_hist=True
    )
except Exception:
    pass

DATA_DIR_CANDIDATES = [
    "/kaggle/input/tabular-playground-series-dec-2021",
    "/kaggle/input",
    "/kaggle/data/tabular-playground-series-dec-2021",
    "/kaggle/data",
]


def _find_file(filename: str) -> str:
    for d in DATA_DIR_CANDIDATES:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find {filename} in any of: {DATA_DIR_CANDIDATES}"
    )


train_path = _find_file("train.csv")
test_path = _find_file("test.csv")
sample_sub_path = _find_file("sample_submission.csv")

_train_head = pd.read_csv(train_path, nrows=5)
feature_cols_all = [c for c in _train_head.columns if c != "Cover_Type"]
dtype_map = {c: np.int16 for c in feature_cols_all if c != "Id"}
dtype_map["Id"] = np.int32
dtype_map["Cover_Type"] = np.int8

train_df = pd.read_csv(train_path, dtype=dtype_map)
test_df = pd.read_csv(
    test_path,
    dtype={
        c: dtype_map.get(c, np.int16) for c in feature_cols_all if c != "Cover_Type"
    },
)
submission = pd.read_csv(sample_sub_path, dtype={"Id": np.int32, "Cover_Type": np.int8})

assert "Cover_Type" in train_df.columns
assert "Id" in train_df.columns and "Id" in test_df.columns
assert list(submission.columns) == [
    "Id",
    "Cover_Type",
], "Sample submission must have columns: Id,Cover_Type"

X = train_df.drop(columns=["Cover_Type"])
y_raw = train_df["Cover_Type"].astype(int)

feature_cols = [c for c in X.columns if c != "Id"]
X_train_full = X[feature_cols]
X_test = test_df[feature_cols]

classes_sorted = np.sort(y_raw.unique())
class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
y = y_raw.map(class_to_idx).astype(int)

DO_VALIDATION_SCORING = False

if DO_VALIDATION_SCORING:
    try:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_train_full, y, test_size=0.02, random_state=SEED, stratify=y
        )
    except ValueError:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_train_full, y, test_size=0.02, random_state=SEED, stratify=None
        )
else:
    X_tr, y_tr = X_train_full, y

numeric_features = feature_cols
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
    ],
    remainder="drop",
)

models = [
    (
        "lr",
        LogisticRegression(
            max_iter=300,
            n_jobs=-1,
            multi_class="auto",
            solver="lbfgs",
            C=2.0,
            random_state=SEED,
        ),
    ),
    (
        "rf",
        RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            n_jobs=-1,
            random_state=SEED,
            class_weight=None,
        ),
    ),
    (
        "et",
        ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            random_state=SEED,
            class_weight=None,
        ),
    ),
]

X_tr_lr = preprocess.fit_transform(X_tr)
X_test_lr = preprocess.transform(X_test)

X_tr_tree = X_tr.to_numpy(copy=False)
X_test_tree = X_test.to_numpy(copy=False)

idx_to_class_arr = np.array(
    [idx_to_class[i] for i in range(len(idx_to_class))], dtype=np.int16
)

y_tr_np = (
    y_tr.to_numpy(dtype=np.int32, copy=False)
    if hasattr(y_tr, "to_numpy")
    else np.asarray(y_tr, dtype=np.int32)
)


def _fit_predict_one(name, clf):
    if name == "lr":
        X_fit, X_tst = X_tr_lr, X_test_lr
    else:
        X_fit, X_tst = X_tr_tree, X_test_tree

    clf.fit(X_fit, y_tr_np)
    test_pred_idx = clf.predict(X_tst).astype(np.int32, copy=False)
    return idx_to_class_arr[test_pred_idx]


predictions = [_fit_predict_one(name, clf) for name, clf in models]

val_scores = {}
val_scores if DO_VALIDATION_SCORING else {"validation_scoring": "skipped"}



## === cell 1
pred_mat = np.column_stack(predictions).astype(np.int32, copy=False)
print(pred_mat.shape)
pred_mat[:5]



## === cell 2
a, b, c = pred_mat[:, 0], pred_mat[:, 1], pred_mat[:, 2]
ensemble_mode = np.where(
    a == b, a, np.where(a == c, a, np.where(b == c, b, np.minimum(np.minimum(a, b), c)))
).astype(np.int32, copy=False)
ensemble_mode[:5]




## === cell 3
def nunique(a, axis):
    return (np.diff(np.sort(a, axis=axis), axis=axis) != 0).sum(axis=axis) + 1




## === cell 4
dif = nunique(pred_mat, 1) - 1
dif[:10], np.unique(dif, return_counts=True)



## === cell 5
submission = submission.copy()
submission = submission.set_index("Id").reindex(test_df["Id"]).reset_index()

assert len(submission) == len(test_df), "Submission rows must match test rows"
assert len(ensemble_mode) == len(test_df), "Ensemble predictions must match test rows"

submission["Cover_Type"] = ensemble_mode.astype(int, copy=False)
submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 6
pass
