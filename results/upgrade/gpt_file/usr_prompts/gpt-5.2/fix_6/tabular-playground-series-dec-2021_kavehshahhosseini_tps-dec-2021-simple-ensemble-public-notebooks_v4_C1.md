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
import numpy as np
import pandas as pd

from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

SEED = 42
np.random.seed(SEED)



## === cell 1
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

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

try:
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train_full, y, test_size=0.02, random_state=SEED, stratify=y
    )
except ValueError:
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train_full, y, test_size=0.02, random_state=SEED, stratify=None
    )

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
X_va_lr = preprocess.transform(X_va)
X_test_lr = preprocess.transform(X_test)

X_tr_tree = X_tr.to_numpy(copy=False)
X_va_tree = X_va.to_numpy(copy=False)
X_test_tree = X_test.to_numpy(copy=False)

idx_to_class_arr = np.array(
    [idx_to_class[i] for i in range(len(idx_to_class))], dtype=int
)

predictions = []
val_scores = {}

y_tr_np = y_tr.to_numpy() if hasattr(y_tr, "to_numpy") else np.asarray(y_tr)
y_va_np = y_va.to_numpy() if hasattr(y_va, "to_numpy") else np.asarray(y_va)

for name, clf in models:
    if name == "lr":
        X_fit, X_val, X_tst = X_tr_lr, X_va_lr, X_test_lr
    else:
        X_fit, X_val, X_tst = X_tr_tree, X_va_tree, X_test_tree

    clf.fit(X_fit, y_tr_np)

    va_pred = clf.predict(X_val)
    val_scores[name] = accuracy_score(y_va_np, va_pred)

    test_pred_idx = clf.predict(X_tst).astype(int)
    test_pred = idx_to_class_arr[test_pred_idx]

    predictions.append(test_pred)

val_scores



## === cell 2
results = pd.DataFrame(index=np.arange(len(test_df)))
model_cols = []

for i, pred in enumerate(predictions):
    col = f"p{i+1}"
    results[col] = pred
    model_cols.append(col)

print(results.shape)
results.head()



## === cell 3
pred_mat = results[model_cols].to_numpy(dtype=np.int32, copy=False)
a, b, c = pred_mat[:, 0], pred_mat[:, 1], pred_mat[:, 2]
ensemble_mode = np.where(
    a == b, a, np.where(a == c, a, np.where(b == c, b, np.minimum(np.minimum(a, b), c)))
)
results["ensemble"] = ensemble_mode.astype(int)
results.head()




## === cell 4
def nunique(a, axis):
    return (np.diff(np.sort(a, axis=axis), axis=axis) != 0).sum(axis=axis) + 1




## === cell 5
results["dif"] = nunique(results[model_cols].values, 1) - 1
results.head()



## === cell 6
results.dif.value_counts()



## === cell 7
submission = submission.copy()
submission = submission.set_index("Id").reindex(test_df["Id"]).reset_index()

assert len(submission) == len(test_df), "Submission rows must match test rows"
assert "ensemble" in results.columns and len(results) == len(
    test_df
), "Ensemble predictions must match test rows"

submission["Cover_Type"] = results["ensemble"].astype(int).values
submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 8
plt.figure(figsize=(10, 5))
ax = sns.countplot(x=submission.Cover_Type)
plt.title("Predictions")
plt.xlabel("Cover Type")
ax.bar_label(ax.containers[0])
plt.show()
