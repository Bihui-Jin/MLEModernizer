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

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)



## === cell 1
BASE_DIR_CANDIDATES = [
    "/kaggle/input/tabular-playground-series-dec-2021",
    "/kaggle/data/tabular-playground-series-dec-2021",
    "/kaggle/input",
    "/kaggle/data",
]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


base_dir = first_existing(BASE_DIR_CANDIDATES)
if base_dir is None:
    raise FileNotFoundError("Could not locate Kaggle input data directory.")

train_path = os.path.join(base_dir, "train.csv")
test_path = os.path.join(base_dir, "test.csv")
sample_path = os.path.join(base_dir, "sample_submission.csv")

for p in [train_path, test_path, sample_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing expected file: {p}")

train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()


def _dtype_map(cols, has_target):
    dtypes = {}
    for c in cols:
        if c == "Id":
            dtypes[c] = np.int32
        elif has_target and c == "Cover_Type":
            dtypes[c] = np.int8
        else:
            dtypes[c] = np.float32
    return dtypes


train = pd.read_csv(
    train_path, dtype=_dtype_map(train_cols, has_target=True), engine="c"
)
test = pd.read_csv(test_path, dtype=_dtype_map(test_cols, has_target=False), engine="c")
submission = pd.read_csv(
    sample_path, dtype={"Id": np.int32, "Cover_Type": np.int8}, engine="c"
)

assert "Cover_Type" in train.columns, "train.csv must contain Cover_Type"
assert (
    "Id" in train.columns and "Id" in test.columns and "Id" in submission.columns
), "Id column missing"
assert len(test) == len(submission), "sample_submission and test row counts must match"

train.shape, test.shape, submission.shape



## === cell 2
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X = train.drop(columns=["Cover_Type"])
y = train["Cover_Type"].astype(int)

X_test = test[X.columns]

num_cols = X.columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ]
            ),
            num_cols,
        )
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

clf = LogisticRegression(
    multi_class="multinomial",
    solver="saga",
    C=2.0,
    max_iter=200,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

model = Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])

subset_n = 200_000  # fixed size; deterministic stratified sampling below
if len(X) > subset_n:
    rng = np.random.RandomState(RANDOM_STATE)
    y_values = y.values
    unique, counts = np.unique(y_values, return_counts=True)
    per_class = np.maximum(1, (counts / counts.sum() * subset_n).astype(int))
    diff = subset_n - per_class.sum()
    if diff != 0:
        order = np.argsort(-counts)
        step = 1 if diff > 0 else -1
        for k in range(abs(diff)):
            per_class[order[k % len(order)]] += step

    idx_parts = []
    for cls, n_take in zip(unique, per_class):
        cls_idx = np.flatnonzero(y_values == cls)
        take = rng.choice(cls_idx, size=min(n_take, cls_idx.size), replace=False)
        idx_parts.append(take)
    subset_idx = np.concatenate(idx_parts)
    rng.shuffle(subset_idx)

    X_s = X.iloc[subset_idx]
    y_s = y.iloc[subset_idx]
else:
    X_s, y_s = X, y

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(model, X_s, y_s, cv=cv, scoring="accuracy", n_jobs=1)
print(
    f"Sanity CV accuracy on subset (3-fold): {cv_scores.mean():.6f} ± {cv_scores.std():.6f}"
)



## === cell 3
model.fit(X, y)

test_pred = model.predict(X_test).astype(int)
test_pred = np.clip(test_pred, 1, 7)



## === cell 4
sub = submission.copy()
if not np.array_equal(sub["Id"].values, test["Id"].values):
    pred_df = pd.DataFrame({"Id": test["Id"].values, "Cover_Type": test_pred})
    sub = sub[["Id"]].merge(pred_df, on="Id", how="left")
else:
    sub["Cover_Type"] = test_pred

assert sub.shape[0] == submission.shape[0]
assert list(sub.columns) == ["Id", "Cover_Type"]
assert sub["Cover_Type"].notna().all()

sub.to_csv("submission.csv", index=False)
sub.head()



## === cell 5
pass
