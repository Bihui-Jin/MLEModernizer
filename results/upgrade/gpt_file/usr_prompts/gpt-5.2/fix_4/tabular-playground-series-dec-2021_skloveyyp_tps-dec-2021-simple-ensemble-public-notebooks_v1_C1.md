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



## === cell 1
BASE_DIR_CANDIDATES = [
    "/kaggle/input/tabular-playground-series-dec-2021",
    "/kaggle/data/tabular-playground-series-dec-2021",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_DIR = None
for d in BASE_DIR_CANDIDATES:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
        os.path.join(d, "test.csv")
    ):
        DATA_DIR = d
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find train.csv/test.csv under expected Kaggle paths. "
        f"Tried: {BASE_DIR_CANDIDATES}"
    )

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_path, test_path, sample_sub_path



## === cell 2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

target_col = "Cover_Type"
id_col = "Id"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
submission = pd.read_csv(sample_sub_path)

X = train.drop(columns=[target_col])
y_raw = train[target_col]
X_test = test.copy()

feature_cols = [c for c in X.columns if c != id_col]

y_raw = pd.to_numeric(y_raw, errors="coerce")
if y_raw.isna().any():
    raise ValueError("Found NaN in target after coercion; cannot train.")
y_raw = y_raw.astype(int)

classes_sorted = np.sort(y_raw.unique())
class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
y = y_raw.map(class_to_idx).astype(np.int32)

X_feat = (
    X[feature_cols]
    .apply(pd.to_numeric, errors="coerce")
    .to_numpy(dtype=np.float32, copy=True)
)
X_test_feat = (
    X_test[feature_cols]
    .apply(pd.to_numeric, errors="coerce")
    .to_numpy(dtype=np.float32, copy=True)
)
y_arr = y.to_numpy(dtype=np.int32, copy=True)

test_ids = X_test[id_col].to_numpy(copy=False)

SEEDS = [0, 1, 2, 3, 4]  # 5 base predictors to vote
predictions = []

pd.Series(y_arr).value_counts().head(), len(classes_sorted)



## === cell 3
n = X_feat.shape[0]
all_idx = np.arange(n, dtype=np.int64)
splits = {}
for seed in SEEDS:
    try:
        tr_idx, val_idx = train_test_split(
            all_idx,
            test_size=0.10,
            random_state=seed,
            stratify=y_arr,
        )
    except ValueError:
        tr_idx, val_idx = train_test_split(
            all_idx,
            test_size=0.10,
            random_state=seed,
            stratify=None,
        )
    splits[seed] = (tr_idx, val_idx)


def _standardize_train_test(X_tr_raw: np.ndarray, X_te_raw: np.ndarray):
    mean_ = X_tr_raw.mean(axis=0, dtype=np.float64)
    var_ = X_tr_raw.var(axis=0, dtype=np.float64)  # ddof=0, same as StandardScaler
    scale_ = np.sqrt(var_, dtype=np.float64)
    scale_[scale_ == 0.0] = 1.0
    X_tr = (X_tr_raw - mean_).astype(np.float32, copy=False)
    X_tr /= scale_.astype(np.float32, copy=False)
    X_te = (X_te_raw - mean_).astype(np.float32, copy=False)
    X_te /= scale_.astype(np.float32, copy=False)
    return X_tr, X_te


for seed in SEEDS:
    tr_idx, val_idx = splits[seed]
    X_tr_raw = X_feat[tr_idx]
    y_tr = y_arr[tr_idx]

    _ = X_feat[val_idx]
    _ = y_arr[val_idx]

    X_tr, X_te = _standardize_train_test(X_tr_raw, X_test_feat)

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=200,
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(X_tr, y_tr)

    test_pred_idx = clf.predict(X_te).astype(np.int32, copy=False)
    p = pd.DataFrame({id_col: test_ids, target_col: test_pred_idx})
    predictions.append(p)

results = pd.DataFrame({id_col: test_ids})
for i, ds in enumerate(predictions):
    results[f"p{i+1}"] = ds[target_col].to_numpy(copy=False)

print(results.shape)
results.head()



## === cell 4
vote_cols = [c for c in results.columns if c.startswith("p")]
pred_mat = results[vote_cols].to_numpy(dtype=np.int32, copy=False)


def majority_vote_rows(a: np.ndarray) -> np.ndarray:
    out = np.empty(a.shape[0], dtype=np.int32)
    for i in range(a.shape[0]):
        vals, cnts = np.unique(a[i], return_counts=True)
        out[i] = vals[np.argmax(cnts)]
    return out


results["ensemble_idx"] = majority_vote_rows(pred_mat)
results["ensemble"] = results["ensemble_idx"].map(idx_to_class).astype(int)
results.head()




## === cell 5
def nunique(a, axis):
    return (np.diff(np.sort(a, axis=axis), axis=axis) != 0).sum(axis=axis) + 1




## === cell 6
results["dif"] = nunique(results[vote_cols].values, 1) - 1
results.head()



## === cell 7
results.dif.value_counts()



## === cell 8
submission[target_col] = results["ensemble"].astype(int).values
submission = submission[[id_col, target_col]]
submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 9
plt.figure(figsize=(10, 5))
ax = sns.countplot(x=submission[target_col])
plt.title("Predictions")
plt.xlabel("Cover Type")
ax.bar_label(ax.containers[0])
plt.show()
