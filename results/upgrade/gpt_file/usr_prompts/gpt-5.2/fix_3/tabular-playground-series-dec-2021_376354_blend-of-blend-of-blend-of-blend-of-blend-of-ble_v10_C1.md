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

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

DATA_DIR = "/kaggle/input/tabular-playground-series-dec-2021"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_PATH), f"Missing {TRAIN_PATH}"
assert os.path.exists(TEST_PATH), f"Missing {TEST_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing {SAMPLE_SUB_PATH}"




## === cell 1
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
submission = pd.read_csv(SAMPLE_SUB_PATH)

target_col = "Cover_Type"
id_col = "Id"

X = train.drop(columns=[target_col])
y = train[target_col].astype(int)

X_test = test.copy()

feature_cols = [c for c in X.columns if c != id_col]
X_feat = X[feature_cols]
X_test_feat = X_test[feature_cols]

try:
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_feat, y, test_size=0.02, random_state=RANDOM_STATE, stratify=y
    )
except ValueError:
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_feat, y, test_size=0.02, random_state=RANDOM_STATE, shuffle=True
    )




## === cell 2
models = []

models.append(
    (
        "lr_saga",
        Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(with_mean=False),
                ),  # sparse-like safety even though data is dense
                (
                    "clf",
                    LogisticRegression(
                        solver="saga",
                        multi_class="multinomial",
                        max_iter=200,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                        C=2.0,
                    ),
                ),
            ]
        ),
    )
)

models.append(
    (
        "lr_lbfgs",
        Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "clf",
                    LogisticRegression(
                        solver="lbfgs",
                        multi_class="multinomial",
                        max_iter=200,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                        C=1.0,
                    ),
                ),
            ]
        ),
    )
)

models.append(
    (
        "gnb",
        Pipeline(
            steps=[("scaler", StandardScaler(with_mean=False)), ("clf", GaussianNB())]
        ),
    )
)

models.append(
    (
        "rf",
        RandomForestClassifier(
            n_estimators=120,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    )
)

pred_matrix = []

for name, model in models:
    model.fit(X_tr, y_tr)
    preds = model.predict(X_test_feat).astype(int)
    pred_matrix.append(preds)

pred_matrix = np.vstack(pred_matrix).T




## === cell 3
def row_mode_int(arr_2d: np.ndarray) -> np.ndarray:
    arr_2d = np.asarray(arr_2d)
    if arr_2d.ndim != 2:
        raise ValueError(f"pred_matrix must be 2D, got shape={arr_2d.shape}")
    out = np.empty(arr_2d.shape[0], dtype=int)
    for i in range(arr_2d.shape[0]):
        vals, counts = np.unique(arr_2d[i], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


ensemble_pred = row_mode_int(pred_matrix).astype(int)

out = submission.copy()
out[id_col] = test[id_col].values  # ensure correct Id alignment
out[target_col] = ensemble_pred

assert list(out.columns) == [
    id_col,
    target_col,
], f"Unexpected submission columns: {out.columns.tolist()}"
assert len(out) == len(test), "Submission row count mismatch with test set"
assert out[target_col].notna().all(), "Found NaNs in predictions"

out.to_csv("submission.csv", index=False)
out.head()
