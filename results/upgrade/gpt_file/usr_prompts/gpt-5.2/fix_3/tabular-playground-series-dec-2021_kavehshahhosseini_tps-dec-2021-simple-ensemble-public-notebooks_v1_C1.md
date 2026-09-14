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

DATA_DIR = "/kaggle/input/tabular-playground-series-dec-2021"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")



## === cell 1
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
submission = pd.read_csv(SAMPLE_SUB_PATH)

assert "Cover_Type" in train.columns
assert "Id" in train.columns and "Id" in test.columns and "Id" in submission.columns

X = train.drop(columns=["Cover_Type"])
y = train["Cover_Type"].astype(int)

X_test = test.copy()

feature_cols = [c for c in X.columns if c != "Id"]
X = X[feature_cols]
X_test = X_test[feature_cols]

print("Train shape:", X.shape, "Test shape:", X_test.shape, "Num classes:", y.nunique())
print("Class counts (min/max):", y.value_counts().min(), y.value_counts().max())



## === cell 2
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score

models = []

models.append(
    (
        "lr",
        Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=200,
                        n_jobs=-1,
                        multi_class="auto",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    )
)

models.append(
    (
        "rf",
        RandomForestClassifier(
            n_estimators=250, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE
        ),
    )
)

models.append(
    (
        "et",
        ExtraTreesClassifier(
            n_estimators=400, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE
        ),
    )
)

models.append(
    ("gnb", Pipeline(steps=[("scaler", StandardScaler()), ("clf", GaussianNB())]))
)

val_scores = {}
try:
    min_count = int(y.value_counts().min())
    if min_count < 2:
        raise ValueError(
            f"Too few samples in at least one class for stratify (min_count={min_count})."
        )

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y
    )
except ValueError as e:
    print("Warning:", e)
    print(
        "Falling back to non-stratified train_test_split for sanity-check validation only."
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_STATE, shuffle=True
    )

for name, clf in models:
    clf.fit(X_tr, y_tr)
    pred_va = clf.predict(X_va)
    val_scores[name] = accuracy_score(y_va, pred_va)

val_scores



## === cell 3
results = pd.DataFrame(index=np.arange(len(X_test)))

trained_models = []
for i, (name, clf) in enumerate(models, start=1):
    clf.fit(X, y)
    trained_models.append((name, clf))
    results[f"p{i}"] = clf.predict(X_test).astype(int)

print("Results shape:", results.shape)
results.head()



## === cell 4
mode_res = stats.mode(results.to_numpy(), axis=1, keepdims=False)
ensemble = np.asarray(mode_res.mode).astype(int).reshape(-1)

results["ensemble"] = ensemble
results.head()



## === cell 5
submission = submission.copy()
submission["Cover_Type"] = results["ensemble"].to_numpy()

assert submission.shape[0] == test.shape[0]
assert list(submission.columns) == ["Id", "Cover_Type"]

submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 6
plt.figure(figsize=(10, 5))
ax = sns.countplot(x=submission["Cover_Type"])
plt.title("Predictions")
plt.xlabel("Cover Type")
ax.bar_label(ax.containers[0])
plt.show()
