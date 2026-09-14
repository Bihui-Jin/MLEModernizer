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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)



## === cell 1
BASE_DIR = "/kaggle/input/tabular-playground-series-dec-2021"
train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission = pd.read_csv(sample_sub_path)

assert "Cover_Type" in train_df.columns
assert (
    "Id" in train_df.columns and "Id" in test_df.columns and "Id" in submission.columns
)

X = train_df.drop(columns=["Cover_Type"])
y = train_df["Cover_Type"].astype(int)

X_test = test_df.copy()

feature_cols = [c for c in X.columns if c != "Id"]
X = X[feature_cols]
X_test = X_test[feature_cols]

n_classes = y.nunique()
print(
    "Train shape:", X.shape, "Test shape:", X_test.shape, "Classes:", sorted(y.unique())
)



## === cell 2


def fit_predict_one(seed: int) -> pd.DataFrame:
    clf = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(with_mean=False),
            ),  # sparse-like safe; data is dense but fine
            (
                "lr",
                LogisticRegression(
                    solver="saga",
                    multi_class="multinomial",
                    max_iter=200,
                    n_jobs=-1,
                    random_state=seed,
                    C=2.0,
                ),
            ),
        ]
    )
    clf.fit(X, y)
    pred = clf.predict(X_test).astype(int)
    return pd.DataFrame({"Id": test_df["Id"].values, "Cover_Type": pred})


seeds = [0, 1, 2, 3, 4]
predictions = [fit_predict_one(s) for s in seeds]

for i, p in enumerate(predictions, start=1):
    assert len(p) == len(test_df)
    assert (p["Id"].values == test_df["Id"].values).all(), f"Id order mismatch in p{i}"

print("Built predictions:", len(predictions))



## === cell 3
results = pd.DataFrame(index=np.arange(len(test_df)))
for i, ds in enumerate(predictions):
    results[f"p{i+1}"] = ds["Cover_Type"].astype(int).values

print(results.shape)
results.head()



## === cell 4
mode_res = stats.mode(results.values, axis=1, keepdims=False)
results["ensemble"] = mode_res.mode.astype(int)
results.head()




## === cell 5
def nunique(a, axis):
    return (np.diff(np.sort(a, axis=axis), axis=axis) != 0).sum(axis=axis) + 1




## === cell 6
results["dif"] = nunique(results.iloc[:, : len(predictions)].values, 1) - 1
results.head()



## === cell 7
vc = results["dif"].value_counts().sort_index()
print(vc)



## === cell 8
ens = pd.DataFrame(
    {"Id": test_df["Id"].values, "Cover_Type": results["ensemble"].values}
)
submission = submission[["Id"]].merge(ens, on="Id", how="left")

if submission["Cover_Type"].isna().any():
    raise RuntimeError("Missing predictions after merge; check Id alignment.")

submission["Cover_Type"] = submission["Cover_Type"].astype(int)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)



## === cell 9
plt.figure(figsize=(10, 5))
ax = sns.countplot(x=submission["Cover_Type"])
plt.title("Predictions")
plt.xlabel("Cover Type")
for container in ax.containers:
    ax.bar_label(container)
plt.show()
