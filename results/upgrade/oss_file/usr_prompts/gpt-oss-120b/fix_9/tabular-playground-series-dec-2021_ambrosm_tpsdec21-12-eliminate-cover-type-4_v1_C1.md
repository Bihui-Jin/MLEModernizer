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
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # use non‑interactive backend to avoid blocking
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.ensemble import ExtraTreesClassifier
from pathlib import Path
import gc  # garbage collection to free memory promptly
import os  # to limit parallelism to available physical cores



## === cell 1
data_dir = Path("..") / "input" / "tabular-playground-series-dec-2021"
train_path = data_dir / "train.csv"
test_path = data_dir / "test.csv"

all_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
feature_cols = [c for c in all_cols if c not in ("Cover_Type", "Id")]

dtype_features = {col: np.float32 for col in feature_cols}
dtype_features["Cover_Type"] = np.int32
dtype_features["Id"] = np.int64

train_df = pd.read_csv(
    train_path,
    usecols=feature_cols + ["Cover_Type", "Id"],
    dtype=dtype_features,
)
y_train = train_df["Cover_Type"].values  # already int32, no copy needed
train_df.drop(columns=["Cover_Type", "Id"], inplace=True)
X_train = train_df.values  # NumPy array for faster downstream processing
del train_df
gc.collect()

dtype_test = {col: np.float32 for col in feature_cols}
dtype_test["Id"] = np.int64

test_df = pd.read_csv(
    test_path,
    usecols=feature_cols + ["Id"],
    dtype=dtype_test,
)
test_ids = test_df["Id"].values
test_df.drop(columns=["Id"], inplace=True)
X_test = test_df.values
del test_df
gc.collect()



## === cell 2
max_jobs = min(os.cpu_count() or 1, 8)

model = ExtraTreesClassifier(
    n_estimators=200,
    max_features="auto",
    n_jobs=max_jobs,
    random_state=42,
)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)

sub = pd.DataFrame({"Id": test_ids, "Cover_Type": test_pred})



## === cell 3
sub.loc[sub.Cover_Type == 4, "Cover_Type"] = 3

output_path = Path("submission_without4.csv")
sub.to_csv(output_path, index=False)

plt.figure(figsize=(10, 3))
plt.hist(
    sub["Cover_Type"],
    bins=np.linspace(0.5, 7.5, 8),
    density=True,
    rwidth=0.7,
    label="Test predictions",
)
probes = [
    [1, 0.38565],
    [2, 0.51259],
    [3, 0.07817],
    [4, 0.00034],
    [6, 0.00701],
    [7, 0.01621],
]
plt.bar(
    [c for c, f in probes],
    [f for c, f in probes],
    label="lb frequencies",
    color="k",
    width=0.2,
)
plt.xticks(
    ticks=range(1, 8),
    labels=[f"{i}\n{(sub['Cover_Type'] == i).mean():.5f}" for i in range(1, 8)],
)
plt.xlabel("Cover_Type")
plt.ylabel("Frequency")
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
plt.legend()
plt.savefig("prediction_histogram.png")
plt.close()

sub.head()
