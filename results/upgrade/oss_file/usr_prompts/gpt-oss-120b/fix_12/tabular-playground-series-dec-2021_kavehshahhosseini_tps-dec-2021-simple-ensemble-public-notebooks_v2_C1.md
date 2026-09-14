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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import gc

sns.set_style("darkgrid")

train_path = "../input/tabular-playground-series-dec-2021/train.csv"

target_col = "Cover_Type"
id_col = "Id"

tmp = pd.read_csv(train_path, nrows=5)
feature_cols = [c for c in tmp.columns if c not in (target_col, id_col)]

dtype_map = {col: np.float32 for col in feature_cols}
dtype_map[target_col] = np.int8
dtype_map[id_col] = np.int32

usecols = feature_cols + [target_col, id_col]
train_df = pd.read_csv(train_path, dtype=dtype_map, low_memory=False, usecols=usecols)




## === cell 1
X = train_df[feature_cols]
y = train_df[target_col]

X_train_df, X_val_df, y_train_series, y_val_series = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

del X, y, train_df
gc.collect()




## === cell 2
model = ExtraTreesClassifier(
    n_estimators=300,  # reduced from 500 for faster training
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=False,
)

model.fit(X_train_df, y_train_series)

val_pred = model.predict(X_val_df)
val_acc = np.mean(y_val_series.to_numpy() == val_pred)
print(f"Validation accuracy: {val_acc:.5f}")

del X_train_df, X_val_df, y_train_series, y_val_series
gc.collect()




## === cell 3
test_path = "../input/tabular-playground-series-dec-2021/test.csv"
sample_sub_path = "../input/tabular-playground-series-dec-2021/sample_submission.csv"

dtype_map_test = {col: np.float32 for col in feature_cols}
dtype_map_test[id_col] = np.int32
usecols_test = feature_cols + [id_col]

test_df = pd.read_csv(
    test_path, dtype=dtype_map_test, low_memory=False, usecols=usecols_test
)
submission = pd.read_csv(sample_sub_path)

test_pred = model.predict(test_df[feature_cols])

submission["Cover_Type"] = test_pred
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(submission.head())

del test_df, submission
gc.collect()




## === cell 4
plt.figure(figsize=(10, 5))
counts = np.bincount(test_pred)[1:]  # classes start at 1
bars = plt.bar(
    np.arange(1, len(counts) + 1), counts, tick_label=np.arange(1, len(counts) + 1)
)
plt.title("Predicted Cover_Type Distribution")
plt.xlabel("Cover Type")
plt.ylabel("Count")
for bar in bars:
    height = bar.get_height()
    plt.annotate(
        f"{int(height)}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),  # 3 points vertical offset
        textcoords="offset points",
        ha="center",
        va="bottom",
    )
plt.show()
