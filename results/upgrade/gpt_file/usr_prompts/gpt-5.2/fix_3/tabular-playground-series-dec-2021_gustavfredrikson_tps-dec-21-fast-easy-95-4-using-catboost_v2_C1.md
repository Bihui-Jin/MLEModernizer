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

catboost==1.2.8
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
import pandas as pd
import numpy as np

import sklearn.model_selection as skl_ms
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)



## === cell 1
BASE_PATH = "../input/tabular-playground-series-dec-2021"
train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
sub_path = os.path.join(BASE_PATH, "sample_submission.csv")

num_cols = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
wilderness_cols = [f"Wilderness_Area{i}" for i in range(1, 5)]
soil_cols = [f"Soil_Type{i}" for i in range(1, 41)]

dtype_train = {c: np.int16 for c in num_cols}
dtype_train.update({c: np.int8 for c in wilderness_cols})
dtype_train.update({c: np.int8 for c in soil_cols})
dtype_train.update({"Id": np.int32, "Cover_Type": np.int8})

dtype_test = dtype_train.copy()
dtype_test.pop("Cover_Type", None)

train_df = pd.read_csv(train_path, dtype=dtype_train, engine="c", low_memory=False)
test_df = pd.read_csv(test_path, dtype=dtype_test, engine="c", low_memory=False)

mask_not5 = train_df["Cover_Type"] != 5
nr5 = int((~mask_not5).sum())
print(f"Nr of cover_type = 5: {nr5}")
train_df = train_df.loc[mask_not5].reset_index(drop=True)

encoder = LabelEncoder()
train_df["Cover_Type"] = encoder.fit_transform(train_df["Cover_Type"])



## === cell 2
test_size = 0.01  # keep identical intent
train, test = skl_ms.train_test_split(
    train_df, test_size=test_size, random_state=RANDOM_STATE, shuffle=True
)

y_train = train.pop("Cover_Type")
X_train = train
y_test = test.pop("Cover_Type")
X_test = test




## === cell 3
def build_model():
    params = dict(
        iterations=5000,
        random_seed=RANDOM_STATE,
        loss_function="MultiClass",
        verbose=False,
    )
    try:
        model_gpu = CatBoostClassifier(**params, task_type="GPU", devices="0")
        model_gpu.fit(X_train, y_train, verbose=False)
        return model_gpu, "GPU"
    except Exception as e:
        print(
            f"GPU training unavailable; falling back to CPU. Reason: {type(e).__name__}: {e}"
        )
        model_cpu = CatBoostClassifier(**params, task_type="CPU", thread_count=-1)
        model_cpu.fit(X_train, y_train, verbose=False)
        return model_cpu, "CPU"


model, used_device = build_model()
print(f"Trained CatBoost using: {used_device}")



## === cell 4
accuracy = model.score(X_test, y_test)
print(f"Accuracy of catboost on holdout data: {accuracy}")



## === cell 5
subm_df = pd.read_csv(sub_path)

preds = model.predict(test_df)
preds = np.asarray(preds).reshape(-1).astype(int)

subm_df["Cover_Type"] = encoder.inverse_transform(preds)

out_path = "submission_cb.csv"
subm_df.to_csv(out_path, index=False)
print(
    f"Wrote submission to: {out_path} with shape {subm_df.shape} and columns {list(subm_df.columns)}"
)
