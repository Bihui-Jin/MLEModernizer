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
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
xgboost==2.0.3

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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")



## === cell 1
skip_cols = ["Soil_Type7", "Soil_Type15"]
train_path = "../input/tabular-playground-series-dec-2021/train.csv"
test_path = "../input/tabular-playground-series-dec-2021/test.csv"

all_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
usecols = [c for c in all_cols if c not in skip_cols]

df_train = pd.read_csv(train_path, usecols=usecols, dtype=np.float32, low_memory=False)
df_test = pd.read_csv(test_path, usecols=usecols, dtype=np.float32, low_memory=False)




## === cell 2
def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).startswith("int") or np.issubdtype(col_type, np.integer):
                if c_min >= 0:
                    if c_max < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif c_max < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if np.iinfo(np.int8).min <= c_min <= np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif np.iinfo(np.int16).min <= c_min <= np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif np.iinfo(np.int32).min <= c_min <= np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
            else:  # float
                df[col] = pd.to_numeric(df[col], downcast="float")
    if verbose:
        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        print(
            f"Mem. usage decreased to {end_mem:5.2f} MB "
            f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )
    return df




## === cell 3
df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)



## === cell 5
targets = df_train["Cover_Type"]
df_train = df_train.drop(["Cover_Type"], axis=1)



## === cell 6
label_encoder = LabelEncoder()
targets_enc = label_encoder.fit_transform(targets.astype(int))

class_counts = np.bincount(targets_enc)
valid_classes = np.where(class_counts >= 2)[0]

mask = np.isin(targets_enc, valid_classes)
df_train = df_train[mask].reset_index(drop=True)
targets_enc = targets_enc[mask]

label_encoder = LabelEncoder()
targets_enc = label_encoder.fit_transform(targets_enc)



## === cell 7
train_X_df, val_X_df, train_y, val_y = train_test_split(
    df_train,
    targets_enc,
    test_size=0.2,
    random_state=1,
    stratify=targets_enc,
)

train_X = train_X_df  # keep as DataFrame for XGBoost
val_X = val_X_df



## === cell 8
params = {
    "objective": "multi:softprob",
    "tree_method": "hist",
    "max_bin": 64,  # smaller bin count speeds up histogram building
    "lambda": 0.5,
    "alpha": 0.01,
    "colsample_bytree": 0.9,
    "subsample": 0.8,
    "learning_rate": 0.05,
    "n_estimators": 1000,  # reduced max trees; early stopping will stop earlier if needed
    "max_depth": 12,
    "min_child_weight": 1,
    "random_state": 42,
    "eval_metric": "mlogloss",
    "num_class": len(label_encoder.classes_),
    "use_label_encoder": False,
    "n_jobs": -1,
}

model = XGBClassifier(**params)

model.fit(
    train_X,
    train_y,
    eval_set=[(val_X, val_y)],
    early_stopping_rounds=50,
    verbose=False,
)

val_pred = np.argmax(model.predict_proba(val_X), axis=1)
acc = accuracy_score(val_y, val_pred)
print(f"validation accuracy_score: {acc}")



## === cell 9
submission = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/sample_submission.csv"
)

test_X = df_test  # use DataFrame directly, no NumPy copy
test_pred_enc = np.argmax(model.predict_proba(test_X), axis=1)
test_pred_orig = label_encoder.inverse_transform(test_pred_enc)
submission["Cover_Type"] = test_pred_orig
submission.to_csv("submission.csv", index=False)
print("submission.csv written, rows:", submission.shape[0])
