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
imbalanced-learn==0.13.0
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
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier  # kept for compatibility if needed elsewhere
import gc  # for explicit memory cleanup



## === cell 1
df_train = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
df_test = pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")
submission = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/sample_submission.csv"
)




## === cell 2
def reduce_mem_usage(df, verbose=True):
    """Downcast numeric columns to the smallest possible dtypes.
    Float columns are cast to float16 when the range fits, otherwise to float32."""
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                else:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df




## === cell 3
df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)
gc.collect()  # free any temporary buffers



## === cell 4
df_train = df_train[(df_train["Cover_Type"] != 4) & (df_train["Cover_Type"] != 5)]

target_col = "Cover_Type"
id_col = "Id"

original_labels = sorted(df_train[target_col].unique())  # e.g. [1,2,3,6,7]
label2idx = {lbl: idx for idx, lbl in enumerate(original_labels)}
idx2label = {idx: lbl for lbl, idx in label2idx.items()}

df_train["Cover_Type_enc"] = df_train[target_col].map(label2idx)

X = df_train.drop(columns=[id_col, target_col, "Cover_Type_enc"])
y = df_train["Cover_Type_enc"]

X_test = df_test.drop(columns=[id_col])



## === cell 5
X_train_df, X_val_df, y_train_series, y_val_series = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dtrain = xgb.DMatrix(data=X_train_df, label=y_train_series)
dval = xgb.DMatrix(data=X_val_df, label=y_val_series)

tree_method = "hist"
predictor = "cpu_predictor"

cpu_cores = os.cpu_count() or 1

xgb_params = {
    "objective": "multi:softprob",
    "num_class": int(y.nunique()),
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1,
    "reg_alpha": 0,
    "tree_method": tree_method,
    "predictor": predictor,
    "max_bin": 64,
    "eval_metric": "mlogloss",
    "nthread": cpu_cores,
    "verbosity": 0,
    "seed": 42,
}

num_boost_round = 800

model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    evals=[],  # no per‑iteration validation to speed up training
    verbose_eval=False,
)



## === cell 6
val_pred_prob = model.predict(dval)  # shape (n_val, n_classes)
val_pred_idx = np.argmax(val_pred_prob, axis=1)

val_pred = pd.Series(val_pred_idx).map(idx2label)
val_true = pd.Series(y_val_series).map(idx2label)

print("Validation accuracy:", accuracy_score(val_true, val_pred))



## === cell 7
dtest = xgb.DMatrix(data=X_test)  # use DataFrame directly, avoid .values copy
test_pred_prob = model.predict(dtest)
test_pred_idx = np.argmax(test_pred_prob, axis=1)
test_pred = pd.Series(test_pred_idx).map(idx2label)



## === cell 8
submission["Cover_Type"] = test_pred
submission.to_csv("Submission.csv", index=False)
print("Submission saved to Submission.csv")
