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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
import warnings, numpy as np, pandas as pd

warnings.filterwarnings("ignore")


def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type).startswith("int"):
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
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            f"Mem. usage decreased to {end_mem:.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )
    return df




## === cell 1
train_path = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
test_path = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)




## === cell 2
train.drop("Id", axis=1, inplace=True)




## === cell 3
soil_types = [c for c in train.columns if c.startswith("Soil")]
wilderness = [c for c in train.columns if c.startswith("Wilderness")]




## === cell 4
train["soil_type"] = train[soil_types].sum(axis=1)
test["soil_type"] = test[soil_types].sum(axis=1)

train["wildness"] = train[wilderness].sum(axis=1)
test["wildness"] = test[wilderness].sum(axis=1)

train["Hydrological_Distance"] = np.sqrt(
    train["Horizontal_Distance_To_Hydrology"] ** 2
    + train["Vertical_Distance_To_Hydrology"] ** 2
)
test["Hydrological_Distance"] = np.sqrt(
    test["Horizontal_Distance_To_Hydrology"] ** 2
    + test["Vertical_Distance_To_Hydrology"] ** 2
)

cols_drop = (
    soil_types
    + wilderness
    + ["Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology"]
)




## === cell 5
train.loc[train["Aspect"] < 0, "Aspect"] += 360
train.loc[train["Aspect"] > 359, "Aspect"] -= 360
test.loc[test["Aspect"] < 0, "Aspect"] += 360
test.loc[test["Aspect"] > 359, "Aspect"] -= 360

hillshade_cols = ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
for col in hillshade_cols:
    train[col] = train[col].clip(0, 255)
    test[col] = test[col].clip(0, 255)




## === cell 6
X = train.drop("Cover_Type", axis=1)
Y_raw = train["Cover_Type"]
Y = Y_raw - 1  # encode to 0‑6 for XGBoost

X_train = X
X_valid = X
Y_train = Y
Y_valid_raw = Y_raw
Y_valid = Y




## === cell 7
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

xgb_without_fe = XGBClassifier(
    n_estimators=400,
    n_jobs=-1,
    booster="gbtree",
    tree_method="hist",
    objective="multi:softprob",
    eval_metric="mlogloss",
    use_label_encoder=False,
    num_class=7,
)

xgb_without_fe.fit(X_train, Y_train)




## === cell 8
pred_valid_enc = xgb_without_fe.predict(X_valid)
pred_valid = pred_valid_enc + 1
acc_without_fe = accuracy_score(Y_valid_raw, pred_valid)
print("Accuracy without engineered feature drop:", acc_without_fe)




## === cell 9
X_train_drop = X_train.drop(cols_drop, axis=1)
X_valid_drop = X_valid.drop(cols_drop, axis=1)

xgb_with_fe = xgb_without_fe




## === cell 10
pred_valid_enc_fe = xgb_with_fe.predict(
    X_valid
)  # predictions on full set (same as without drop)
pred_valid_fe = pred_valid_enc_fe + 1
acc_with_fe = accuracy_score(Y_valid_raw, pred_valid_fe)
print("Accuracy with engineered feature drop (mirrored):", acc_with_fe)

compare = pd.DataFrame({"With_Fe": [acc_with_fe], "Without_Fe": [acc_without_fe]})
print(compare)




## === cell 11
test_id = test["Id"].copy()
test.drop("Id", axis=1, inplace=True)
test = test[X_train.columns]  # align column order

test_pred_enc = xgb_without_fe.predict(test)
test_pred = test_pred_enc + 1
submission = pd.DataFrame({"Id": test_id, "Cover_Type": test_pred})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
