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
import numpy as np
import pandas as pd



## === cell 1
train_df = pd.read_csv("/kaggle/input/tabular-playground-series-dec-2021/train.csv")
test_df = pd.read_csv("/kaggle/input/tabular-playground-series-dec-2021/test.csv")



## === cell 2
FEATURES = [c for c in train_df.columns if c not in ("Cover_Type", "Id")]
X = train_df[FEATURES]
y = train_df["Cover_Type"].astype(np.int64)

y0 = y - 1  # map 1..7 -> 0..6
X_test = test_df[FEATURES]

print("Shapes:", X.shape, y0.shape, X_test.shape)



## === cell 3
from sklearn.model_selection import train_test_split

vc = y0.value_counts()
can_stratify = vc.min() >= 2

split_kwargs = dict(
    test_size=0.01,
    random_state=42,
    shuffle=True,
)

if can_stratify:
    split_kwargs["stratify"] = y0
else:
    print(
        "Warning: Cannot stratify split because min class count is",
        int(vc.min()),
        "- using non-stratified split.",
    )

x_train, x_val, y_train, y_val = train_test_split(X, y0, **split_kwargs)

if not isinstance(y_train, pd.Series):
    y_train = pd.Series(y_train, index=x_train.index)
if not isinstance(y_val, pd.Series):
    y_val = pd.Series(y_val, index=x_val.index)

print(
    "Split shapes:",
    x_train.shape,
    x_val.shape,
    y_train.shape,
    y_val.shape,
    "n_classes_train:",
    np.unique(y_train).size,
    "n_classes_val:",
    np.unique(y_val).size,
)



## === cell 4
from xgboost import XGBClassifier

common_params = dict(
    n_estimators=20000,
    n_jobs=4,
    learning_rate=0.01,
    objective="multi:softprob",
    num_class=7,
    eval_metric="mlogloss",
)

try:
    model_xgbc = XGBClassifier(
        **common_params,
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        gpu_id=0,
    )
    model_xgbc.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
except Exception as e:
    print("GPU training not available or failed; falling back to CPU. Error:", repr(e))
    model_xgbc = XGBClassifier(
        **common_params,
        tree_method="hist",
        predictor="cpu_predictor",
    )
    model_xgbc.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )



## === cell 5
y_pred0 = model_xgbc.predict(X_test)
y_predict_xgbc = (y_pred0 + 1).astype(np.int64)

print(
    "Pred summary:",
    "min=",
    int(y_predict_xgbc.min()),
    "max=",
    int(y_predict_xgbc.max()),
    "n=",
    len(y_predict_xgbc),
)



## === cell 6
result = pd.DataFrame({"Id": test_df["Id"].values, "Cover_Type": y_predict_xgbc})
print(result.head())



## === cell 7
assert list(result.columns) == ["Id", "Cover_Type"]
assert result.shape[0] == test_df.shape[0]
assert result["Cover_Type"].between(1, 7).all()

result.to_csv("/kaggle/working/submission.csv", index=False)
print("Done. Wrote /kaggle/working/submission.csv with shape:", result.shape)
