# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict the fare amount for a taxi ride given the pickup and dropoff locations.

## Metric
Root mean-squared error.

## Submission Format
For each `key` in the test set, you must predict a value for the `fare_amount` variable. The file should contain a header and have the following format:

```
key,fare_amount
2015-01-27 13:08:24.0000002,11.00
2015-02-27 13:08:24.0000002,12.05
2015-03-27 13:08:24.0000002,11.23
2015-04-27 13:08:24.0000002,14.17
2015-05-27 13:08:24.0000002,15.12
etc
```

## Dataset
- **train.csv** - Input features and target `fare_amount` values for the training set (about 55M rows).
- **test.csv** - Input features for the test set (about 10K rows). Your goal is to predict `fare_amount` for each row.
- **sample_submission.csv** - a sample submission file in the correct format (columns `key` and `fare_amount`). This file 'predicts' `fare_amount` to be $`11.35` for all rows, which is the mean `fare_amount` from the training set.

### Data fields
**ID**

- **key** - Unique `string` identifying each row in both the training and test sets. Comprised of **pickup_datetime** plus a unique integer, but this doesn't matter, it should just be used as a unique ID field.Required in your submission CSV. Not necessarily needed in the training set, but could be useful to simulate a 'submission file' while doing cross-validation within the training set.

**Features**

- **pickup_datetime** - `timestamp` value indicating when the taxi ride started.
- **pickup_longitude** - `float` for longitude coordinate of where the taxi ride started.
- **pickup_latitude** - `float` for latitude coordinate of where the taxi ride started.
- **dropoff_longitude** - `float` for longitude coordinate of where the taxi ride ended.
- **dropoff_latitude** - `float` for latitude coordinate of where the taxi ride ended.
- **passenger_count** - `integer` indicating the number of passengers in the taxi ride.

**Target**

- **fare_amount** - `float` dollar amount of the cost of the taxi ride. This value is only in the training set; this is what you are predicting in the test set and it is required in your submission CSV.

# 2. Python version

3.7

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
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        input/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
```

-> data/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 5. Target score

3.27133

# 6. Current score

4.65185

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.55737) has done: 'The crash happens because `KFold` defaults to `shuffle=False` in scikit-learn 1.2+, and providing `random_state` while not shuffling now raises a `ValueError`. The minimal fix is to explicitly enable shuffling so the existing intent (a deterministic split controlled by `random_state=0`) remains valid. This preserves the same cross-validation approach and keeps `cv_split` compatible with the next cell’s `cross_validate` call. No other logic is changed.'
- What this solution (achieved 4.44594) has done: 'Diagnosis: `cross_validate` only returns train scores when `return_train_score=True`; otherwise the `'train_score'` key is absent, causing the `KeyError`. Additionally, this notebook prints `tune_model.*` in cell 16, but `tune_model` is not defined anywhere in cells 0–15 (it appears tuning is done later as `submit_xgb` in cell 17), which would immediately cause a `NameError` after fixing the first issue.  
Patch summary: Update cell 16 to (1) compute and print only the always-present test score from `base_results`, and (2) guard the “After XGB …” prints so they run only if a `tune_model` object exists, preserving intended behavior when/if it’s defined elsewhere. This is the minimal change to stop the crash without altering model training or evaluation semantics.  
Updated cells: Only cell 16 is changed.  
Compatibility notes for cell k+1: No variables used by cell 17 are modified; `xgb`, `base_results`, `X`, `y`, and `cv_split` remain intact.  
Assumptions: The intent of cell 16 is reporting; it is acceptable to skip “After XGB …” reporting when `tune_model` is not defined yet.'
- What this solution (achieved 4.63234) has done: 'You’re currently far from the target RMSE (4.44594 vs 3.27133, lower is better), so we should make a small, legitimate improvement without changing the overall modeling approach (still XGBoost + same features + same CV/tuning structure). The biggest score drag is that your data “range filters” use `|` (OR), which effectively keeps almost all invalid coordinates; switching these to `&` (AND) is a minimal bug fix that preserves intent and typically improves RMSE materially. Also, training uses only 100k sampled rows; modestly increasing this sample size (still from the same 6M already loaded) improves generalization while keeping runtime reasonable. Finally, align the tuned model’s `random_state` with the baseline for stability, without changing the architecture or training semantics, and keep the submission format unchanged.'
- What this solution (achieved 4.65185) has done: 'Your RMSE (4.63234) is worse than the target (3.27133), so we need a small, legitimate improvement without changing the overall approach (still: same engineered features + XGBRegressor + CV/tuning + same submission semantics). The biggest low-risk gain here is to remove a training/inference mismatch: you filter invalid/outlier coordinates only in train, but not in test, which can cause extreme distances and unstable predictions; we add the same basic coordinate/passenger sanity checks to test and then clip predictions to a reasonable positive range. Additionally, your `pickup_datetime` parsing format is too strict for this dataset (often no `" UTC"` suffix), which can silently create NaTs and weaken time features; we switch to `pd.to_datetime(..., errors="coerce", utc=True)` to robustly parse while keeping the same time-derived features. These changes preserve the core model/training loop and feature set, but reduce noise/outliers to move RMSE downward toward the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

from math import radians, cos, sin, asin, sqrt

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GridSearchCV

import warnings

warnings.filterwarnings("ignore")



## === cell 1
original_train_data = pd.read_csv("../input/train.csv", nrows=6000000)
train_data = original_train_data.sample(n=300000, random_state=0)
train_data.info()



## === cell 2
test_data = pd.read_csv("../input/test.csv")
test_data.info()



## === cell 3
train_data.isnull().sum()



## === cell 4
train_data.dropna(axis=0, inplace=True)



## === cell 5
train_data.describe()



## === cell 6
train_data = train_data[train_data["fare_amount"] > 0]

train_data = train_data[
    (train_data["passenger_count"] <= 6) & (train_data["passenger_count"] > 0)
]

train_data = train_data[
    (train_data["pickup_latitude"] > -90) & (train_data["pickup_latitude"] <= 90)
]
train_data = train_data[
    (train_data["dropoff_latitude"] > -90) & (train_data["dropoff_latitude"] <= 90)
]

train_data = train_data[
    (train_data["pickup_longitude"] >= -180) & (train_data["pickup_longitude"] <= 180)
]
train_data = train_data[
    (train_data["dropoff_longitude"] >= -180) & (train_data["dropoff_longitude"] <= 180)
]



## === cell 7
train_data.shape



## === cell 8
train_data.info()



## === cell 9
train_data.head(5)




## === cell 10
def distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Return distance along great radius between pickup and dropoff coordinates.
    """
    R_earth = 6371
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(
        np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    )
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon / 2.0) ** 2
    )

    return 2 * R_earth * np.arcsin(np.sqrt(a))


def date_time_info(data):
    data["pickup_datetime"] = pd.to_datetime(
        data["pickup_datetime"], errors="coerce", utc=True
    )

    data["hour"] = data["pickup_datetime"].dt.hour
    data["day"] = data["pickup_datetime"].dt.day
    data["month"] = data["pickup_datetime"].dt.month
    data["weekday"] = data["pickup_datetime"].dt.weekday
    data["year"] = data["pickup_datetime"].dt.year

    return data


train_data = date_time_info(train_data)
train_data["distance"] = distance(
    train_data["pickup_latitude"],
    train_data["pickup_longitude"],
    train_data["dropoff_latitude"],
    train_data["dropoff_longitude"],
)

train_data.head()



## === cell 11
train_data.drop(["key", "pickup_datetime"], axis=1, inplace=True)
train_data.head()



## === cell 12
test_data.head()



## === cell 13
test_data = date_time_info(test_data)

valid_mask = (
    (test_data["passenger_count"].between(1, 6, inclusive="both"))
    & (test_data["pickup_latitude"].between(-90, 90, inclusive="right"))
    & (test_data["dropoff_latitude"].between(-90, 90, inclusive="right"))
    & (test_data["pickup_longitude"].between(-180, 180, inclusive="both"))
    & (test_data["dropoff_longitude"].between(-180, 180, inclusive="both"))
)

test_data["distance"] = distance(
    test_data["pickup_latitude"],
    test_data["pickup_longitude"],
    test_data["dropoff_latitude"],
    test_data["dropoff_longitude"],
)

test_key = test_data["key"]
x_pred = test_data.drop(columns=["key", "pickup_datetime"])

feature_medians = x_pred[valid_mask].median(numeric_only=True)
x_pred.loc[~valid_mask, feature_medians.index] = feature_medians.values



## === cell 14
y = train_data["fare_amount"]
X = train_data.drop(["fare_amount"], axis=1)

cv_split = KFold(n_splits=10, shuffle=True, random_state=0)



## === cell 15
xgb = XGBRegressor(random_state=0)
base_results = cross_validate(xgb, X, y, cv=cv_split)
xgb.fit(X, y)



## === cell 16
print("Best XGB parameters: ", xgb.get_params())

print(
    "Before XGB Testing score mean: {:.2f}".format(
        base_results["test_score"].mean() * 100
    )
)
print("#" * 20)

if "tune_model" in globals():
    print("After XGB Parameters: ", tune_model.best_params_)
    print(
        "After XGB Training score mean: {:.2f}".format(
            tune_model.cv_results_["mean_train_score"][tune_model.best_index_] * 100
        )
    )
    print(
        "After XGB Testing score mean: {:.2f}".format(
            tune_model.cv_results_["mean_test_score"][tune_model.best_index_] * 100
        )
    )
    print("#" * 20)



## === cell 17
params = {
    "max_depth": [8],  # Result of tuning with CV
    "eta": [0.03],  # Result of tuning with CV
    "subsample": [1],  # Result of tuning with CV
    "colsample_bytree": [0.8],  # Result of tuning with CV
    "objective": ["reg:linear"],
    "eval_metric": ["rmse"],
    "silent": [1],
}

submit_xgb = GridSearchCV(
    XGBRegressor(random_state=0),
    param_grid=params,
    scoring="neg_mean_squared_error",
    cv=cv_split,
)
submit_xgb.fit(X, y)



## === cell 18
prediction = submit_xgb.predict(x_pred)

prediction = np.clip(prediction, 0.0, 500.0)



## === cell 19
submission = pd.DataFrame({"key": test_key, "fare_amount": prediction.round(2)})

submission.to_csv("taxi_fare_submission.csv", index=False)
submission.head()
