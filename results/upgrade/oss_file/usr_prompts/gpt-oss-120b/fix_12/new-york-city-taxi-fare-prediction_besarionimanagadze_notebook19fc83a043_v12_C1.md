# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.12

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
sklearn-pandas==2.2.0

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

5.51477

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 12.27188) has done: 'I replace the simple linear regression with a Gradient Boosting Regressor, which can capture non‑linear relationships in the features (hour, distance, passenger count, etc.) and therefore dramatically lower the RMSE toward the target. I also add a quick internal RMSE calculation so you can see the improvement before creating the Kaggle submission. All other preprocessing steps remain unchanged, and the script still writes a valid `submission.csv` file.'
- What this solution (achieved 11.52499) has done: 'The fix keeps the same preprocessing and model type but reduces the GradientBoostingRegressor’s workload by lowering the number of trees (n_estimators) and enabling subsampling, which speeds up fitting dramatically while preserving the algorithm’s core logic and overall accuracy. No other parts of the pipeline are altered.'

# 9. Code solution

## === cell 0
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ImportError:
    pass

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # plotting library with python
from sklearn.model_selection import train_test_split

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor as GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import os



## === cell 1
usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtypes = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
train_data_set = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=2_000_000,
    parse_dates=["pickup_datetime"],
    usecols=usecols,
    dtype=dtypes,
)

train_data_set.head(5)



## === cell 2
print(train_data_set.dtypes)



## === cell 3
old_len = len(train_data_set)
train_data_set = train_data_set[train_data_set.fare_amount >= 0.1]
new_len = len(train_data_set)
print(f"Removed {(old_len-new_len)} entities from the dataset")



## === cell 4
old_len = len(train_data_set)
train_data_set = train_data_set.dropna(how="any", axis="rows")
new_len = len(train_data_set)
print(f"Removed {(old_len-new_len)} entities from the dataset")




## === cell 5
def select_within_boundingbox(df, box):
    return (
        (df.pickup_longitude >= box[0])
        & (df.pickup_longitude <= box[1])
        & (df.pickup_latitude >= box[2])
        & (df.pickup_latitude <= box[3])
        & (df.dropoff_longitude >= box[0])
        & (df.dropoff_longitude <= box[1])
        & (df.dropoff_latitude >= box[2])
        & (df.dropoff_latitude <= box[3])
    )


new_york_box = (-74.763379, -72.856164, 40.502009, 41.915509)

old_len = len(train_data_set)
train_data_set = train_data_set[select_within_boundingbox(train_data_set, new_york_box)]
new_len = len(train_data_set)
print(f"Removed {(old_len-new_len)} entities from the dataset")




## === cell 6
def distance_on_the_sphere(lat1, lon1, lat2, lon2):
    earth_radius = 6371  # km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return earth_radius * c


train_data_set["distance"] = distance_on_the_sphere(
    train_data_set["pickup_latitude"],
    train_data_set["pickup_longitude"],
    train_data_set["dropoff_latitude"],
    train_data_set["dropoff_longitude"],
)

train_data_set.head(5)



## === cell 7
train_data_set["hour"] = train_data_set["pickup_datetime"].dt.hour
train_data_set["year"] = train_data_set["pickup_datetime"].dt.year
train_data_set["day_of_week"] = train_data_set["pickup_datetime"].dt.dayofweek

hour = train_data_set["hour"]
train_data_set["is_rush_hour"] = (
    ((hour >= 7) & (hour <= 10)) | ((hour >= 16) & (hour <= 19))
).astype(int)

train_data_set.head(5)



## === cell 8
idx = train_data_set.passenger_count != 0

features = [
    "hour",
    "year",
    "distance",
    "passenger_count",
    "is_rush_hour",
    "day_of_week",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
target = "fare_amount"

X = train_data_set.loc[idx, features].values.astype(np.float32)
y = train_data_set.loc[idx, target].values.astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

y_train_log = np.log1p(y_train).astype(np.float32)

model = GradientBoostingRegressor(
    max_iter=200,  # n_estimators → max_iter for HistGradientBoosting
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=42,
)

model.fit(X_train, y_train_log)

y_pred_log = model.predict(X_val)
y_pred = np.expm1(y_pred_log)

rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f"Validation RMSE (back‑transformed): {rmse:.4f}")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/66043220.py in <cell line: 0>()
     22 y_train_log = np.log1p(y_train).astype(np.float32)
     23 
---> 24 model = GradientBoostingRegressor(
     25     max_iter=200,  # n_estimators → max_iter for HistGradientBoosting
     26     learning_rate=0.05,

TypeError: HistGradientBoostingRegressor.__init__() got an unexpected keyword argument 'subsample'

## === cell 9
test_data_set = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    dtype={
        "pickup_longitude": np.float32,
        "pickup_latitude": np.float32,
        "dropoff_longitude": np.float32,
        "dropoff_latitude": np.float32,
        "passenger_count": np.int8,
    },
)

test_data_set["distance"] = distance_on_the_sphere(
    test_data_set["pickup_latitude"],
    test_data_set["pickup_longitude"],
    test_data_set["dropoff_latitude"],
    test_data_set["dropoff_longitude"],
)

test_data_set["hour"] = test_data_set["pickup_datetime"].dt.hour
test_data_set["year"] = test_data_set["pickup_datetime"].dt.year
test_data_set["day_of_week"] = test_data_set["pickup_datetime"].dt.dayofweek

hour_test = test_data_set["hour"]
test_data_set["is_rush_hour"] = (
    ((hour_test >= 7) & (hour_test <= 10)) | ((hour_test >= 16) & (hour_test <= 19))
).astype(int)



## === cell 10
XTEST = test_data_set[features].values.astype(np.float32)

y_pred_test_log = model.predict(XTEST)
y_pred_test = np.expm1(y_pred_test_log)

submission = pd.DataFrame(
    {"key": test_data_set["key"], "fare_amount": y_pred_test},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1219617481.py in <cell line: 0>()
      1 XTEST = test_data_set[features].values.astype(np.float32)
      2 
----> 3 y_pred_test_log = model.predict(XTEST)
      4 y_pred_test = np.expm1(y_pred_test_log)
      5 

NameError: name 'model' is not defined
