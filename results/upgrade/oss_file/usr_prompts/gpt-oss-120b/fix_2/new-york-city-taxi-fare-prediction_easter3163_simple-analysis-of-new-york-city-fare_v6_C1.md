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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

4.73089

# 6. Current score

5.55028

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5.55028) has done: 'The fix updates deprecated datetime access, adds useful time‑based and haversine distance features, and switches to a stronger XGBoost regressor (while keeping the overall pipeline unchanged). This resolves the AttributeError, improves model performance, and writes a correctly‑named CSV submission.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from pathlib import Path

train_path = Path("../input/train.csv")
test_path = Path("../input/test.csv")
sample_sub_path = Path("../input/sample_submission.csv")

train = pd.read_csv(train_path, nrows=300_000)
test = pd.read_csv(test_path)




## === cell 1
train.shape




## === cell 2
train.head()




## === cell 3
train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"])
train["hour"] = train["pickup_datetime"].dt.hour
train["day"] = train["pickup_datetime"].dt.day
train["week"] = train["pickup_datetime"].dt.isocalendar().week.astype(int)
train["month"] = train["pickup_datetime"].dt.month
train["day_of_year"] = train["pickup_datetime"].dt.dayofyear
train["week_of_year"] = train["pickup_datetime"].dt.isocalendar().week.astype(int)




## === cell 4
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])
test["hour"] = test["pickup_datetime"].dt.hour
test["day"] = test["pickup_datetime"].dt.day
test["week"] = test["pickup_datetime"].dt.isocalendar().week.astype(int)
test["month"] = test["pickup_datetime"].dt.month
test["day_of_year"] = test["pickup_datetime"].dt.dayofyear
test["week_of_year"] = test["pickup_datetime"].dt.isocalendar().week.astype(int)




## === cell 5
train = train.dropna(how="any", axis="rows")




## === cell 6
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] < 200)]
train = train.loc[(train["pickup_longitude"] > -75) & (train["pickup_longitude"] < 75)]
train = train.loc[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 45)]
train = train.loc[
    (train["dropoff_longitude"] > -75) & (train["dropoff_longitude"] < 75)
]
train = train.loc[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 45)]
train = train.loc[train["passenger_count"] <= 8]




## === cell 7
train["abs_diff_longitude"] = (
    train["pickup_longitude"] - train["dropoff_longitude"]
).abs()
train["abs_diff_latitude"] = (
    train["pickup_latitude"] - train["dropoff_latitude"]
).abs()
test["abs_diff_longitude"] = (
    test["pickup_longitude"] - test["dropoff_longitude"]
).abs()
test["abs_diff_latitude"] = (test["pickup_latitude"] - test["dropoff_latitude"]).abs()




## === cell 8
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Vectorized haversine distance in kilometers.
    """
    R = 6371.0  # Earth radius in km
    lon1_rad, lat1_rad = np.radians(lon1), np.radians(lat1)
    lon2_rad, lat2_rad = np.radians(lon2), np.radians(lat2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


train["distance_km"] = haversine_np(
    train["pickup_longitude"],
    train["pickup_latitude"],
    train["dropoff_longitude"],
    train["dropoff_latitude"],
)

test["distance_km"] = haversine_np(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["dropoff_longitude"],
    test["dropoff_latitude"],
)




## === cell 9
feature_names = [
    "abs_diff_longitude",
    "abs_diff_latitude",
    "distance_km",
    "hour",
    "day",
    "week",
    "month",
    "day_of_year",
    "week_of_year",
    "passenger_count",
]
print("Using features:", feature_names)




## === cell 10
label_name = "fare_amount"
print("Label:", label_name)




## === cell 11
X_train = train[feature_names]
y_train = train[label_name]
X_test = test[feature_names]




## === cell 12
from sklearn.model_selection import train_test_split
import xgboost as xgb

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)




## === cell 13
xgb_reg = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=4,
    random_state=42,
    reg_lambda=1.0,
)
xgb_reg.fit(X_tr, y_tr)




## === cell 14
val_pred = xgb_reg.predict(X_val)
val_pred = np.clip(val_pred, 0, None)




## === cell 15
test_pred = xgb_reg.predict(X_test)
test_pred = np.clip(test_pred, 0, None)  # ensure no negative fares




## === cell 16
submission = pd.read_csv(sample_sub_path)
submission["fare_amount"] = test_pred
submission.head()




## === cell 17
output_path = Path("./simplenewyorktaxi.csv")
submission.to_csv(output_path, index=False)
print(f"Submission saved to {output_path.resolve()}")
