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

3.9

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

5.52625

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def locate(relative_path):
    candidates = [
        pathlib.Path("../input") / relative_path,
        pathlib.Path("data") / relative_path,
        pathlib.Path("/kaggle/input/new-york-city-taxi-fare-prediction")
        / relative_path,
        pathlib.Path("/kaggle/input") / relative_path,
        pathlib.Path(relative_path),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {relative_path}")


dtype_spec = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int32",
}

train_path = locate("train.csv")
test_path = locate("test.csv")
sample_submission_path = locate("sample_submission.csv")

train_df = pd.read_csv(train_path, nrows=500_000, dtype=dtype_spec, parse_dates=False)
test_df = pd.read_csv(test_path, dtype=dtype_spec, parse_dates=False)

train_df.dtypes




## === cell 1
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)


## === cell 2
print("Train nulls:\n", train_df.isnull().sum())
print("Test nulls:\n", test_df.isnull().sum())

print("Old train size:", len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New train size:", len(train_df))

print("Old test size:", len(test_df))
test_df = test_df.dropna(how="any", axis="rows")
print("New test size:", len(test_df))


## === cell 3
print("Filtering extreme distances – train")
print("Before:", len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("After:", len(train_df))

print("Filtering extreme distances – test")
print("Before:", len(test_df))
test_df = test_df[
    (test_df.abs_diff_longitude < 5.0) & (test_df.abs_diff_latitude < 5.0)
]
print("After:", len(test_df))


## === cell 4
train_dt = pd.to_datetime(train_df["pickup_datetime"])
train_df["pickup_time"] = train_dt.dt.hour * 100 + train_dt.dt.minute
train_df["Weekday_int"] = train_dt.dt.weekday

test_dt = pd.to_datetime(test_df["pickup_datetime"])
test_df["pickup_time"] = test_dt.dt.hour * 100 + test_dt.dt.minute
test_df["Weekday_int"] = test_dt.dt.weekday


## === cell 5
train_df.drop(columns=["pickup_datetime"], inplace=True)
test_df.drop(columns=["pickup_datetime"], inplace=True)


## === cell 6
train_one_hot = pd.get_dummies(train_df["Weekday_int"], prefix="Weekday")
test_one_hot = pd.get_dummies(test_df["Weekday_int"], prefix="Weekday")

train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)

train_df.drop(columns=["Weekday_int"], inplace=True)
test_df.drop(columns=["Weekday_int"], inplace=True)


## === cell 7
R = 6373.0  # Earth radius in km
lat_air = np.radians(40.6413)
lon_air = np.radians(-73.7781)

lat1 = np.radians(train_df["pickup_latitude"].values)
lon1 = np.radians(train_df["pickup_longitude"].values)
lat2 = np.radians(train_df["dropoff_latitude"].values)
lon2 = np.radians(train_df["dropoff_longitude"].values)

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
train_df["Distance"] = (R * c * 0.621).round(2)  # miles

dlon_pickup = lon_air - lon1
dlat_pickup = lat_air - lat1
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat_air) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
train_df["Pickup_Distance_airport"] = (R * c1 * 0.621).round(2)

dlon_dropoff = lon_air - lon2
dlat_dropoff = lat_air - lat2
a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat_air) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
train_df["Dropoff_Distance_airport"] = (R * c2 * 0.621).round(2)

lat1_t = np.radians(test_df["pickup_latitude"].values)
lon1_t = np.radians(test_df["pickup_longitude"].values)
lat2_t = np.radians(test_df["dropoff_latitude"].values)
lon2_t = np.radians(test_df["dropoff_longitude"].values)

dlon_t = lon2_t - lon1_t
dlat_t = lat2_t - lat1_t
a_t = (
    np.sin(dlat_t / 2) ** 2 + np.cos(lat1_t) * np.cos(lat2_t) * np.sin(dlon_t / 2) ** 2
)
c_t = 2 * np.arctan2(np.sqrt(a_t), np.sqrt(1 - a_t))
test_df["Distance"] = (R * c_t * 0.621).round(2)

dlon_pickup_t = lon_air - lon1_t
dlat_pickup_t = lat_air - lat1_t
a1_t = (
    np.sin(dlat_pickup_t / 2) ** 2
    + np.cos(lat1_t) * np.cos(lat_air) * np.sin(dlon_pickup_t / 2) ** 2
)
c1_t = 2 * np.arctan2(np.sqrt(a1_t), np.sqrt(1 - a1_t))
test_df["Pickup_Distance_airport"] = (R * c1_t * 0.621).round(2)

dlon_dropoff_t = lon_air - lon2_t
dlat_dropoff_t = lat_air - lat2_t
a2_t = (
    np.sin(dlat_dropoff_t / 2) ** 2
    + np.cos(lat2_t) * np.cos(lat_air) * np.sin(dlon_dropoff_t / 2) ** 2
)
c2_t = 2 * np.arctan2(np.sqrt(a2_t), np.sqrt(1 - a2_t))
test_df["Dropoff_Distance_airport"] = (R * c2_t * 0.621).round(2)


## === cell 8
cols_to_drop = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
train_df.drop(columns=cols_to_drop, inplace=True)
test_df.drop(columns=cols_to_drop, inplace=True)


## === cell 9
abs_lon_mean = train_df["abs_diff_longitude"].mean()
abs_lon_var = train_df["abs_diff_longitude"].var()
train_df["abs_diff_longitude"] = (
    train_df["abs_diff_longitude"] - abs_lon_mean
) / abs_lon_var

abs_lat_mean = train_df["abs_diff_latitude"].mean()
abs_lat_var = train_df["abs_diff_latitude"].var()
train_df["abs_diff_latitude"] = (
    train_df["abs_diff_latitude"] - abs_lat_mean
) / abs_lat_var

test_df["abs_diff_longitude"] = (
    test_df["abs_diff_longitude"] - abs_lon_mean
) / abs_lon_var
test_df["abs_diff_latitude"] = (
    test_df["abs_diff_latitude"] - abs_lat_mean
) / abs_lat_var


## === cell 10
train_features = train_df.drop(["key", "fare_amount"], axis=1).columns
missing_in_test = set(train_features) - set(test_df.columns)
for col in missing_in_test:
    test_df[col] = 0

test_df = test_df[["key"] + list(train_features)]


## === cell 11
X = train_df.drop(["key", "fare_amount"], axis=1).values.astype(np.float32, copy=False)
y = train_df["fare_amount"].values.astype(np.float32, copy=False)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=80)

gbr = GradientBoostingRegressor(
    random_state=80, n_estimators=200, learning_rate=0.05, max_depth=4
)
gbr.fit(X_train, y_train)

val_pred = gbr.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {rmse:.4f}")

pred = gbr.predict(test_df.drop("key", axis=1).values.astype(np.float32, copy=False))
pred = np.clip(pred, 0, None)


## === cell 12
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
