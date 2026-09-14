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

5.68914

# 6. Current score

6.28463

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'I remove the deprecated `normalize=True` argument from the `LinearRegression` constructor so the model can be instantiated, which resolves the `TypeError`. This also restores the `lr` variable, allowing subsequent prediction, submission creation, and CSV export to run without further errors.'
- What this solution (achieved 11.74409) has done: 'I replace the simple LinearRegression with a more powerful HistGradientBoostingRegressor, which better captures non‑linear relationships in the engineered features while keeping the overall pipeline unchanged. This change is expected to dramatically lower the RMSE from the current ~937 toward the target ~5.7, and the rest of the code (feature engineering, CSV creation) remains intact.'
- What this solution (achieved 6.28463) has done: 'Implemented missing imports, corrected data paths, added safe sampling for the large training set, and ensured all variables are defined before use. The workflow now loads data, engineers features, trains a HistGradientBoostingRegressor (with a log‑transform target), generates predictions for the test set, and writes a properly formatted `submission.csv`. This fixes the runtime errors and produces a valid submission file while keeping the original modeling approach.'

# 9. Code solution

## === cell 0
import os, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
data_dir = "/kaggle/input"
assert os.path.isdir(data_dir), "Data directory not found"



## === cell 1
dtype = {
    "key": "category",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
train_path = os.path.join(data_dir, "new-york-city-taxi-fare-prediction", "train.csv")
test_path = os.path.join(data_dir, "new-york-city-taxi-fare-prediction", "test.csv")

train_data = pd.read_csv(train_path, dtype=dtype, nrows=2_000_000)
test_data = pd.read_csv(
    test_path,
    dtype={k: v for k, v in dtype.items() if k != "fare_amount"},
)



## === cell 2
train_data["Difference_longitude"] = np.abs(
    train_data["pickup_longitude"] - train_data["dropoff_longitude"]
).astype("float32")
train_data["Difference_latitude"] = np.abs(
    train_data["pickup_latitude"] - train_data["dropoff_latitude"]
).astype("float32")

test_data["Difference_longitude"] = np.abs(
    test_data["pickup_longitude"] - test_data["dropoff_longitude"]
).astype("float32")
test_data["Difference_latitude"] = np.abs(
    test_data["pickup_latitude"] - test_data["dropoff_latitude"]
).astype("float32")



## === cell 3
print(f"Before dropping nulls: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After dropping nulls: {len(train_data)}")



## === cell 4
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]



## === cell 5
train_dt = pd.to_datetime(train_data["pickup_datetime"])
test_dt = pd.to_datetime(test_data["pickup_datetime"])

train_data["pickuptime"] = (train_dt.dt.hour * 100 + train_dt.dt.minute).astype(int)
test_data["pickuptime"] = (test_dt.dt.hour * 100 + test_dt.dt.minute).astype(int)

train_data["Weekday"] = train_dt.dt.weekday
test_data["Weekday"] = test_dt.dt.weekday

train_data.drop(columns="pickup_datetime", inplace=True)
test_data.drop(columns="pickup_datetime", inplace=True)



## === cell 6
train_one_hot = pd.get_dummies(train_data["Weekday"], prefix="Weekday")
test_one_hot = pd.get_dummies(test_data["Weekday"], prefix="Weekday")
train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)

train_data.drop(columns="Weekday", inplace=True)
test_data.drop(columns="Weekday", inplace=True)



## === cell 7
train_data["pickuptime"] = train_data["pickuptime"].astype("int16")
test_data["pickuptime"] = test_data["pickuptime"].astype("int16")



## === cell 8
R = 6373.0  # earth radius in km
lat1 = np.radians(train_data["pickup_latitude"].astype("float32"))
lon1 = np.radians(train_data["pickup_longitude"].astype("float32"))
lat2 = np.radians(train_data["dropoff_latitude"].astype("float32"))
lon2 = np.radians(train_data["dropoff_longitude"].astype("float32"))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
train_data["Distance"] = (R * c * 0.621).astype("float32")

lat1 = np.radians(test_data["pickup_latitude"].astype("float32"))
lon1 = np.radians(test_data["pickup_longitude"].astype("float32"))
lat2 = np.radians(test_data["dropoff_latitude"].astype("float32"))
lon2 = np.radians(test_data["dropoff_longitude"].astype("float32"))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
test_data["Distance"] = (R * c * 0.621).astype("float32")



## === cell 9
lat_air = np.radians(40.6413111)
lon_air = np.radians(-73.7781391)

lat1 = np.radians(train_data["pickup_latitude"].astype("float32"))
lon1 = np.radians(train_data["pickup_longitude"].astype("float32"))
lat2 = np.radians(train_data["dropoff_latitude"].astype("float32"))
lon2 = np.radians(train_data["dropoff_longitude"].astype("float32"))

dlon_pickup = lon_air - lon1
dlat_pickup = lat_air - lat1
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat_air) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
train_data["Pickup_Distance_airport"] = (R * c1 * 0.621).astype("float32")

dlon_dropoff = lon_air - lon2
dlat_dropoff = lat_air - lat2
a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat_air) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
train_data["Dropoff_Distance_airport"] = (R * c2 * 0.621).astype("float32")

lat1 = np.radians(test_data["pickup_latitude"].astype("float32"))
lon1 = np.radians(test_data["pickup_longitude"].astype("float32"))
lat2 = np.radians(test_data["dropoff_latitude"].astype("float32"))
lon2 = np.radians(test_data["dropoff_longitude"].astype("float32"))

dlon_pickup = lon_air - lon1
dlat_pickup = lat_air - lat1
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat_air) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
test_data["Pickup_Distance_airport"] = (R * c1 * 0.621).astype("float32")

dlon_dropoff = lon_air - lon2
dlat_dropoff = lat_air - lat2
a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat_air) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
test_data["Dropoff_Distance_airport"] = (R * c2 * 0.621).astype("float32")



## === cell 10
for col in ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]:
    train_data[col] = np.round(train_data[col], 2)
    test_data[col] = np.round(test_data[col], 2)



## === cell 11
train_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 12
mean_dl = train_data["Difference_longitude"].mean()
var_dl = train_data["Difference_longitude"].var()
train_data["Difference_longitude"] = (
    np.abs(train_data["Difference_longitude"] - mean_dl) / var_dl
)

mean_dlat = train_data["Difference_latitude"].mean()
var_dlat = train_data["Difference_latitude"].var()
train_data["Difference_latitude"] = (
    np.abs(train_data["Difference_latitude"] - mean_dlat) / var_dlat
)

mean_dl_test = test_data["Difference_longitude"].mean()
var_dl_test = test_data["Difference_longitude"].var()
test_data["Difference_longitude"] = (
    np.abs(test_data["Difference_longitude"] - mean_dl_test) / var_dl_test
)

mean_dlat_test = test_data["Difference_latitude"].mean()
var_dlat_test = test_data["Difference_latitude"].var()
test_data["Difference_latitude"] = (
    np.abs(test_data["Difference_latitude"] - mean_dlat_test) / var_dlat_test
)



## === cell 13
from sklearn.model_selection import train_test_split

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]

valid_mask = X.notnull().all(axis=1) & np.isfinite(X).all(axis=1) & y.notnull()
X = X[valid_mask]
y = y[valid_mask]

test_features = test_data.drop("key", axis=1).copy()
test_features = test_features.fillna(test_features.median())

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=80
)



## === cell 14
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

y_train = y_train.clip(lower=0)
y_train_log = np.log1p(y_train)

hgb = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.05,
    max_depth=10,
    max_bins=255,
    random_state=42,
)
hgb.fit(X_train, y_train_log)

valid_pred_log = hgb.predict(X_valid)
valid_pred = np.expm1(valid_pred_log)

rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
print(f"Validation RMSE (original scale): {rmse:.5f}")



## === cell 15
test_features = test_features.reindex(columns=X_train.columns, fill_value=0)

test_pred_log = hgb.predict(test_features)
test_pred = np.round(np.expm1(test_pred_log), 2)



## === cell 16
Submission = pd.DataFrame({"key": test_data["key"], "fare_amount": test_pred})
Submission = Submission[["key", "fare_amount"]]
Submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
