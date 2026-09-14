# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import pandas as pd
import numpy as np
from pathlib import Path




## === cell 1
def read_csv_fallback(filename: str, **kwargs):
    """Read a CSV from /kaggle/input or a sub‑folder if needed."""
    base = Path("/kaggle/input")
    possible = [
        base / filename,
        base / "new-york-city-taxi-fare-prediction" / filename,
    ]
    for p in possible:
        if p.exists():
            return pd.read_csv(p, **kwargs)
    raise FileNotFoundError(f"{filename} not found in /kaggle/input")


train_dtype = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
test_dtype = {
    "key": "object",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

train_data = read_csv_fallback("train.csv", dtype=train_dtype, low_memory=False)
test_data = read_csv_fallback("test.csv", dtype=test_dtype, low_memory=False)



## === cell 2
train_data.head()



## === cell 3
test_data.head()



## === cell 4
train_data["Difference_longitude"] = np.abs(
    train_data["pickup_longitude"] - train_data["dropoff_longitude"]
)
train_data["Difference_latitude"] = np.abs(
    train_data["pickup_latitude"] - train_data["dropoff_latitude"]
)

test_data["Difference_longitude"] = np.abs(
    test_data["pickup_longitude"] - test_data["dropoff_longitude"]
)
test_data["Difference_latitude"] = np.abs(
    test_data["pickup_latitude"] - test_data["dropoff_latitude"]
)



## === cell 5
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")
train_data = train_data[train_data["fare_amount"] > 0].reset_index(drop=True)
print(f"After removing non‑positive fares: {len(train_data)}")



## === cell 6
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]




## === cell 7
def extract_time(series: pd.Series) -> pd.Series:
    return series.str[11:16]


train_data["pickuptime"] = extract_time(train_data["pickup_datetime"])
test_data["pickuptime"] = extract_time(test_data["pickup_datetime"])




## === cell 8
def weekday_int(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.str[:19]).dt.weekday


train_data["Weekday"] = weekday_int(train_data["pickup_datetime"])
test_data["Weekday"] = weekday_int(test_data["pickup_datetime"])



## === cell 9
train_data.drop(columns=["pickup_datetime"], inplace=True)
test_data.drop(columns=["pickup_datetime"], inplace=True)



## === cell 10
weekday_names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
train_data["Weekday"] = train_data["Weekday"].replace(
    {i: name for i, name in enumerate(weekday_names)}
)
test_data["Weekday"] = test_data["Weekday"].replace(
    {i: name for i, name in enumerate(weekday_names)}
)



## === cell 11
train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])
train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 12
train_data.drop(columns=["Weekday"], inplace=True)
test_data.drop(columns=["Weekday"], inplace=True)




## === cell 13
def hm_to_int(series: pd.Series) -> pd.Series:
    hrs = series.str.slice(0, 2).astype(np.int16)
    mins = series.str.slice(3, 5).astype(np.int16)
    return hrs * 100 + mins


train_data["pickuptime"] = hm_to_int(train_data["pickuptime"])
test_data["pickuptime"] = hm_to_int(test_data["pickuptime"])



## === cell 14
R = 6373.0  # Earth radius in km


def haversine(df):
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c * 0.621  # convert km → miles


train_data["Distance"] = haversine(train_data)
test_data["Distance"] = haversine(test_data)



## === cell 15
airport_lat = np.radians(40.6413111)
airport_lon = np.radians(-73.7781391)


def airport_distance(lat_col, lon_col):
    lat = np.radians(lat_col)
    lon = np.radians(lon_col)
    dlon = airport_lon - lon
    dlat = airport_lat - lat
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat) * np.cos(airport_lat) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c * 0.621


train_data["Pickup_Distance_airport"] = airport_distance(
    train_data["pickup_latitude"], train_data["pickup_longitude"]
)
train_data["Dropoff_Distance_airport"] = airport_distance(
    train_data["dropoff_latitude"], train_data["dropoff_longitude"]
)

test_data["Pickup_Distance_airport"] = airport_distance(
    test_data["pickup_latitude"], test_data["pickup_longitude"]
)
test_data["Dropoff_Distance_airport"] = airport_distance(
    test_data["dropoff_latitude"], test_data["dropoff_longitude"]
)



## === cell 16
dist_cols = ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]
for col in dist_cols:
    train_data[col] = np.round(train_data[col], 2)
    test_data[col] = np.round(test_data[col], 2)



## === cell 17
num_cols = [
    "Difference_longitude",
    "Difference_latitude",
    "Distance",
    "Pickup_Distance_airport",
    "Dropoff_Distance_airport",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "pickuptime",
    "passenger_count",
]
for col in num_cols:
    mean = train_data[col].mean()
    std = train_data[col].std()
    train_data[col] = (train_data[col] - mean) / std
    test_data[col] = (train_data[col] - mean) / std  # apply train stats to test

for base in ["Distance", "passenger_count", "pickuptime"]:
    train_data[f"{base}_sq"] = train_data[base] ** 2
    test_data[f"{base}_sq"] = test_data[base] ** 2



## === cell 18
train_features = set(train_data.columns) - {"key", "fare_amount"}
test_features = set(test_data.columns) - {"key"}

for col in train_features - test_features:
    test_data[col] = 0
for col in test_features - train_features:
    train_data[col] = 0



## === cell 19
X = train_data.drop(columns=["key", "fare_amount"])
y = train_data["fare_amount"]
X_test = test_data.drop(columns=["key"])

X_array = X.values.astype(np.float32)
X_test_array = X_test.values.astype(np.float32)



## === cell 20
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_array, y, test_size=0.01, random_state=80
)



## === cell 21
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

alpha_val = 0.001
ridge_raw = Ridge(
    alpha=alpha_val,
    random_state=80,
    solver="sag",
    max_iter=100,
    tol=1e-3,
    fit_intercept=True,
)  # removed unsupported n_jobs
ridge_raw.fit(X_train, y_train)
pred_raw_val = ridge_raw.predict(X_val)
rmse_raw = mean_squared_error(y_val, pred_raw_val, squared=False)

ridge_log = Ridge(
    alpha=alpha_val,
    random_state=80,
    solver="sag",
    max_iter=100,
    tol=1e-3,
    fit_intercept=True,
)  # removed unsupported n_jobs
y_train_log = np.log1p(y_train.clip(lower=0))
ridge_log.fit(X_train, y_train_log)
pred_log_val = np.expm1(ridge_log.predict(X_val))
rmse_log = mean_squared_error(y_val, pred_log_val, squared=False)

if rmse_log < rmse_raw:
    chosen_model = ridge_log
    use_log = True
    best_rmse = rmse_log
else:
    chosen_model = ridge_raw
    use_log = False
    best_rmse = rmse_raw

print(f"Validation RMSE (raw): {rmse_raw:.4f}")
print(f"Validation RMSE (log): {rmse_log:.4f}")
print(
    f"Chosen model uses {'log‑target' if use_log else 'raw'} with RMSE: {best_rmse:.4f}"
)



## === cell 22
if use_log:
    y_full = np.log1p(y.clip(lower=0))
else:
    y_full = y

chosen_model.fit(X_array, y_full)



## === cell 23
test_pred = chosen_model.predict(X_test_array)
if use_log:
    test_pred = np.expm1(test_pred)

test_pred = np.clip(test_pred, 0, 300)
test_pred = np.round(test_pred, 2)



## === cell 24
Submission = pd.DataFrame({"key": test_data["key"], "fare_amount": test_pred})
Submission = Submission[["key", "fare_amount"]]



## === cell 25
output_path = Path("/kaggle/working/Submission.csv")
Submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
