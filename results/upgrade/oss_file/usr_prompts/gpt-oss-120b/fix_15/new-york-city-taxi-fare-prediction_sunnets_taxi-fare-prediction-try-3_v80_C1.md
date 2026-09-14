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

3.7

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
        input/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
```

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> input/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> input/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> input/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> working/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

np.random.seed(42)  # ensure reproducibility for any NumPy randomness
pd.options.mode.chained_assignment = None  # silence SettingWithCopy warnings



## === cell 1
TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/labels.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"
SUBMISSION_PATH = f"/kaggle/working/{SUBMISSION_NAME}"




## === cell 2
def load_data(train_path, test_path, sample_n=800_000, random_state=42):
    """Read the massive training CSV in chunks, sample `sample_n` rows,
    and load the test set with proper datetime parsing."""
    dtypes = {
        "key": "object",
        "fare_amount": "float32",
        "pickup_datetime": "object",  # keep as string for train, will convert later
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int8",
    }

    chunksize = 500_000
    target_rows = sample_n * 2  # read enough rows to get a good random sample
    collected = []
    rows_read = 0

    for chunk in pd.read_csv(
        train_path,
        dtype=dtypes,
        usecols=list(dtypes.keys()),
        chunksize=chunksize,
    ):
        collected.append(chunk)
        rows_read += len(chunk)
        if rows_read >= target_rows:
            break

    train_full = pd.concat(collected, ignore_index=True)
    train = train_full.sample(n=sample_n, random_state=random_state).reset_index(
        drop=True
    )
    train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"], utc=True)

    test_dtypes = {
        k: v for k, v in dtypes.items() if k != "fare_amount" and k != "pickup_datetime"
    }  # keep datetime column as parsed, not forced to object
    test = pd.read_csv(
        test_path,
        dtype=test_dtypes,
        parse_dates=["pickup_datetime"],
    )
    return train, test




## === cell 3
def clean_data(df, is_train=True):
    df = df.dropna(
        subset=[
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]
    )
    lon_min, lon_max = -74.5, -73.0
    lat_min, lat_max = 40.0, 41.0
    cond = (
        (df["pickup_longitude"].between(lon_min, lon_max))
        & (df["dropoff_longitude"].between(lon_min, lon_max))
        & (df["pickup_latitude"].between(lat_min, lat_max))
        & (df["dropoff_latitude"].between(lat_min, lat_max))
    )
    df = df[cond]
    df = df[df["passenger_count"].between(1, 6)]
    if is_train:
        df = df[df["fare_amount"].between(0, 200)]
    return df.reset_index(drop=True)


def add_time_features(df):
    df["hour"] = df["pickup_datetime"].dt.hour.astype(np.int8)
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek.astype(np.int8)
    df["month"] = df["pickup_datetime"].dt.month.astype(np.int8)
    return df


def haversine_np(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_distance_features(df):
    df["haversine_km"] = haversine_np(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )
    km_per_deg_lat = 111.0
    km_per_deg_lon = 85.0  # average at NYC latitude
    df["manhattan_km"] = (
        np.abs(df["pickup_latitude"] - df["dropoff_latitude"]) * km_per_deg_lat
        + np.abs(df["pickup_longitude"] - df["dropoff_longitude"]) * km_per_deg_lon
    )
    df["euclidean_km"] = np.sqrt(
        (np.abs(df["pickup_latitude"] - df["dropoff_latitude"]) * km_per_deg_lat) ** 2
        + (np.abs(df["pickup_longitude"] - df["dropoff_longitude"]) * km_per_deg_lon)
        ** 2
    )
    return df


def prepare_features(df, is_train=True):
    df = add_time_features(df)
    df = add_distance_features(df)
    feature_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "hour",
        "dayofweek",
        "month",
        "haversine_km",
        "manhattan_km",
        "euclidean_km",
    ]
    X = df[feature_cols].astype(np.float32)
    if is_train:
        y = np.log1p(df["fare_amount"].values)  # log‑transform target
        return X, y
    else:
        return X




## === cell 4
train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

train_df = clean_data(train_df, is_train=True)
test_df = clean_data(test_df, is_train=False)  # test has no fare column

train_split, val_split = train_test_split(
    train_df, test_size=0.2, random_state=42, shuffle=True
)

X_train, y_train = prepare_features(train_split, is_train=True)
X_val, y_val = prepare_features(val_split, is_train=True)
X_test = prepare_features(test_df, is_train=False)



## === cell 5
model = GradientBoostingRegressor(
    n_estimators=800,  # more trees for better fit
    learning_rate=0.03,  # smaller step size
    max_depth=5,  # slightly deeper trees
    subsample=0.9,  # a bit more data per iteration
    max_features=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

val_pred_log = model.predict(X_val)
val_pred = np.expm1(val_pred_log)
val_true = np.expm1(y_val)
rmse = np.sqrt(mean_squared_error(val_true, val_pred))
print(f"Validation RMSE: {rmse:.5f}")



## === cell 6
test_pred_log = model.predict(X_test)
test_pred = np.expm1(test_pred_log)

submission = pd.DataFrame(
    {
        "key": test_df["key"],
        "fare_amount": np.clip(test_pred, 0, None),  # ensure non‑negative fares
    }
)
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}")
