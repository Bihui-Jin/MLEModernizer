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
geopy==2.4.1
lightgbm==4.6.0
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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "0"

INPUT_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
WORKING_DIR = "/kaggle/working"

print("Listing /kaggle/input (truncated):")
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))
    break



## === cell 1
NROWS = 2_000_000  # keep bounded for <600s on Kaggle CPU
CHUNKSIZE = 250_000  # reliability: chunked read avoids memory spikes/timeouts while keeping same data amount

TRAIN_COLS = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
TEST_COLS = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

DTYPES_TRAIN = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
    "key": "string",
}
DTYPES_TEST = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
    "key": "string",
}

train_path = os.path.join(INPUT_DIR, "train.csv")

train_chunks = []
read_rows = 0
for chunk in pd.read_csv(
    train_path,
    usecols=TRAIN_COLS,
    dtype=DTYPES_TRAIN,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
    chunksize=CHUNKSIZE,
):
    train_chunks.append(chunk)
    read_rows += len(chunk)
    if read_rows >= NROWS:
        break
train = pd.concat(train_chunks, ignore_index=True)
if len(train) > NROWS:
    train = train.iloc[:NROWS].copy()

test = pd.read_csv(
    os.path.join(INPUT_DIR, "test.csv"),
    usecols=TEST_COLS,
    dtype=DTYPES_TEST,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
)
sample_submission = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

print(train.shape, test.shape, sample_submission.shape)



## === cell 2
train.isnull().sum()



## === cell 3
train.dropna(inplace=True)



## === cell 4
train.describe()



## === cell 5
train.query("passenger_count > 6")



## === cell 6
train.query("passenger_count < 1")



## === cell 7
train.query("fare_amount < 0")



## === cell 8
train.query("pickup_longitude < -180 or pickup_longitude > 180")



## === cell 9
train.query("dropoff_longitude < -180 or dropoff_longitude > 180")



## === cell 10
train.query("pickup_latitude < -90 or pickup_latitude > 90")



## === cell 11
train.query("dropoff_latitude < -90 or dropoff_latitude > 90")



## === cell 12
train = train.query(
    "1 <= passenger_count <= 6 and "
    "0 <= fare_amount <= 250 and "
    "-180 <= pickup_longitude <= 180 and "
    "-180 <= dropoff_longitude <= 180 and "
    "-90 <= pickup_latitude <= 90 and "
    "-90 <= dropoff_latitude <= 90"
)

train = train[
    (train["pickup_longitude"].between(-74.5, -72.8))
    & (train["dropoff_longitude"].between(-74.5, -72.8))
    & (train["pickup_latitude"].between(40.5, 41.8))
    & (train["dropoff_latitude"].between(40.5, 41.8))
].copy()

train = train[
    ~(
        (train["pickup_longitude"].abs() < 1e-6)
        & (train["pickup_latitude"].abs() < 1e-6)
        & (train["dropoff_longitude"].abs() < 1e-6)
        & (train["dropoff_latitude"].abs() < 1e-6)
    )
].copy()

train = train[
    ~(
        (train["pickup_longitude"] == train["dropoff_longitude"])
        & (train["pickup_latitude"] == train["dropoff_latitude"])
        & (train["fare_amount"] > 20.0)
    )
].copy()

train.describe()



## === cell 13
train.reset_index(drop=True, inplace=True)
train.head()




## === cell 14
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["key"] = df["key"].astype(str)

    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["pickup_hour"] = dt.dt.hour.astype("float32").fillna(-1.0)
    df["pickup_dayofweek"] = dt.dt.dayofweek.astype("float32").fillna(-1.0)
    df["pickup_month"] = dt.dt.month.astype("float32").fillna(-1.0)
    df["pickup_year"] = dt.dt.year.astype("float32").fillna(-1.0)

    df = df.drop("pickup_datetime", axis=1)
    return df


train_fe = add_time_features(train)
test_fe = add_time_features(test)

train_fe.head()




## === cell 15
def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0088 * c  # Earth radius in km


def bearing_rad(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance"] = haversine_km(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    df["abs_lon_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()

    df["manhattan_km"] = haversine_km(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
    ) + haversine_km(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["pickup_longitude"],
    )

    df["center_lat"] = ((df["pickup_latitude"] + df["dropoff_latitude"]) / 2.0).astype(
        "float32"
    )
    df["center_lon"] = (
        (df["pickup_longitude"] + df["dropoff_longitude"]) / 2.0
    ).astype("float32")
    df["bearing"] = bearing_rad(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    ).astype("float32")
    df["bearing_abs"] = np.abs(df["bearing"]).astype("float32")

    return df


train_fe = add_geo_features(train_fe)
test_fe = add_geo_features(test_fe)

train_fe.head()



## === cell 16
test_fe["pickup_longitude"] = test_fe["pickup_longitude"].clip(-74.5, -72.8)
test_fe["dropoff_longitude"] = test_fe["dropoff_longitude"].clip(-74.5, -72.8)
test_fe["pickup_latitude"] = test_fe["pickup_latitude"].clip(40.5, 41.8)
test_fe["dropoff_latitude"] = test_fe["dropoff_latitude"].clip(40.5, 41.8)

train_fe = train_fe[train_fe["distance"].between(0.0, 200.0)].copy()
train_fe.reset_index(drop=True, inplace=True)

test_fe["distance"] = test_fe["distance"].clip(0.0, 200.0)

train_fe.shape, test_fe.shape



## === cell 17
train_fe = train_fe[
    ~((train_fe["distance"] < 0.05) & (train_fe["fare_amount"] > 50.0))
].copy()
train_fe.reset_index(drop=True, inplace=True)
train_fe.shape



## === cell 18
y_train = train_fe["fare_amount"].astype(float)

X_train = train_fe.drop(["fare_amount", "key"], axis=1)
X_test = test_fe.drop(["key"], axis=1)

X_train.head()



## === cell 19
from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train),), dtype=float)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

categorical_features = []



## === cell 20
import lightgbm as lgb

params = {
    "objective": "regression",
    "metric": "rmse",
    "max_bin": 300,
    "learning_rate": 0.05,
    "num_leaves": 40,
    "feature_fraction": 0.9,
    "min_child_samples": 25,
    "seed": 0,
    "num_threads": 1,
    "deterministic": True,
    "feature_pre_filter": False,
}

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
    X_tr = X_train.iloc[train_index, :]
    X_val = X_train.iloc[valid_index, :]
    y_tr = y_train.iloc[train_index]
    y_val = y_train.iloc[valid_index]

    lgb_train = lgb.Dataset(
        X_tr, y_tr, categorical_feature=categorical_features, free_raw_data=False
    )
    lgb_eval = lgb.Dataset(
        X_val,
        y_val,
        reference=lgb_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=10),
        ],
    )

    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    y_preds.append(y_pred)
    models.append(model)



## === cell 21
pd.DataFrame(oof_train).to_csv(
    os.path.join(WORKING_DIR, "oof_train_kfold.csv"), index=False
)

scores = [m.best_score["valid_1"]["rmse"] for m in models]
score = sum(scores) / len(scores)
print("===CV scores===")
print(scores)
print("mean:", score)



## === cell 22
from sklearn.metrics import mean_squared_error

y_pred_oof = oof_train
print("OOF RMSE:", np.sqrt(mean_squared_error(y_train, y_pred_oof)))



## === cell 23
len(y_preds)



## === cell 24
y_preds[0][:10]



## === cell 25
y_sub = sum(y_preds) / len(y_preds)
y_sub = np.clip(y_sub, 0.0, 500.0)
y_sub[:10]



## === cell 26
sub_lgb = pd.DataFrame({"key": test["key"].astype(str).values, "fare_amount": y_sub})

assert list(sub_lgb.columns) == ["key", "fare_amount"]
assert len(sub_lgb) == len(test), (len(sub_lgb), len(test))
assert sub_lgb["key"].isnull().sum() == 0

submission_path = os.path.join(WORKING_DIR, "submission.csv")
sub_lgb.to_csv(submission_path, index=False)

print("Wrote:", submission_path, "shape:", sub_lgb.shape)
sub_lgb.head()
