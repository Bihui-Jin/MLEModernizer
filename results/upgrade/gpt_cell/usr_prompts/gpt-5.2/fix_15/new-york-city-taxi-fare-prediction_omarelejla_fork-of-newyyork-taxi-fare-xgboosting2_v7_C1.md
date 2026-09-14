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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

print(
    "Skipping TensorFlow import due to TensorFlow/protobuf incompatibility in this environment."
)



## === cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

train_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

TRAIN_NROWS = 2_000_000

dataset_test = pd.read_csv(test_iop_path, index_col="key")
dataset_train = pd.read_csv(
    train_iop_path,
    usecols=[
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
).sample(n=TRAIN_NROWS, random_state=RANDOM_STATE)

dataset_train = dataset_train.set_index("key")

print("train rows:", len(dataset_train), "test rows:", len(dataset_test))




## === cell 2
def clean_train_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    num_cols = [
        "fare_amount",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=num_cols + ["pickup_datetime"])

    out = out[(out["fare_amount"] > 0) & (out["fare_amount"] <= 250)]
    out = out[(out["passenger_count"] >= 1) & (out["passenger_count"] <= 6)]

    out = out[
        (out["pickup_longitude"].between(-74.3, -73.7))
        & (out["dropoff_longitude"].between(-74.3, -73.7))
        & (out["pickup_latitude"].between(40.5, 41.0))
        & (out["dropoff_latitude"].between(40.5, 41.0))
    ]

    out = out[
        ~(
            (out["pickup_longitude"] == out["dropoff_longitude"])
            & (out["pickup_latitude"] == out["dropoff_latitude"])
        )
    ]

    return out


print("dataset_train old size", len(dataset_train))
dataset_train = clean_train_df(dataset_train)
print("dataset_train new size (cleaned)", len(dataset_train))

print("dataset_test size (kept intact)", len(dataset_test))




## === cell 3
def get_year(pickup_date):
    return pickup_date.year


def get_month(pickup_date):
    return pickup_date.month


def get_day(pickup_date):
    return pickup_date.day


def get_hour(pickup_date):
    return pickup_date.hour




## === cell 4
def _haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def preparedataset2(datasetname: pd.DataFrame) -> pd.DataFrame:
    df = datasetname.copy()

    for c in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")

    df["pickup_year"] = df["pickup_datetime"].dt.year
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_day"] = df["pickup_datetime"].dt.day
    df["pickup_hour"] = df["pickup_datetime"].dt.hour

    df["x_dis"] = df["dropoff_longitude"] - df["pickup_longitude"]
    df["y_dis"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["dis"] = np.sqrt(
        (df["dropoff_longitude"] - df["pickup_longitude"]) ** 2
        + (df["dropoff_latitude"] - df["pickup_latitude"]) ** 2
    )

    df["haversine_km"] = _haversine_km(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    )

    df["abs_x_dis"] = np.abs(df["x_dis"])
    df["abs_y_dis"] = np.abs(df["y_dis"])
    df["manhattan_dis"] = df["abs_x_dis"] + df["abs_y_dis"]

    nyc_lon, nyc_lat = -73.985428, 40.748817  # Midtown Manhattan (approx)
    df["pickup_to_center_km"] = _haversine_km(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        np.full(len(df), nyc_lon),
        np.full(len(df), nyc_lat),
    )
    df["dropoff_to_center_km"] = _haversine_km(
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
        np.full(len(df), nyc_lon),
        np.full(len(df), nyc_lat),
    )

    df = df.drop(["pickup_datetime"], axis=1)

    return df




## === cell 5
df = preparedataset2(dataset_train)
test_df = preparedataset2(dataset_test)

df = df.replace([np.inf, -np.inf], np.nan)
test_df = test_df.replace([np.inf, -np.inf], np.nan)

if "haversine_km" in df.columns:
    df = df[(df["haversine_km"] > 0) & (df["haversine_km"] <= 60)].copy()
if "abs_x_dis" in df.columns and "abs_y_dis" in df.columns:
    df = df[(df["abs_x_dis"] <= 0.30) & (df["abs_y_dis"] <= 0.30)].copy()

if "haversine_km" in df.columns and "pickup_hour" in df.columns:
    df["fare_amount"] = pd.to_numeric(df["fare_amount"], errors="coerce")
    fare_per_km = df["fare_amount"] / df["haversine_km"].clip(lower=0.1)
    df = df[fare_per_km.between(0.5, 50.0)].copy()

df["fare_amount"] = pd.to_numeric(df["fare_amount"], errors="coerce")

feature_cols = [c for c in df.columns if c != "fare_amount"]
train_feature_medians = df[feature_cols].median(numeric_only=True)

df[feature_cols] = df[feature_cols].fillna(train_feature_medians)
df = df.dropna(subset=["fare_amount"]).copy()

test_df = test_df.fillna(train_feature_medians)

print("Prepared train shape:", df.shape)
print("Prepared test shape:", test_df.shape)
df.head(3)



## === cell 6
y = df["fare_amount"]
X = df.drop("fare_amount", axis=1)

X = X.apply(pd.to_numeric, errors="coerce").astype(np.float32)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print("X_train:", X_train.shape, "X_valid:", X_valid.shape)



## === cell 7
best_n_estimators = None
best_learning_rate = None
best_rmse = float("inf")

common_params = dict(
    objective="reg:squarederror",
    max_depth=8,
    min_child_weight=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

for lr in [0.02, 0.03, 0.05, 0.07]:
    for ns in [900, 1200, 1600, 2000]:
        my_model = XGBRegressor(
            n_estimators=ns,
            learning_rate=lr,
            **common_params,
        )
        my_model.fit(X_train, y_train)

        preds = my_model.predict(X_valid)
        rmse = mean_squared_error(y_valid, preds, squared=False)

        if rmse < best_rmse:
            best_rmse = rmse
            best_n_estimators = ns
            best_learning_rate = lr
            print("better found ->", "n_estimators:", ns, "lr:", lr, "RMSE:", rmse)

print("best_n_estimators:", best_n_estimators)
print("best_learning_rate:", best_learning_rate)
print("Best validation RMSE:", best_rmse)



## === cell 8
final_model = XGBRegressor(
    n_estimators=best_n_estimators,
    learning_rate=best_learning_rate,
    **common_params,
)

final_model.fit(X, y)

valid_preds = final_model.predict(X_valid)
valid_rmse = mean_squared_error(y_valid, valid_preds, squared=False)
print("Validation RMSE (refit-on-all model, evaluated on X_valid):", valid_rmse)



## === cell 9
test_df_aligned = test_df.reindex(columns=X.columns)
test_df_aligned = test_df_aligned.apply(pd.to_numeric, errors="coerce").astype(
    np.float32
)
test_df_aligned = test_df_aligned.fillna(train_feature_medians.reindex(X.columns))

test_preds = final_model.predict(test_df_aligned)

test_preds = np.clip(test_preds, 0.0, None)

output = pd.DataFrame(
    {
        "key": test_df_aligned.index,
        "fare_amount": test_preds,
    }
)

output = output[["key", "fare_amount"]]
print("Submission rows:", len(output), "Expected:", len(dataset_test))

output.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
