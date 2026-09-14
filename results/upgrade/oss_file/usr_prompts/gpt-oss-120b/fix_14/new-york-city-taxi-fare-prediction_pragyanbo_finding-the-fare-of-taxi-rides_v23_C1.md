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

# 5. Code solution

## === cell 0
import os, gc
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

sns.set_style("whitegrid")


def locate_file(fname):
    possible_dirs = [
        "/kaggle/input/new-york-city-taxi-fare-prediction",
        "/kaggle/input",
        os.path.abspath(os.path.join(os.getcwd(), "..", "..", "input")),
    ]
    for d in possible_dirs:
        p = os.path.join(d, fname)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Could not locate {fname}")


def distance(lat1, lon1, lat2, lon2):
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))


train_usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_map = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
train_path = locate_file("train.csv")

chunk_iter = pd.read_csv(
    train_path,
    usecols=train_usecols,
    dtype=dtype_map,
    parse_dates=False,
    low_memory=False,
    chunksize=500_000,
)

feat_cols = [
    "distance",
    "passenger_count",
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
    "is_weekend",
]

train_chunks = []
target_chunks = []

for chunk in chunk_iter:
    chunk.dropna(inplace=True)
    chunk = chunk[chunk["fare_amount"] > 0]

    chunk["distance"] = distance(
        chunk["pickup_latitude"],
        chunk["pickup_longitude"],
        chunk["dropoff_latitude"],
        chunk["dropoff_longitude"],
    )
    chunk = chunk[chunk["distance"] < 15]

    dt = pd.to_datetime(chunk["pickup_datetime"], errors="raise")
    chunk["pickup_hour"] = dt.dt.hour.astype("int8")
    chunk["pickup_weekday"] = dt.dt.weekday.astype("int8")
    chunk["pickup_month"] = dt.dt.month.astype("int8")
    chunk["is_weekend"] = (chunk["pickup_weekday"] >= 5).astype("int8")

    train_chunks.append(chunk[feat_cols].astype("float32"))
    target_chunks.append(chunk["fare_amount"])

X = pd.concat(train_chunks, ignore_index=True)
y = pd.concat(target_chunks, ignore_index=True)
del train_chunks, target_chunks, chunk_iter
gc.collect()
print("Training data shape after cleaning:", X.shape)



## === cell 1
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_dtype_map = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
test_path = locate_file("test.csv")
test_df = pd.read_csv(
    test_path,
    usecols=test_usecols,
    dtype=test_dtype_map,
    parse_dates=False,
    low_memory=False,
)
print("Test shape:", test_df.shape)



## === cell 2
test_df["distance"] = distance(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
)
dt_test = pd.to_datetime(test_df["pickup_datetime"], errors="raise")
test_df["pickup_hour"] = dt_test.dt.hour.astype("int8")
test_df["pickup_weekday"] = dt_test.dt.weekday.astype("int8")
test_df["pickup_month"] = dt_test.dt.month.astype("int8")
test_df["is_weekend"] = (test_df["pickup_weekday"] >= 5).astype("int8")
test_df.drop(columns=["pickup_datetime"], inplace=True)
gc.collect()



## === cell 3
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)



## === cell 4
import xgboost as xgb


def train_xgboost(X_tr, X_va, y_tr, y_va, num_rounds=800):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 42,
        "eta": 0.05,
        "max_depth": 8,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "max_bin": 128,
        "nthread": -1,
        "predictor": "cpu_predictor",
    }
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        evals=[(dval, "validation")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    return booster




## === cell 5
xgb_model = train_xgboost(X_train, X_val, y_train, y_val)



## === cell 6
test_pred = xgb_model.predict(xgb.DMatrix(test_df[feat_cols].astype("float32")))



## === cell 7
test_pred = np.clip(test_pred, a_min=0, a_max=None)
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
