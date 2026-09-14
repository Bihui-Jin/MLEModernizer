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
ipywidgets==8.1.5
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
protobuf==6.33.0
seaborn==0.12.2
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
tf_keras==2.18.0
tqdm==4.67.1

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

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import HistGradientBoostingRegressor




## === cell 1
def read_csv_sampled(path, frac, chunksize=5 * 10**5, random_state=None):
    """
    Read a CSV in chunks and return a random fraction of rows from each chunk.
    Uses explicit dtypes and column selection to minimise I/O and memory overhead.
    """
    usecols = [
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    dtypes = {
        "key": "string",
        "fare_amount": "float32",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int8",
    }
    samples = []
    for df_chunk in tqdm(
        pd.read_csv(
            path,
            usecols=usecols,
            dtype=dtypes,
            parse_dates=["pickup_datetime"],
            chunksize=chunksize,
        )
    ):
        samples.append(df_chunk.sample(frac=frac, random_state=random_state))
    df = pd.concat(samples, ignore_index=True)
    return df




## === cell 2
df_train_sample = read_csv_sampled("../input/train.csv", 0.01, random_state=1989)



## === cell 3
df_test = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])



## === cell 4
df_train_sample.dropna(inplace=True)

is_weird = df_train_sample["fare_amount"] < 0
is_weird |= ~df_train_sample["pickup_latitude"].between(40, 42)
is_weird |= ~df_train_sample["pickup_longitude"].between(-75, -72)
is_weird |= ~df_train_sample["dropoff_latitude"].between(40, 42)
is_weird |= ~df_train_sample["dropoff_longitude"].between(-75, -72)
is_weird |= df_train_sample["passenger_count"] == 0
df_train_sample = df_train_sample[~is_weird]




## === cell 5
def prep_data(df, shuffle=False, random_state=None):
    """
    Returns feature matrix X and target vector y (if present).
    Adds haversine distance as an extra numerical feature.
    """
    X_cat = np.vstack(
        [
            df["pickup_datetime"].dt.hour,  # 0‑23
            df["pickup_datetime"].dt.weekday + 24,  # 24‑30
            df["pickup_datetime"].dt.dayofyear + 30,  # 31‑396
            df["pickup_datetime"].dt.isocalendar().week + 396,  # 397‑449
            df["pickup_datetime"].dt.year - 2009 + 450,  # 450‑456
            df["passenger_count"] + 456,  # 457‑463
        ]
    ).T.astype(np.float32)

    X_deg = df[
        ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    ].values.astype(np.float32)
    X_deg /= 180.0  # roughly [-1, 1]

    lat1 = np.deg2rad(df["pickup_latitude"].values.astype(np.float32))
    lon1 = np.deg2rad(df["pickup_longitude"].values.astype(np.float32))
    lat2 = np.deg2rad(df["dropoff_latitude"].values.astype(np.float32))
    lon2 = np.deg2rad(df["dropoff_longitude"].values.astype(np.float32))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    dist = 6371.0 * c  # Earth radius in km
    dist = (dist / 100.0).astype(np.float32).reshape(-1, 1)

    X = np.concatenate([X_deg, dist, X_cat], axis=1)

    if "fare_amount" in df.columns:
        y = df["fare_amount"].values.astype(np.float32)
    else:
        y = None

    if shuffle:
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(X))
        X = X[idx]
        if y is not None:
            y = y[idx]

    return X, y




## === cell 6
np.random.seed(1989)
X_train, y_train = prep_data(df_train_sample, shuffle=True, random_state=1989)
y_train = np.log1p(y_train)



## === cell 7
model = HistGradientBoostingRegressor(
    max_iter=2000,
    learning_rate=0.01,
    max_depth=8,
    random_state=1989,
    l2_regularization=0.01,
    loss="squared_error",
)



## === cell 8
model.fit(X_train, y_train)



## === cell 9
X_test, _ = prep_data(df_test, shuffle=False)



## === cell 10
y_test_pred_log = model.predict(X_test)
y_test_pred = np.expm1(y_test_pred_log)
y_test_pred = np.clip(y_test_pred, 0, None)  # fare cannot be negative



## === cell 11
df_sub = pd.DataFrame({"key": df_test["key"].values, "fare_amount": y_test_pred})
df_sub.to_csv("submission.csv", index=False)
