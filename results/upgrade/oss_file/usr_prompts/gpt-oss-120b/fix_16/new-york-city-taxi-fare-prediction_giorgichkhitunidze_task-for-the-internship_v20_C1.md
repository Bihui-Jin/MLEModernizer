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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

3.33144

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.32296) has done: 'I fix the prediction step by removing the unsupported `ntree_limit` argument from `model.predict`. This aligns the call with the current XGBoost API, allowing the script to generate predictions, flatten them if needed, and write a proper CSV submission file.'
- What this solution (achieved 5.17255) has done: 'I increase the model capacity and add early‑stopping to let XGBoost pick the optimal number of trees, then clip any negative fare predictions to zero (fares can’t be negative). These modest tweaks keep the original pipeline intact while aiming to lower the validation RMSE toward the target.'

# 9. Code solution

## === cell 0
import pathlib, os
import pandas as pd
import numpy as np
import gc


def get_path(*parts):
    base_paths = [
        pathlib.Path("/kaggle/input"),
        pathlib.Path("../input"),
        pathlib.Path("./input"),
    ]
    for base in base_paths:
        candidate = base.joinpath(*parts)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"File not found: {'/'.join(parts)}")




## === cell 1
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
dtype = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
train_path = get_path("new-york-city-taxi-fare-prediction", "train.csv")
test_path = get_path("new-york-city-taxi-fare-prediction", "test.csv")

train_usecols = [c for c in usecols if c != "key"]
chunks = []
CHUNK_SIZE = 2_000_000
for chunk in pd.read_csv(
    train_path,
    usecols=train_usecols,
    dtype={k: v for k, v in dtype.items() if k != "key"},
    low_memory=True,
    chunksize=CHUNK_SIZE,
):
    chunk = chunk.dropna(
        axis=0,
        subset=[
            "dropoff_longitude",
            "dropoff_latitude",
            "pickup_longitude",
            "pickup_latitude",
        ],
    )
    mask = (
        (chunk["pickup_longitude"] >= -180)
        & (chunk["pickup_longitude"] <= 180)
        & (chunk["pickup_latitude"] >= -90)
        & (chunk["pickup_latitude"] <= 90)
        & (chunk["dropoff_longitude"] >= -180)
        & (chunk["dropoff_longitude"] <= 180)
        & (chunk["dropoff_latitude"] >= -90)
        & (chunk["dropoff_latitude"] <= 90)
        & (chunk["pickup_longitude"] >= -75)
        & (chunk["pickup_longitude"] <= -72)
        & (chunk["dropoff_longitude"] >= -75)
        & (chunk["dropoff_longitude"] <= -72)
        & (chunk["pickup_latitude"] >= 40)
        & (chunk["pickup_latitude"] <= 42)
        & (chunk["dropoff_latitude"] >= 40)
        & (chunk["dropoff_latitude"] <= 42)
        & (chunk["passenger_count"] > 0)
        & (chunk["fare_amount"] > 0)
    )
    chunk = chunk.loc[mask]
    chunks.append(chunk)

train_df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

idx = train_df[train_df["pickup_longitude"] >= 40].index
if not idx.empty:
    train_df.loc[idx, ["pickup_longitude", "pickup_latitude"]] = train_df.loc[
        idx, ["pickup_latitude", "pickup_longitude"]
    ].values
    train_df.loc[idx, ["dropoff_longitude", "dropoff_latitude"]] = train_df.loc[
        idx, ["dropoff_latitude", "dropoff_longitude"]
    ].values

test_df = pd.read_csv(
    test_path,
    usecols=[c for c in usecols if c != "fare_amount"],
    dtype={k: v for k, v in dtype.items() if k != "fare_amount"},
    low_memory=True,
)




## === cell 2
pd.set_option("display.float_format", lambda x: "%.5f" % x)




## === cell 3
train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)
test_df["pickup_datetime"] = pd.to_datetime(
    test_df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)


def date_splitter(df):
    df.dropna(subset=["pickup_datetime"], inplace=True)
    df["Year"] = df["pickup_datetime"].dt.year.astype(np.int16)
    df["Month"] = df["pickup_datetime"].dt.month.astype(np.int8)
    df["Day"] = df["pickup_datetime"].dt.day.astype(np.int8)
    df["Weekday"] = df["pickup_datetime"].dt.dayofweek.astype(np.int8)
    df["Hour"] = df["pickup_datetime"].dt.hour.astype(np.int8)


date_splitter(train_df)
date_splitter(test_df)

train_df.drop(columns=["pickup_datetime"], inplace=True)
test_df.drop(columns=["pickup_datetime"], inplace=True)




## === cell 4
def haversine_distance(df):
    phi1 = np.radians(df["pickup_latitude"])
    phi2 = np.radians(df["dropoff_latitude"])
    lambda1 = np.radians(df["pickup_longitude"])
    lambda2 = np.radians(df["dropoff_longitude"])
    R = 6371.0
    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["Distance"] = (R * c).astype(np.float32)


haversine_distance(train_df)
haversine_distance(test_df)




## === cell 5
features = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "Year",
    "Month",
    "Day",
    "Weekday",
    "Hour",
    "Distance",
]




## === cell 6
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

np.random.seed(42)

X = train_df[features]
y = train_df["fare_amount"]
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_train = X_train.to_numpy(dtype=np.float32)
X_valid = X_valid.to_numpy(dtype=np.float32)
y_train = y_train.to_numpy(dtype=np.float32)
y_valid = y_valid.to_numpy(dtype=np.float32)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2859411909.py in <cell line: 0>()
      7 X = train_df[features]
      8 y = train_df["fare_amount"]
----> 9 X_train, X_valid, y_train, y_valid = train_test_split(
     10     X, y, test_size=0.30, random_state=42
     11 )

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in train_test_split(test_size, train_size, random_state, shuffle, stratify, *arrays)
   2560 
   2561     n_samples = _num_samples(arrays[0])
-> 2562     n_train, n_test = _validate_shuffle_split(
   2563         n_samples, test_size, train_size, default_test_size=0.25
   2564     )

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in _validate_shuffle_split(n_samples, test_size, train_size, default_test_size)
   2234 
   2235     if n_train == 0:
-> 2236         raise ValueError(
   2237             "With n_samples={}, test_size={} and train_size={}, the "
   2238             "resulting train set will be empty. Adjust any of the "

ValueError: With n_samples=0, test_size=0.3 and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters.

## === cell 7
test_features = test_df[features].to_numpy(dtype=np.float32)




## === cell 8
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=12,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    n_jobs=5,
    random_state=42,
    verbosity=0,
    tree_method="hist",
    max_bin=256,
)




## === cell 9
model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    early_stopping_rounds=30,
    verbose=False,
)
val_pred = model.predict(X_valid)
rmse = mean_squared_error(y_valid, val_pred, squared=False)
print(f"Validation RMSE: {rmse:.5f}")




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/860471682.py in <cell line: 0>()
      1 model.fit(
----> 2     X_train,
      3     y_train,
      4     eval_set=[(X_valid, y_valid)],
      5     early_stopping_rounds=30,

NameError: name 'X_train' is not defined

## === cell 10
prediction = model.predict(test_features)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_55/1158422961.py in <cell line: 0>()
----> 1 prediction = model.predict(test_features)
      2 
      3 

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in predict(self, X, output_margin, validate_features, base_margin, iteration_range)
   1166             if self._can_use_inplace_predict():
   1167                 try:
-> 1168                     predts = self.get_booster().inplace_predict(
   1169                         data=X,
   1170                         iteration_range=iteration_range,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in get_booster(self)
    723             from sklearn.exceptions import NotFittedError
    724 
--> 725             raise NotFittedError("need to call fit or load_model beforehand")
    726         return self._Booster
    727 

NotFittedError: need to call fit or load_model beforehand

## === cell 11
prediction = np.clip(prediction, 0, None)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3382540159.py in <cell line: 0>()
----> 1 prediction = np.clip(prediction, 0, None)
      2 
      3 

NameError: name 'prediction' is not defined

## === cell 12
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": prediction})
submission_path = "taxi_fare_submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1920709038.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"key": test_df["key"], "fare_amount": prediction})
      2 submission_path = "taxi_fare_submission.csv"
      3 submission.to_csv(submission_path, index=False)
      4 print(f"Submission written to {submission_path}")

NameError: name 'prediction' is not defined
