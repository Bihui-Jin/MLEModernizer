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

3.12

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

5.54066

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import matplotlib.pyplot as plt  # plotting library
from sklearn.linear_model import Ridge  # regularized linear regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def distance_on_the_sphere(lat1, lon1, lat2, lon2):
    earth_radius = 6371.0  # km
    phi1 = np.radians(lat1.astype(np.float32))
    phi2 = np.radians(lat2.astype(np.float32))
    delta_phi = np.radians((lat2 - lat1).astype(np.float32))
    delta_lambda = np.radians((lon2 - lon1).astype(np.float32))
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return earth_radius * c


nyc_down_town = (-74.0063889, 40.7141667)  # (lon, lat)

features = [
    "hour",
    "year",
    "distance",
    "log_distance",
    "passenger_count",
    "is_rush_hour",
    "day_of_week",
    "distance_to_downtown",
]
target = "fare_amount"

train_path = os.path.join(
    "/kaggle/input",
    "new-york-city-taxi-fare-prediction",
    "train.csv",
)

dtype = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
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

new_york_box = (-74.763379, -72.856164, 40.502009, 41.915509)

chunks = pd.read_csv(
    train_path,
    usecols=usecols,
    dtype=dtype,
    memory_map=True,
    chunksize=5_000_000,  # was 500_000
)

X_parts = []
y_log_parts = []

for chunk in chunks:
    chunk = chunk.drop(columns=["key"])
    chunk = chunk[chunk["fare_amount"] >= 0.1]
    chunk = chunk.dropna()

    mask = (
        (chunk["pickup_longitude"] >= new_york_box[0])
        & (chunk["pickup_longitude"] <= new_york_box[1])
        & (chunk["pickup_latitude"] >= new_york_box[2])
        & (chunk["pickup_latitude"] <= new_york_box[3])
        & (chunk["dropoff_longitude"] >= new_york_box[0])
        & (chunk["dropoff_longitude"] <= new_york_box[1])
        & (chunk["dropoff_latitude"] >= new_york_box[2])
        & (chunk["dropoff_latitude"] <= new_york_box[3])
    )
    chunk = chunk[mask]

    chunk["distance"] = distance_on_the_sphere(
        chunk["pickup_latitude"],
        chunk["pickup_longitude"],
        chunk["dropoff_latitude"],
        chunk["dropoff_longitude"],
    ).astype(np.float32)

    chunk["log_distance"] = np.log1p(chunk["distance"]).astype(np.float32)

    chunk["distance_to_downtown"] = distance_on_the_sphere(
        nyc_down_town[1],
        nyc_down_town[0],
        chunk["pickup_latitude"],
        chunk["pickup_longitude"],
    ).astype(np.float32)

    chunk["pickup_datetime"] = pd.to_datetime(chunk["pickup_datetime"])
    chunk["hour"] = chunk["pickup_datetime"].dt.hour.astype(np.uint8)
    chunk["year"] = chunk["pickup_datetime"].dt.year.astype(np.int16)
    chunk["day_of_week"] = chunk["pickup_datetime"].dt.dayofweek.astype(np.uint8)
    chunk["is_rush_hour"] = (
        ((chunk["hour"] >= 7) & (chunk["hour"] <= 10))
        | ((chunk["hour"] >= 16) & (chunk["hour"] <= 19))
    ).astype(np.uint8)

    row_mask = (chunk["passenger_count"] != 0) & (chunk["distance_to_downtown"] < 15)

    X_chunk = chunk.loc[row_mask, features].values.astype(np.float32, copy=False)
    y_chunk = np.log1p(
        chunk.loc[row_mask, target].values.astype(np.float32, copy=False)
    )

    X_parts.append(X_chunk)
    y_log_parts.append(y_chunk)

X = np.ascontiguousarray(np.concatenate(X_parts, axis=0), dtype=np.float32)
y_log = np.ascontiguousarray(np.concatenate(y_log_parts, axis=0), dtype=np.float32)

print(f"Total training samples after all filtering: {X.shape[0]}")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1357579358.py in <cell line: 0>()
    113     chunk["log_distance"] = np.log1p(chunk["distance"]).astype(np.float32)
    114 
--> 115     chunk["distance_to_downtown"] = distance_on_the_sphere(
    116         nyc_down_town[1],
    117         nyc_down_town[0],

/tmp/ipykernel_11/1357579358.py in distance_on_the_sphere(lat1, lon1, lat2, lon2)
     14 def distance_on_the_sphere(lat1, lon1, lat2, lon2):
     15     earth_radius = 6371.0  # km
---> 16     phi1 = np.radians(lat1.astype(np.float32))
     17     phi2 = np.radians(lat2.astype(np.float32))
     18     delta_phi = np.radians((lat2 - lat1).astype(np.float32))

AttributeError: 'float' object has no attribute 'astype'

## === cell 1
X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.25, random_state=42
)

linear_model = make_pipeline(
    StandardScaler(), Ridge(alpha=0.1, random_state=42, solver="sag")
)
linear_model.fit(X_train, y_train_log)

y_val_pred_log = linear_model.predict(X_val)
y_val_pred = np.expm1(y_val_pred_log)
y_val = np.expm1(y_val_log)
val_rmse = np.sqrt(((y_val - y_val_pred) ** 2).mean())
print(f"Validation RMSE (raw fare): {val_rmse:.5f}")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1518858942.py in <cell line: 0>()
      1 X_train, X_val, y_train_log, y_val_log = train_test_split(
----> 2     X, y_log, test_size=0.25, random_state=42
      3 )
      4 
      5 linear_model = make_pipeline(

NameError: name 'X' is not defined

## === cell 2
test_path = os.path.join(
    "/kaggle/input",
    "new-york-city-taxi-fare-prediction",
    "test.csv",
)
test_dtype = {
    "key": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

test_data_set = pd.read_csv(
    test_path,
    usecols=test_usecols,
    dtype=test_dtype,
    memory_map=True,
)

test_data_set["distance"] = distance_on_the_sphere(
    test_data_set["pickup_latitude"],
    test_data_set["pickup_longitude"],
    test_data_set["dropoff_latitude"],
    test_data_set["dropoff_longitude"],
).astype(np.float32)

test_data_set["log_distance"] = np.log1p(test_data_set["distance"]).astype(np.float32)

test_data_set["distance_to_downtown"] = distance_on_the_sphere(
    nyc_down_town[1],
    nyc_down_town[0],
    test_data_set["pickup_latitude"],
    test_data_set["pickup_longitude"],
).astype(np.float32)

test_data_set["pickup_datetime"] = pd.to_datetime(test_data_set["pickup_datetime"])
test_data_set["hour"] = test_data_set["pickup_datetime"].dt.hour.astype(np.uint8)
test_data_set["year"] = test_data_set["pickup_datetime"].dt.year.astype(np.int16)
test_data_set["day_of_week"] = test_data_set["pickup_datetime"].dt.dayofweek.astype(
    np.uint8
)
test_data_set["is_rush_hour"] = (
    ((test_data_set["hour"] >= 7) & (test_data_set["hour"] <= 10))
    | ((test_data_set["hour"] >= 16) & (test_data_set["hour"] <= 19))
).astype(np.uint8)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4238339466.py in <cell line: 0>()
     38 test_data_set["log_distance"] = np.log1p(test_data_set["distance"]).astype(np.float32)
     39 
---> 40 test_data_set["distance_to_downtown"] = distance_on_the_sphere(
     41     nyc_down_town[1],
     42     nyc_down_town[0],

/tmp/ipykernel_11/1357579358.py in distance_on_the_sphere(lat1, lon1, lat2, lon2)
     14 def distance_on_the_sphere(lat1, lon1, lat2, lon2):
     15     earth_radius = 6371.0  # km
---> 16     phi1 = np.radians(lat1.astype(np.float32))
     17     phi2 = np.radians(lat2.astype(np.float32))
     18     delta_phi = np.radians((lat2 - lat1).astype(np.float32))

AttributeError: 'float' object has no attribute 'astype'

## === cell 3
X_test = test_data_set[features].values.astype(np.float32, copy=False)
y_pred_log = linear_model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # inverse of log1p
y_pred = np.maximum(y_pred, 0.0)

submission = pd.DataFrame(
    {"key": test_data_set["key"], "fare_amount": y_pred},
    columns=["key", "fare_amount"],
)

os.makedirs("./output", exist_ok=True)
submission.to_csv("./output/submission.csv", index=False)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1497339186.py in <cell line: 0>()
----> 1 X_test = test_data_set[features].values.astype(np.float32, copy=False)
      2 y_pred_log = linear_model.predict(X_test)
      3 y_pred = np.expm1(y_pred_log)  # inverse of log1p
      4 y_pred = np.maximum(y_pred, 0.0)
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['hour', 'year', 'is_rush_hour', 'day_of_week', 'distance_to_downtown'] not in index"
