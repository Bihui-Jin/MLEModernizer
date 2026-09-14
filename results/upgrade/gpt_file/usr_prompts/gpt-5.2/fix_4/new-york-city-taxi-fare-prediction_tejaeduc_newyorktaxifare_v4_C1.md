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

3.7

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

# 5. Target score

3.33042

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 6.63306) has done: 'I keep your XGBoost approach and existing features, but fix two issues that are strongly hurting RMSE: (1) your train/test feature columns are not guaranteed to align after `get_dummies` (different year/day values in train vs test), and (2) you are rounding predictions to cents, which adds extra error under RMSE. I also remove early stopping (it violates your “no early stopping” requirement) while keeping the same number of boosting rounds so the training approach stays the same. Finally, I update the model objective to the non-deprecated equivalent and set a fixed random seed for stable splits and training.'
- What this solution (achieved 5.52924) has done: 'You’re already using a reasonable XGBoost regressor, so the biggest remaining RMSE loss comes from (1) the extremely slow per-row distance/feature engineering (risking timeouts and limiting practical training size) and (2) mild target noise/outliers that survive current filters. I keep the exact same feature set and training approach, but vectorize the distance and datetime-derived features (same formulas, just faster) so you can safely raise `nrows` (more signal) within the 600s budget. I also add one minimal, standard NYC-taxi cleanup: remove zero-distance rides with non-trivial fare (and vice versa) which are frequent data errors and disproportionately hurt RMSE. The submission file path/format stays identical and still writes `finaloutput.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost
import os

print(os.listdir("../input"))



## === cell 1
TRAIN_NROWS = 6_000_000

train_path = "../input/train.csv"
test_path = "../input/test.csv"

usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

df = pd.read_csv(train_path, nrows=TRAIN_NROWS, usecols=usecols_train)
test = pd.read_csv(test_path, usecols=usecols_test)



## === cell 2
testkey = test.key



## === cell 3
df = df.dropna(how="any", axis="rows")



## === cell 4
coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
df = df[(df[coord_cols] != 0.0).all(axis=1)].copy()
test = test[(test[coord_cols] != 0.0).all(axis=1)].copy()



## === cell 5
l = df[
    (df.pickup_latitude > 42.0)
    | (df.pickup_latitude < 40.0)
    | (df.dropoff_latitude > 42.0)
    | (df.dropoff_latitude < 40.0)
    | (df.pickup_longitude > -73.0)
    | (df.pickup_longitude < -75.0)
    | (df.dropoff_longitude > -73.0)
    | (df.dropoff_longitude < -75.0)
].index
df = df.drop(l, axis=0)



## === cell 6
z = df[
    (df.fare_amount > 300.0)
    | (df.fare_amount < 0.0)
    | (df.passenger_count > 7.0)
    | (df.passenger_count < 1.0)
].index
df = df.drop(z, axis=0)



## === cell 7
for c in [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]:
    df[c] = df[c].astype("float32")
    test[c] = test[c].astype("float32")
df["passenger_count"] = df["passenger_count"].astype("int16")
test["passenger_count"] = test["passenger_count"].astype("int16")
df["fare_amount"] = df["fare_amount"].astype("float32")

len(df)




## === cell 8
def haversine_vectorized(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype("float64"))
    lat1 = np.radians(lat1.astype("float64"))
    lon2 = np.radians(lon2.astype("float64"))
    lat2 = np.radians(lat2.astype("float64"))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2.0) ** 2) + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return 6373.0 * c


df["dist"] = haversine_vectorized(
    df["pickup_longitude"].values,
    df["pickup_latitude"].values,
    df["dropoff_longitude"].values,
    df["dropoff_latitude"].values,
).astype("float32")
test["dist"] = haversine_vectorized(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
).astype("float32")



## === cell 9
bad_dist_fare = df[
    ((df["dist"] < 0.01) & (df["fare_amount"] > 5.0))
    | ((df["dist"] > 0.5) & (df["fare_amount"] < 2.5))
].index
df = df.drop(bad_dist_fare, axis=0)



## === cell 10
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])



## === cell 11
df["latenights"] = (df["pickup_datetime"].dt.hour < 5).astype(np.int8)
test["latenights"] = (test["pickup_datetime"].dt.hour < 5).astype(np.int8)

df["weekday"] = (df["pickup_datetime"].dt.weekday > 4).astype(np.int8)
test["weekday"] = (test["pickup_datetime"].dt.weekday > 4).astype(np.int8)

df["year"] = df["pickup_datetime"].dt.year.astype(np.int16)
test["year"] = test["pickup_datetime"].dt.year.astype(np.int16)

df["day"] = df["pickup_datetime"].dt.day.astype(np.int8)
test["day"] = test["pickup_datetime"].dt.day.astype(np.int8)



## === cell 12
feat = df.drop(["key", "pickup_datetime"], axis=1)
test_feat = test.drop(["key", "pickup_datetime"], axis=1)

label = feat["fare_amount"].copy()



## === cell 13
combined = pd.concat(
    [feat.drop(["fare_amount"], axis=1), test_feat], axis=0, ignore_index=True
)

combined = pd.get_dummies(combined, columns=["year", "day"], prefix=["year", "day"])

X = combined.iloc[: len(feat), :].copy()
X_test = combined.iloc[len(feat) :, :].copy()

for col in X.columns:
    if X[col].dtype == "float64":
        X[col] = X[col].astype("float32")
        X_test[col] = X_test[col].astype("float32")



## === cell 14
xtr, xts, ytr, yts = train_test_split(X, label, test_size=0.25, random_state=42)

xgbtrain = xgboost.DMatrix(xtr, label=ytr)
xgbtest = xgboost.DMatrix(xts, label=yts)
xgbfinaltest = xgboost.DMatrix(X_test)



## === cell 15
params = {
    "eval_metric": "rmse",
    "objective": "reg:squarederror",
    "seed": 42,
    "tree_method": "hist",
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eta": 0.1,
}

xgbmodel = xgboost.train(
    params,
    dtrain=xgbtrain,
    num_boost_round=150,
    evals=[(xgbtest, "test")],
    verbose_eval=False,
)



## === cell 16
pred = xgbmodel.predict(xgbfinaltest)

pred = np.clip(pred, 0.0, 300.0)

finalset = pd.DataFrame({"key": testkey, "fare_amount": pred})
finalset = finalset[["key", "fare_amount"]]
finalset.to_csv("finaloutput.csv", index=False)
print("Wrote submission:", "finaloutput.csv", "rows:", len(finalset))

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3692074056.py in <cell line: 0>()
      4 pred = np.clip(pred, 0.0, 300.0)
      5 
----> 6 finalset = pd.DataFrame({"key": testkey, "fare_amount": pred})
      7 finalset = finalset[["key", "fare_amount"]]
      8 finalset.to_csv("finaloutput.csv", index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    688                     f"length {len(index)}"
    689                 )
--> 690                 raise ValueError(msg)
    691         else:
    692             index = default_index(lengths[0])

ValueError: array length 9695 does not match index length 9914
