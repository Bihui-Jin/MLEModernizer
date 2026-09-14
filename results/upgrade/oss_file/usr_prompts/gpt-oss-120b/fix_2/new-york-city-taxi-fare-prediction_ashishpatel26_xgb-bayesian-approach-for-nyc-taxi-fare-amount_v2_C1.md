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
geopy==2.4.1
lightgbm==4.6.0
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

# 5. Target score

4.22957

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopy.distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

plt.style.use("fivethirtyeight")
print("Input directory contents:", os.listdir("../input"))
import gc




## === cell 1
def load_Data():
    train = pd.read_csv("../input/train.csv", nrows=1_000_000, low_memory=True)
    test = pd.read_csv("../input/test.csv", nrows=1_000_000, low_memory=True)
    return train, test


def prepare_distance_features(df):
    df["longitude_distance"] = np.abs(df["pickup_longitude"] - df["dropoff_longitude"])
    df["latitude_distance"] = np.abs(df["pickup_latitude"] - df["dropoff_latitude"])
    df["distance_travelled"] = np.sqrt(
        df["longitude_distance"] ** 2 + df["latitude_distance"] ** 2
    )
    df["distance_travelled_sin"] = np.sin(df["distance_travelled"])
    df["distance_travelled_cos"] = np.cos(df["distance_travelled"])
    df["distance_travelled_sin_sqrd"] = np.sin(df["distance_travelled"]) ** 2
    df["distance_travelled_cos_sqrd"] = np.cos(df["distance_travelled"]) ** 2

    R = 6371e3  # metres
    phi1 = np.radians(df["pickup_latitude"])
    phi2 = np.radians(df["dropoff_latitude"])
    phi_chg = np.radians(df["pickup_latitude"] - df["dropoff_latitude"])
    delta_chg = np.radians(df["pickup_longitude"] - df["dropoff_longitude"])
    a = np.sin(phi_chg / 2) + np.cos(phi1) * np.cos(phi2) * np.sin(delta_chg / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["haversine"] = R * c

    y = np.sin(delta_chg * np.cos(phi2))
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    df["bearing"] = np.arctan2(y, x)
    return df


def prepare_time_features(df):
    df["pickup_datetime"] = df["pickup_datetime"].str.replace(" UTC", "")
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    df["hour_of_day"] = df["pickup_datetime"].dt.hour
    df["week"] = df["pickup_datetime"].dt.isocalendar().week.astype(int)
    df["month"] = df["pickup_datetime"].dt.month
    df["year"] = df["pickup_datetime"].dt.year
    df["day_of_year"] = df["pickup_datetime"].dt.dayofyear
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["quarter"] = df["pickup_datetime"].dt.quarter
    df["day_of_month"] = df["pickup_datetime"].dt.day
    return df




## === cell 2
train, test = load_Data()



## === cell 3
train = prepare_time_features(train)
train = prepare_distance_features(train)

test = prepare_time_features(test)
test = prepare_distance_features(test)




## === cell 4
def compute_geodesic(row):
    return geopy.distance.geodesic(
        (row["pickup_latitude"], row["pickup_longitude"]),
        (row["dropoff_latitude"], row["dropoff_longitude"]),
    ).km


train["distance"] = train.apply(compute_geodesic, axis=1)
test["distance"] = test.apply(compute_geodesic, axis=1)

train.drop(
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_datetime",
    ],
    axis=1,
    inplace=True,
)
test.drop(
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_datetime",
    ],
    axis=1,
    inplace=True,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2172584988.py in <cell line: 0>()
      7 
      8 
----> 9 train["distance"] = train.apply(compute_geodesic, axis=1)
     10 test["distance"] = test.apply(compute_geodesic, axis=1)
     11 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
  10372             kwargs=kwargs,
  10373         )
> 10374         return op.apply().__finalize__(self, method="apply")
  10375 
  10376     def map(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
    914             return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
    915 
--> 916         return self.apply_standard()
    917 
    918     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1061     def apply_standard(self):
   1062         if self.engine == "python":
-> 1063             results, res_index = self.apply_series_generator()
   1064         else:
   1065             results, res_index = self.apply_series_numba()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_series_generator(self)
   1079             for i, v in enumerate(series_gen):
   1080                 # ignore SettingWithCopy here in case the user mutates
-> 1081                 results[i] = self.func(v, *self.args, **self.kwargs)
   1082                 if isinstance(results[i], ABCSeries):
   1083                     # If we have a view on v, we need to make a copy because

/tmp/ipykernel_11/2172584988.py in compute_geodesic(row)
      1 # Compute geodesic distance (km) using geopy (replaces removed VincentyDistance)
      2 def compute_geodesic(row):
----> 3     return geopy.distance.geodesic(
      4         (row["pickup_latitude"], row["pickup_longitude"]),
      5         (row["dropoff_latitude"], row["dropoff_longitude"]),

/usr/local/lib/python3.11/dist-packages/geopy/distance.py in __init__(self, *args, **kwargs)
    538         self.set_ellipsoid(kwargs.pop('ellipsoid', 'WGS-84'))
    539         major, minor, f = self.ELLIPSOID
--> 540         super().__init__(*args, **kwargs)
    541 
    542     def set_ellipsoid(self, ellipsoid):

/usr/local/lib/python3.11/dist-packages/geopy/distance.py in __init__(self, *args, **kwargs)
    274         elif len(args) > 1:
    275             for a, b in util.pairwise(args):
--> 276                 kilometers += self.measure(a, b)
    277 
    278         kilometers += units.kilometers(**kwargs)

/usr/local/lib/python3.11/dist-packages/geopy/distance.py in measure(self, a, b)
    554 
    555     def measure(self, a, b):
--> 556         a, b = Point(a), Point(b)
    557         _ensure_same_altitude(a, b)
    558         lat1, lon1 = a.latitude, a.longitude

/usr/local/lib/python3.11/dist-packages/geopy/point.py in __new__(cls, latitude, longitude, altitude)
    173                     )
    174                 else:
--> 175                     return cls.from_sequence(seq)
    176 
    177         if single_arg:

/usr/local/lib/python3.11/dist-packages/geopy/point.py in from_sequence(cls, seq)
    470             raise ValueError('When creating a Point from sequence, it '
    471                              'must not have more than 3 items.')
--> 472         return cls(*args)
    473 
    474     @classmethod

/usr/local/lib/python3.11/dist-packages/geopy/point.py in __new__(cls, latitude, longitude, altitude)
    186 
    187         latitude, longitude, altitude = \
--> 188             _normalize_coordinates(latitude, longitude, altitude)
    189 
    190         self = super().__new__(cls)

/usr/local/lib/python3.11/dist-packages/geopy/point.py in _normalize_coordinates(latitude, longitude, altitude)
     72                       '(latitude, longitude) or (y, x) in Cartesian terms.',
     73                       UserWarning, stacklevel=3)
---> 74         raise ValueError('Latitude must be in the [-90; 90] range.')
     75 
     76     if abs(longitude) > 180:

ValueError: Latitude must be in the [-90; 90] range.

## === cell 5
train["key2"] = pd.to_datetime(train["key"], errors="coerce")
test["key2"] = pd.to_datetime(test["key"], errors="coerce")

for df in [train, test]:
    df["year"] = df["key2"].dt.year
    df["month"] = df["key2"].dt.month
    df["day"] = df["key2"].dt.day
    df["day_of_week"] = df["key2"].dt.weekday
    df["hour"] = df["key2"].dt.hour



## === cell 6
feature_cols = [
    "passenger_count",
    "distance",
    "year",
    "month",
    "day",
    "day_of_week",
    "hour",
]
X = train[feature_cols]
y = train["fare_amount"]



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3106547318.py in <cell line: 0>()
      9     "hour",
     10 ]
---> 11 X = train[feature_cols]
     12 y = train["fare_amount"]
     13 

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

KeyError: "['distance'] not in index"

## === cell 7
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3013028905.py in <cell line: 0>()
      1 # Train‑validation split for a quick sanity check
----> 2 X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
      3 

NameError: name 'X' is not defined

## === cell 8
lgb = LGBMRegressor(
    objective="regression",
    num_leaves=5,
    learning_rate=0.05,
    n_estimators=720,
    max_bin=55,
    bagging_fraction=0.8,
    bagging_freq=5,
    feature_fraction=0.2319,
    feature_fraction_seed=9,
    bagging_seed=9,
    min_data_in_leaf=6,
    min_sum_hessian_in_leaf=11,
    verbose=-1,
)

lgb.fit(X_train, y_train)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1297470974.py in <cell line: 0>()
     16 )
     17 
---> 18 lgb.fit(X_train, y_train)
     19 

NameError: name 'X_train' is not defined

## === cell 9
val_pred = lgb.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {rmse:.4f}")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/266847622.py in <cell line: 0>()
      1 # Validation RMSE (for reference)
----> 2 val_pred = lgb.predict(X_val)
      3 rmse = np.sqrt(mean_squared_error(y_val, val_pred))
      4 print(f"Validation RMSE: {rmse:.4f}")
      5 

NameError: name 'X_val' is not defined

## === cell 10
lgb.fit(X, y)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1616175307.py in <cell line: 0>()
      1 # Retrain on full training data
----> 2 lgb.fit(X, y)
      3 

NameError: name 'X' is not defined

## === cell 11
X_test = test[feature_cols]
test_pred = lgb.predict(X_test)

submission = pd.DataFrame({"key": test["key"], "fare_amount": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/99906564.py in <cell line: 0>()
      1 # Prepare test features and generate predictions
----> 2 X_test = test[feature_cols]
      3 test_pred = lgb.predict(X_test)
      4 
      5 submission = pd.DataFrame({"key": test["key"], "fare_amount": test_pred})

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

KeyError: "['distance'] not in index"
