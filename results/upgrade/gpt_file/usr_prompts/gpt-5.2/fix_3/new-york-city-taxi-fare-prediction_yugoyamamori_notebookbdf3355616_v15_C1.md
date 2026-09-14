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

3.11

# 3. Installed packages

geopandas==0.14.4
geopy==2.4.1
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

3.9197

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 5.11239) has done: 'Your score gap is large (5.56948 vs target 3.9197; lower is better), so we should improve generalization with minimal, metric-aligned fixes while keeping the same overall approach (RandomForest on engineered distance + basic cleaning). The biggest issue is the extremely slow/variable `geopy.geodesic` loop and a default RandomForest with untuned capacity; switching to a fast, standard Haversine distance (same feature intent) lets us safely scale trees and control depth to reduce RMSE without changing the modeling paradigm. I also keep your existing filtering but add a couple of standard NYC-taxi sanity filters (nonzero trip distance and removing extreme distance outliers) that typically reduce label noise and improve RMSE. Finally, I ensure the submission uses the test `key` in the correct order by preserving it before dropping columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



## === cell 2
train_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=100000
)
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 3
train_df



## === cell 4
test_df



## === cell 5
train_df.columns



## === cell 6
test_df.columns



## === cell 7
train_df.isnull().sum().sort_values(ascending=False)



## === cell 8
test_df.isnull().sum().sort_values(ascending=False)



## === cell 9
a = train_df[
    train_df["dropoff_longitude"].isnull() | train_df["dropoff_latitude"].isnull()
]
print(a)



## === cell 10
train_df.drop(a.index, axis=0, inplace=True)



## === cell 11
train_df.isnull().sum().sort_values(ascending=False)



## === cell 12
train_df.describe()



## === cell 13
sns.histplot(data=train_df, x="fare_amount", kde=True, bins=100)



## === cell 14
sns.boxplot(data=train_df, y="fare_amount")



## === cell 15
train_df = train_df[(train_df["fare_amount"] >= 2.5) & (train_df["fare_amount"] < 75)]



## === cell 16
train_df["fare_amount"].describe()



## === cell 17
sns.histplot(data=train_df, x="fare_amount", kde=True, bins=100)
plt.xlim(0, 80)
plt.ylim(0, 150000)



## === cell 18
sns.boxplot(data=train_df, y="fare_amount")



## === cell 19
train_df["passenger_count"].describe()



## === cell 20
train_df = train_df[train_df["passenger_count"] <= 6]
sns.histplot(data=train_df, x="passenger_count")
plt.ylim(0, 1000000)




## === cell 21
def apply_basic_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["passenger_count"].between(1, 6, inclusive="both")]
    df = df[df["pickup_longitude"].between(-74.5, -73.0, inclusive="both")]
    df = df[df["dropoff_longitude"].between(-74.5, -73.0, inclusive="both")]
    df = df[df["pickup_latitude"].between(40.5, 42.0, inclusive="both")]
    df = df[df["dropoff_latitude"].between(40.5, 42.0, inclusive="both")]
    return df


train_df = apply_basic_filters(train_df)
test_df = apply_basic_filters(test_df)



## === cell 22
same_loc = (train_df["pickup_longitude"] == train_df["dropoff_longitude"]) & (
    train_df["pickup_latitude"] == train_df["dropoff_latitude"]
)
train_df = train_df[~same_loc].copy()

same_loc_test = (test_df["pickup_longitude"] == test_df["dropoff_longitude"]) & (
    test_df["pickup_latitude"] == test_df["dropoff_latitude"]
)
test_df = test_df[~same_loc_test].copy()



## === cell 23
train_df.describe()



## === cell 24
train_df




## === cell 25
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["pickup_datetime"], utc=True, errors="coerce")
    df["pickup_hour"] = dt.dt.hour.astype("float32")
    df["pickup_dow"] = dt.dt.dayofweek.astype("float32")
    df["pickup_month"] = dt.dt.month.astype("float32")
    df[["pickup_hour", "pickup_dow", "pickup_month"]] = df[
        ["pickup_hour", "pickup_dow", "pickup_month"]
    ].fillna(0.0)
    return df


train_df = add_time_features(train_df)
test_df = add_time_features(test_df)



## === cell 26
test_keys = test_df["key"].copy()

train_df.drop(["key", "pickup_datetime"], axis=1, inplace=True)
test_df.drop(["key", "pickup_datetime"], axis=1, inplace=True)



## === cell 27
train_df.columns



## === cell 28
train_len = len(train_df)
print(train_len)



## === cell 29
test_df.columns



## === cell 30
test_df.describe()



## === cell 31
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)




## === cell 32
def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0
    return R * c




## === cell 33
print(len(df))



## === cell 34
df["distance"] = haversine_km(
    df["pickup_latitude"].values,
    df["pickup_longitude"].values,
    df["dropoff_latitude"].values,
    df["dropoff_longitude"].values,
)



## === cell 35
print(df["distance"].head(20).tolist())



## === cell 36
print(float(df["distance"].mean()))



## === cell 37
print(len(df["distance"]))



## === cell 38
df



## === cell 39
train = df.iloc[:train_len].copy()
test = df.iloc[train_len:].copy()



## === cell 40
train = train[(train["distance"] > 0) & (train["distance"] < 100)].copy()
test = test[(test["distance"] > 0) & (test["distance"] < 100)].copy()



## === cell 41
train



## === cell 42
train.isnull().sum().sort_values(ascending=False)



## === cell 43
test



## === cell 44
test.isnull().sum().sort_values(ascending=False)



## === cell 45
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



## === cell 46
train_X = train.drop("fare_amount", axis=1).values
train_y = train["fare_amount"].values



## === cell 47
test_X = test.drop("fare_amount", axis=1).values
test_keys_aligned = test_keys.loc[test.index].values

train_x, valid_x, y_tr, y_va = train_test_split(
    train_X, train_y, test_size=0.3, random_state=0
)



## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/605122736.py in <cell line: 0>()
      1 # Align test keys with the filtered test rows to keep submission row order consistent.
      2 test_X = test.drop("fare_amount", axis=1).values
----> 3 test_keys_aligned = test_keys.loc[test.index].values
      4 
      5 train_x, valid_x, y_tr, y_va = train_test_split(

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1189             maybe_callable = com.apply_if_callable(key, self.obj)
   1190             maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
-> 1191             return self._getitem_axis(maybe_callable, axis=axis)
   1192 
   1193     def _is_scalar_access(self, key: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1418                     raise ValueError("Cannot index with multidimensional key")
   1419 
-> 1420                 return self._getitem_iterable(key, axis=axis)
   1421 
   1422             # nested tuple slicing

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_iterable(self, key, axis)
   1358 
   1359         # A collection of keys
-> 1360         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361         return self.obj._reindex_with_indexers(
   1362             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: 'None of [RangeIndex(start=96324, stop=105865, step=1)] are in the [index]'

## === cell 48
model = RandomForestRegressor(
    n_estimators=300,
    random_state=0,
    n_jobs=-1,
    max_depth=18,
    min_samples_leaf=2,
)
model.fit(train_x, y_tr)
valid_x_predict = model.predict(valid_x)
print("score:" + str(np.sqrt(mean_squared_error(y_va, valid_x_predict))))



## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1425960612.py in <cell line: 0>()
      6     min_samples_leaf=2,
      7 )
----> 8 model.fit(train_x, y_tr)
      9 valid_x_predict = model.predict(valid_x)
     10 print("score:" + str(np.sqrt(mean_squared_error(y_va, valid_x_predict))))

NameError: name 'train_x' is not defined

## === cell 49
test_pred = model.predict(test_X)
test_pred = np.clip(test_pred, 0.0, None)

sub = pd.DataFrame({"key": test_keys_aligned, "fare_amount": test_pred})



## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/659548313.py in <cell line: 0>()
      1 # Change (metric-safety): clip negative fares to 0 (invalid predictions can hurt RMSE).
----> 2 test_pred = model.predict(test_X)
      3 test_pred = np.clip(test_pred, 0.0, None)
      4 
      5 sub = pd.DataFrame({"key": test_keys_aligned, "fare_amount": test_pred})

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py in predict(self, X)
    977             The predicted values.
    978         """
--> 979         check_is_fitted(self)
    980         # Check data
    981         X = self._validate_X_predict(X)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This RandomForestRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 50
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with", len(sub), "rows")

## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/668780180.py in <cell line: 0>()
----> 1 sub.to_csv("submission.csv", index=False)
      2 print(sub.head())
      3 print("Wrote submission.csv with", len(sub), "rows")

NameError: name 'sub' is not defined
