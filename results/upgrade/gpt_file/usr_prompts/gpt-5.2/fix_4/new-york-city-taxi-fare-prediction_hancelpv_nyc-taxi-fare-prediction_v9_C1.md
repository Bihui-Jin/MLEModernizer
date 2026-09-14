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

3.99125

# 6. Current score

4.75609

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.00081) has done: 'I fix the pandas datetime API break (`weekday_name` removed) and ensure the engineered time features exist, so the feature selection in your pipeline no longer fails. I also prevent the `pickup_datetime` timestamp column from leaking into the model matrix (it caused the RandomForest `float()`/Timestamp error) by reading only the needed columns for train/test and then selecting only numeric/dummy-encoded features for modeling. Finally, I keep the same core logic (same features, one-hot encoding, RandomForestRegressor fit/predict) and make sure a valid `submission.csv` with columns `key,fare_amount` is always written.'
- What this solution (achieved 4.75609) has done: 'The timeout is dominated by fitting a 300-tree RandomForest on 1,000,000 rows with very wide one-hot features (string categories) and by expensive pandas operations (concat/copies, slow `.map(week_num)`, and `get_dummies` on object columns). The core model/training loop remain identical, but we reduce overhead around it: avoid unnecessary copies/concat, vectorize `week_of_month` creation, convert time parts to pandas `category` (same semantics, faster/leaner dummies), and avoid materializing extra intermediate DataFrames. We also enable Intel scikit-learn acceleration if available (same estimator semantics) and pass `X` as numpy arrays for faster fit/predict without changing results. These changes preserve the same data filters, the same engineered features, and the same RandomForest hyperparameters.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import os

INPUT_DIR = "/kaggle/input" if os.path.exists("/kaggle/input") else "../input"
print("INPUT_DIR:", INPUT_DIR)
print(os.listdir(INPUT_DIR))

np.random.seed(42)



## === cell 1
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

cols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

test_cols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]



## === cell 2
train = pd.read_csv(
    f"{INPUT_DIR}/train.csv",
    nrows=1000000,
    usecols=cols,
    dtype=types,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
)
test = pd.read_csv(
    f"{INPUT_DIR}/test.csv",
    usecols=test_cols,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
)
samp = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")



## === cell 3
train.dropna(how="any", axis="rows", inplace=True)
train = train[train.fare_amount > 0]
train = train[train["passenger_count"] <= 6]



## === cell 4
latitude_mask_pickup = (train.pickup_latitude > -90) & (train.pickup_latitude < 90)
train = train[latitude_mask_pickup]

latitude_mask_dropoff = (train.dropoff_latitude > -90) & (train.dropoff_latitude < 90)
train = train[latitude_mask_dropoff]



## === cell 5
longitude_mask_pickup = (train.pickup_longitude > -180) & (train.pickup_longitude < 180)
train = train[longitude_mask_pickup]

longitude_mask_dropoff = (train.dropoff_longitude > -180) & (
    train.dropoff_longitude < 180
)
train = train[longitude_mask_dropoff]



## === cell 6
y = train.fare_amount.values
n_train = len(train)
n_test = len(test)
test_id = test.key

train_Xbase = train.drop(columns=["fare_amount"])
test_Xbase = test.drop(columns=["key"])
all_data = pd.concat((train_Xbase, test_Xbase), axis=0, ignore_index=True, copy=False)




## === cell 7
def week_num(day):
    """
    given the day of the month, return the week number of the month
    """
    if day <= 7:
        return "first"
    if (day > 7) and (day <= 14):
        return "second"
    if (day > 14) and (day <= 21):
        return "third"
    if (day > 21) and (day <= 28):
        return "fourth"
    return "fifth"




## === cell 8
def add_time_features(data):
    data = data.copy()

    dt = data["pickup_datetime"]
    if not np.issubdtype(dt.dtype, np.datetime64):
        data["pickup_datetime"] = pd.to_datetime(dt, errors="coerce")
        dt = data["pickup_datetime"]

    data["hour"] = dt.dt.hour
    data["day_of_week"] = dt.dt.day_name()
    dom = dt.dt.day
    data["month"] = dt.dt.month
    data["year"] = dt.dt.year

    data["week_of_month"] = pd.cut(
        dom,
        bins=[0, 7, 14, 21, 28, 31],
        labels=["first", "second", "third", "fourth", "fifth"],
        include_lowest=True,
        right=True,
    ).astype(object)

    data["hour"] = data["hour"].astype("Int64").astype("string").astype(object)
    data["month"] = data["month"].astype("Int64").astype("string").astype(object)
    data["year"] = data["year"].astype("Int64").astype("string").astype(object)

    data.drop("pickup_datetime", axis=1, inplace=True)
    return data




## === cell 9
def add_geo_features(data):
    data = data.copy()

    plon = data["pickup_longitude"].to_numpy()
    plat = data["pickup_latitude"].to_numpy()
    dlon = data["dropoff_longitude"].to_numpy()
    dlat = data["dropoff_latitude"].to_numpy()

    abs_diff_long = np.abs(dlon - plon)
    abs_diff_lat = np.abs(dlat - plat)

    data["abs_diff_longitude"] = abs_diff_long
    data["abs_diff_latitude"] = abs_diff_lat
    data["manhattan_distance"] = abs_diff_long + abs_diff_lat

    data["squared_long"] = abs_diff_long * abs_diff_long
    data["squared_lat"] = abs_diff_lat * abs_diff_lat
    data["euclid_disance"] = np.sqrt(
        data["squared_long"].to_numpy() + data["squared_lat"].to_numpy()
    )

    return data




## === cell 10
all_data = add_time_features(all_data)
all_data = add_geo_features(all_data)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3543095580.py in <cell line: 0>()
      1 # Speed: feature engineering without creating multiple large intermediate DataFrames.
----> 2 all_data = add_time_features(all_data)
      3 all_data = add_geo_features(all_data)
      4 

/tmp/ipykernel_11/3013636123.py in add_time_features(data)
      7     dt = data["pickup_datetime"]
      8     # In case parse_dates didn't coerce some values (rare), keep original semantics.
----> 9     if not np.issubdtype(dt.dtype, np.datetime64):
     10         data["pickup_datetime"] = pd.to_datetime(dt, errors="coerce")
     11         dt = data["pickup_datetime"]

/usr/local/lib/python3.11/dist-packages/numpy/core/numerictypes.py in issubdtype(arg1, arg2)
    415     """
    416     if not issubclass_(arg1, generic):
--> 417         arg1 = dtype(arg1).type
    418     if not issubclass_(arg2, generic):
    419         arg2 = dtype(arg2).type

TypeError: Cannot interpret 'datetime64[ns, UTC]' as a data type

## === cell 11
features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "euclid_disance",
    "manhattan_distance",
]

all_data = all_data[features]

cat_cols = ["hour", "day_of_week", "week_of_month", "month", "year"]
for c in cat_cols:
    all_data[c] = all_data[c].astype("category")

all_data = all_data.fillna(
    {
        "hour": "nan",
        "day_of_week": "nan",
        "week_of_month": "nan",
        "month": "nan",
        "year": "nan",
    }
)
num_cols = [
    "abs_diff_longitude",
    "abs_diff_latitude",
    "euclid_disance",
    "manhattan_distance",
]
all_data[num_cols] = all_data[num_cols].fillna(0.0)

all_data = pd.get_dummies(all_data)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3625734199.py in <cell line: 0>()
     12 ]
     13 
---> 14 all_data = all_data[features]
     15 
     16 # Speed: set categorical dtype before get_dummies to reduce memory/time.

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

KeyError: "['hour', 'day_of_week', 'week_of_month', 'month', 'year', 'abs_diff_longitude', 'abs_diff_latitude', 'euclid_disance', 'manhattan_distance'] not in index"

## === cell 12
x = all_data.iloc[:n_train]
x_test = all_data.iloc[n_train:]

x_test = x_test.reindex(columns=x.columns, fill_value=0)

X = x.to_numpy(dtype=np.float32, copy=False)
X_test = x_test.to_numpy(dtype=np.float32, copy=False)



## === cell 13
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
)



## === cell 14
model.fit(X, y)



## === cell 15
test_pred = model.predict(X_test)

sub = pd.DataFrame()
sub["key"] = test_id.values
sub["fare_amount"] = test_pred.astype(np.float32)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
