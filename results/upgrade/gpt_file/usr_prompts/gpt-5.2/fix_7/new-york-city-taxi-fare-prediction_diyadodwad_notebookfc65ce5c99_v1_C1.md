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

3.9

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

5.689

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 936.94064) has done: 'I fix the immediate runtime error by removing the deprecated `normalize` argument from `LinearRegression` (scikit-learn 1.2+) and keep the same linear-regression core model. To preserve comparable behavior, I wrap `LinearRegression` in a `Pipeline` with `StandardScaler`, which is the modern equivalent of the old normalization path and is score-positive without changing the model family. I also make the one-hot weekday columns consistent between train/test to avoid hidden shape mismatches, and ensure we always write a valid `submission.csv` with the required `key,fare_amount` columns. All other feature engineering and training flow are kept intact.'
- What this solution (achieved 15.08847) has done: 'Your RMSE is exploding because the model is being trained on badly scaled/invalid targets and because train/test feature scaling is inconsistent: you normalize `Difference_*` using each dataset’s own mean/variance (data shift), and you never filter impossible/negative/outlier `fare_amount` values that dominate RMSE. I keep the same linear-regression + StandardScaler pipeline and the same feature set, but (1) make the `Difference_*` normalization use train statistics applied to both train and test, (2) apply minimal, standard NYC Taxi sanity filters (fare>0, passenger_count 1–6, lat/lon bounds) to remove catastrophic label noise, and (3) clip predictions to a reasonable non-negative range to avoid huge RMSE from occasional extreme negatives/positives. These are small, directly score-relevant fixes that don’t change the modeling family or training approach and should move RMSE sharply toward your target. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 15.63752) has done: 'Your current RMSE (15.09) is still far from the target (5.689), so we need a meaningful but still “core-logic-preserving” fix. The single biggest remaining score drag is that your LinearRegression is trying to model `fare_amount` on the raw scale with heavy-tailed noise; switching the target to `log1p(fare_amount)` and then inverting with `expm1` at prediction time typically reduces RMSE substantially on this competition without changing the model family, features, or training loop. I also (minimally) fix the “variance used as scale” bug in your Difference_* normalization (use std instead of var) while keeping the same idea and train-stat application to test. Submission writing remains identical (`key,fare_amount`, `submission.csv`).'
- What this solution (achieved 15.90204) has done: 'Your current RMSE (15.64) is still far above the target (5.689), so the smallest “core-logic-preserving” move is to improve feature signal without changing the model family or training loop. I keep your LinearRegression+StandardScaler pipeline and the same overall preprocessing flow, but add two standard NYC Taxi engineered features that are computed directly from your existing coordinates: (1) a Manhattan-distance proxy and (2) the trip bearing angle; both are strong for linear models and typically reduce RMSE materially. I also fix a subtle but impactful issue: rounding distance features to 2 decimals throws away signal for a linear regressor, so I stop rounding (this doesn’t change the feature definitions, just preserves precision). Everything else (filters, log1p target, clipping, submission format/path) stays the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved 15.90212) has done: 'Your current RMSE (15.90) is far above the target (5.689), so we need a small but meaningful signal gain without changing the model family or training loop. The biggest low-risk improvement for this competition with a linear model is to use a more appropriate distance feature: compute Haversine using Earth radius in miles directly (your current `Distance` uses an inconsistent radius/conversion), which typically moves RMSE substantially downward while keeping the same “distance feature” idea. I also remove the final rounding of predictions (rounding throws away accuracy and increases RMSE) while keeping the same clipping and log1p/expm1 target transform. Everything else (filters, one-hot weekdays, LinearRegression+StandardScaler pipeline, submission format/path) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_data = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=10_000_000
)
train_data.head()



## === cell 2
train_data.shape



## === cell 3
train_data.info()



## === cell 4
test_data = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
test_data.head()



## === cell 5
test_data.info()



## === cell 6
train_data.isna().sum()



## === cell 7
train_data["Difference_longitude"] = np.abs(
    np.asarray(train_data["pickup_longitude"] - train_data["dropoff_longitude"])
)
train_data["Difference_latitude"] = np.abs(
    np.asarray(train_data["pickup_latitude"] - train_data["dropoff_latitude"])
)

test_data["Difference_longitude"] = np.abs(
    np.asarray(test_data["pickup_longitude"] - test_data["dropoff_longitude"])
)
test_data["Difference_latitude"] = np.abs(
    np.asarray(test_data["pickup_latitude"] - test_data["dropoff_latitude"])
)



## === cell 8
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")



## === cell 9
try:
    plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")
except Exception:
    plot = None



## === cell 10
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]



## === cell 11
train_data = train_data[
    (train_data["fare_amount"] > 0.0)
    & (train_data["fare_amount"] < 250.0)
    & (train_data["passenger_count"] >= 1)
    & (train_data["passenger_count"] <= 6)
    & (train_data["pickup_longitude"].between(-75, -72))
    & (train_data["dropoff_longitude"].between(-75, -72))
    & (train_data["pickup_latitude"].between(40, 42))
    & (train_data["dropoff_latitude"].between(40, 42))
].copy()



## === cell 12
train_data["_pickup_dt"] = pd.to_datetime(
    train_data["pickup_datetime"], utc=True, errors="coerce"
)
test_data["_pickup_dt"] = pd.to_datetime(
    test_data["pickup_datetime"], utc=True, errors="coerce"
)

train_data = train_data.dropna(subset=["_pickup_dt"]).copy()

train_data["pickup_hour"] = train_data["_pickup_dt"].dt.hour.astype(np.int16)
test_data["pickup_hour"] = test_data["_pickup_dt"].dt.hour.astype(np.int16)

train_data["pickup_dow"] = train_data["_pickup_dt"].dt.dayofweek.astype(np.int16)
test_data["pickup_dow"] = test_data["_pickup_dt"].dt.dayofweek.astype(np.int16)



## === cell 13
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_data["pickuptime"] = ls1

ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_data["pickuptime"] = ls1



## === cell 14
train_data.head()



## === cell 15
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
train_data["Weekday"] = ls1

ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
test_data["Weekday"] = ls1



## === cell 16
train_data.head()



## === cell 17
test_data.head()



## === cell 18
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 19
train_data["Weekday"] = train_data["Weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
)
test_data["Weekday"] = test_data["Weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
)



## === cell 20
train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)

all_dummy_cols = sorted(set(train_one_hot.columns).union(set(test_one_hot.columns)))
for c in all_dummy_cols:
    if c not in train_data.columns:
        train_data[c] = 0
    if c not in test_data.columns:
        test_data[c] = 0



## === cell 21
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 22
ls1 = list(train_data["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
train_data["pickuptime"] = ls1

ls1 = list(test_data["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
test_data["pickuptime"] = ls1



## === cell 23
train_data.head()



## === cell 24
R_miles = 3958.8
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
train_data["Distance"] = R_miles * c

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
test_data["Distance"] = R_miles * c



## === cell 25
R = 6373.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

lat3 = np.zeros(len(train_data)) + np.radians(40.6413111)
lon3 = np.zeros(len(train_data)) + np.radians(-73.7781391)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff = lon3 - lon2
d_lat_dropoff = lat3 - lat2
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
train_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

lat3 = np.zeros(len(test_data)) + np.radians(40.6413111)
lon3 = np.zeros(len(test_data)) + np.radians(-73.7781391)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff = lon3 - lon2
d_lat_dropoff = lat3 - lat2
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
test_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621




## === cell 26
def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    lat1 = np.radians(df["pickup_latitude"].astype(float).to_numpy())
    lon1 = np.radians(df["pickup_longitude"].astype(float).to_numpy())
    lat2 = np.radians(df["dropoff_latitude"].astype(float).to_numpy())
    lon2 = np.radians(df["dropoff_longitude"].astype(float).to_numpy())

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    latm = 0.5 * (lat1 + lat2)
    miles_per_rad_lat = 3958.8  # Earth radius in miles
    miles_per_rad_lon = 3958.8 * np.cos(latm)
    df["Manhattan_Distance"] = (
        np.abs(dlat) * miles_per_rad_lat + np.abs(dlon) * miles_per_rad_lon
    )

    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    df["Bearing"] = np.arctan2(y, x)

    return df


train_data = add_geo_features(train_data)
test_data = add_geo_features(test_data)



## === cell 28
dl_mean = float(train_data["Difference_longitude"].mean())
dl_std = float(train_data["Difference_longitude"].std(ddof=0))
dl_std = dl_std if dl_std > 0 else 1.0

dlat_mean = float(train_data["Difference_latitude"].mean())
dlat_std = float(train_data["Difference_latitude"].std(ddof=0))
dlat_std = dlat_std if dlat_std > 0 else 1.0

train_data["Difference_longitude"] = (
    np.abs(train_data["Difference_longitude"] - dl_mean) / dl_std
)
train_data["Difference_latitude"] = (
    np.abs(train_data["Difference_latitude"] - dlat_mean) / dlat_std
)

test_data["Difference_longitude"] = (
    np.abs(test_data["Difference_longitude"] - dl_mean) / dl_std
)
test_data["Difference_latitude"] = (
    np.abs(test_data["Difference_latitude"] - dlat_mean) / dlat_std
)



## === cell 29
train_data.shape



## === cell 30
test_data.shape



## === cell 31
feature_cols_train = [c for c in train_data.columns if c not in ["fare_amount"]]
feature_cols_test = list(test_data.columns)

all_cols = sorted(set(feature_cols_train).union(set(feature_cols_test)))
for c in all_cols:
    if c not in train_data.columns:
        train_data[c] = 0
    if c not in test_data.columns:
        test_data[c] = 0

train_data = train_data.reindex(columns=sorted(train_data.columns))
test_data = test_data.reindex(columns=sorted(test_data.columns))



## === cell 32
from sklearn.model_selection import train_test_split

X = train_data.drop(["key", "fare_amount"], axis=1)

y = np.log1p(train_data["fare_amount"].astype(float))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 33
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lr = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", LinearRegression()),
    ]
)

lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/423343160.py in <cell line: 0>()
     10 )
     11 
---> 12 lr.fit(X_train, y_train)
     13 print(lr.score(X_test, y_test))
     14 

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in fit(self, X, y, **fit_params)
    399         """
    400         fit_params_steps = self._check_fit_params(**fit_params)
--> 401         Xt = self._fit(X, y, **fit_params_steps)
    402         with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
    403             if self._final_estimator != "passthrough":

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit(self, X, y, **fit_params_steps)
    357                 cloned_transformer = clone(transformer)
    358             # Fit or load from cache the current transformer
--> 359             X, fitted_transformer = fit_transform_one_cached(
    360                 cloned_transformer,
    361                 X,

/usr/local/lib/python3.11/dist-packages/joblib/memory.py in __call__(self, *args, **kwargs)
    324 
    325     def __call__(self, *args, **kwargs):
--> 326         return self.func(*args, **kwargs)
    327 
    328     def call_and_shelve(self, *args, **kwargs):

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit_transform_one(transformer, X, y, weight, message_clsname, message, **fit_params)
    891     with _print_elapsed_time(message_clsname, message):
    892         if hasattr(transformer, "fit_transform"):
--> 893             res = transformer.fit_transform(X, y, **fit_params)
    894         else:
    895             res = transformer.fit(X, y, **fit_params).transform(X)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in fit_transform(self, X, y, **fit_params)
    879         else:
    880             # fit method of arity 2 (supervised transformation)
--> 881             return self.fit(X, y, **fit_params).transform(X)
    882 
    883 

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in fit(self, X, y, sample_weight)
    822         # Reset internal state before fitting
    823         self._reset()
--> 824         return self.partial_fit(X, y, sample_weight)
    825 
    826     def partial_fit(self, X, y=None, sample_weight=None):

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in partial_fit(self, X, y, sample_weight)
    859 
    860         first_call = not hasattr(self, "n_samples_seen_")
--> 861         X = self._validate_data(
    862             X,
    863             accept_sparse=("csr", "csc"),

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    563             raise ValueError("Validation should be done on X, y or both.")
    564         elif not no_val_X and no_val_y:
--> 565             X = check_array(X, input_name="X", **check_params)
    566             out = X
    567         elif no_val_X and not no_val_y:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    808         # Use the original dtype for conversion if dtype is None
    809         new_dtype = dtype_orig if dtype is None else dtype
--> 810         array = array.astype(new_dtype)
    811         # Since we converted here, we do not need to convert again later
    812         dtype = None

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    235 
    236     try:
--> 237         new_values = astype_array(values, dtype, copy=copy)
    238     except (ValueError, TypeError):
    239         # e.g. _astype_nansafe can fail on object-dtype of strings

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    177     if not isinstance(values, np.ndarray):
    178         # i.e. ExtensionArray
--> 179         values = values.astype(dtype, copy=copy)
    180 
    181     else:

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimes.py in astype(self, dtype, copy)
    737         elif isinstance(dtype, PeriodDtype):
    738             return self.to_period(freq=dtype.freq)
--> 739         return dtl.DatetimeLikeArrayMixin.astype(self, dtype, copy)
    740 
    741     # -----------------------------------------------------------------

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimelike.py in astype(self, dtype, copy)
    492             # and conversions for any datetimelike to float
    493             msg = f"Cannot cast {type(self).__name__} to dtype {dtype}"
--> 494             raise TypeError(msg)
    495         else:
    496             return np.asarray(self, dtype=dtype)

TypeError: Cannot cast DatetimeArray to dtype float64

## === cell 34
pred_log = lr.predict(test_data.drop("key", axis=1))
pred = np.expm1(pred_log)

pred = np.clip(pred, 0.0, 250.0)



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1263559211.py in <cell line: 0>()
----> 1 pred_log = lr.predict(test_data.drop("key", axis=1))
      2 pred = np.expm1(pred_log)
      3 
      4 pred = np.clip(pred, 0.0, 250.0)
      5 

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in predict(self, X, **predict_params)
    478         Xt = X
    479         for _, name, transform in self._iter(with_final=False):
--> 480             Xt = transform.transform(Xt)
    481         return self.steps[-1][1].predict(Xt, **predict_params)
    482 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in transform(self, X, copy)
    990 
    991         copy = copy if copy is not None else self.copy
--> 992         X = self._validate_data(
    993             X,
    994             reset=False,

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    563             raise ValueError("Validation should be done on X, y or both.")
    564         elif not no_val_X and no_val_y:
--> 565             X = check_array(X, input_name="X", **check_params)
    566             out = X
    567         elif no_val_X and not no_val_y:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    808         # Use the original dtype for conversion if dtype is None
    809         new_dtype = dtype_orig if dtype is None else dtype
--> 810         array = array.astype(new_dtype)
    811         # Since we converted here, we do not need to convert again later
    812         dtype = None

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    235 
    236     try:
--> 237         new_values = astype_array(values, dtype, copy=copy)
    238     except (ValueError, TypeError):
    239         # e.g. _astype_nansafe can fail on object-dtype of strings

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    177     if not isinstance(values, np.ndarray):
    178         # i.e. ExtensionArray
--> 179         values = values.astype(dtype, copy=copy)
    180 
    181     else:

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimes.py in astype(self, dtype, copy)
    737         elif isinstance(dtype, PeriodDtype):
    738             return self.to_period(freq=dtype.freq)
--> 739         return dtl.DatetimeLikeArrayMixin.astype(self, dtype, copy)
    740 
    741     # -----------------------------------------------------------------

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimelike.py in astype(self, dtype, copy)
    492             # and conversions for any datetimelike to float
    493             msg = f"Cannot cast {type(self).__name__} to dtype {dtype}"
--> 494             raise TypeError(msg)
    495         else:
    496             return np.asarray(self, dtype=dtype)

TypeError: Cannot cast DatetimeArray to dtype float64

## === cell 35
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 36
Submission = pd.DataFrame(data=pred, columns=["fare_amount"])
Submission["key"] = test_data["key"]
Submission = Submission[["key", "fare_amount"]]



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1618083723.py in <cell line: 0>()
----> 1 Submission = pd.DataFrame(data=pred, columns=["fare_amount"])
      2 Submission["key"] = test_data["key"]
      3 Submission = Submission[["key", "fare_amount"]]
      4 

NameError: name 'pred' is not defined

## === cell 37
submission_path = "submission.csv"
Submission.to_csv(submission_path, index=False)
print(f"Wrote submission to: {submission_path} with shape {Submission.shape}")
print(Submission.head())

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2705105323.py in <cell line: 0>()
      1 submission_path = "submission.csv"
----> 2 Submission.to_csv(submission_path, index=False)
      3 print(f"Wrote submission to: {submission_path} with shape {Submission.shape}")
      4 print(Submission.head())

NameError: name 'Submission' is not defined
