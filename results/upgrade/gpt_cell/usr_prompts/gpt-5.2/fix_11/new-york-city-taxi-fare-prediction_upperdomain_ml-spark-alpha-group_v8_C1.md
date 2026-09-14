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
scipy==1.15.3
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

# 5. Code solution

## === cell 0
from sklearn.impute import SimpleImputer as Imputer
import numpy as np  # linear algebra
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import math
import os

print(os.listdir("../input"))



## === cell 1
df = pd.read_csv("../input/train.csv", nrows=2_000_000)
df.head()



## === cell 2
df = df[df.passenger_count > 0]
df = df[df.fare_amount > 0]
df.head()



## === cell 3
alpha_ang = 0.506


def distance_travel(df):
    lat1 = df["pickup_latitude"].astype("float64")
    lon1 = df["pickup_longitude"].astype("float64")
    lat2 = df["dropoff_latitude"].astype("float64")
    lon2 = df["dropoff_longitude"].astype("float64")

    dlat = lat2 - lat1
    dlon = (lon2 - lon1) * np.cos(alpha_ang * (lat2 + lat1))

    df["distance_travel"] = np.sqrt(dlat * dlat + dlon * dlon).astype("float32")


def filter_nyc_coords(frame):
    frame = frame[
        (frame.pickup_longitude.between(-74.3, -73.6))
        & (frame.dropoff_longitude.between(-74.3, -73.6))
        & (frame.pickup_latitude.between(40.4, 41.0))
        & (frame.dropoff_latitude.between(40.4, 41.0))
    ]
    frame = frame[
        (frame.pickup_longitude != 0)
        & (frame.dropoff_longitude != 0)
        & (frame.pickup_latitude != 0)
        & (frame.dropoff_latitude != 0)
    ]
    return frame


def valid_nyc_mask(frame):
    m = (
        frame.pickup_longitude.between(-74.3, -73.6)
        & frame.dropoff_longitude.between(-74.3, -73.6)
        & frame.pickup_latitude.between(40.4, 41.0)
        & frame.dropoff_latitude.between(40.4, 41.0)
        & (frame.pickup_longitude != 0)
        & (frame.dropoff_longitude != 0)
        & (frame.pickup_latitude != 0)
        & (frame.dropoff_latitude != 0)
    )
    return m


def add_time_features(frame):
    dt = pd.to_datetime(frame["pickup_datetime"], errors="coerce", utc=True)
    frame["pickup_hour"] = dt.dt.hour.astype("float32")
    frame["pickup_weekday"] = dt.dt.weekday.astype("float32")
    frame["pickup_month"] = dt.dt.month.astype("float32")
    return frame


def add_manhattan(frame):
    frame["manhattan"] = (
        (
            frame["pickup_latitude"].astype("float64")
            - frame["dropoff_latitude"].astype("float64")
        ).abs()
        + (
            frame["pickup_longitude"].astype("float64")
            - frame["dropoff_longitude"].astype("float64")
        ).abs()
    ).astype("float32")
    return frame


df = filter_nyc_coords(df)
df = add_time_features(df)
df = add_manhattan(df)

distance_travel(df)
df = df[df.distance_travel > 0]
df.head()



## === cell 4
test = df[df.passenger_count == 1]
plot = test.iloc[: len(test)].plot.scatter("distance_travel", "fare_amount")



## === cell 5
df = df[df.distance_travel < 30]
df = df[df.fare_amount < 100]
plot = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")



## === cell 6
l = len(df)
print(l)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

df_train = df[: int(0.7 * l)]
df_test = df[int(0.7 * l) :]

train_X = np.column_stack(
    (
        df_train.distance_travel,
        df_train.manhattan,
        df_train.passenger_count,
        df_train.pickup_hour,
        df_train.pickup_weekday,
        df_train.pickup_month,
        np.ones(len(df_train)),
    )
)
test_X = np.column_stack(
    (
        df_test.distance_travel,
        df_test.manhattan,
        df_test.passenger_count,
        df_test.pickup_hour,
        df_test.pickup_weekday,
        df_test.pickup_month,
        np.ones(len(df_test)),
    )
)

train_y = np.array(df_train.fare_amount)
test_y = np.array(df_test.fare_amount)
train_y_log = np.log1p(train_y)
test_y_log = np.log1p(test_y)



## === cell 7
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

imp = Imputer(missing_values=np.nan, strategy="mean")
imp = imp.fit(train_X)
train_X_imp = imp.transform(train_X)
test_X_imp = imp.transform(test_X)

regr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)
regr.fit(train_X_imp, train_y_log)

val_pred = np.expm1(regr.predict(test_X_imp))
rmse = float(np.sqrt(mean_squared_error(test_y, val_pred)))
print("Validation RMSE (approx):", rmse)



## === cell 8
tdf = pd.read_csv("../input/test.csv", nrows=10_00_000)

valid_mask = valid_nyc_mask(tdf)

tdf_work = tdf.copy()
tdf_work.loc[~valid_mask, "pickup_longitude"] = np.nan
tdf_work.loc[~valid_mask, "pickup_latitude"] = np.nan
tdf_work.loc[~valid_mask, "dropoff_longitude"] = np.nan
tdf_work.loc[~valid_mask, "dropoff_latitude"] = np.nan

tdf_work = add_time_features(tdf_work)
tdf_work = add_manhattan(tdf_work)

distance_travel(tdf_work)

tdf_work.loc[~valid_mask, "distance_travel"] = np.nan
tdf_work.loc[~valid_mask, "manhattan"] = np.nan

tdf_work.head()



## === cell 9
ttrain_X = np.column_stack(
    (
        tdf_work.distance_travel,
        tdf_work.manhattan,
        tdf_work.passenger_count,
        tdf_work.pickup_hour,
        tdf_work.pickup_weekday,
        tdf_work.pickup_month,
        np.ones(len(tdf_work)),
    )
)
ttrain_X = imp.transform(ttrain_X)

output_log = regr.predict(ttrain_X)
output = np.expm1(output_log)

lo = float(np.percentile(train_y, 0.5))
hi = float(np.percentile(train_y, 99.5))
output = np.clip(output, lo, hi)

print(output)



## === cell 10
my_submission = pd.DataFrame({"key": tdf.key, "fare_amount": output})

if len(my_submission) != 9914:
    all_test = pd.read_csv("../input/test.csv", usecols=["key"])
    my_submission = all_test.merge(my_submission, on="key", how="left")
    my_submission["fare_amount"] = my_submission["fare_amount"].fillna(
        np.median(train_y)
    )

my_submission.to_csv("submission.csv", index=False)
my_submission.head()
