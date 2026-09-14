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
import os

print(os.listdir("../input"))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import math

from sklearnex import patch_sklearn

patch_sklearn()


def chunck_generator(filename, chunk_size=5_000_000):
    """Yield pandas DataFrame chunks from a CSV file, loading only needed columns."""
    usecols = [
        "fare_amount",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    dtype = {
        "fare_amount": np.float32,
        "pickup_longitude": np.float32,
        "pickup_latitude": np.float32,
        "dropoff_longitude": np.float32,
        "dropoff_latitude": np.float32,
        "passenger_count": np.int8,
    }
    for chunk in pd.read_csv(
        filename,
        usecols=usecols,
        dtype=dtype,
        iterator=True,
        chunksize=chunk_size,
        low_memory=False,
    ):
        yield chunk




## === cell 1
alpha_ang = 0.506  # retained for compatibility; not used in the streamlined version


def distance_travel(df):
    """Add only the haversine distance as the travel feature (float32)."""
    lon1 = np.radians(df["pickup_longitude"].to_numpy(dtype=np.float32))
    lat1 = np.radians(df["pickup_latitude"].to_numpy(dtype=np.float32))
    lon2 = np.radians(df["dropoff_longitude"].to_numpy(dtype=np.float32))
    lat2 = np.radians(df["dropoff_latitude"].to_numpy(dtype=np.float32))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_miles = 3956.0
    df["distance_travel"] = (earth_radius_miles * c).astype(np.float32)
    return df




## === cell 2
def data_clean(df):
    """Basic cleaning: single mask for passenger count, fare, and distance."""
    mask = (
        (df["passenger_count"] > 0)
        & (df["fare_amount"] > 0)
        & (df["distance_travel"] > 0)
    )
    df = df.loc[mask].copy()
    df["fare_amount"] = df["fare_amount"].astype(np.float32)
    return df




## === cell 3
def remove_outliers(df):
    """Trim extreme distance and fare values with a single mask."""
    mask = (df["distance_travel"] < 30) & (df["fare_amount"] < 100)
    return df.loc[mask].copy()




## === cell 4
def graph_present(df):
    """Optional quick visualisation (not used in final pipeline)."""
    test = df[df.passenger_count == 1]
    _ = test.plot.scatter("distance_travel", "fare_amount")
    _ = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")




## === cell 5
def incremental_training(train_X, train_y, regr):
    """Fit the regressor on the supplied aggregated data."""
    regr.fit(train_X, train_y)
    return regr




## === cell 6
filename = r"../input/train.csv"
gen = chunck_generator(filename=filename)

regr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    warm_start=False,
)

df = next(gen)
df = distance_travel(df)
df = data_clean(df)
df = remove_outliers(df)

l = len(df)
df_train = df[: int(0.9 * l)]

train_X = np.empty((len(df_train), 3), dtype=np.float32)
train_X[:, 0] = df_train["distance_travel"].to_numpy(dtype=np.float32)
train_X[:, 1] = df_train["passenger_count"].astype(np.float32).to_numpy()
train_X[:, 2] = 1.0  # bias term

train_y = df_train["fare_amount"].to_numpy(dtype=np.float32)

col_means = np.nanmean(train_X, axis=0)

regr = incremental_training(train_X, train_y, regr)

print(
    "Training completed on the first processed chunk. Model R² on last chunk not shown."
)




## === cell 7
tdf = pd.read_csv("../input/test.csv")
distance_travel(tdf)  # add distance features to test set




## === cell 8
ttrain_X = np.empty((len(tdf), 3), dtype=np.float32)
ttrain_X[:, 0] = tdf["distance_travel"].to_numpy(dtype=np.float32)
ttrain_X[:, 1] = tdf["passenger_count"].astype(np.float32).to_numpy()
ttrain_X[:, 2] = 1.0

ttrain_X = np.nan_to_num(ttrain_X, nan=col_means)

output = regr.predict(ttrain_X)
print("Predictions shape:", output.shape)




## === cell 9
my_submission = pd.DataFrame({"key": tdf.key, "fare_amount": output})
my_submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
