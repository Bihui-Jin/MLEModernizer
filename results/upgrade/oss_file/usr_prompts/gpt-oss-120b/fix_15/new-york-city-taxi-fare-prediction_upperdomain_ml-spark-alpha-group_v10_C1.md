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

# 5. Target score

4.11442

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.33511) has done: 'Implemented fixes:
- Replaced deprecated `Imputer` with `SimpleImputer`.
- Corrected import errors and ensured all required libraries are loaded.
- Minor clean‑ups to keep the workflow intact and produce a valid `submission.csv`.
- Added a small increase to `GradientBoostingRegressor` complexity for modest score improvement.'
- What this solution (achieved 66.77421) has done: 'Implemented fixes to correctly parse the `pickup_datetime` column in the test set, enabling the datetime‑based feature creation and preventing attribute errors. Adjusted the test CSV read to use `parse_dates=[1]` (the correct column index) and retained the original workflow for feature engineering and prediction. The script now runs end‑to‑end and writes a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

from sklearnex import patch_sklearn

patch_sklearn()

input_dir = "./input"
if not os.path.isdir(input_dir):
    input_dir = "../input"
print("Input directory contents:", os.listdir(input_dir))


def chunk_generator(filename, chunk_size=2_000_000):
    """Yield CSV chunks as DataFrames with explicit dtypes for faster parsing.
    The datetime column is read as string to avoid pandas' costly automatic parsing."""
    dtype_spec = {
        "fare_amount": np.float32,
        "pickup_longitude": np.float32,
        "pickup_latitude": np.float32,
        "dropoff_longitude": np.float32,
        "dropoff_latitude": np.float32,
        "passenger_count": np.int8,
        "pickup_datetime": object,  # read as raw string
    }
    usecols = [
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    for chunk in pd.read_csv(
        filename,
        usecols=usecols,
        iterator=True,
        chunksize=chunk_size,
        dtype=dtype_spec,
        low_memory=False,
    ):
        yield chunk




## === cell 1
alpha_ang = 0.506


def distance_travel(df):
    """Vectorised distance features (float32). Operates on NumPy arrays to avoid pandas overhead."""
    lon_pick = df["pickup_longitude"].values.astype(np.float32)
    lon_drop = df["dropoff_longitude"].values.astype(np.float32)
    lat_pick = df["pickup_latitude"].values.astype(np.float32)
    lat_drop = df["dropoff_latitude"].values.astype(np.float32)

    abs_diff_longitude = np.abs(lon_drop - lon_pick) * np.float32(50)
    abs_diff_latitude = np.abs(lat_drop - lat_pick) * np.float32(69)

    displacement_vector = np.sqrt(abs_diff_latitude**2 + abs_diff_longitude**2)

    theta = np.arctan2(abs_diff_longitude, abs_diff_latitude) - alpha_ang
    actual_long = np.abs(displacement_vector * np.sin(theta))
    actual_lat = np.abs(displacement_vector * np.cos(theta))

    df["abs_diff_longitude"] = abs_diff_longitude
    df["abs_diff_latitude"] = abs_diff_latitude
    df["displacement_vector"] = displacement_vector
    df["actual_long"] = actual_long
    df["actual_lat"] = actual_lat
    df["distance_travel"] = actual_long + actual_lat
    return df


def haversine_distance(df):
    """Vectorised great‑circle distance (km) using NumPy."""
    R = 6371.0  # Earth radius in kilometers

    lat1 = np.radians(df["pickup_latitude"].values.astype(np.float64))
    lat2 = np.radians(df["dropoff_latitude"].values.astype(np.float64))
    dlat = lat2 - lat1
    dlon = np.radians(
        df["dropoff_longitude"].values.astype(np.float64)
        - df["pickup_longitude"].values.astype(np.float64)
    )

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["haversine_km"] = R * c
    return df


def add_datetime_features(df):
    """Parse datetime once with a known format and extract hour & weekday."""
    dt = pd.to_datetime(
        df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
    )
    df["pickup_hour"] = dt.dt.hour.astype(np.float32)
    df["pickup_dayofweek"] = dt.dt.dayofweek.astype(np.float32)
    return df




## === cell 2
def data_clean(df):
    """Basic cleaning, type casting and distance feature generation."""
    df = df[df["passenger_count"] > 0]
    df["fare_amount"] = df["fare_amount"].astype(np.float64)
    df = df[df["fare_amount"] > 0]
    distance_travel(df)
    haversine_distance(df)
    df = df[df["distance_travel"] > 0]
    df = df[df["haversine_km"] > 0]
    return df




## === cell 3
def remove_outliers(df):
    """Filter unrealistic distances and fares."""
    df = df[df["distance_travel"] < 30]
    df = df[df["fare_amount"] < 100]
    return df




## === cell 4
train_path = os.path.join(input_dir, "train.csv")
gen = chunk_generator(train_path)

max_chunks_to_sample = 2
sampled_X, sampled_y = [], []

while max_chunks_to_sample > 0:
    try:
        df = next(gen)
    except StopIteration:
        break

    df = add_datetime_features(df)
    df = data_clean(df)
    df = remove_outliers(df)

    if df.empty:
        continue

    chunk_X = np.column_stack(
        (
            df["distance_travel"].values.astype(np.float32),
            df["haversine_km"].values.astype(np.float32),
            df["passenger_count"].values.astype(np.float32),
            df["pickup_hour"].values.astype(np.float32),
            df["pickup_dayofweek"].values.astype(np.float32),
            np.ones(len(df), dtype=np.float32),  # bias term
        )
    )
    sampled_X.append(chunk_X)
    sampled_y.append(df["fare_amount"].values)

    max_chunks_to_sample -= 1

train_X = np.vstack(sampled_X).astype(np.float32)
train_y = np.concatenate(sampled_y).astype(
    np.float32
)  # cast target to float32 for speed

imp = SimpleImputer(missing_values=np.nan, strategy="mean")
train_X = imp.fit_transform(train_X).astype(np.float32)

regr = GradientBoostingRegressor(
    n_estimators=200,  # kept from original logic
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
    n_jobs=-1,  # <-- parallel execution
)
regr.fit(train_X, train_y)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3260650060.py in <cell line: 0>()
     42 
     43 # Use all CPU cores for Gradient Boosting to cut runtime dramatically
---> 44 regr = GradientBoostingRegressor(
     45     n_estimators=200,  # kept from original logic
     46     learning_rate=0.05,

TypeError: GradientBoostingRegressor.__init__() got an unexpected keyword argument 'n_jobs'

## === cell 5
test_path = os.path.join(input_dir, "test.csv")
tdf = pd.read_csv(
    test_path,
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    dtype={
        "pickup_longitude": np.float32,
        "pickup_latitude": np.float32,
        "dropoff_longitude": np.float32,
        "dropoff_latitude": np.float32,
        "passenger_count": np.int8,
        "pickup_datetime": object,
    },
    low_memory=False,
)

tdf = distance_travel(tdf)
haversine_distance(tdf)
tdf = add_datetime_features(tdf)

ttrain_X = np.column_stack(
    (
        tdf["distance_travel"].values.astype(np.float32),
        tdf["haversine_km"].values.astype(np.float32),
        tdf["passenger_count"].values.astype(np.float32),
        tdf["pickup_hour"].values.astype(np.float32),
        tdf["pickup_dayofweek"].values.astype(np.float32),
        np.ones(len(tdf), dtype=np.float32),
    )
)
ttrain_X = imp.transform(ttrain_X).astype(np.float32)
output = regr.predict(ttrain_X)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/25012505.py in <cell line: 0>()
     37 )
     38 ttrain_X = imp.transform(ttrain_X).astype(np.float32)
---> 39 output = regr.predict(ttrain_X)
     40 
     41 

NameError: name 'regr' is not defined

## === cell 6
submission = pd.DataFrame({"key": tdf["key"], "fare_amount": output})
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3290843912.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"key": tdf["key"], "fare_amount": output})
      2 submission.to_csv("submission.csv", index=False)
      3 print("Submission file written to submission.csv")

NameError: name 'output' is not defined
