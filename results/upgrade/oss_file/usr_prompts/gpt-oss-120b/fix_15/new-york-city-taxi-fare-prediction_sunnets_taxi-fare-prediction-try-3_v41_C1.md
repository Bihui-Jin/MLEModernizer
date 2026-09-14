# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
        input/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
```

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> input/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> input/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> input/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> working/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 5. Target score

4.25412

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 15.22047) has done: 'Implemented fixes to resolve runtime errors and improve model performance:
- Updated `DATASET_SIZE` to use more training rows.
- Corrected optimizer creation (`optimizers.Adam` with `learning_rate` argument).
- Simplified `remove_datapoints_from_water` to bypass external image loading.
- Adjusted evaluation to use scaled test data.
- Fixed submission output to handle prediction shape.
- Guarded optional visualization import.
- Minor cleanup for consistency.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor


def resolve_path(*parts):
    """
    Locate a file in common Kaggle locations.
    Falls back to a recursive search from the current directory if not found.
    """
    base_paths = [
        os.path.join("input", "new-york-city-taxi-fare-prediction"),
        os.path.join("working", "new-york-city-taxi-fare-prediction"),
        os.path.join("data", "new-york-city-taxi-fare-prediction"),
        os.path.join("kaggle", "input", "new-york-city-taxi-fare-prediction"),
    ]
    for base in base_paths:
        p = os.path.join(base, *parts)
        if os.path.exists(p):
            return p
    target = os.path.join(*parts)
    for root, _, files in os.walk("."):
        if os.path.basename(target) in files:
            return os.path.abspath(os.path.join(root, os.path.basename(target)))
    raise FileNotFoundError(
        f"Unable to locate {'/'.join(parts)} in any known directory."
    )


TRAIN_PATH = resolve_path("labels.csv")
TEST_PATH = resolve_path("test.csv")
SUBMISSION_NAME = "submission.csv"

DATASET_SIZE = 200_000
EPOCHS = 5  # kept for compatibility, not used with RandomForest
BATCH_SIZE = 256  # kept for compatibility, not used with RandomForest
LEARNING_RATE = 1e-3  # kept for compatibility, not used with RandomForest




## === cell 1
datatypes = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
train_raw = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    dtype=datatypes,
    usecols=[
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)
test_raw = pd.read_csv(
    TEST_PATH,
    dtype=datatypes,
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)




## === cell 2
train_df, validation_df = train_test_split(train_raw, test_size=0.10, random_state=1)




## === cell 3
def remove_datapoints_from_water(df):
    return df


def clean(df, is_training=True):
    """
    Clean a dataframe.
    If `is_training` is False (i.e., the test set), skip filters that require the
    target column `fare_amount`.
    """
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != df["pickup_longitude"])
        & (df["dropoff_latitude"] != df["pickup_latitude"])
    ]
    print(" New size after removing same long/lat: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    ]
    print(" New size after removing zero coords: %d" % len(df))

    MinMax = (-74.5, -72.8, 40.5, 41.8)
    df = df[
        (MinMax[0] <= df["pickup_longitude"]) & (df["pickup_longitude"] <= MinMax[1])
    ]
    df = df[
        (MinMax[0] <= df["dropoff_longitude"]) & (df["dropoff_longitude"] <= MinMax[1])
    ]
    df = df[(MinMax[2] <= df["pickup_latitude"]) & (df["pickup_latitude"] <= MinMax[3])]
    df = df[
        (MinMax[2] <= df["dropoff_latitude"]) & (df["dropoff_latitude"] <= MinMax[3])
    ]
    print(" New size after NYC bbox filter: %d" % len(df))

    if is_training and "fare_amount" in df.columns:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" New size after fare/passenger filters: %d" % len(df))

    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)

    for coord in [nyc_coord, fk_coord, ewr_coord, lga_coord, sol_coord]:
        df = df[
            (coord[1] != df["pickup_longitude"]) | (coord[0] != df["pickup_latitude"])
        ]
        df = df[
            (coord[1] != df["dropoff_longitude"]) | (coord[0] != df["dropoff_latitude"])
        ]

    print(" New size after airport/landmark filters: %d" % len(df))
    print("Old size before water removal: %d" % len(df))
    df = remove_datapoints_from_water(df)
    print("New size after water removal (no‑op): %d" % len(df))
    return df




## === cell 4
print("Cleaning training set")
train_df = clean(train_df, is_training=True)
print("Cleaning validation set")
validation_df = clean(validation_df, is_training=True)
print("Cleaning test set")
test_df = clean(test_raw.copy(), is_training=False)


def add_features(df):
    """Add haversine, Manhattan distance (km) and simple datetime features."""
    R = 6371.0  # Earth radius in km
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["haversine_km"] = R * c

    lat_km = 111.0
    lon_km = 111.0 * np.cos(lat1)
    df["manhattan_km"] = (np.abs(dlat) * lat_km) + (np.abs(dlon) * lon_km)

    dt = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.weekday
    return df


train_df = add_features(train_df)
validation_df = add_features(validation_df)
test_df = add_features(test_df)




## === cell 5
test_keys = test_df["key"].values

drop_cols = ["key", "pickup_datetime"]
train_df = train_df.drop(columns=drop_cols)
validation_df = validation_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)




## === cell 6
train_labels = np.log1p(train_df["fare_amount"].values)
validation_labels = np.log1p(validation_df["fare_amount"].values)

train_df = train_df.drop(columns=["fare_amount"])
validation_df = validation_df.drop(columns=["fare_amount"])




## === cell 7
scaler = preprocessing.MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
validation_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)




## === cell 8
rf_model = RandomForestRegressor(
    n_estimators=800,  # more trees for modest performance gain
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    verbose=0,
)
rf_model.fit(train_scaled, train_labels)




## === cell 9
importances = rf_model.feature_importances_
print("Feature importances (top 5):")
for idx in np.argsort(importances)[-5:][::-1]:
    print(f"{train_df.columns[idx]:30s}: {importances[idx]:.4f}")




## === cell 10
pred_log = rf_model.predict(test_scaled)
pred = np.expm1(pred_log).ravel()
submission = pd.DataFrame({"key": test_keys, "fare_amount": pred})
submission.to_csv(SUBMISSION_NAME, index=False)
print(f"Submission written to {SUBMISSION_NAME}")
