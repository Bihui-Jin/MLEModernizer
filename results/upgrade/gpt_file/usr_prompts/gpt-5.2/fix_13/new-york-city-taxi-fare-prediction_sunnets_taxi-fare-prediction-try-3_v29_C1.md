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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import urllib.request
from PIL import Image
from io import BytesIO

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 60  # keep identical training budget
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

ENABLE_WATER_MASK = False

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)

print(
    "Using scikit-learn MLPRegressor backend (TensorFlow disabled due to protobuf crash)."
)
print("Train path exists:", os.path.exists(TRAIN_PATH))
print("Test path exists:", os.path.exists(TEST_PATH))



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

train_cols = [
    "key",
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


def read_train_reservoir_sampled(
    path, n_sample, dtype, usecols, seed=0, chunksize=250_000
):
    rng = np.random.RandomState(seed)

    reservoir = None
    seen = 0  # number of rows processed so far

    for chunk in pd.read_csv(path, dtype=dtype, usecols=usecols, chunksize=chunksize):
        if chunk is None or len(chunk) == 0:
            continue

        if reservoir is None:
            take = min(n_sample, len(chunk))
            reservoir = chunk.iloc[:take].copy()
            reservoir = reservoir.reset_index(drop=True)
            seen = take

            start_idx = take
        else:
            start_idx = 0

        for row_idx in range(start_idx, len(chunk)):
            if seen < n_sample:
                reservoir.loc[seen] = chunk.iloc[row_idx].values
                seen += 1
                continue

            j = rng.randint(0, seen + 1)
            if j < n_sample:
                reservoir.iloc[j] = chunk.iloc[row_idx].values
            seen += 1

    if reservoir is None:
        reservoir = pd.DataFrame(columns=usecols)

    reservoir = reservoir.reindex(columns=usecols)
    return reservoir


trainKaggle = read_train_reservoir_sampled(
    TRAIN_PATH, DATASET_SIZE, dtype=datatypes, usecols=train_cols, seed=0
)
testKaggle = pd.read_csv(TEST_PATH, dtype=datatypes, usecols=test_cols)

print("Loaded train rows:", len(trainKaggle), "test rows:", len(testKaggle))



## === cell 2
trainKaggle["_pickup_dt"] = pd.to_datetime(
    trainKaggle["pickup_datetime"], errors="coerce", utc=True
)
trainKaggle = (
    trainKaggle.dropna(subset=["_pickup_dt"])
    .sort_values("_pickup_dt")
    .reset_index(drop=True)
)

n = len(trainKaggle)
n_test = int(0.20 * n)  # holdout test from the latest times
n_valid = int(0.10 * (n - n_test))  # validation from the remaining latest times

train_df = trainKaggle.iloc[: n - n_test - n_valid].copy()
validation_df = trainKaggle.iloc[n - n_test - n_valid : n - n_test].copy()
test_df = trainKaggle.iloc[n - n_test :].copy()

for _df in (train_df, validation_df, test_df):
    _df.drop(columns=["_pickup_dt"], inplace=True)



## === cell 3
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("validation_df Size %d" % len(validation_df))
print("test_df Size %d" % len(test_df))



## === cell 4
train_df.describe(include="all")



## === cell 5
validation_df.describe(include="all")



## === cell 6
test_df.describe(include="all")




## === cell 7
def remove_datapoints_from_water(df):
    if not ENABLE_WATER_MASK:
        return df

    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)
    url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"

    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            img = Image.open(BytesIO(resp.read())).convert("RGB")
        nyc_mask = (np.asarray(img)[:, :, 0] / 255.0) > 0.9

        pickup_x, pickup_y = lonlat_to_xy(
            df.pickup_longitude,
            df.pickup_latitude,
            nyc_mask.shape[1],
            nyc_mask.shape[0],
            BB,
        )
        dropoff_x, dropoff_y = lonlat_to_xy(
            df.dropoff_longitude,
            df.dropoff_latitude,
            nyc_mask.shape[1],
            nyc_mask.shape[0],
            BB,
        )

        pickup_x = np.clip(pickup_x, 0, nyc_mask.shape[1] - 1)
        dropoff_x = np.clip(dropoff_x, 0, nyc_mask.shape[1] - 1)
        pickup_y = np.clip(pickup_y, 0, nyc_mask.shape[0] - 1)
        dropoff_y = np.clip(dropoff_y, 0, nyc_mask.shape[0] - 1)

        idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]
        return df[idx]
    except Exception as e:
        print(
            f"Water-mask unavailable or failed to load ({type(e).__name__}: {e}). Skipping water filtering."
        )
        return df


def clean(df):
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != df["pickup_longitude"])
        & (df["dropoff_latitude"] != df["pickup_latitude"])
    ]
    print(" New size after removing same long lat: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    ]
    print(" New size after removing 0 long lat: %d" % len(df))

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
    print(" New size after only NYC: %d" % len(df))

    if "fare_amount" in df.columns:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

    eps = 1e-6

    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)

    def not_exact_point(df_in, lat_col, lon_col, lat0, lon0):
        return ~(
            (np.abs(df_in[lat_col] - lat0) <= eps)
            & (np.abs(df_in[lon_col] - lon0) <= eps)
        )

    df = df[
        not_exact_point(
            df, "pickup_latitude", "pickup_longitude", nyc_coord[0], nyc_coord[1]
        )
    ]
    df = df[
        not_exact_point(
            df, "dropoff_latitude", "dropoff_longitude", nyc_coord[0], nyc_coord[1]
        )
    ]
    print(" New size after NY exact-point removed: %d" % len(df))

    df = df[
        not_exact_point(
            df, "pickup_latitude", "pickup_longitude", fk_coord[0], fk_coord[1]
        )
    ]
    df = df[
        not_exact_point(
            df, "dropoff_latitude", "dropoff_longitude", fk_coord[0], fk_coord[1]
        )
    ]
    print(" New size after jfk exact-point removed: %d" % len(df))

    df = df[
        not_exact_point(
            df, "pickup_latitude", "pickup_longitude", ewr_coord[0], ewr_coord[1]
        )
    ]
    df = df[
        not_exact_point(
            df, "dropoff_latitude", "dropoff_longitude", ewr_coord[0], ewr_coord[1]
        )
    ]
    print(" New size after ewr exact-point removed: %d" % len(df))

    df = df[
        not_exact_point(
            df, "pickup_latitude", "pickup_longitude", lga_coord[0], lga_coord[1]
        )
    ]
    df = df[
        not_exact_point(
            df, "dropoff_latitude", "dropoff_longitude", lga_coord[0], lga_coord[1]
        )
    ]
    print(" New size after lga exact-point removed: %d" % len(df))

    df = df[
        not_exact_point(
            df, "pickup_latitude", "pickup_longitude", sol_coord[0], sol_coord[1]
        )
    ]
    df = df[
        not_exact_point(
            df, "dropoff_latitude", "dropoff_longitude", sol_coord[0], sol_coord[1]
        )
    ]
    print(" New size after sol exact-point removed: %d" % len(df))

    print("Old size: %d" % len(df))
    df = remove_datapoints_from_water(df)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df


def clean_kaggle_test_no_drop(df):
    df = df.copy()

    for c in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    pc = df["passenger_count"]
    valid_pc = pc[(pc > 0) & (pc <= 6)]
    fill_pc = float(valid_pc.median()) if len(valid_pc) else 1.0
    df["passenger_count"] = pc.fillna(fill_pc).clip(1, 6).astype("float32")

    MinMax = (-74.5, -72.8, 40.5, 41.8)
    df["pickup_longitude"] = df["pickup_longitude"].clip(MinMax[0], MinMax[1])
    df["dropoff_longitude"] = df["dropoff_longitude"].clip(MinMax[0], MinMax[1])
    df["pickup_latitude"] = df["pickup_latitude"].clip(MinMax[2], MinMax[3])
    df["dropoff_latitude"] = df["dropoff_latitude"].clip(MinMax[2], MinMax[3])

    for c in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]:
        med = float(df[c].median())
        df[c] = df[c].fillna(med).astype("float32")

    df["pickup_datetime"] = df["pickup_datetime"].astype(str)

    return df


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    if dt.isna().any():
        med_ts = dt.dropna().median()
        dt = dt.fillna(med_ts)

    df["year"] = dt.dt.year.astype("int16")
    df["month"] = dt.dt.month.astype("int8")
    df["day"] = dt.dt.day.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["weekday"] = dt.dt.weekday.astype("int8")
    df["pickup_datetime"] = dt.astype(str)

    hour = df["hour"].astype(np.int16)
    weekday = df["weekday"].astype(np.int16)

    df["late_night"] = ((hour <= 3) | (hour >= 23)).astype("int8")
    df["night"] = (((hour >= 20) | (hour <= 6)) & (weekday < 5)).astype("int8")
    df["rush_hour"] = (
        (((hour >= 7) & (hour <= 10)) | ((hour >= 16) & (hour <= 20))) & (weekday < 5)
    ).astype("int8")
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["latdiff"] = lat1 - lat2
    df["londiff"] = lon1 - lon2
    return df


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["distance"] = distance(lat1, lon1, lat2, lon2)
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    pred = np.asarray(prediction).reshape(-1)
    df = pd.DataFrame({id_column: raw_test[id_column].values, prediction_column: pred})
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows=", len(df))


def plot_loss_accuracy_rmse(history):
    print(
        "Plot skipped: sklearn MLPRegressor does not provide Keras-like history dict."
    )




## === cell 8
print("train_df clean")
train_df = clean(train_df)
print("validation_df clean")
validation_df = clean(validation_df)

print("test_df clean")
test_df = clean(test_df)

print("testKaggle clean (no-drop, for consistent feature distribution at inference)")
testKaggle_raw = testKaggle.copy()
testKaggle = clean_kaggle_test_no_drop(testKaggle)



## === cell 9
train_df.describe(include="all")



## === cell 10
validation_df.describe(include="all")



## === cell 11
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 12
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
validation_df = add_coordinate_features(validation_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 13
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)
print("Done with Adding features")



## === cell 14
train_df.describe(include="all")



## === cell 15
validation_df.describe(include="all")




## === cell 16
def cap_distance_features(train_d, valid_d, test_d, kaggle_d):
    for col in ["distance", "manhattan"]:
        if col in train_d.columns:
            hi = float(train_d[col].quantile(0.999))
            lo = float(train_d[col].quantile(0.001))
            train_d[col] = train_d[col].clip(lo, hi)
            valid_d[col] = valid_d[col].clip(lo, hi)
            test_d[col] = test_d[col].clip(lo, hi)
            kaggle_d[col] = kaggle_d[col].clip(lo, hi)
            print(f"Capped {col}: [{lo:.6f}, {hi:.6f}]")
    return train_d, valid_d, test_d, kaggle_d


train_df, validation_df, test_df, testKaggle = cap_distance_features(
    train_df, validation_df, test_df, testKaggle
)

dropped_columns = ["passenger_count", "pickup_datetime"]

train_df = train_df.drop(dropped_columns + ["key"], axis=1)
test_df = test_df.drop(dropped_columns + ["key"], axis=1)
validation_df = validation_df.drop(dropped_columns + ["key"], axis=1)

testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)
print("Done with dropped_columns and key removal")



## === cell 17
train_labels = train_df["fare_amount"].values.astype(np.float32)
validation_labels = validation_df["fare_amount"].values.astype(np.float32)
test_labels = test_df["fare_amount"].values.astype(np.float32)

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)
print("Done with Labels")



## === cell 18
train_df.shape



## === cell 19
test_df.shape



## === cell 20
validation_df.shape




## === cell 21
def ensure_numeric_and_align(train_x, valid_x, test_x, kaggle_x):
    common_cols = list(train_x.columns)
    valid_x = valid_x.reindex(columns=common_cols)
    test_x = test_x.reindex(columns=common_cols)
    kaggle_x = kaggle_x.reindex(columns=common_cols)

    def to_numeric_df(df):
        out = df.copy()
        for c in out.columns:
            if not pd.api.types.is_numeric_dtype(out[c]):
                out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    train_x = to_numeric_df(train_x)
    valid_x = to_numeric_df(valid_x)
    test_x = to_numeric_df(test_x)
    kaggle_x = to_numeric_df(kaggle_x)

    med = train_x.median(numeric_only=True)
    train_x = train_x.fillna(med)
    valid_x = valid_x.fillna(med)
    test_x = test_x.fillna(med)
    kaggle_x = kaggle_x.fillna(med)

    return train_x, valid_x, test_x, kaggle_x


train_df, validation_df, test_df, testKaggle_clean = ensure_numeric_and_align(
    train_df, validation_df, test_df, testKaggle_clean
)

scaler = preprocessing.RobustScaler(
    with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)
)
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)

print(
    "Scaled shapes:",
    train_df_scaled.shape,
    validation_df_scaled.shape,
    test_scaled.shape,
    testKaggle_scaled.shape,
)




## === cell 22
def rmse_np(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))




## === cell 23
y_scaler = preprocessing.StandardScaler()
train_y_scaled = y_scaler.fit_transform(train_labels.reshape(-1, 1)).ravel()

mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32, 8),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=BATCH_SIZE,
    learning_rate_init=LEARNING_RATE,
    max_iter=EPOCHS,
    shuffle=True,
    random_state=0,
    early_stopping=False,
    verbose=True,
)

print("Dataset size:", DATASET_SIZE)
print("Epochs (max_iter):", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Batch size:", BATCH_SIZE)
print("Input dimension:", train_df_scaled.shape[1])
print("Features used:", list(train_df.columns))

mlp.fit(train_df_scaled, train_y_scaled)

val_pred_scaled = mlp.predict(validation_df_scaled)
val_pred = y_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()
print("Validation RMSE:", rmse_np(validation_labels, val_pred))



## === cell 24
print("Model visualization skipped (not applicable for sklearn MLPRegressor).")



## === cell 25
history = None
if history is not None:
    plot_loss_accuracy_rmse(history)



## === cell 26
prediction_scaled = mlp.predict(test_scaled)
predictionKaggle_scaled = mlp.predict(testKaggle_scaled)

prediction = y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).ravel()
predictionKaggle = y_scaler.inverse_transform(
    predictionKaggle_scaled.reshape(-1, 1)
).ravel()

prediction = np.clip(prediction, 0.0, None)
predictionKaggle = np.clip(predictionKaggle, 0.0, None)

print("Pred shapes:", prediction.shape, predictionKaggle.shape)



## === cell 27
y_true = test_labels[:1000]
y_pred = prediction[:1000].reshape(-1)

mse_vals = (y_true - y_pred) ** 2
print("Mean MSE on 1000-sample holdout:", float(np.mean(mse_vals)))
print("Approx RMSE on 1000-sample holdout:", float(np.sqrt(np.mean(mse_vals))))



## === cell 28
full = pd.DataFrame(
    {"key": testKaggle_raw["key"].values, "fare_amount": predictionKaggle}
)
full["fare_amount"] = full["fare_amount"].astype(np.float32)
full["fare_amount"] = np.clip(full["fare_amount"].values, 0.0, None)

full.to_csv(SUBMISSION_NAME, index=False)
print("Output complete:", SUBMISSION_NAME, "rows=", len(full))

idx = min(10000, len(prediction) - 1, len(test_labels) - 1)
print(
    "Example pred vs label (holdout):", float(prediction[idx]), float(test_labels[idx])
)
print("Submission preview:")
print(pd.read_csv(SUBMISSION_NAME).head())
