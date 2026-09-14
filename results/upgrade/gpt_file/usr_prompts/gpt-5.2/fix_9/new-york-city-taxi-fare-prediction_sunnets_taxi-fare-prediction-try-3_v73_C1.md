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

4.67002

# 6. Current score

5.39893

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 194.65358) has done: 'I fix the two blockers that prevent an end-to-end run and a valid submission: (1) the `tf_keras` import crash (protobuf MessageFactory issue) by switching to the standard `tensorflow.keras` API that is available in Kaggle, and (2) the timezone-aware datetime conversion error in `add_time_features` by explicitly dropping timezone info before converting to string. Then I make the feature columns consistent across train/validation/test (the scaler error came from missing time-feature columns in `test_df` after cleaning) by reindexing to the training feature list before scaling. Finally, I ensure the script always writes `submissiontry_water.csv` with the exact required columns `key,fare_amount`.'
- What this solution (achieved 194.65358) has done: 'I fix the two execution blockers that prevent training and submission generation: (1) the `tf_keras`/protobuf crash by switching imports to the available `tensorflow.keras`, and (2) the scaler failure caused by leftover string columns by ensuring `key` is dropped from all model feature matrices and that all feature columns are numeric before scaling. I keep the same model architecture, loss, and training loop semantics, only making the minimal changes required for correctness and end-to-end execution. Finally, I ensure the pipeline always writes `submissiontry_water.csv` with exactly `key,fare_amount` and aligned row order with the raw Kaggle test set.'
- What this solution (achieved 15.62952) has done: 'I fix the TensorFlow/Keras import crash (`MessageFactory.GetPrototype`) by avoiding the incompatible `tensorflow` stack in this environment and using scikit-learn’s `MLPRegressor` to keep the same “multi-layer dense regressor trained with minibatches for many epochs” core logic. Then I correct the data path to the actual Kaggle-mounted dataset location and keep your existing cleaning + feature-engineering pipeline unchanged. Finally, I ensure the submission is generated with exactly `key,fare_amount` in the original test row order, and I clip negative predictions to 0 (score-improving but evaluation-consistent since fares can’t be negative). These changes are directly aimed at (1) unblocking end-to-end execution and (2) bringing RMSE down dramatically from ~194 toward the ~4.67 target band.'
- What this solution (achieved 5.36397) has done: 'Your current RMSE (15.63) is far worse than the target (4.67), so we should improve (lower) it with the smallest changes that don’t alter your core pipeline. The biggest fix is to prevent a distribution mismatch: right now you train on a cleaned subset but score on raw test rows; instead, apply the *same* cleaning to the Kaggle test set and then predict only for the remaining rows, filling dropped rows with a safe fallback (training median) so the submission still has all keys. Second, the scaler currently fills NaNs in Kaggle test using train medians but does not do the same for validation/test splits, which can degrade training/validation fit; we fill NaNs in all splits consistently (using train medians) without changing features/models. These changes keep your feature engineering and MLPRegressor intact while aligning train/test preprocessing and improving generalization toward the 4.67 target band.'
- What this solution (achieved 5.40641) has done: 'Your current RMSE (5.36397) is worse than the target (4.67002), so we should improve (lower) it with the smallest safe changes that keep your MLPRegressor + feature pipeline intact. The biggest avoidable error comes from applying an extra `add_time_features/add_coordinate_features/add_distances_features` to `testKaggle` earlier, then re-cleaning and re-featurizing again in cell 53; this double-pass can distort time strings and unnecessarily increases mismatch risk. I make the Kaggle-test path consistent: keep `testKaggle` raw until cell 53, then apply the *same* `clean + feature` steps exactly once, using the same `medians/scaler/feature_cols` learned from training. Finally, I clip predictions to a reasonable range (0–50) consistent with your training outlier filter, which typically reduces RMSE for this competition without changing the model/training logic.'
- What this solution (achieved 5.39893) has done: 'We’re currently worse than the target RMSE (5.40641 vs 4.67002; lower is better), so the smallest safe improvement is to make train/validation/testKaggle preprocessing more consistent and remove avoidable label-noise introduced by cleaning rules that are meant for training labels but are being applied to the test set via a dummy `fare_amount`. Concretely: (1) fix `clean()` so that fare-based outlier filtering runs only when `fare_amount` is truly present (training/validation) and is skipped for Kaggle test, (2) use a stable mean fallback (instead of median) for rows dropped during test cleaning to reduce squared-error penalty, and (3) avoid an O(N) Python dict loop by doing a vectorized merge back to the original test row order (same predictions, just safer alignment). These changes keep the same model (MLPRegressor), features, scaler, training loop, and prediction semantics, while typically reducing public LB RMSE for this competition.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

SEED = 1
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
from urllib.request import urlopen
from PIL import Image


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
    print(" New size after NYC lang lot: %d" % len(df))

    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["pickup_longitude"] != 0)]
    df = df[(df["dropoff_longitude"] != 0)]
    df = df[(df["dropoff_latitude"] != 0)]
    print(" New size after lang lot > 0: %d" % len(df))

    df = df[((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > 0.001)]
    df = df[((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > 0.001)]
    print(" New size after lang - lot > 0.001: %d" % len(df))

    print(" New size after only NYC: %d" % len(df))

    if "fare_amount" in df.columns:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))
    else:
        print(" Skipping fare_amount outlier filtering (fare_amount not present).")

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)  # Statue of Liberty

    df = df[
        (nyc_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != nyc_coord[0])
    ]
    df = df[
        (nyc_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != nyc_coord[0])
    ]
    print(" New size after NY airport: %d" % len(df))

    df = df[
        (fk_coord[1] != df["pickup_longitude"]) & (df["pickup_latitude"] != fk_coord[0])
    ]
    df = df[
        (fk_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != fk_coord[0])
    ]
    print(" New size after jfk airport: %d" % len(df))

    df = df[
        (ewr_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != ewr_coord[0])
    ]
    df = df[
        (ewr_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != ewr_coord[0])
    ]
    print(" New size after ewr airport: %d" % len(df))

    df = df[
        (lga_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != lga_coord[0])
    ]
    df = df[
        (lga_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != lga_coord[0])
    ]
    print(" New size after lgr airport: %d" % len(df))

    df = df[
        (sol_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != sol_coord[0])
    ]
    df = df[
        (sol_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != sol_coord[0])
    ]
    print(" New size after sol removed: %d" % len(df))

    print("Old size: %d" % len(df))
    df = remove_datapoints_from_water(df)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df


def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    mask_local_candidates = [
        "/kaggle/input/nyc-mask/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/input/new-york-city-taxi-fare-prediction/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/working/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "nyc_mask-74.5_-72.8_40.5_41.8.png",
    ]

    nyc_mask = None
    for pth in mask_local_candidates:
        if os.path.exists(pth):
            try:
                nyc_mask = np.array(Image.open(pth))[:, :, 0] > 230
                break
            except Exception:
                nyc_mask = None

    if nyc_mask is None:
        return df

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


def late_night(row):
    h = row["hour"]
    return 1 if (h <= 3 or h >= 23) else 0


def night(row):
    h = row["hour"]
    wd = row["weekday"]
    return 1 if ((h >= 20 or h <= 6) and (wd < 5)) else 0


def rush_hour(row):
    h = row["hour"]
    wd = row["weekday"]
    return 1 if ((h <= 20) and (h >= 16) and (wd < 5)) else 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)

    if dt.isna().any():
        dt2 = pd.to_datetime(df.loc[dt.isna(), "pickup_datetime"], errors="coerce")
        try:
            dt.loc[dt.isna()] = dt2.dt.tz_localize(
                "UTC", nonexistent="NaT", ambiguous="NaT"
            )
        except Exception:
            pass

    df["year"] = dt.dt.year.astype("int16")
    df["month"] = dt.dt.month.astype("int8")
    df["day"] = dt.dt.day.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["weekday"] = dt.dt.weekday.astype("int8")

    dt_naive = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    df["pickup_datetime"] = dt_naive.astype(str)

    df["night"] = df.apply(lambda x: night(x), axis=1).astype("int8")
    df["late_night"] = df.apply(lambda x: late_night(x), axis=1).astype("int8")
    df["rush_hour"] = df.apply(lambda x: rush_hour(x), axis=1).astype("int8")
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["latdiff"] = (lat1 - lat2).abs()
    df["londiff"] = (lon1 - lon2).abs()
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["distance"] = np.sqrt(
        np.abs(df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
        + np.abs(df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
    )
    return df


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


def distanceP(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    pred = np.asarray(prediction).reshape(-1)
    df = pd.DataFrame({id_column: raw_test[id_column].values, prediction_column: pred})
    df.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df))


def plot_loss_accuracy_rmse(history):
    try:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper right")
        plt.show()
    except Exception:
        print("Plotting history skipped (non-Keras history).")




## === cell 2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

TRAIN_PATH_CANDIDATES = [
    "/kaggle/data/train.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/input/train.csv",
]
TEST_PATH_CANDIDATES = [
    "/kaggle/data/test.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/input/test.csv",
]


def pick_existing(cands):
    for p in cands:
        if os.path.exists(p):
            return p
    return cands[0]


TRAIN_PATH = pick_existing(TRAIN_PATH_CANDIDATES)
TEST_PATH = pick_existing(TEST_PATH_CANDIDATES)

SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

print("Using TRAIN_PATH:", TRAIN_PATH)
print("Using TEST_PATH:", TEST_PATH)



## === cell 3
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

trainKaggle = pd.read_csv(
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

testKaggle = pd.read_csv(
    TEST_PATH,
    dtype={
        "key": "str",
        "pickup_datetime": "str",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
)



## === cell 4
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000]



## === cell 5
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))



## === cell 6
train_df.describe()



## === cell 7
test_df.describe()



## === cell 8
print("train_df clean")
train_df = clean(train_df)
test_df = clean(test_df)



## === cell 9
train_df.describe()



## === cell 10
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)

print("Skipping testKaggle add_time_features here to avoid double-processing.")



## === cell 11
train_df.describe()



## === cell 12
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features Disabled!")
test_df = add_coordinate_features(test_df)

print("Skipping testKaggle add_coordinate_features here to avoid double-processing.")



## === cell 13
train_df.describe()



## === cell 14
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)

print("Skipping testKaggle add_distances_features here to avoid double-processing.")

print("Done with Adding features")



## === cell 15
train_df.describe()



## === cell 16
try:
    _ = train_df.iloc[:2000].plot.scatter("latdiff", "londiff")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "weekday")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "night")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "late_night")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "rush_hour")
    plt.close("all")
except Exception as e:
    print("Plotting skipped:", repr(e))



## === cell 17
pass



## === cell 18
dropped_columns = ["pickup_datetime", "key"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)

print("Done with dropped_columns")



## === cell 19
train_df.shape



## === cell 20
train_df.describe()



## === cell 21
test_df.describe()



## === cell 22
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## === cell 23
train_df_main = train_df
validation_df_main = validation_df



## === cell 24
validation_df.describe()



## === cell 25
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 26
test_labels



## === cell 27
pass



## === cell 28
pass



## === cell 29
train_df.describe()



## === cell 30
validation_df.describe()



## === cell 31
test_df.describe()



## === cell 32
feature_cols = list(train_df.columns)
validation_df = validation_df.reindex(columns=feature_cols)
test_df = test_df.reindex(columns=feature_cols)

for c in feature_cols:
    train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
    validation_df[c] = pd.to_numeric(validation_df[c], errors="coerce")
    test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

medians = train_df.median(numeric_only=True)
train_df = train_df.fillna(medians)
validation_df = validation_df.fillna(medians)
test_df = test_df.fillna(medians)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)



## === cell 33
test_scaled



## === cell 34
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32, 8),
    activation="relu",
    solver="adam",
    alpha=0.0001,  # mild L2 regularization; analogous role to regularizers in Keras
    batch_size=BATCH_SIZE,
    learning_rate_init=LEARNING_RATE,
    max_iter=EPOCHS,
    shuffle=True,
    random_state=SEED,
    early_stopping=False,
    validation_fraction=0.0,  # keep explicit validation split semantics
    n_iter_no_change=EPOCHS + 1,
)

print("Dataset size:", DATASET_SIZE)
print("Epochs (max_iter):", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Batch size:", BATCH_SIZE)
print("Input dimension:", train_df_scaled.shape[1])
print("Features used:", list(train_df.columns))

mlp.fit(train_df_scaled, train_labels)



## === cell 35
print("Model diagram skipped (non-Keras model).")



## === cell 36
print("Training complete (MLPRegressor).")



## === cell 37
train_pred = mlp.predict(train_df_scaled)
train_rmse = mean_squared_error(train_labels, train_pred, squared=False)
train_mae = np.mean(np.abs(train_pred - train_labels))
train_mse = mean_squared_error(train_labels, train_pred, squared=True)
print([train_mse, train_mae, train_rmse, train_mse])
print("train mean_squared_error:", train_mse)
print("train mae:", train_mae)
print("train rmse:", train_rmse)
print("train mse:", train_mse)



## === cell 38
val_pred = mlp.predict(validation_df_scaled)
val_rmse = mean_squared_error(validation_labels, val_pred, squared=False)
val_mae = np.mean(np.abs(val_pred - validation_labels))
val_mse = mean_squared_error(validation_labels, val_pred, squared=True)
print([val_mse, val_mae, val_rmse, val_mse])
print("Validation mean_squared_error:", val_mse)
print("Validation mae:", val_mae)
print("Validation rmse:", val_rmse)
print("Validation mse:", val_mse)



## === cell 39
test_predictions = mlp.predict(test_scaled)
test_rmse = mean_squared_error(test_labels, test_predictions, squared=False)
test_mae = np.mean(np.abs(test_predictions - test_labels))
test_mse = mean_squared_error(test_labels, test_predictions, squared=True)
print([test_mse, test_mae, test_rmse, test_mse])
print("Test mean_squared_error:", test_mse)
print("Test mae:", test_mae)
print("Test rmse:", test_rmse)
print("Test mse:", test_mse)



## === cell 40
pass



## === cell 41
validation_predictions = val_pred.flatten()

plt.scatter(validation_labels, validation_predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot(
    [validation_predictions.min(), validation_predictions.max()],
    [validation_predictions.min(), validation_predictions.max()],
    "k--",
    lw=4,
)
plt.show()



## === cell 42
test_predictions = test_predictions.flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot(
    [test_predictions.min(), test_predictions.max()],
    [test_predictions.min(), test_predictions.max()],
    "k--",
    lw=4,
)
plt.show()



## === cell 43
pass



## === cell 44
print(np.argmax(test_predictions))
print(test_predictions[np.argmax(test_predictions)])
print(test_labels[np.argmax(test_predictions)])
test_df.iloc[np.argmax(test_predictions)]



## === cell 45
print(np.argmin(test_predictions))
print(test_predictions[np.argmin(test_predictions)])
print(test_labels[np.argmin(test_predictions)])
test_df.iloc[np.argmin(test_predictions)]



## === cell 46
fig, ax = plt.subplots()
ax.scatter(test_labels, test_predictions)
ax.plot(
    [test_labels.min(), test_labels.max()],
    [test_labels.min(), test_labels.max()],
    "k--",
    lw=4,
)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.show()



## === cell 47
plt.figure(figsize=(20, 10))
plt.plot(validation_labels[:100])
plt.plot(validation_predictions[:100])
plt.title("Prediction vs Actual")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()

validation_df_scaled[52]
validation_df.iloc[52]



## === cell 48
plt.figure(figsize=(20, 10))
plt.plot(test_labels[:100])
plt.plot(test_predictions[:100])
plt.title("Prediction vs Actual")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()



## === cell 49
error = validation_predictions - validation_labels
plt.hist(error, bins=100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## === cell 50
error = test_predictions - test_labels
plt.hist(error, bins=50)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## === cell 51
errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
print(len(error))
print(len(errorGreaterZero))

plt.hist(errorGreaterZero, bins=100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## === cell 52
pass



## === cell 53
testKaggle_for_clean = testKaggle.copy()

testKaggle_cleaned_full = clean(testKaggle_for_clean)

testKaggle_cleaned_full = add_time_features(testKaggle_cleaned_full)
testKaggle_cleaned_full = add_coordinate_features(testKaggle_cleaned_full)
testKaggle_cleaned_full = add_distances_features(testKaggle_cleaned_full)

testKaggle_cleaned_X = testKaggle_cleaned_full.drop(["pickup_datetime", "key"], axis=1)
testKaggle_cleaned_X = testKaggle_cleaned_X.reindex(columns=feature_cols)

for c in feature_cols:
    testKaggle_cleaned_X[c] = pd.to_numeric(testKaggle_cleaned_X[c], errors="coerce")
testKaggle_cleaned_X = testKaggle_cleaned_X.fillna(medians)

testKaggle_cleaned_scaled = scaler.transform(testKaggle_cleaned_X)

prediction_cleaned = mlp.predict(testKaggle_cleaned_scaled).reshape(-1)
prediction_cleaned = np.clip(prediction_cleaned, 0.0, 50.0)

fallback_fare = float(np.mean(train_labels))

pred_map = pd.DataFrame(
    {"key": testKaggle_cleaned_full["key"].values, "fare_amount": prediction_cleaned}
)
pred_full_df = pd.DataFrame({"key": testKaggle["key"].values}).merge(
    pred_map, on="key", how="left"
)
predictionKaggle = (
    pred_full_df["fare_amount"].fillna(fallback_fare).to_numpy(dtype=np.float32)
)
predictionKaggle = np.clip(predictionKaggle, 0.0, 50.0)



## === cell 54
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
print("Submission head:")
print(pd.read_csv(SUBMISSION_NAME).head())
