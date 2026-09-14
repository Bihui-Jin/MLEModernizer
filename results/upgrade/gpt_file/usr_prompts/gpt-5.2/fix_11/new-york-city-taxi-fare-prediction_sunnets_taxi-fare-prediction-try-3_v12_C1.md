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

4.01943

# 6. Current score

257.39645

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 352.17609) has done: 'I fix the two hard runtime blockers: the NYC water-mask loading currently tries to read an external URL (not allowed) and the optimizer construction uses a deprecated `optimizers.adam` API in Keras 3. To keep core logic intact while improving RMSE toward your target, I make the water-mask step gracefully skip (or use a local cached mask if present) instead of crashing, and I correct the time-feature functions’ logical conditions (they currently always return 1/0 incorrectly), which is a minimal bug fix that should materially improve model signal. I also ensure CSV paths work in Kaggle (`/kaggle/input/...`) and keep the submission format exactly `key,fare_amount` with a `.csv` suffix.'
- What this solution (achieved 63.07303) has done: 'I fix the hard runtime blocker that prevents imports/model training by forcing Keras to use the pure-Python protobuf implementation (this avoids the `MessageFactory.GetPrototype` crash seen in Kaggle for some protobuf builds). I also correct the dataset paths to point at the actual provided `/kaggle/input/new-york-city-taxi-fare-prediction/...` files (your current `/kaggle/input/train.csv` paths not exist in this environment), which is required to run end-to-end and generate a submission. These changes preserve your model/training/feature logic exactly and should materially improve score simply by letting the intended pipeline run on real data instead of failing early. Finally, I keep the submission formatting (`key,fare_amount`) and ensure the output filename ends with `.csv`.'
- What this solution (achieved 1554.80391) has done: 'I fix the protobuf-related import crash by avoiding `tf_keras` (which triggers the `MessageFactory.GetPrototype` issue in this environment) and using the already-installed `keras` (Keras 3) backend instead, while keeping the exact same model architecture, loss, optimizer type, and training loop. I also fix a major data-loading logic bug: the training CSV is currently read without the `fare_amount` column, which makes the model train on the wrong target (or crash/behave badly), and this is the main reason the RMSE is extremely poor. Finally, I keep all feature engineering and cleaning logic intact and ensure the submission is written as a valid `.csv` with `key,fare_amount`.'
- What this solution (achieved 55.93349) has done: 'I fix the hard runtime blocker causing the protobuf `MessageFactory.GetPrototype` crash by avoiding TensorFlow/Keras protobuf initialization entirely and switching to scikit-learn’s `MLPRegressor` while keeping the same feed-forward dense-network core logic (multi-layer ReLU MLP trained with Adam-like optimization). I also correct the data path to the actual Kaggle-provided `/kaggle/input/new-york-city-taxi-fare-prediction/...` files and ensure `key` is preserved so the submission matches `key,fare_amount`. To move RMSE drastically down toward the target, I keep your existing cleaning + feature engineering, and fix one major score-killing issue: the model must be trained on the real `fare_amount` target and then used to predict on the real test set. Finally, I always write a valid `.csv` submission to the working directory with the exact required columns.'
- What this solution (achieved 220.60718) has done: 'Your RMSE is extremely high mainly because the model is being trained with too little usable data (heavy filtering + only 80k rows) and because the engineered time flags are computed very slowly with `df.apply`, which limits practical dataset size. To move score substantially toward the 4.019 target while preserving your core model/feature logic, I (1) vectorize the time-feature flags (same semantics, far faster), (2) increase `DATASET_SIZE` to use more training signal within the 600s limit, and (3) add a minimal, metric-aligned safeguard to clip negative fare predictions to 0 (fares are non-negative; this typically reduces RMSE without changing model logic). Everything else (cleaning rules, features, scaling, MLP architecture/solver) stays the same, and the script still writes a valid `key,fare_amount` submission `.csv`.'
- What this solution (achieved 220.60718) has done: 'I fix the NaN crash by applying the same cleaning/NA-handling to the held-out `test_df` (the internal split from train) as you already do for `train_df` and `validation_df`, and by ensuring the datetime-derived feature columns never introduce NaNs. This preserves your core model, features, scaling, and training loop, but makes `mlp.predict(test_scaled)` run reliably. I also align the internal split columns (drop `passenger_count` and `pickup_datetime` consistently after feature engineering) and add a minimal post-scale NaN/inf guard (should be score-neutral unless upstream parsing produces rare invalid rows). The submission writing stays identical (`key,fare_amount` and `.csv`).'
- What this solution (achieved 230.89335) has done: 'I fix the scaler crash by ensuring no string columns (notably `key`, and any leftover datetime strings) are present when fitting `MinMaxScaler`, while keeping your existing feature engineering and MLP training logic unchanged. Concretely, I drop `key` from `train_df/test_df/validation_df` at the same time you already drop `pickup_datetime`, and I add a small “numeric-only + inf/NaN guard” right before scaling to prevent hidden object dtypes from slipping in. This unblocks the downstream cells (model fit/predict/submission) and produce a valid `key,fare_amount` submission CSV. No modeling changes are introduced beyond this correctness fix, so score changes should come only from running the intended pipeline end-to-end.'
- What this solution (achieved 257.39645) has done: 'Your RMSE is far from the target mainly because the current pipeline trains on a *randomly sampled* subset via `skiprows`, which breaks the relationship between pickup_datetime/locations and fare (it becomes effectively noise), and because `test_df` is incorrectly “cleaned” using fare-based rules (it contains `fare_amount`, but in real test it won’t). To move the score sharply down toward the target while preserving your core feature engineering and MLPRegressor logic, I switch sampling to a reproducible **random row sample using `skiprows` as a boolean mask** (so we truly sample across the whole file), and I ensure the train/validation/test split is done after sampling without inadvertently using fare-based filters on a pseudo-test. Finally, I add a minimal guard to ensure `submissiontry_water.csv` is always a valid `.csv` with the correct row order and columns.'
- What this solution (achieved 257.39645) has done: 'Your RMSE is still extremely high mainly because `MinMaxScaler` is being fit on `train_df` while `validation_df/holdout_df/testKaggle` can silently end up with a *different* set/order of numeric columns after cleaning (rows dropped) and feature generation, so you may be scaling/predicting on misaligned features (effectively garbage input). I make one minimal, score-critical fix: enforce a single canonical feature column list from `train_df` and reindex `validation_df/holdout_df/testKaggle_clean` to that list (same order, missing columns filled with 0), preserving your model/feature logic. I also add a tiny safeguard to keep the submission keys aligned to the original test order (no accidental reordering), and keep your non-negative clipping unchanged. These changes are intended to sharply reduce RMSE toward the target without changing the model architecture, loss, or training approach.'

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001

DATASET_SIZE = 500000

np.random.seed(1)



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

usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

TRAIN_ROWS_EXCL_HEADER = 55423856
rng = np.random.RandomState(1)

keep_prob = min(1.0, float(DATASET_SIZE) / float(TRAIN_ROWS_EXCL_HEADER))
skip_mask = rng.rand(TRAIN_ROWS_EXCL_HEADER) > keep_prob
skiprows = np.where(skip_mask)[0] + 1  # +1 because row 1 is first data row after header

trainKaggle = pd.read_csv(
    TRAIN_PATH,
    dtype=datatypes,
    usecols=usecols_train,
    skiprows=skiprows.tolist(),
)

if len(trainKaggle) > DATASET_SIZE:
    trainKaggle = trainKaggle.sample(n=DATASET_SIZE, random_state=1).reset_index(
        drop=True
    )

if "key" in trainKaggle.columns:
    trainKaggle = trainKaggle.drop_duplicates(subset=["key"], keep="first").reset_index(
        drop=True
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



## === cell 2
train_df, holdout_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)



## === cell 3
testKaggle.head()



## === cell 4
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## === cell 5
train_df.head()



## === cell 6
holdout_df.head()



## === cell 7
validation_df.head()




## === cell 8
def remove_datapoints_from_water(df):
    """
    Bugfix: original code attempted to read a remote PNG via plt.imread(URL),
    which fails in Kaggle (no external network + matplotlib doesn't open URLs).
    Minimal, score-neutral change: if a local mask exists, use it; otherwise skip.
    """

    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        x = (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int")
        y = (dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])).astype("int")
        return x, y

    BB = (-74.5, -72.8, 40.5, 41.8)

    local_candidates = [
        "/kaggle/input/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/working/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "nyc_mask-74.5_-72.8_40.5_41.8.png",
    ]
    mask_path = next((p for p in local_candidates if os.path.exists(p)), None)
    if mask_path is None:
        return df

    nyc_mask = plt.imread(mask_path)
    if nyc_mask.ndim == 3:
        nyc_mask = nyc_mask[:, :, 0]
    nyc_mask = nyc_mask > 0.9

    pickup_x, pickup_y = lonlat_to_xy(
        df.pickup_longitude.values,
        df.pickup_latitude.values,
        nyc_mask.shape[1],
        nyc_mask.shape[0],
        BB,
    )
    dropoff_x, dropoff_y = lonlat_to_xy(
        df.dropoff_longitude.values,
        df.dropoff_latitude.values,
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


def late_night(row):
    return 1 if (row["hour"] <= 3) else 0


def night(row):
    return (
        1 if ((row["hour"] > 20) or (row["hour"] < 6)) and (row["weekday"] < 5) else 0
    )


def rush_hour(row):
    return 1 if (16 <= row["hour"] <= 20) and (row["weekday"] < 5) else 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)

    df["year"] = dt.dt.year.fillna(0).astype("int16")
    df["month"] = dt.dt.month.fillna(0).astype("int8")
    df["day"] = dt.dt.day.fillna(0).astype("int8")
    df["hour"] = dt.dt.hour.fillna(0).astype("int8")
    df["weekday"] = dt.dt.weekday.fillna(0).astype("int8")

    df["pickup_datetime"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")

    hour = df["hour"].astype(np.int16)
    weekday = df["weekday"].astype(np.int16)

    df["night"] = (((hour > 20) | (hour < 6)) & (weekday < 5)).astype("int8")
    df["late_night"] = (hour <= 3).astype("int8")
    df["rush_hour"] = ((hour >= 16) & (hour <= 20) & (weekday < 5)).astype("int8")
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["latdiff"] = lat1 - lat2
    df["londiff"] = lon1 - lon2
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["distance"] = np.sqrt(np.abs(lon1 - lon2) ** 2 + np.abs(lat1 - lat2) ** 2)
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column].values
    df[[id_column, prediction_column]].to_csv((file_name), index=False)
    print("Output complete:", file_name, "rows=", len(df))


def plot_loss_accuracy(history_dict):
    if history_dict is None:
        print("No Keras history available (sklearn model). Skipping plot.")
        return




## === cell 9
validation_df.head()



## === cell 10
print("train_df clean")
train_df = clean(train_df)
print("validation_df clean")
validation_df = clean(validation_df)

print("holdout_df clean")
holdout_df = clean(holdout_df)

train_df.describe()

print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)
print("holdout_df add_time_features")
holdout_df = add_time_features(holdout_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 11
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
validation_df = add_coordinate_features(validation_df)
print("holdout_df add_coordinate_features")
holdout_df = add_coordinate_features(holdout_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 12
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)
print("holdout_df add_distances_features")
holdout_df = add_distances_features(holdout_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## === cell 13
dropped_columns = ["key", "passenger_count", "pickup_datetime"]

train_df = train_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)
holdout_df = holdout_df.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(dropped_columns, axis=1)

print("Done with dropped_columns")



## === cell 14
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
holdout_labels = holdout_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
holdout_df = holdout_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 15
train_df.shape



## === cell 16
holdout_df.shape



## === cell 17
validation_df.shape




## === cell 18
def _ensure_numeric_frame(df, name):
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        print(f"[warn] Dropping non-numeric columns from {name}: {non_numeric}")
        df = df.drop(columns=non_numeric)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)
    return df


train_df = _ensure_numeric_frame(train_df, "train_df")
validation_df = _ensure_numeric_frame(validation_df, "validation_df")
holdout_df = _ensure_numeric_frame(holdout_df, "holdout_df")
testKaggle_clean = _ensure_numeric_frame(testKaggle_clean, "testKaggle_clean")

feature_cols = list(train_df.columns)
validation_df = validation_df.reindex(columns=feature_cols, fill_value=0.0)
holdout_df = holdout_df.reindex(columns=feature_cols, fill_value=0.0)
testKaggle_clean = testKaggle_clean.reindex(columns=feature_cols, fill_value=0.0)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
holdout_scaled = scaler.transform(holdout_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)

train_df_scaled = np.nan_to_num(train_df_scaled, nan=0.0, posinf=0.0, neginf=0.0)
validation_df_scaled = np.nan_to_num(
    validation_df_scaled, nan=0.0, posinf=0.0, neginf=0.0
)
holdout_scaled = np.nan_to_num(holdout_scaled, nan=0.0, posinf=0.0, neginf=0.0)
testKaggle_scaled = np.nan_to_num(testKaggle_scaled, nan=0.0, posinf=0.0, neginf=0.0)



## === cell 19
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32, 8),
    activation="relu",
    solver="adam",
    alpha=0.0,  # keep regularization minimal; original used activity_regularizer, not weight decay
    batch_size=BATCH_SIZE,
    learning_rate_init=LEARNING_RATE,
    max_iter=EPOCHS,
    shuffle=True,
    random_state=1,
    verbose=True,
)

print("Dataset size: %s" % DATASET_SIZE)
print("Epochs (max_iter): %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size: %s" % BATCH_SIZE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % list(train_df.columns))

mlp.fit(train_df_scaled, train_labels)



## === cell 20
plot_loss_accuracy(None)



## === cell 21
holdout_pred = mlp.predict(holdout_scaled)



## === cell 22
holdout_labels.shape



## === cell 23
predictionKaggle = mlp.predict(testKaggle_scaled)



## === cell 24
testKaggle.head()



## === cell 25
predictionKaggle = np.clip(predictionKaggle, 0.0, None)

predictionKaggle = np.asarray(predictionKaggle, dtype=np.float32).reshape(-1, 1)
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
