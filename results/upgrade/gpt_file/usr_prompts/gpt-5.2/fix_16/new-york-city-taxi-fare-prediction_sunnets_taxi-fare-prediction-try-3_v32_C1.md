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

4.20882

# 6. Current score

6.18149

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 524.81175) has done: 'I fix the runtime blockers caused by (1) importing the standalone `keras` package (which triggers the protobuf `MessageFactory` error here) by switching to `tf_keras` (TensorFlow Keras), (2) trying to read a mask image from an internet URL (not available) by loading it locally if present and otherwise skipping that filter safely, and (3) using deprecated optimizer API (`optimizers.adam`) by using `Adam(learning_rate=...)`. I also correct the broken time-feature logic (night/late_night conditions were always true) and ensure pickup_datetime parsing works with the dataset format, which should materially improve RMSE toward your target without changing the model architecture/training loop. Finally, I remove/guard optional visualization and fix the loss check shape mismatch so the notebook runs end-to-end and writes a valid `submissiontry_water.csv`.'
- What this solution (achieved 1374.18038) has done: 'I fix the runtime blocker happening at the very start by avoiding the protobuf-incompatible Keras import path and forcing TensorFlow-backed `tf_keras` to be used cleanly (this is what triggers the `MessageFactory.GetPrototype` crash in this environment). I also correct the train/test file paths to the actual Kaggle-mounted locations you listed (`/kaggle/input/...`) so the notebook reliably reads data on Kaggle without depending on relative paths. Finally, I fix the MSE sanity-check call to use a compatible API (and matching shapes) so it can’t crash after training; the model, features, training loop, and submission format remain unchanged, but these fixes should eliminate the catastrophic failure/garbage predictions that led to the very poor RMSE.'
- What this solution (achieved 5.36317) has done: 'Your current notebook likely didn’t yield a Kaggle score because it writes `submissiontry_water.csv` into the current working directory, while Kaggle only reliably picks up files saved under `/kaggle/working/`. I make the smallest change to always write the submission to `/kaggle/working/submissiontry_water.csv` and also ensure the submission `key` aligns with the *filtered* `testKaggle` rows used for prediction (so row counts never mismatch). Additionally, because filtering can drop some test rows, I re-expand predictions back to the full original test set (missing rows filled with the training mean fare), ensuring the submission always has exactly the expected number of rows and keys. These changes preserve your model, features, training loop, and loss/metric semantics while making the pipeline robust and score-producing.'
- What this solution (achieved 5.36317) has done: 'To move RMSE down toward your 4.20882 target (current 5.36317; lower is better) without changing the model/training loop, I make two minimal, score-relevant fixes: (1) ensure the scaler and MLP see consistent feature columns in the same order across train/valid/test/testKaggle (avoids silent column misalignment that can materially hurt RMSE), and (2) add a very small, standard NYC-taxi cleanup that removes obviously bad coordinate pairs (lat/lon out of range / NaNs already handled) which otherwise injects heavy noise into this MLP. Everything else (feature engineering functions, MinMaxScaler usage, MLPRegressor architecture/optimizer/max_iter, and submission semantics) stays the same, and it still writes `/kaggle/working/submissiontry_water.csv`.'
- What this solution (achieved 5.34342) has done: 'Your current RMSE (5.363) is worse than the target (4.209), so we should nudge performance upward with minimal, score-relevant changes while preserving the same MLPRegressor approach and feature set. The biggest “free” RMSE wins here typically come from (1) fixing a subtle bug in the cleaning step where you accidentally remove almost all non-axis-aligned trips, and (2) scaling the target (`fare_amount`) during training and then inverse-transforming predictions (keeps the exact same model/loop/loss, but makes optimization numerically easier). I keep your architecture, max_iter/EPOCHS, solver, and feature engineering unchanged, and only adjust these two items plus ensure predictions are inverse-scaled and clipped the same way. The submission path, row alignment, and fill strategy remain identical.'
- What this solution (achieved 6.37302) has done: 'Your current RMSE (5.343) is above the target (4.209), so we should make a small, legitimate improvement without changing the model/feature set. The biggest score drag left is that `clean()` is applied to `testKaggle`, dropping rows and then filling missing predictions with the train mean; this degrades Kaggle RMSE, so we keep training cleaning strict but make test cleaning non-dropping (only fix invalid values) and predict for all test rows. Additionally, we remove the airport/statue “exact coordinate equals” filters (they almost never trigger due to float precision and can remove valid rows unpredictably) while keeping the core NYC bounding-box/outlier/passenger filters unchanged. Finally, we compute time features vectorized (same semantics) to avoid any row-order surprises and keep runtime comfortably under the limit.'
- What this solution (achieved 5.61763) has done: 'Your current RMSE (6.373) is still above the target (4.209), so we should make a small, legitimate improvement without changing the MLPRegressor approach, features, or training loop. The biggest remaining score drag is that the model is trained on a randomly sampled 80k from 55M rows, which is very noisy; increasing the sample size is the most direct “same logic, better signal” change and should move RMSE down toward the target while staying within the 600s budget. I also make the sampling deterministic-but-stratified over `fare_amount` quantiles (still just sampling) so the 120k rows better represent the overall distribution, improving generalization without altering the model. Everything else (cleaning rules, feature engineering, scaling, model hyperparameters, and submission writing) stays the same.'
- What this solution (achieved 6.18149) has done: 'To move RMSE down toward your target with minimal disruption, I make two score-relevant fixes that keep the same MLPRegressor, feature set, and training loop: (1) reduce distribution shift by fitting scalers on train+validation features (still no test leakage) so validation/test features land in-range more consistently, and (2) switch the feature scaler from MinMaxScaler to StandardScaler (same “scale then MLP” pipeline, but typically much more stable for MLP optimization on this task). I also ensure `add_time_features()` never drops rows for test/validation by imputing bad datetimes instead of filtering them out, preventing accidental train/valid mismatch and preserving full test coverage. Everything else (cleaning rules, engineered features, model hyperparameters, target scaling + inverse-transform, submission format/path) stays the same.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

np.random.seed(1)
random.seed(1)
os.environ["PYTHONHASHSEED"] = "1"


def _first_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


TRAIN_PATH = _first_existing_path(
    [
        "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
        "/kaggle/input/train.csv",
        "/kaggle/data/new-york-city-taxi-fare-prediction/train.csv",
        "/kaggle/data/train.csv",
    ]
)
TEST_PATH = _first_existing_path(
    [
        "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
        "/kaggle/input/test.csv",
        "/kaggle/data/new-york-city-taxi-fare-prediction/test.csv",
        "/kaggle/data/test.csv",
    ]
)

if TRAIN_PATH is None or TEST_PATH is None:
    raise FileNotFoundError(
        f"Could not locate train/test CSV. TRAIN_PATH={TRAIN_PATH}, TEST_PATH={TEST_PATH}"
    )

SUBMISSION_NAME = "/kaggle/working/submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001

DATASET_SIZE = 120000

print("Resolved TRAIN_PATH:", TRAIN_PATH)
print("Resolved TEST_PATH:", TEST_PATH)
print("Submission path:", SUBMISSION_NAME)
print("DATASET_SIZE:", DATASET_SIZE)



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

trainKaggle = pd.read_csv(
    TRAIN_PATH,
    dtype=datatypes,
    usecols=[
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)

if len(trainKaggle) > DATASET_SIZE:
    y = trainKaggle["fare_amount"].astype("float32")
    qbins = pd.qcut(y, q=20, duplicates="drop")
    trainKaggle = (
        trainKaggle.groupby(qbins, group_keys=False)
        .apply(
            lambda g: g.sample(
                n=max(1, int(round(DATASET_SIZE * (len(g) / len(trainKaggle))))),
                random_state=1,
            )
        )
        .reset_index(drop=True)
    )
    if len(trainKaggle) > DATASET_SIZE:
        trainKaggle = trainKaggle.sample(n=DATASET_SIZE, random_state=1).reset_index(
            drop=True
        )
    elif len(trainKaggle) < DATASET_SIZE:
        topup = DATASET_SIZE - len(trainKaggle)
        extra = (
            pd.read_csv(
                TRAIN_PATH,
                dtype=datatypes,
                usecols=[
                    "fare_amount",
                    "pickup_datetime",
                    "pickup_longitude",
                    "pickup_latitude",
                    "dropoff_longitude",
                    "dropoff_latitude",
                    "passenger_count",
                ],
            )
            .sample(n=topup, random_state=1)
            .reset_index(drop=True)
        )
        trainKaggle = pd.concat([trainKaggle, extra], axis=0, ignore_index=True)

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

testKaggle_raw = testKaggle.copy()



## === cell 2
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)



## === cell 3
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## === cell 4
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("validation_df Size %d" % len(validation_df))
print("test_df Size %d" % len(test_df))



## === cell 5
train_df.describe()



## === cell 6
validation_df.describe()



## === cell 7
test_df.describe()




## === cell 8
def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    possible_paths = [
        "/kaggle/input/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/input/new-york-city-taxi-fare-prediction/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "../input/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "../input/new-york-city-taxi-fare-prediction/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "./nyc_mask-74.5_-72.8_40.5_41.8.png",
    ]
    mask_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if mask_path is None:
        return df

    nyc_mask = plt.imread(mask_path)[:, :, 0] > 0.9

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


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")

    if dt.notna().any():
        fill_dt = dt[dt.notna()].median()
    else:
        fill_dt = pd.Timestamp("2012-01-01 00:00:00")

    dt = dt.fillna(fill_dt)

    df = df.copy()
    df["year"] = dt.dt.year.astype("int16")
    df["month"] = dt.dt.month.astype("int8")
    df["day"] = dt.dt.day.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["weekday"] = dt.dt.weekday.astype("int8")
    df["pickup_datetime"] = df["pickup_datetime"].astype(str)

    h = df["hour"].astype("int16")
    wd = df["weekday"].astype("int16")

    df["late_night"] = ((h <= 3) | (h >= 22)).astype("int8")
    df["night"] = (((h >= 20) | (h <= 6)) & (wd < 5)).astype("int8")
    df["rush_hour"] = (((h >= 16) & (h <= 20)) & (wd < 5)).astype("int8")
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
    prediction = np.asarray(prediction).reshape(-1).astype("float32")
    prediction = np.clip(prediction, 0.0, None)

    df = pd.DataFrame(
        {
            id_column: raw_test[id_column].values,
            prediction_column: prediction,
        }
    )
    if not file_name.lower().endswith(".csv"):
        file_name = file_name + ".csv"
    df.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df))


def plot_loss_accuracy_rmse(history):
    return


def clean(df, is_test=False):
    print(" Old size: %d" % len(df))

    feature_cols = [
        c
        for c in [
            "pickup_datetime",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]
        if c in df.columns
    ]
    if is_test:
        df = df.copy()
        for c in feature_cols:
            if c != "pickup_datetime":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        num_cols = [c for c in feature_cols if c != "pickup_datetime"]
        for c in num_cols:
            med = float(np.nanmedian(df[c].values))
            if not np.isfinite(med):
                med = 0.0
            df[c] = df[c].fillna(med).astype("float32")
        df["passenger_count"] = df["passenger_count"].fillna(1).astype("uint8")
        print(" New size after test imputation (no drop): %d" % len(df))
    else:
        df = df.dropna(subset=feature_cols, how="any", axis="rows")
        print(" New size after dropna: %d" % len(df))

    for col in ["pickup_latitude", "dropoff_latitude"]:
        if is_test:
            df[col] = df[col].clip(-90.0, 90.0)
        else:
            df = df[(df[col] >= -90.0) & (df[col] <= 90.0)]
    for col in ["pickup_longitude", "dropoff_longitude"]:
        if is_test:
            df[col] = df[col].clip(-180.0, 180.0)
        else:
            df = df[(df[col] >= -180.0) & (df[col] <= 180.0)]
    print(" New size after lat/lon handling: %d" % len(df))

    if not is_test:
        df = df[
            (df["dropoff_longitude"] != df["pickup_longitude"])
            | (df["dropoff_latitude"] != df["pickup_latitude"])
        ]
        print(" New size after removing same long+lat: %d" % len(df))

        df = df[
            (df["dropoff_longitude"] != 0)
            & (df["pickup_longitude"] != 0)
            & (df["dropoff_latitude"] != 0)
            & (df["pickup_latitude"] != 0)
        ]
        print(" New size after removing 0 long lat: %d" % len(df))

    MinMax = (-74.5, -72.8, 40.5, 41.8)
    if is_test:
        df["pickup_longitude"] = df["pickup_longitude"].clip(MinMax[0], MinMax[1])
        df["dropoff_longitude"] = df["dropoff_longitude"].clip(MinMax[0], MinMax[1])
        df["pickup_latitude"] = df["pickup_latitude"].clip(MinMax[2], MinMax[3])
        df["dropoff_latitude"] = df["dropoff_latitude"].clip(MinMax[2], MinMax[3])
    else:
        df = df[
            (MinMax[0] <= df["pickup_longitude"])
            & (df["pickup_longitude"] <= MinMax[1])
        ]
        df = df[
            (MinMax[0] <= df["dropoff_longitude"])
            & (df["dropoff_longitude"] <= MinMax[1])
        ]
        df = df[
            (MinMax[2] <= df["pickup_latitude"]) & (df["pickup_latitude"] <= MinMax[3])
        ]
        df = df[
            (MinMax[2] <= df["dropoff_latitude"])
            & (df["dropoff_latitude"] <= MinMax[3])
        ]
    print(" New size after only NYC: %d" % len(df))

    if "fare_amount" in df.columns:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))

    if is_test:
        df["passenger_count"] = df["passenger_count"].clip(1, 6).astype("uint8")
    else:
        df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" New size after passenger_count handling: %d" % len(df))

    if not is_test:
        print("Old size: %d" % len(df))
        df = remove_datapoints_from_water(df)
        print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df




## === cell 9
print("train_df clean")
train_df = clean(train_df, is_test=False)
print("validation_df clean")
validation_df = clean(validation_df, is_test=False)

print("test_df clean")
test_df = clean(test_df, is_test=False)

print("testKaggle clean (no dropping keys)")
testKaggle = clean(testKaggle, is_test=True)



## === cell 10
train_df.describe()



## === cell 11
validation_df.describe()



## === cell 12
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 13
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
validation_df = add_coordinate_features(validation_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 14
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)
print("Done with Adding features")



## === cell 15
train_df.describe()



## === cell 16
validation_df.describe()



## === cell 17
dropped_columns = ["passenger_count", "pickup_datetime"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)

testKaggle_features = testKaggle.drop(dropped_columns + ["key"], axis=1)

feature_cols = list(train_df.drop(["fare_amount"], axis=1).columns)
validation_df = validation_df[["fare_amount"] + feature_cols]
test_df = test_df[["fare_amount"] + feature_cols]
testKaggle_features = testKaggle_features[feature_cols]

print("Done with dropped_columns + enforced feature alignment")



## === cell 18
train_labels = train_df["fare_amount"].values.astype("float32")
validation_labels = validation_df["fare_amount"].values.astype("float32")
test_labels = test_df["fare_amount"].values.astype("float32")

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 19
train_df.shape



## === cell 20
test_df.shape



## === cell 21
validation_df.shape



## === cell 22
X_scale_fit = pd.concat([train_df, validation_df], axis=0, ignore_index=False)

scaler = preprocessing.StandardScaler()
train_df_scaled = scaler.fit_transform(X_scale_fit.loc[train_df.index])
validation_df_scaled = scaler.transform(X_scale_fit.loc[validation_df.index])
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_features)

y_scaler = preprocessing.MinMaxScaler()
train_labels_scaled = (
    y_scaler.fit_transform(train_labels.reshape(-1, 1)).reshape(-1).astype("float32")
)
validation_labels_scaled = (
    y_scaler.transform(validation_labels.reshape(-1, 1)).reshape(-1).astype("float32")
)
test_labels_scaled = (
    y_scaler.transform(test_labels.reshape(-1, 1)).reshape(-1).astype("float32")
)



## === cell 23
hidden_layer_sizes = (256, 128, 64, 32, 8)

model = MLPRegressor(
    hidden_layer_sizes=hidden_layer_sizes,
    activation="relu",
    solver="adam",
    alpha=0.0,  # keep as-is
    batch_size="auto",
    learning_rate_init=LEARNING_RATE,
    max_iter=EPOCHS,
    shuffle=False,  # preserve original shuffle=False
    random_state=1,
    early_stopping=False,
    tol=0.0,
    n_iter_no_change=EPOCHS + 1,
    verbose=True,
)

print("Dataset size: %s" % DATASET_SIZE)
print("Epochs (mapped to max_iter): %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size (not used by sklearn Adam in the same way): %s" % BATCH_SIZE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % list(train_df.columns))
print("Training MLPRegressor hidden layers:", hidden_layer_sizes)

model.fit(train_df_scaled, train_labels_scaled)

val_pred_scaled = model.predict(validation_df_scaled).astype("float32").reshape(-1, 1)
val_pred = y_scaler.inverse_transform(val_pred_scaled).reshape(-1).astype("float32")
val_pred = np.clip(val_pred, 0.0, None)
val_rmse = float(np.sqrt(mean_squared_error(validation_labels, val_pred)))
print("Validation RMSE:", val_rmse)



## === cell 24
plot_loss_accuracy_rmse(None)



## === cell 25
prediction_scaled = model.predict(test_scaled).astype("float32").reshape(-1, 1)
predictionKaggle_scaled = (
    model.predict(testKaggle_scaled).astype("float32").reshape(-1, 1)
)

prediction = y_scaler.inverse_transform(prediction_scaled).reshape(-1).astype("float32")
predictionKaggle = (
    y_scaler.inverse_transform(predictionKaggle_scaled).reshape(-1).astype("float32")
)

prediction = np.clip(prediction, 0.0, None)
predictionKaggle = np.clip(predictionKaggle, 0.0, None)



## === cell 26
mse_check = float(
    mean_squared_error(
        test_labels[:1000].astype("float32").reshape(-1),
        np.asarray(prediction[:1000]).astype("float32").reshape(-1),
    )
)
print("MSE check (first 1000):", mse_check)



## === cell 27
submission = pd.DataFrame(
    {"key": testKaggle_raw["key"].values, "fare_amount": predictionKaggle.reshape(-1)}
)
submission["fare_amount"] = submission["fare_amount"].astype("float32")
submission["fare_amount"] = np.clip(submission["fare_amount"].values, 0.0, None)

submission.to_csv(SUBMISSION_NAME, index=False)
print("Output complete:", SUBMISSION_NAME, "rows:", len(submission))



## === cell 28
print("Example prediction:", float(np.asarray(prediction).reshape(-1)[0]))
print("Example true label:", float(test_labels[0]))
print("Wrote submission file:", SUBMISSION_NAME)
print(pd.read_csv(SUBMISSION_NAME).head())
print("Submission rows:", len(pd.read_csv(SUBMISSION_NAME)))
