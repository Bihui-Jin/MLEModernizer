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

4.25412

# 6. Current score

13.22861

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 90.83131) has done: 'I fix the runtime errors caused by (1) incompatible `keras` imports in this Kaggle environment, (2) attempting to load an external URL image mask for “water removal” (no internet / `plt.imread` URL limitation), and (3) some Keras API changes (optimizer naming, removed `vis_utils`). I keep your model architecture and training loop intact, but correct the optimizer construction and metrics so `compile()/fit()/predict()` run end-to-end. To move RMSE toward your target (lower is better) with minimal semantic change, I fix a bug in your “late_night/night” logic (it currently returns 1 almost always) and avoid a wrong datetime parsing format that produces NaTs. Finally, I ensure a valid `submissiontry_water.csv` with `key,fare_amount` is written.'
- What this solution (achieved 129.74256) has done: 'I fix the two blockers that prevent an end-to-end run and a valid submission: (1) the `MessageFactory/GetPrototype` crash triggered by importing `tf_keras` (protobuf incompatibility), by switching to `tensorflow.keras` which is available in Kaggle and keeps the same model/training semantics; and (2) the scaling failure caused by keeping `pickup_datetime` as a string after feature extraction, by dropping it (and the `key`) before scaling so only numeric features reach `MinMaxScaler`. I keep your architecture, optimizer, loss, epochs, batch size, and feature engineering intact, but make datetime feature creation robust and ensure all train/val/test frames have identical numeric columns. Finally, I ensure the submission is written as `submissiontry_water.csv` with exactly `key,fare_amount`.'
- What this solution (achieved 126.31673) has done: 'I first fix the runtime crash (`MessageFactory.GetPrototype`) by avoiding TensorFlow/Keras import combinations that trigger protobuf incompatibility in this environment, while keeping the same Keras model and training loop. Then I fix a major logic bug that hurts score: your `late_night()` feature is currently almost always 1 because `row["hour"] <= 3` is always true due to Python chained comparison behavior in your original code; I make it a correct hour-range check. Finally, I keep the feature set identical but make time-feature generation vectorized (no `apply`) to avoid slowdowns and ensure consistent dtypes, and I keep the submission writing exactly `key,fare_amount` to `submissiontry_water.csv`.'
- What this solution (achieved 221.63157) has done: 'I fix the environment crash by removing the incompatible `tf_keras` import and switching to `tensorflow.keras`, which keeps the same model/training semantics but avoids the protobuf `MessageFactory.GetPrototype` error. I also fix the major logic issue that makes your “test” evaluation meaningless (you currently set `test_df = validation_df.copy()`), replacing it with the actual Kaggle test set for inference while keeping validation for monitoring. To move RMSE substantially toward your target with minimal core-logic change, I also remove an accidental feature drop (`passenger_count`) that is a real predictive input and should be kept (this is a small, legitimate feature-set correction rather than a modeling rewrite). Finally, I keep the submission format exactly `key,fare_amount` and ensure the `.csv` is written.'
- What this solution (achieved 19.61735) has done: 'I fix the TensorFlow/Keras import crash (`MessageFactory` protobuf issue) by avoiding TensorFlow entirely and switching to scikit-learn’s `MLPRegressor`, while keeping your overall pipeline semantics the same: same cleaned dataset slice, same engineered features, same MinMax scaling, and same train/validation split. This change is necessary because your current code cannot run end-to-end in the provided environment, so no valid submission (and thus no competitive score) can be reliably produced. To move RMSE strongly toward the target (lower is better) with minimal conceptual change, I train an MLP regressor (still a feed-forward neural net) and keep the same feature set and preprocessing. Finally, I ensure a correct `submissiontry_water.csv` is written with exactly `key,fare_amount` and 9914 rows aligned to the test keys.'
- What this solution (achieved 5.34789) has done: 'Your current gap to the target is large (19.62 vs 4.25 RMSE; lower is better), so we need a meaningful but still minimal change that keeps your pipeline semantics intact. The biggest lever without changing the model type is to make the cleaning consistent across train/validation/test: right now you clean only train/validation, but not the test set, which can create out-of-distribution inputs and unstable predictions that hurt RMSE. I apply the same `clean()` to `testKaggle` (without touching labels) and then re-align the submission to include every test `key` by predicting on the cleaned subset and filling removed rows with a safe fallback (train mean fare), ensuring the submission remains exactly 9914 rows. This should improve generalization and reduce extreme prediction errors while preserving your feature engineering, scaling, model, and training loop.'
- What this solution (achieved 5.37141) has done: 'To reduce RMSE toward your 4.25412 target without changing the model/training loop, I make two small, high-impact fixes in preprocessing that commonly dominate this competition’s score: compute a more realistic geographic distance (haversine in km) instead of raw-degree Euclidean, and ensure datetime-derived features are consistent by parsing the UTC “Z” timestamps correctly. These changes keep your same feature-engineering structure (still “add distances/time features”), the same scaler, and the same MLPRegressor configuration, but provide the model with better-behaved numeric inputs that typically improves generalization. I also clamp negative fare predictions to 0.0 at inference time (a legitimate post-process consistent with the target domain) to avoid large-error outliers that inflate RMSE.'
- What this solution (achieved 5.5647) has done: 'Your current RMSE (5.37) is worse than the target (4.254), so we should make a small, legitimate accuracy improvement without changing the model type or training loop. The biggest low-risk gain here is to stabilize the target distribution by training on `log1p(fare_amount)` and inverting with `expm1` at inference; this is a standard trick for heavy-tailed regression that typically reduces RMSE outliers while keeping identical evaluation semantics (still predicting fare in dollars). To avoid hurting score via train/test mismatch, we keep your exact feature set and cleaning, but compute the fallback fill value in the same target space (log-space median) and then invert, so cleaned-out test rows get a more robust default than the raw mean. All changes are localized around label preparation and prediction post-processing; data paths, features, scaler, and MLPRegressor configuration remain the same, and the script still writes a valid `submissiontry_water.csv`.'
- What this solution (achieved 5.60278) has done: 'We need to reduce RMSE from 5.5647 toward 4.25412 (lower is better), so we make small, low-risk improvements that keep your model/training loop and overall feature set intact. The biggest likely gain with minimal semantic change is to add two standard, cheap geographic features (pickup/dropoff distance to NYC center) and a simple interaction (distance × passenger_count), which helps an MLP capture fare structure without changing the approach. We also make the train/test cleaning more consistent by not dropping test rows (keep all test rows and only mask invalid coordinates to avoid distribution shift + fallback overuse), while keeping the same cleaning rules for training labels/outliers. All changes are localized to feature engineering and test preprocessing; the MLPRegressor, scaling, log1p target trick, and submission writing remain unchanged.'
- What this solution (achieved 13.22861) has done: 'I fix the runtime error in `haversine_km` by allowing scalar latitude/longitude inputs (NYC center) without calling `.astype()` on Python floats. This is a minimal, local bug fix that unblocks feature creation and lets the pipeline run end-to-end. I keep the model, training loop, preprocessing, and submission-writing logic unchanged to preserve evaluation semantics and maintain the current score-improvement intent. The result again write a valid `submissiontry_water.csv` with `key,fare_amount` and the correct row alignment.'
- What this solution (achieved 13.22861) has done: 'Your current RMSE (13.23) is far worse than the target (4.254), so we should make a small, high-impact fix rather than tuning the model. The biggest issue is that your log1p/expm1 pipeline can create extreme outliers when the MLP predicts large values in log-space, and those few outliers dominate RMSE on this competition. I add a minimal, domain-valid post-processing cap on predicted fares (e.g., 0–250) to suppress catastrophic errors while preserving your model, features, training loop, and submission format. This should move your public RMSE substantially downward without changing the core approach.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 1000
EPOCHS = 30
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

np.random.seed(1)

print(
    "Using scikit-learn MLPRegressor (TensorFlow disabled due to protobuf MessageFactory crash in this environment)."
)



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

train_usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

trainKaggle = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    dtype=datatypes,
    usecols=train_usecols,
)
testKaggle = pd.read_csv(
    TEST_PATH,
    dtype={k: v for k, v in datatypes.items() if k != "fare_amount"},
    usecols=test_usecols,
)



## === cell 2
train_df, validation_df = train_test_split(trainKaggle, test_size=0.10, random_state=1)

test_df = None



## === cell 3
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("validation_df Size %d" % len(validation_df))



## === cell 4
train_df.describe()



## === cell 5
validation_df.describe()




## === cell 6
def remove_datapoints_from_water(df):
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

    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)

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


def clean_test_keep_rows(df):
    df = df.copy()
    df = df.dropna(how="any", axis="rows")

    MinMax = (-74.5, -72.8, 40.5, 41.8)
    bad = (df["dropoff_longitude"] == df["pickup_longitude"]) & (
        df["dropoff_latitude"] == df["pickup_latitude"]
    )
    bad |= (
        (df["dropoff_longitude"] == 0)
        | (df["pickup_longitude"] == 0)
        | (df["dropoff_latitude"] == 0)
        | (df["pickup_latitude"] == 0)
    )
    bad |= ~(
        (MinMax[0] <= df["pickup_longitude"])
        & (df["pickup_longitude"] <= MinMax[1])
        & (MinMax[0] <= df["dropoff_longitude"])
        & (df["dropoff_longitude"] <= MinMax[1])
        & (MinMax[2] <= df["pickup_latitude"])
        & (df["pickup_latitude"] <= MinMax[3])
        & (MinMax[2] <= df["dropoff_latitude"])
        & (df["dropoff_latitude"] <= MinMax[3])
    )
    bad |= ~((df["passenger_count"] > 0) & (df["passenger_count"] <= 6))

    coord_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    df.loc[bad, coord_cols] = np.nan
    return df


def late_night_hour(h):
    return ((h >= 0) & (h <= 3)).astype("int8")


def night_hour_weekday(h, wd):
    return (((h >= 20) | (h <= 5)) & (wd < 5)).astype("int8")


def rush_hour_hour_weekday(h, wd):
    return ((h >= 16) & (h <= 20) & (wd < 5)).astype("int8")


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype="float64"))
    lon1 = np.radians(np.asarray(lon1, dtype="float64"))
    lat2 = np.radians(np.asarray(lat2, dtype="float64"))
    lon2 = np.radians(np.asarray(lon2, dtype="float64"))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return (6371.0 * c).astype("float32")


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True).dt.tz_convert(
        None
    )

    df["year"] = dt.dt.year.fillna(0).astype("int16")
    df["month"] = dt.dt.month.fillna(0).astype("int8")
    df["day"] = dt.dt.day.fillna(0).astype("int8")
    df["hour"] = dt.dt.hour.fillna(0).astype("int8")
    df["weekday"] = dt.dt.weekday.fillna(0).astype("int8")

    df["pickup_datetime"] = df["pickup_datetime"].astype(str)

    h = df["hour"].astype("int16")
    wd = df["weekday"].astype("int16")
    df["night"] = night_hour_weekday(h, wd)
    df["late_night"] = late_night_hour(h)
    df["rush_hour"] = rush_hour_hour_weekday(h, wd)
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
    df["distance"] = haversine_km(lat1, lon1, lat2, lon2)
    return df


def add_geo_context_features(df):
    nyc_lat, nyc_lon = 40.7141667, -74.0063889
    df["pickup_nyc_km"] = haversine_km(
        df["pickup_latitude"], df["pickup_longitude"], nyc_lat, nyc_lon
    )
    df["dropoff_nyc_km"] = haversine_km(
        df["dropoff_latitude"], df["dropoff_longitude"], nyc_lat, nyc_lon
    )
    df["dist_x_passengers"] = df["distance"] * df["passenger_count"].astype("float32")
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    pred = np.asarray(prediction).reshape(-1)
    df = pd.DataFrame({id_column: raw_test[id_column].values, prediction_column: pred})
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df))


def plot_loss_accuracy_rmse(history):
    pass




## === cell 7
print("train_df clean")
train_df = clean(train_df)
print("validation_df clean")
validation_df = clean(validation_df)

print("testKaggle clean (keep rows)")
testKaggle_raw = testKaggle.copy()
testKaggle = clean_test_keep_rows(testKaggle)



## === cell 8
train_df.describe()



## === cell 9
validation_df.describe()



## === cell 10
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 11
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
validation_df = add_coordinate_features(validation_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 12
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("train_df add_geo_context_features")
train_df = add_geo_context_features(train_df)
print("validation_df add_geo_context_features")
validation_df = add_geo_context_features(validation_df)
print("testKaggle add_geo_context_features")
testKaggle = add_geo_context_features(testKaggle)

print("Done with Adding features")



## === cell 13
train_df.describe()



## === cell 14
validation_df.describe()



## === cell 15
dropped_columns = ["pickup_datetime", "key"]

train_df = train_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns, axis=1)

print("Done with dropped_columns")



## === cell 16
train_labels = np.log1p(train_df["fare_amount"].values.astype("float32"))
validation_labels = np.log1p(validation_df["fare_amount"].values.astype("float32"))

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)

print("Done with Labels (log1p transformed)")



## === cell 17
train_df.shape



## === cell 18
validation_df.shape



## === cell 19
assert (
    list(train_df.columns)
    == list(validation_df.columns)
    == list(testKaggle_clean.columns)
)

train_medians = train_df.median(numeric_only=True)
train_df = train_df.fillna(train_medians)
validation_df = validation_df.fillna(train_medians)
testKaggle_clean = testKaggle_clean.fillna(train_medians)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df.astype("float32"))
validation_df_scaled = scaler.transform(validation_df.astype("float32"))
testKaggle_scaled = scaler.transform(testKaggle_clean.astype("float32"))



## === cell 20
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32, 8),
    activation="relu",
    solver="adam",
    alpha=0.0001,  # L2 regularization (closest analogue to keep training stable)
    batch_size=BATCH_SIZE,
    learning_rate_init=LEARNING_RATE,
    max_iter=EPOCHS,
    shuffle=True,
    random_state=1,
    early_stopping=False,  # Do NOT introduce early stopping per constraints
    n_iter_no_change=EPOCHS + 1,
    verbose=True,
)

print("Dataset size: %s" % DATASET_SIZE)
print("Epochs(max_iter): %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size: %s" % BATCH_SIZE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % list(train_df.columns))

mlp.fit(train_df_scaled, train_labels)



## === cell 21
print("Model visualization skipped (TensorFlow/Keras not used).")



## === cell 22
print("Training finished. (No Keras history to plot.)")



## === cell 23
val_pred_log = mlp.predict(validation_df_scaled).astype("float32")
val_pred = np.expm1(val_pred_log)
val_true = np.expm1(validation_labels.astype("float32"))

val_rmse = float(np.sqrt(mean_squared_error(val_true, val_pred)))
val_mae = float(np.mean(np.abs(val_true - val_pred)))
print([val_rmse**2, val_mae, val_rmse])
print("Validation loss (MSE):", val_rmse**2)
print("Validation mae:", val_mae)
print("Validation rmse:", val_rmse)



## === cell 24
predictionKaggle_log = mlp.predict(testKaggle_scaled).astype("float32")
predictionKaggle = np.expm1(predictionKaggle_log).astype("float32")

predictionKaggle = np.clip(predictionKaggle, 0.0, 250.0)



## === cell 25
output_submission(
    testKaggle_raw, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME
)



## === cell 26
print(
    "Submission exists:",
    os.path.exists(SUBMISSION_NAME),
    "size:",
    os.path.getsize(SUBMISSION_NAME) if os.path.exists(SUBMISSION_NAME) else None,
)
sub = pd.read_csv(SUBMISSION_NAME)
print(sub.head())
print("Submission rows:", len(sub), "unique keys:", sub["key"].nunique())
assert list(sub.columns) == ["key", "fare_amount"]
assert len(sub) == len(testKaggle_raw)
assert sub["key"].iloc[0] == testKaggle_raw["key"].iloc[0]
