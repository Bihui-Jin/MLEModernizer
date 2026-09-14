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

4.40796

# 6. Current score

12.98804

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.27203) has done: 'I make the code compatible with the current Kaggle environment by (1) fixing the Keras 3 optimizer API usage and removing/guarding unsupported visualization imports, (2) replacing the URL-based NYC water mask with a local-only fallback so cleaning no longer crashes without internet, and (3) fixing datetime parsing and a couple of time-feature logic bugs that currently generate wrong indicators and hurt RMSE. I also ensure the train/test CSV paths match your provided `/kaggle/input/...` structure and that the submission file is written correctly as `key,fare_amount`. These changes keep the same overall model/feature approach but should materially improve score from the current 15.3 toward your ~4.4 target by preventing broken cleaning and correcting feature engineering.'
- What this solution (achieved 151.79366) has done: 'I fix the immediate runtime error caused by Keras 3’s backend API change by implementing the RMSE metric using `tf_keras.backend` (which still provides `sqrt/mean/square`) so `model.fit()` runs. I also fix the input file paths to match your actual `/kaggle/input/new-york-city-taxi-fare-prediction/...` location so data loading works in the Kaggle environment you described. Finally, I make the plotting cell conditional on `history` existing, so the notebook won’t crash if training fails for any other reason, and I ensure the submission CSV is written with the required `key,fare_amount` columns.'
- What this solution (achieved 351.92591) has done: 'I fix the immediate runtime crash in the first import cell by avoiding the standalone `keras` package (which is triggering a protobuf `MessageFactory.GetPrototype` incompatibility in this environment) and consistently using `tf_keras` instead. This keeps the same Sequential/Dense/BatchNorm architecture, training loop, and loss, but makes imports/optimizer/loss calls stable under Python 3.7 on Kaggle. I also remove the inappropriate `"accuracy"` metric (it is for classification and can distort training/metrics logging for regression), keeping RMSE/MAE which align with the competition RMSE objective; this is a minimal change that should move score substantially toward the 4.41 target from 151.79. Finally, I ensure the submission is always written as a valid `key,fare_amount` CSV to the specified filename.'
- What this solution (achieved 6.43427) has done: 'I fix the immediate crash caused by the protobuf/Keras incompatibility by removing all `tf_keras` imports and switching to `tensorflow.keras` (which is stable on Kaggle for this competition) while keeping the exact same model architecture, loss, optimizer type, and training loop. I also correct the input data loading so the `key` column is preserved for the test set (needed for submission) and ensure the submission writer aligns predictions 1:1 with `test.csv` rows. Finally, I add a small safety clamp on predictions (non-negative fares) to prevent obviously invalid outputs that can explode RMSE, which should move the score substantially toward your 4.41 target without changing the core approach.'
- What this solution (achieved 8.93681) has done: 'I fix the TensorFlow/Keras import crash by avoiding the standalone Keras 3 stack entirely and using the `tf_keras` package that is installed in your environment, keeping the same model, optimizer type, loss, and training loop. I also fix the `mean_squared_error` API call in the debug MSE cell (it was removed in newer Keras) by computing MSE with NumPy/TensorFlow ops without changing training semantics. To nudge RMSE down toward your 4.41 target without changing the core approach, I apply the same `clean()` filtering to the held-out `test_df` before feature engineering so evaluation isn’t polluted by obviously invalid rows that the model never sees during training. The script still write a valid `key,fare_amount` submission CSV with the same filename.'
- What this solution (achieved 403.09104) has done: 'I fix the immediate runtime crash in the first cell caused by the protobuf/Keras incompatibility by removing the `tf_keras` stack and using `tensorflow.keras`, which is stable in this Kaggle environment. To move RMSE down toward your 4.41 target with minimal semantic change, I apply the same `clean()` step to the actual Kaggle test set (`testKaggle`) and then ensure the submission still contains all original test rows by predicting on the full test set and only using the cleaned-test predictions where available, filling the remaining rows with the median prediction (score-improving vs leaving them misaligned). I also make datetime feature extraction robust to parsing failures by dropping rows with unparseable datetimes during cleaning (train/val/internal test only) while keeping the original test keys intact. The model architecture, loss, optimizer type, and training loop remain unchanged.'
- What this solution (achieved 59.196) has done: 'I fix the immediate crash in the first cell caused by an incompatibility between `tensorflow.keras` and the protobuf runtime in this Kaggle image by switching the model imports to the already-installed `tf_keras` package (keeping the same Sequential/Dense/BatchNorm architecture, optimizer type, loss, and training loop). I also harden datetime feature extraction so it never produces NaNs/overflows for the integer time features (which can otherwise explode RMSE), while preserving the same engineered features. Finally, I keep the submission alignment logic but ensure the filled prediction array is always fully dense (no NaNs) and written as a valid `key,fare_amount` CSV.'
- What this solution (achieved 64.92911) has done: 'I fix the immediate crash (`MessageFactory.GetPrototype`) by removing the incompatible TensorFlow/tf_keras imports and switching to scikit-learn’s MLPRegressor, which keeps the same “dense MLP on engineered numeric features with MSE objective” core logic but runs reliably in this Kaggle/Python 3.7 image. I also make the train/test CSV loading consistent by ensuring `fare_amount` is included in the training columns (it was missing, which would otherwise break or poison training). Finally, I keep your existing cleaning, feature engineering, scaling, and submission alignment logic intact so the pipeline runs end-to-end and writes a valid `key,fare_amount` CSV.'
- What this solution (achieved 26.27189) has done: 'Your current RMSE is extremely high because the feature pipeline is dropping `passenger_count` and also uses `df.apply(...)` for time flags, which is both slow and error-prone; together this can badly degrade the model and produce poorly-calibrated predictions. I keep the same overall approach (clean → time/coordinate/distance features → MinMax scaling → dense MLPRegressor with Adam) but make two minimal, score-relevant fixes: (1) keep `passenger_count` as a feature, and (2) vectorize the `night/late_night/rush_hour` flags to match your intended logic without row-wise apply. I also ensure training uses `shuffle=True` (MLPRegressor default and closer to Keras’ behavior) because `shuffle=False` can harm convergence; this is a small training-semantics adjustment that should reduce RMSE substantially toward your 4.41 target without changing the model family. The script still runs end-to-end and writes `submissiontry_water.csv` with `key,fare_amount`.'
- What this solution (achieved 24.94181) has done: 'Your RMSE is far from the 4.41 target, so the biggest “minimal” win is to stop throwing away training signal and stop training on an arbitrary half-split: we train on the full sampled training set (keeping the same model/feature pipeline) and only hold out a small validation slice for monitoring. We also fix a subtle bug in your “airport removal” filters: they currently remove almost nothing because they use `&` (both lat and lon must match exactly) instead of excluding rows where both match an airport coordinate; correcting that improves data cleaning without changing feature semantics. Finally, we ensure the datetime parsing is done once per dataframe and reused consistently, and keep the submission alignment logic unchanged so `key,fare_amount` matches `test.csv` 1:1.'
- What this solution (achieved 12.98804) has done: 'Your RMSE is far from the target, so we should make a couple of minimal, high-impact fixes that keep your exact model family and feature set but remove obvious training/feature bugs. First, your cleaning currently drops almost all “normal” trips because it uses `&` when checking same lat/long; we change that to drop only rows where both lat and lon are identical (a true zero-distance trip). Second, your “airport” filters almost never trigger because they check exact float equality; we make them remove only points extremely close to those coordinates using a tiny epsilon, preserving the intended semantics without rewriting the approach. Finally, we apply the same basic non-target cleaning (dropna, bounds, passenger_count, zero-distance) consistently to the Kaggle test features while keeping 1:1 alignment with `test.csv` keys so the submission stays valid.'

# 9. Code solution

## === cell 0
import os
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
DATASET_SIZE = 80000

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

trainKaggle = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
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



## === cell 2
train_df, validation_df = train_test_split(trainKaggle, test_size=0.10, random_state=1)
train_df, test_df = train_test_split(train_df, test_size=0.10, random_state=1)



## === cell 3
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("validation_df Size %d" % len(validation_df))
print("test_df Size %d" % len(test_df))



## === cell 4
train_df.describe()



## === cell 5
validation_df.describe()



## === cell 6
test_df.describe()




## === cell 7
def remove_datapoints_from_water(df):
    return df


def _parse_pickup_datetime(df):
    return pd.to_datetime(
        df["pickup_datetime"], errors="coerce", infer_datetime_format=True, utc=False
    )


def _near_coord(df, lat_col, lon_col, coord_lat, coord_lon, eps=1e-4):
    return (np.abs(df[lat_col] - coord_lat) <= eps) & (
        np.abs(df[lon_col] - coord_lon) <= eps
    )


def clean(df):
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    if "pickup_datetime" in df.columns:
        dt = _parse_pickup_datetime(df)
        before = len(df)
        df = df.loc[~dt.isna()].copy()
        print(
            " New size after removing bad datetimes: %d (dropped %d)"
            % (len(df), before - len(df))
        )

    df = df[
        ~(
            (df["dropoff_longitude"] == df["pickup_longitude"])
            & (df["dropoff_latitude"] == df["pickup_latitude"])
        )
    ]
    print(" New size after removing same long lat (true zero-distance): %d" % len(df))

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
        ~_near_coord(
            df, "pickup_latitude", "pickup_longitude", nyc_coord[0], nyc_coord[1]
        )
    ]
    df = df[
        ~_near_coord(
            df, "dropoff_latitude", "dropoff_longitude", nyc_coord[0], nyc_coord[1]
        )
    ]
    print(" New size after NY coord removed: %d" % len(df))

    df = df[
        ~_near_coord(
            df, "pickup_latitude", "pickup_longitude", fk_coord[0], fk_coord[1]
        )
    ]
    df = df[
        ~_near_coord(
            df, "dropoff_latitude", "dropoff_longitude", fk_coord[0], fk_coord[1]
        )
    ]
    print(" New size after jfk coord removed: %d" % len(df))

    df = df[
        ~_near_coord(
            df, "pickup_latitude", "pickup_longitude", ewr_coord[0], ewr_coord[1]
        )
    ]
    df = df[
        ~_near_coord(
            df, "dropoff_latitude", "dropoff_longitude", ewr_coord[0], ewr_coord[1]
        )
    ]
    print(" New size after ewr coord removed: %d" % len(df))

    df = df[
        ~_near_coord(
            df, "pickup_latitude", "pickup_longitude", lga_coord[0], lga_coord[1]
        )
    ]
    df = df[
        ~_near_coord(
            df, "dropoff_latitude", "dropoff_longitude", lga_coord[0], lga_coord[1]
        )
    ]
    print(" New size after lga coord removed: %d" % len(df))

    df = df[
        ~_near_coord(
            df, "pickup_latitude", "pickup_longitude", sol_coord[0], sol_coord[1]
        )
    ]
    df = df[
        ~_near_coord(
            df, "dropoff_latitude", "dropoff_longitude", sol_coord[0], sol_coord[1]
        )
    ]
    print(" New size after sol coord removed: %d" % len(df))

    print("Old size: %d" % len(df))
    df = remove_datapoints_from_water(df)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = _parse_pickup_datetime(df)
    if dt.isna().any():
        df = df.loc[~dt.isna()].copy()
        dt = dt.loc[~dt.isna()]

    df["year"] = dt.dt.year.astype("int16")
    df["month"] = dt.dt.month.astype("int8")
    df["day"] = dt.dt.day.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["weekday"] = dt.dt.weekday.astype("int8")

    df["pickup_datetime"] = dt.astype(str)

    hour = df["hour"].astype(np.int16)
    weekday = df["weekday"].astype(np.int16)
    df["night"] = ((hour >= 20) & (weekday < 5)).astype("int8")
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
    df = pd.DataFrame(
        {
            id_column: raw_test[id_column].values,
            prediction_column: prediction.reshape(-1),
        }
    )
    df.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows=", len(df))


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 6))
    plt.plot(history.history.get("loss", []))
    plt.plot(history.history.get("val_loss", []))
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()

    if "rmse" in history.history:
        plt.figure(figsize=(20, 6))
        plt.plot(history.history.get("rmse", []))
        plt.plot(history.history.get("val_rmse", []))
        plt.title("Model rmse")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper right")
        plt.show()




## === cell 8
print("train_df clean")
train_df = clean(train_df)
print("validation_df clean")
validation_df = clean(validation_df)

print("test_df clean")
test_df = clean(test_df)

print("testKaggle clean (for consistent feature engineering)")
testKaggle_cleaned = clean(testKaggle.drop(columns=["key"]).copy())
testKaggle_cleaned_with_key = testKaggle.loc[testKaggle_cleaned.index].copy()



## === cell 9
train_df.describe()



## === cell 10
validation_df.describe()



## === cell 11
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)

print("testKaggle add_time_features (full)")
testKaggle = add_time_features(testKaggle)
print("testKaggle_cleaned add_time_features (cleaned subset)")
testKaggle_cleaned_with_key = add_time_features(testKaggle_cleaned_with_key)



## === cell 12
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
validation_df = add_coordinate_features(validation_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)

print("testKaggle add_coordinate_features (full)")
testKaggle = add_coordinate_features(testKaggle)
print("testKaggle_cleaned add_coordinate_features (cleaned subset)")
testKaggle_cleaned_with_key = add_coordinate_features(testKaggle_cleaned_with_key)



## === cell 13
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)

print("testKaggle add_distances_features (full)")
testKaggle = add_distances_features(testKaggle)
print("testKaggle_cleaned add_distances_features (cleaned subset)")
testKaggle_cleaned_with_key = add_distances_features(testKaggle_cleaned_with_key)

print("Done with Adding features")



## === cell 14
train_df.describe()



## === cell 15
validation_df.describe()



## === cell 16
dropped_columns = ["pickup_datetime"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)

testKaggle_full_features = testKaggle.drop(dropped_columns + ["key"], axis=1)
testKaggle_clean_features = testKaggle_cleaned_with_key.drop(
    dropped_columns + ["key"], axis=1
)

print("Done with dropped_columns")



## === cell 17
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

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
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)

testKaggle_full_scaled = scaler.transform(testKaggle_full_features)
testKaggle_clean_scaled = scaler.transform(testKaggle_clean_features)




## === cell 22
def rmse_np(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))




## === cell 23
model = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32, 8),
    activation="relu",
    solver="adam",
    alpha=0.0,
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

model.fit(train_df_scaled, train_labels)

val_pred = model.predict(validation_df_scaled)
print("Validation RMSE:", rmse_np(validation_labels, val_pred))



## === cell 24
print("Skipping plot: sklearn MLPRegressor does not provide a Keras History object.")



## === cell 25
prediction = model.predict(test_scaled).reshape(-1, 1)

predictionKaggle_full = model.predict(testKaggle_full_scaled).reshape(-1, 1)
predictionKaggle_clean = model.predict(testKaggle_clean_scaled).reshape(-1, 1)

prediction = np.maximum(prediction, 0.0)
predictionKaggle_full = np.maximum(predictionKaggle_full, 0.0)
predictionKaggle_clean = np.maximum(predictionKaggle_clean, 0.0)

predictionKaggle_filled = predictionKaggle_full.copy()
if len(testKaggle_cleaned_with_key) > 0:
    predictionKaggle_filled[testKaggle_cleaned_with_key.index.values] = (
        predictionKaggle_clean
    )

fill_value = (
    float(np.median(predictionKaggle_clean))
    if len(predictionKaggle_clean) > 0
    else float(np.median(predictionKaggle_full))
)
pred_flat = predictionKaggle_filled.reshape(-1)
nan_mask = np.isnan(pred_flat)
if nan_mask.any():
    pred_flat[nan_mask] = fill_value
predictionKaggle_filled = pred_flat.reshape(-1, 1)



## === cell 26
y_true = test_labels[:1000].reshape(-1, 1).astype("float32")
y_pred = prediction[:1000].astype("float32")
mse_sample = np.mean((y_pred - y_true) ** 2)
print("Sample MSE (first 1000):", float(mse_sample))



## === cell 27
output_submission(
    testKaggle, predictionKaggle_filled, "key", "fare_amount", SUBMISSION_NAME
)



## === cell 28
idx = min(10000, len(prediction) - 1)
print(prediction[idx])
print(test_labels[idx])
