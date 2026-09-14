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

No external packages required in the script and installed.

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

4.051660609378363

# 6. Current score

8.0535

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 158.01017) has done: 'I fix the two blockers that prevent end-to-end execution: (1) the Keras/TensorFlow import crash (protobuf incompatibility) by switching to `tensorflow.keras`, and (2) the missing custom Mehrak CSV paths by reading the provided Kaggle `train.csv`/`test.csv` instead. I also remove the online NYC water mask dependency (no internet) by making `remove_datapoints_from_water` a safe no-op, keeping the rest of your cleaning logic intact. Finally, I ensure the pipeline always produces a valid `key,fare_amount` submission CSV aligned to the Kaggle test set, using the trained model’s predictions.'
- What this solution (achieved 186.66872) has done: 'I fix the TensorFlow/protobuf crash causing the `MessageFactory.GetPrototype` error by avoiding TensorFlow entirely (it’s not usable in this Kaggle runtime as-is) and replacing the training backend with scikit-learn’s `MLPRegressor`, keeping the same core idea: a multi-layer perceptron trained on the same engineered and scaled features with RMSE-aligned loss. I also correct a major scaling logic bug where the scaler was incorrectly refit per-column (destroying feature scaling consistency), by fitting one scaler on all feature columns and transforming train/valid/test consistently. Finally, I keep the same cleaning/feature logic and ensure a valid `key,fare_amount` submission CSV is always written with the expected filename suffix.'
- What this solution (achieved 7.81825) has done: 'Your current RMSE (186) indicates the model is producing extreme/invalid fares; the smallest safe fix is to make preprocessing consistent between train/validation/test and prevent feature/label misalignment after cleaning. I (1) clean the full sampled training set before splitting so the split doesn’t create distribution/row-drop artifacts, (2) ensure we never fit/transform with NaNs/Infs and keep the exact same feature columns in the same order everywhere, and (3) clip predictions to a reasonable competition range (0–50) to avoid a few huge predictions dominating RMSE. These changes keep your core feature engineering and MLPRegressor approach intact, but should drastically reduce the catastrophic-error tail and move RMSE down toward the ~4 target band.'
- What this solution (achieved 7.94078) has done: 'Your current RMSE (7.818) is still far above the ~4.05 target, so we should make a small, safe improvement that reduces large systematic errors without changing your model/feature set. The biggest issue is that `MLPRegressor` is sensitive to target scale and long-tailed fares; you already clip outputs, but training is still dominated by outliers up to 50. I keep the same architecture and training loop, but add a standard, competition-safe target transform: train the MLP on `log1p(fare_amount)` and invert with `expm1` at prediction time (then keep your 0–50 clip). This typically reduces RMSE materially on this competition by stabilizing learning while preserving evaluation semantics.'
- What this solution (achieved 8.0535) has done: 'We’re still far above the target RMSE (7.94 vs 4.05), so we need a small but meaningful improvement without changing your feature set or model architecture. The biggest low-risk gain here is to make the MLP optimization better-behaved by standardizing inputs (MLPs typically perform much better with zero-mean/unit-variance features than MinMax scaling), while keeping the same features and training procedure. I switch the scaler from `MinMaxScaler` to `StandardScaler` (fit on train only, transform val/test/Kaggle consistently), keep your `log1p` target transform and prediction clipping, and keep all paths/output format unchanged. This should reduce systematic under/over-shoot and bring RMSE down toward the target band without any semantic changes to evaluation.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


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
    print(" New size after NYC lang lot: %d" % len(df))

    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["pickup_latitude"] != 0)]
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

    df = df[(df["passenger_count"] > 0)]
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


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=True
    )
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["minute"] = df["pickup_datetime"].dt.minute
    df["second"] = df["pickup_datetime"].dt.second
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
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(
        {id_column: raw_test[id_column].values, prediction_column: prediction}
    )
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df))


def plot_loss_accuracy_rmse(history):
    pass




## === cell 1
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

TRAIN_PATH = "../input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "../input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 1000
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

np.random.seed(1)




## === cell 2
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

train_df_full = pd.read_csv(
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
    dtype={k: v for k, v in datatypes.items() if k != "fare_amount"},
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

print("train_df_full clean (before split)")
train_df_full = clean(train_df_full)

train_df, test_df = train_test_split(train_df_full, test_size=0.10, random_state=1)




## === cell 3
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))




## === cell 4
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)

train_df = train_df.dropna(subset=["pickup_datetime"])
test_df = test_df.dropna(subset=["pickup_datetime"])
testKaggle = testKaggle.dropna(subset=["pickup_datetime"])




## === cell 5
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)

print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")




## === cell 6
dropped_columns = ["pickup_datetime"]
train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")




## === cell 7
feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "passenger_count",
    "latdiff",
    "londiff",
    "manhattan",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
]


def _sanitize_numeric(df, cols):
    df = df.copy()
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=cols)


train_df = _sanitize_numeric(
    train_df,
    feature_cols + (["fare_amount"] if "fare_amount" in train_df.columns else []),
)
test_df = _sanitize_numeric(
    test_df,
    feature_cols + (["fare_amount"] if "fare_amount" in test_df.columns else []),
)
testKaggle_clean = _sanitize_numeric(testKaggle_clean, feature_cols)

train_df_scaled = train_df.copy()
test_df_scaled = test_df.copy()
testKaggle_scaled = testKaggle_clean.copy()

scaler = preprocessing.StandardScaler()
scaler.fit(train_df_scaled[feature_cols])

train_df_scaled[feature_cols] = scaler.transform(train_df_scaled[feature_cols])
test_df_scaled[feature_cols] = scaler.transform(test_df_scaled[feature_cols])
testKaggle_scaled[feature_cols] = scaler.transform(testKaggle_scaled[feature_cols])




## === cell 8
train_df_scaled, validation_df_scaled = train_test_split(
    train_df_scaled, test_size=0.10, random_state=1
)




## === cell 9
print(train_df_scaled.shape)
print(validation_df_scaled.shape)
print(test_df_scaled.shape)
print(testKaggle_scaled.shape)




## === cell 10
train_labels = np.log1p(train_df_scaled["fare_amount"].values)
validation_labels = np.log1p(validation_df_scaled["fare_amount"].values)
test_labels = np.log1p(test_df_scaled["fare_amount"].values)

X_train = train_df_scaled[feature_cols]
X_val = validation_df_scaled[feature_cols]
X_test = test_df_scaled[feature_cols]

print("Done with Labels")




## === cell 11
model = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32, 8),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=BATCH_SIZE,
    learning_rate_init=LEARNING_RATE,
    max_iter=EPOCHS,
    shuffle=True,
    random_state=1,
    verbose=True,
)

model.fit(X_train.values, train_labels)

train_pred_log = model.predict(X_train.values)
val_pred_log = model.predict(X_val.values)
test_pred_log = model.predict(X_test.values)

train_pred = np.expm1(train_pred_log)
val_pred = np.expm1(val_pred_log)
test_pred = np.expm1(test_pred_log)

train_pred = np.clip(train_pred, 0.0, 50.0)
val_pred = np.clip(val_pred, 0.0, 50.0)
test_pred = np.clip(test_pred, 0.0, 50.0)

train_rmse = mean_squared_error(np.expm1(train_labels), train_pred, squared=False)
val_rmse = mean_squared_error(np.expm1(validation_labels), val_pred, squared=False)
test_rmse = mean_squared_error(np.expm1(test_labels), test_pred, squared=False)

print({"rmse": train_rmse, "split": "train"})
print({"rmse": val_rmse, "split": "validation"})
print({"rmse": test_rmse, "split": "test"})




## === cell 12
pass




## === cell 13
pass




## === cell 14
predictionKaggle_log = model.predict(testKaggle_scaled[feature_cols].values).reshape(-1)
predictionKaggle = np.expm1(predictionKaggle_log)

predictionKaggle = np.clip(predictionKaggle, 0.0, 50.0)

output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)

sub_check = pd.read_csv(SUBMISSION_NAME)
print(sub_check.head())
print("Submission shape:", sub_check.shape)
print("Submission columns:", list(sub_check.columns))
