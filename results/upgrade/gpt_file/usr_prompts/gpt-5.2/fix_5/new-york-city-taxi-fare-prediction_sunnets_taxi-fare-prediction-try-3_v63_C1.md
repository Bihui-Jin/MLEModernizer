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

5.00327

# 6. Current score

11.84496

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 103.33706) has done: 'I fix the runtime blockers without changing the model/feature logic: (1) make Keras imports compatible with the installed `tf_keras`/Keras 3 stack and fix the optimizer constructor, (2) remove the external URL dependency in the water-mask cleaning step (no internet in Kaggle) while keeping the rest of your cleaning intact, and (3) correct pathing to `/kaggle/input/...` so files load. I also make the datetime parsing robust to the actual NYC taxi format, and ensure the script always writes a valid `key,fare_amount` submission CSV. These fixes should run end-to-end and typically improve RMSE substantially versus the previous failing/partially-running pipeline (current 15.28) because training complete and the heavy cleaning no longer crashes.'
- What this solution (achieved 763.57764) has done: 'I fix the immediate runtime crash in the first cell caused by an incompatible protobuf version that breaks `tf_keras` import (`MessageFactory.GetPrototype`). To keep core modeling logic intact while restoring execution, I switch the Keras stack to `tensorflow.keras` (available in Kaggle’s TensorFlow runtime) and keep the same Sequential architecture, optimizer, loss, and training loop. I also correct the `late_night`, `night`, and `rush_hour` time-feature logic (currently always/never true due to wrong inequalities), which is a minimal semantic bug fix that should significantly reduce the RMSE from the very poor current score. Finally, I ensure file paths fall back to the provided `/kaggle/input/new-york-city-taxi-fare-prediction/...` location if needed and the submission CSV is always written with `key,fare_amount`.'
- What this solution (achieved 25.73788) has done: 'I fix the immediate runtime blocker in the first import cell by removing the TensorFlow dependency that’s triggering the protobuf `MessageFactory.GetPrototype` crash, and switch the exact same Keras `Sequential` model/training loop to scikit-learn’s `MLPRegressor` (still a dense neural network regressor with MSE objective) so the notebook runs end-to-end in this environment. I also keep your existing feature engineering/cleaning logic intact, but make the train file read include the `fare_amount` column (it’s currently accidentally excluded, which can silently break labels and destroy RMSE). Finally, I ensure predictions are finite and clipped to a reasonable positive range before writing `key,fare_amount` to a `.csv` submission file, which is score-stable and prevents invalid outputs.'
- What this solution (achieved 11.84496) has done: 'Your current RMSE (25.74) is far worse than the target (5.00), so we need a modest, legitimate accuracy boost without changing the overall pipeline structure. The biggest score drag here is an implicit bug: the train CSV is read without the `key`, but later the cleaning/feature logic expects consistent handling, and more importantly your model is forced to learn raw lat/long without any scale-stable geodesic feature—your current “distance” is in degrees and poorly correlated with fare. I keep the same sklearn MLPRegressor approach and training loop, but add a single robust Haversine distance feature (in km) alongside your existing distance/manhattan features, and I also stop converting `pickup_datetime` back to string (keep derived columns and drop the datetime column later as you already do). These are minimal, metric-aligned feature fixes that typically move NYC Taxi Fare baselines from ~20–30 RMSE into the single digits without changing the model family.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor

TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

SUBMISSION_NAME = "submissiontry_water.csv"  # keep original name; ensure .csv suffix

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

np.random.seed(1)




## === cell 1
def remove_datapoints_from_water(df):
    """
    Kaggle notebooks do not allow external network access. The original version
    attempted to download a NYC land/water mask from a URL, which crashes.
    Minimal, score-stable fix: skip the mask filtering and return df unchanged.
    """
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
    return 1 if ((row["hour"] >= 20) and (row["weekday"] < 5)) else 0


def rush_hour(row):
    return (
        1
        if ((row["hour"] >= 16) and (row["hour"] <= 20) and (row["weekday"] < 5))
        else 0
    )


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.deg2rad(lat1.astype(np.float64))
    lon1 = np.deg2rad(lon1.astype(np.float64))
    lat2 = np.deg2rad(lat2.astype(np.float64))
    lon2 = np.deg2rad(lon2.astype(np.float64))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    R = 6371.0
    return (R * c).astype(np.float32)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["pickup_datetime"] = dt

    df["year"] = df["pickup_datetime"].dt.year.astype("Int64")
    df["month"] = df["pickup_datetime"].dt.month.astype("Int64")
    df["day"] = df["pickup_datetime"].dt.day.astype("Int64")
    df["hour"] = df["pickup_datetime"].dt.hour.astype("Int64")
    df["weekday"] = df["pickup_datetime"].dt.weekday.astype("Int64")

    df["night"] = df.apply(lambda x: night(x), axis=1)
    df["late_night"] = df.apply(lambda x: late_night(x), axis=1)
    df["rush_hour"] = df.apply(lambda x: rush_hour(x), axis=1)
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
    ).astype(np.float32)

    df["haversine_km"] = haversine_km(lat1, lon1, lat2, lon2)
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    pred = np.asarray(prediction).reshape(-1)
    df = pd.DataFrame(
        {
            id_column: raw_test[id_column].values,
            prediction_column: pred.astype(np.float32),
        }
    )
    df.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df))


def plot_loss_accuracy_rmse(history):
    print("Skipping plots: sklearn MLPRegressor does not provide keras History object.")




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
trainKaggle = pd.read_csv(
    TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes, usecols=train_usecols
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




## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000]




## === cell 4
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))




## === cell 5
train_df.describe()




## === cell 6
test_df.describe()




## === cell 7
print("train_df clean")
train_df = clean(train_df)
test_df = clean(test_df)




## === cell 8
train_df.describe()




## === cell 9
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)




## === cell 10
train_df.describe()




## === cell 11
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features Disabled!")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)




## === cell 12
train_df.describe()




## === cell 13
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")




## === cell 14
train_df.describe()




## === cell 15
print("Plots skipped.")




## === cell 16
dropped_columns = ["passenger_count", "pickup_datetime"]

if "key" in train_df.columns:
    dropped_columns_train = dropped_columns + ["key"]
else:
    dropped_columns_train = dropped_columns

if "key" in test_df.columns:
    dropped_columns_test = dropped_columns + ["key"]
else:
    dropped_columns_test = dropped_columns

train_df = train_df.drop(dropped_columns_train, axis=1)
test_df = test_df.drop(dropped_columns_test, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")




## === cell 17
train_df.shape




## === cell 18
train_df.describe()




## === cell 19
test_df.describe()




## === cell 20
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)




## === cell 21
train_df.describe()




## === cell 22
validation_df.describe()




## === cell 23
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")




## === cell 24
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)




## === cell 25
model = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32, 8),
    activation="relu",
    solver="adam",
    alpha=0.01,  # keep identical regularization setting
    learning_rate_init=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    max_iter=EPOCHS,
    shuffle=True,
    random_state=1,
    early_stopping=False,
    verbose=True,
)

print("Dataset size: %s" % DATASET_SIZE)
print("Epochs(max_iter): %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size: %s" % BATCH_SIZE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % list(train_df.columns))

model.fit(train_df_scaled, train_labels)




## === cell 26
print("Model visualization skipped (tensorflow/keras not used).")




## === cell 27
plot_loss_accuracy_rmse(None)




## === cell 28
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


train_pred = model.predict(train_df_scaled)
val_pred = model.predict(validation_df_scaled)
test_pred = model.predict(test_scaled)

print("Train RMSE:", rmse(train_labels, train_pred))
print("Val RMSE:", rmse(validation_labels, val_pred))
print("Test RMSE:", rmse(test_labels, test_pred))




## === cell 29
validation_predictions = val_pred.flatten()

plt.scatter(validation_labels, validation_predictions, s=5)
plt.xlabel("True Values [$]")
plt.ylabel("Predictions [$]")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])




## === cell 30
test_predictions = test_pred.flatten()

plt.scatter(test_labels, test_predictions, s=5)
plt.xlabel("True Values [$]")
plt.ylabel("Predictions [$]")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])




## === cell 31
predictionKaggle = model.predict(testKaggle_scaled)

predictionKaggle = np.asarray(predictionKaggle, dtype=np.float32)
predictionKaggle = np.nan_to_num(predictionKaggle, nan=11.35, posinf=50.0, neginf=0.0)
predictionKaggle = np.clip(predictionKaggle, 0.0, 200.0)




## === cell 32
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)

print("Saved to:", os.path.abspath(SUBMISSION_NAME))
print(pd.read_csv(SUBMISSION_NAME).head())
