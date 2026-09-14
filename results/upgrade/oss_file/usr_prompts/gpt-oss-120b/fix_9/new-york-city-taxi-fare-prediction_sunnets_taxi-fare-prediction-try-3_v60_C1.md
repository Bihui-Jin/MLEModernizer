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

4.30351

# 6. Current score

10.12988

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.27447) has done: 'The changes fix the image‑loading error in `remove_datapoints_from_water`, update the optimizer to the current Keras API, correct variable names for the predictions, and add small robustness tweaks (e.g., safe imports, increasing the sampled training size for a slightly better model). These fixes allow the notebook to run end‑to‑end and produce a valid `.csv` submission while keeping the original modelling approach.'
- What this solution (achieved 98.75948) has done: 'The update fixes the Keras import errors by switching to `tensorflow.keras`, restores the missing backend functions (e.g., `sqrt`), adds a modest EarlyStopping callback, and expands the training sample size and epochs so the model can reach a lower RMSE while keeping the original architecture unchanged. The script now runs end‑to‑end and writes a proper `.csv` submission.'
- What this solution (achieved 15.44038) has done: 'Implemented fixes to unblock execution and improve model performance:
- Switched all TensorFlow‑Keras imports to the standalone `keras` package to avoid protobuf errors.
- Corrected the `late_night` logic (now true only for hours ≤ 3).
- Improved datetime parsing in `add_time_features` and dropped rows with any resulting NaNs.
- Updated the backend import for the custom RMSE metric.
- Minor clean‑up comments kept unchanged.'
- What this solution (achieved 297.48078) has done: 'The fix switches all Keras imports to `tensorflow.keras`, which restores the missing backend functions (e.g., `K.sqrt`) and resolves the protobuf import errors. Only the import lines and the backend import are changed, preserving the original model architecture and training logic.'
- What this solution (achieved 14.15588) has done: 'Implemented missing preprocessing, feature‑engineering, and submission utilities, added NaN handling, and ensured the GradientBoostingRegressor receives clean data. The new functions (`clean`, `add_time_features`, `add_coordinate_features`, `add_distances_features`, `output_submission`) safely process timestamps, compute hour/weekday, coordinate differences, and haversine distance, then drop any rows with missing values. Predictions are now written to a proper `submissiontry_water.csv` with the required columns, allowing the script to run end‑to‑end and produce a valid submission while keeping the original modeling approach unchanged.'
- What this solution (achieved 10.12988) has done: 'The changes reduce the data sampled for training (from 500 k to 200 k rows) and lower the number of trees in the GradientBoostingRegressor (from 800 to 200). Both adjustments keep the same preprocessing, feature engineering, and model type, only trimming size‑related hyperparameters to fit comfortably within the 600 s limit while preserving prediction semantics.'

# 9. Code solution

## === cell 0
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import math

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256  # retained for possible future use
EPOCHS = 100  # retained for possible future use
LEARNING_RATE = 0.001  # retained for possible future use
DATASET_SIZE = 200000  # smaller sample for faster execution


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with clearly invalid values."""
    df = df.dropna()
    lon_min, lon_max = -74.5, -73.0
    lat_min, lat_max = 40.0, 41.0
    cond = (
        (df["pickup_longitude"] >= lon_min)
        & (df["pickup_longitude"] <= lon_max)
        & (df["dropoff_longitude"] >= lon_min)
        & (df["dropoff_longitude"] <= lon_max)
        & (df["pickup_latitude"] >= lat_min)
        & (df["pickup_latitude"] <= lat_max)
        & (df["dropoff_latitude"] >= lat_min)
        & (df["dropoff_latitude"] <= lat_max)
    )
    return df.loc[cond].reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse pickup_datetime and add hour & weekday."""
    df = df.copy()
    df["pickup_datetime_parsed"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce"
    )
    df = df.dropna(subset=["pickup_datetime_parsed"])
    df["hour"] = df["pickup_datetime_parsed"].dt.hour.astype("uint8")
    df["weekday"] = df["pickup_datetime_parsed"].dt.weekday.astype("uint8")
    df["month"] = df["pickup_datetime_parsed"].dt.month.astype("uint8")
    df = df.drop(columns=["pickup_datetime_parsed"])
    return df


def add_coordinate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple coordinate deltas."""
    df = df.copy()
    df["lon_diff"] = (df["dropoff_longitude"] - df["pickup_longitude"]).astype(
        "float32"
    )
    df["lat_diff"] = (df["dropoff_latitude"] - df["pickup_latitude"]).astype("float32")
    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in kilometers."""
    R = 6371.0  # Earth radius in km
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def add_distances_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add haversine distance as a feature."""
    df = df.copy()
    df["haversine_km"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    ).astype("float32")
    return df


def output_submission(
    original_test_df: pd.DataFrame,
    predictions: np.ndarray,
    key_col: str,
    target_col: str,
    filename: str,
):
    """Write a Kaggle‑compatible submission file."""
    sub = pd.DataFrame(
        {
            key_col: original_test_df[key_col].values,
            target_col: predictions.ravel(),
        }
    )
    sub.to_csv(filename, index=False)
    print(f"Submission written to {filename}")




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
)



## === cell 2
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000]  # keep validation size modest



## === cell 3
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))



## === cell 4
train_df = clean(train_df)
test_df = clean(test_df)



## === cell 5
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 6
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 7
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)
print("Done with Adding features")



## === cell 8
train_df = train_df.dropna().reset_index(drop=True)
test_df = test_df.dropna().reset_index(drop=True)
testKaggle = testKaggle.dropna().reset_index(drop=True)

dropped_columns = ["pickup_datetime"]
train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")



## === cell 9
train_labels = train_df["fare_amount"].values
validation_labels = test_df["fare_amount"].values  # validation set
train_df = train_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Labels prepared")



## === cell 10
train_df_scaled = train_df.values
validation_df_scaled = test_df.values
testKaggle_scaled = testKaggle_clean.values




## === cell 11
def rmse_metric(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_pred - y_true) ** 2))




## === cell 12
gbr = GradientBoostingRegressor(
    n_estimators=200,  # fewer trees to meet time limit
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    random_state=42,
)
gbr.fit(train_df_scaled, train_labels)

print("Model training complete")
print(f"Dataset size: {DATASET_SIZE}")
print(f"Features used: {train_df.columns}")



## === cell 13
train_pred = gbr.predict(train_df_scaled)
val_pred = gbr.predict(validation_df_scaled)
train_score = [
    rmse_metric(train_labels, train_pred),  # RMSE
    np.mean(np.abs(train_labels - train_pred)),  # MAE
]
val_score = [
    rmse_metric(validation_labels, val_pred),
    np.mean(np.abs(validation_labels - val_pred)),
]
print("Train scores (RMSE, MAE):", train_score)
print("Validation scores (RMSE, MAE):", val_score)



## === cell 14
predictionKaggle = gbr.predict(testKaggle_scaled).reshape(-1, 1)



## === cell 15
print("Prediction max:", predictionKaggle.max())
print("Prediction min:", predictionKaggle.min())



## === cell 16
fig, ax = plt.subplots()
ax.scatter(validation_labels, val_pred, alpha=0.5, s=10)
ax.plot(
    [validation_labels.min(), validation_labels.max()],
    [validation_labels.min(), validation_labels.max()],
    "k--",
    lw=2,
)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.show()



## === cell 17
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
