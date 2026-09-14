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

3.12

# 3. Installed packages

geopandas==0.14.4
geopy==2.4.1
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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0
xgboost==2.0.3

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

3.61245

# 6. Current score

5.53493

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5.53493) has done: 'I fix the TensorFlow/Keras crash caused by an incompatible `protobuf==6.x` (it triggers the `MessageFactory.GetPrototype` error in TF 2.18) by removing the neural-net section from execution so the notebook runs end-to-end. Your extremely bad Kaggle score is consistent with a `key`/prediction misalignment or invalid predictions; I ensure predictions are finite, clipped to reasonable positive fares, and written using the `sample_submission.csv` key order to guarantee correct alignment. I also replace the very slow `geopy.great_circle` loop with a vectorized haversine distance (same feature intent: trip distance) to improve both runtime and RMSE without changing the overall approach (still distance/time + XGBoost). Finally, the script always write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

INPUT_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

print("Files in /kaggle/input/new-york-city-taxi-fare-prediction:")
for f in sorted(os.listdir(INPUT_DIR))[:20]:
    print(" -", f)



## === cell 1
train_df = pd.read_csv(TRAIN_PATH, nrows=1_000_000)
test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print(train_df.shape, test_df.shape, sample_sub.shape)
print(train_df.head(2))
print(test_df.head(2))



## === cell 2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor



## === cell 3
train_df = train_df.dropna()

train_df = train_df[train_df["passenger_count"] < 8]
train_df = train_df[train_df["fare_amount"] > 0]

print("After basic cleaning:", train_df.shape)
print(train_df.isnull().sum())



## === cell 4
RANGE = (-74.26, -72.99, 40.56, 41.71)


def select_within_boundingbox(df, rng):
    return (
        (df.pickup_longitude >= rng[0])
        & (df.pickup_longitude <= rng[1])
        & (df.pickup_latitude >= rng[2])
        & (df.pickup_latitude <= rng[3])
        & (df.dropoff_longitude >= rng[0])
        & (df.dropoff_longitude <= rng[1])
        & (df.dropoff_latitude >= rng[2])
        & (df.dropoff_latitude <= rng[3])
    )


old_size = len(train_df)
train_df = train_df[select_within_boundingbox(train_df, RANGE)]
print(f"Bounding box filter: {old_size} -> {len(train_df)}")




## === cell 5
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.7613  # Earth radius in miles
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def add_features(df):
    df = df.copy()
    df["distance"] = haversine_miles(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["hour"] = dt.dt.hour.astype("float32")
    df["day"] = dt.dt.day.astype("float32")
    df["month"] = dt.dt.month.astype("float32")
    df["year"] = dt.dt.year.astype("float32")
    return df


train_df = add_features(train_df)
test_df = add_features(test_df)

train_df = train_df.dropna(subset=["distance", "hour", "day", "month", "year"])
test_df = test_df.dropna(subset=["distance", "hour", "day", "month", "year"])

print("Feature-added shapes:", train_df.shape, test_df.shape)



## === cell 6
train_df = train_df[(train_df["distance"] < 25) & (train_df["distance"] > 0.1)]
print("After distance filter:", train_df.shape)



## === cell 7
features = [
    "passenger_count",
    "distance",
    "hour",
    "day",
    "month",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
X = train_df[features].astype("float32")
y = train_df["fare_amount"].astype("float32")

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 8
model = XGBRegressor(
    learning_rate=0.1,
    n_estimators=300,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

y_val_pred = model.predict(X_valid)
rmse = float(np.sqrt(mean_squared_error(y_valid, y_val_pred)))
print("Validation RMSE:", rmse)



## === cell 9
X_test = test_df[features].astype("float32")
y_test_pred = model.predict(X_test)

y_test_pred = np.asarray(y_test_pred, dtype="float64")
y_test_pred = np.where(np.isfinite(y_test_pred), y_test_pred, np.nan)
median_fare = float(np.nanmedian(y)) if np.isfinite(np.nanmedian(y)) else 11.35
y_test_pred = np.nan_to_num(
    y_test_pred, nan=median_fare, posinf=median_fare, neginf=median_fare
)
y_test_pred = np.clip(y_test_pred, 0.0, 500.0)

pred_df = pd.DataFrame({"key": test_df["key"].values, "fare_amount": y_test_pred})
submission_df = sample_sub[["key"]].merge(pred_df, on="key", how="left")

submission_df["fare_amount"] = submission_df["fare_amount"].fillna(median_fare)

submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)



## === cell 10
print(
    "NN section skipped due to TF/protobuf incompatibility; submission.csv generated via XGBoost."
)
