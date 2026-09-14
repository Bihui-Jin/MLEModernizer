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
protobuf==6.33.0
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

26.87094

# 6. Current score

21.81223

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 305.10296) has done: 'I fixed the logical‑OR errors in the cleaning routine, added a safe protobuf setting before importing TensorFlow to avoid the protobuf incompatibility, and reorganised the notebook cells so the data loading, preprocessing, model building, training, and submission steps run sequentially without missing variables. These changes let the script execute end‑to‑end and produce a proper **submission.csv** while preserving the original model architecture and training logic.'
- What this solution (achieved 21.81223) has done: 'The update speeds up the pipeline by loading a much smaller chunk of the massive training file (200 k rows instead of ~2 M) and by configuring the RandomForest to train each tree on a random 30 % subsample of the data (using `max_samples=0.3`). This keeps the same model type and hyper‑parameters (200 trees, no depth limit, same random state) while drastically cutting the amount of work each tree does, allowing the whole script to finish well under the 600 s limit without altering any feature engineering or evaluation logic.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import gc
import subprocess

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"



## === cell 1
p = subprocess.Popen(
    ["wc", "-l", TRAIN_PATH], stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
result, err = p.communicate()
if p.returncode != 0:
    raise IOError(err)
n_rows = int(result.strip().split()[0]) + 1




## === cell 2
def compute_haversine_distance(
    df,
    lat1="pickup_latitude",
    long1="pickup_longitude",
    lat2="dropoff_latitude",
    long2="dropoff_longitude",
):
    R = 3959  # radius of earth in miles
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])

    delta_phi = np.radians(df[lat2] - df[lat1])
    delta_lambda = np.radians(df[long2] - df[long1])

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    df["distance"] = d.astype("float32")




## === cell 3
def add_date_features(df, test=False):
    df["pickup_datetime_clone"] = df["pickup_datetime"].values
    df.pickup_datetime_clone = df.pickup_datetime_clone.str.slice(0, 16)
    df.pickup_datetime_clone = pd.to_datetime(
        df.pickup_datetime_clone, utc=True, format="%Y-%m-%d %H:%M"
    )
    df["year"] = df.pickup_datetime_clone.dt.year.astype("uint8")
    df["month"] = df.pickup_datetime_clone.dt.month.astype("uint8")
    df["day"] = df.pickup_datetime_clone.dt.day.astype("uint8")
    df["dayofweek"] = df.pickup_datetime_clone.dt.dayofweek.astype("uint8")
    df["hour"] = df.pickup_datetime_clone.dt.hour.astype("uint8")
    df["minute"] = df.pickup_datetime_clone.dt.minute.astype("uint8")
    df.drop(columns=["pickup_datetime_clone"], inplace=True)




## === cell 4
MIN_FARE = 2.50
MAX_FARE = 500

MIN_PASSENGER = 1
MAX_PASSENGER = 6


def clean_data(df, test=False):
    compute_haversine_distance(df)
    add_date_features(df, test)

    if not test:
        df.dropna(inplace=True)

        fare_mask = (df.fare_amount > MAX_FARE) | (df.fare_amount < MIN_FARE)
        df.drop(df[fare_mask].index, inplace=True)

        passenger_mask = (df.passenger_count > MAX_PASSENGER) | (
            df.passenger_count < MIN_PASSENGER
        )
        df.drop(df[passenger_mask].index, inplace=True)

        lat_mask = (df.pickup_latitude > 90) | (df.pickup_latitude < -90)
        df.drop(df[lat_mask].index, inplace=True)

        lon_mask = (df.pickup_longitude > 180) | (df.pickup_longitude < -180)
        df.drop(df[lon_mask].index, inplace=True)

        drop_lat_mask = (df.dropoff_latitude > 90) | (df.dropoff_latitude < -90)
        df.drop(df[drop_lat_mask].index, inplace=True)

        drop_lon_mask = (df.dropoff_longitude > 180) | (df.dropoff_longitude < -180)
        df.drop(df[drop_lon_mask].index, inplace=True)

        dist_mask = (df.distance > 100) | (df.distance <= 0)
        df.drop(df[dist_mask].index, inplace=True)

    if not test:
        df.drop(columns=["pickup_datetime"], inplace=True)




## === cell 5
traintypes = {
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
cols = list(traintypes.keys())
chunksize = 200_000  # smaller chunk for speed
total_chunk = n_rows // chunksize + 1
df_list = []
i = 0

for df_chunk in pd.read_csv(
    TRAIN_PATH, usecols=cols, dtype=traintypes, chunksize=chunksize
):
    i += 1
    print(f"DataFrame Chunk {i:02d}/{total_chunk}")
    clean_data(df_chunk)
    df_list.append(df_chunk)
    del df_chunk
    break  # keep a single, smaller chunk for speed
print("Complete")



## === cell 6
X = pd.concat(df_list, ignore_index=True)
del df_list
gc.collect()
print("Loaded training data shape:", X.shape)



## === cell 7
minmax = {}  # store (min, max) for each column
norm = pd.DataFrame()

float32cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "distance",
]
float16cols = [
    "passenger_count",
    "year",
    "month",
    "day",
    "dayofweek",
    "hour",
    "minute",
]

for col in float32cols:
    col_min = X[col].min()
    col_max = X[col].max()
    minmax[col] = (col_min, col_max)
    norm[col] = ((X[col] - col_min) / (col_max - col_min)).astype("float32")

for col in float16cols:
    col_min = X[col].min()
    col_max = X[col].max()
    minmax[col] = (col_min, col_max)
    norm[col] = ((X[col] - col_min) / (col_max - col_min)).astype("float16")

norm["fare_amount"] = X["fare_amount"]
X = norm
del norm
gc.collect()
print("After scaling:")
X.head()
X.info()



## === cell 8
X = X.sample(frac=1, random_state=42).reset_index(drop=True)



## === cell 9
y = X["fare_amount"]
X.drop(columns="fare_amount", inplace=True)
print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)



## === cell 10
validation_portion = 2.5 / 100
index = int(X.shape[0] * validation_portion)
print("training:\t%d\nvalidation:\t%d" % (X.shape[0] - index, index))



## === cell 11
val_X = X.iloc[:index].copy()
X.drop(X.index[:index], inplace=True)

val_y = y.iloc[:index].copy()
y.drop(y.index[:index], inplace=True)

print("Train shape:", X.shape, "Val shape:", val_X.shape)



## === cell 12
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    min_samples_leaf=1,
    max_samples=0.3,  # speed up training while preserving algorithmic intent
)



## === cell 13
model.fit(X.values, y.values)

val_pred = model.predict(val_X.values)
val_rmse = math.sqrt(mean_squared_error(val_y.values, val_pred))
val_mae = mean_absolute_error(val_y.values, val_pred)
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation MAE : {val_mae:.4f}")



## === cell 14
import matplotlib.pyplot as plt

errors = val_pred - val_y.values
plt.figure(figsize=(8, 4))
plt.hist(errors, bins=30, edgecolor="k")
plt.title("Validation Prediction Error Distribution")
plt.xlabel("Error (predicted - actual)")
plt.ylabel("Count")
plt.show()



## === cell 15
print("Sample predictions vs actual:")
for i in range(5):
    print(f"actual: {val_y.values[i]:.2f} \t pred: {val_pred[i]:.2f}")



## === cell 16
traintypes_test = {
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
cols_test = list(traintypes_test.keys())
cols_test.append("key")

X_test = pd.read_csv(TEST_PATH, usecols=cols_test, dtype=traintypes_test)
clean_data(X_test, test=True)
X_test_key = X_test["key"].copy()
X_test.drop(columns=["pickup_datetime", "key"], inplace=True)

for col in float16cols:
    col_min, col_max = minmax[col]
    X_test[col] = ((X_test[col] - col_min) / (col_max - col_min)).astype("float16")

for col in float32cols:
    col_min, col_max = minmax[col]
    X_test[col] = ((X_test[col] - col_min) / (col_max - col_min)).astype("float32")



## === cell 17
pred = model.predict(X_test.values).flatten()
pred = np.round(pred, 2)



## === cell 18
results = pd.DataFrame({"key": X_test_key.astype(str), "fare_amount": pred})
results.info()
results.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
