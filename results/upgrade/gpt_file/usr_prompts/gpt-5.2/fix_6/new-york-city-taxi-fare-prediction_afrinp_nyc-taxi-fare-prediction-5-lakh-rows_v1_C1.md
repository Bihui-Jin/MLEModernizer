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

3.8

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

3.45343

# 6. Current score

14.03156

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 567.82654) has done: 'I fix the runtime crash caused by importing `keras` in this Kaggle environment by switching to the installed `tf_keras` backend while keeping the exact same Sequential/Dense architecture and training loop. I also fix a subtle preprocessing bug: `scale(test)` was being fit on the test set instead of using the training scaling parameters, which harms RMSE; I replace it with a `StandardScaler` fit on training features and applied to both train and test (same semantics as scaling, but correct). Finally, I ensure the submission file has exactly the required columns (`key,fare_amount`) and that predictions are written as a 1D float array.'
- What this solution (achieved 679.75833) has done: 'I fix the crash in the `tf_keras` import that’s caused by an environment protobuf incompatibility by switching to the already-available `tensorflow` Keras backend (same Sequential/Dense model, same training loop). I also keep the corrected scaling (fit scaler on train, apply to both train and test) and ensure the test columns match the train feature columns exactly to avoid silent misalignment. Finally, I keep the submission format strictly as `key,fare_amount` and write `submission.csv` in the working directory so Kaggle accepts it.'
- What this solution (achieved 1363.71157) has done: 'I fix the runtime crash caused by importing `tensorflow` (protobuf incompatibility) by switching the model backend to the already-installed `tf_keras`, keeping the exact same Sequential/Dense architecture, compile settings, and training loop. I also add deterministic seeding and force CPU usage to avoid GPU/TF initialization paths that can trigger the same protobuf issue, without changing the modeling semantics. Finally, I keep the corrected scaling (fit on train, apply to test) and ensure the submission is written as `submission.csv` with exactly `key,fare_amount` and aligned row order.'
- What this solution (achieved 74.36088) has done: 'I fix the crash in the import cell by removing the dependency on `tf_keras` (which is triggering a protobuf `MessageFactory` error in this Kaggle image) and switching to scikit-learn’s `MLPRegressor`, keeping the same “dense MLP on scaled engineered numeric features trained with MSE” core approach. I keep all existing feature engineering and the train-fit scaler / test-transform scaler logic intact, and only change the modeling backend so the notebook runs end-to-end reliably. I also ensure predictions are finite and written to `submission.csv` with exactly `key,fare_amount` in the test row order. This should dramatically reduce RMSE compared to the currently-broken/degraded run and move the score toward the target.'
- What this solution (achieved 14.03156) has done: 'Your current RMSE (74) is far above the target (3.45), so we should improve performance with the smallest changes that don’t alter the overall “scaled engineered numeric features → dense MLP regressor trained with MSE” approach. The biggest issue is that your model is learning on raw latitude/longitude differences and an unscaled Euclidean distance in degrees, which is poorly aligned with true trip distance; adding a proper haversine distance feature (in km) is a minimal feature-engineering fix that usually yields a large RMSE drop on this competition. I also apply the exact same passenger_count cleanup to the test set (you currently only fix it in train), and I clip predictions to a more reasonable upper bound (100, consistent with your training label filter) to reduce extreme outliers that hurt RMSE. Everything else (data size, scaler-fit-on-train, MLPRegressor, training loop) stays the same.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # keep CPU-only for stability
random.seed(42)
np.random.seed(42)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    parse_dates=["pickup_datetime"],
    nrows=500000,
)
df.dropna(inplace=True)



## === cell 2
df



## === cell 3
test = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)
test



## === cell 4
df.describe()



## === cell 5
test.describe()



## === cell 6
nyc_min_longitude = -74.3
nyc_max_longitude = -72
nyc_min_latitude = 40.63
nyc_max_latitude = 42



## === cell 7
for long in ["pickup_longitude", "dropoff_longitude"]:
    df = df[(df[long] > nyc_min_longitude) & (df[long] < nyc_max_longitude)]
for lat in ["pickup_latitude", "dropoff_latitude"]:
    df = df[(df[lat] > nyc_min_latitude) & (df[lat] < nyc_max_latitude)]



## === cell 8
df.loc[df["passenger_count"] == 0, "passenger_count"] = 1

test.loc[test["passenger_count"] == 0, "passenger_count"] = 1



## === cell 9
df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 100)]



## === cell 10
df.describe()



## === cell 11
df["longitude_diff"] = df["dropoff_longitude"] - df["pickup_longitude"]
df["latitude_diff"] = df["dropoff_latitude"] - df["pickup_latitude"]
df.describe()



## === cell 12
test["longitude_diff"] = test["dropoff_longitude"] - test["pickup_longitude"]
test["latitude_diff"] = test["dropoff_latitude"] - test["pickup_latitude"]
test.describe()



## === cell 13
df["year"] = df["pickup_datetime"].dt.year
df["month"] = df["pickup_datetime"].dt.month
df["day"] = df["pickup_datetime"].dt.day
df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
df["hour"] = df["pickup_datetime"].dt.hour
df = df.drop(["pickup_datetime"], axis=1)



## === cell 14
test["year"] = test["pickup_datetime"].dt.year
test["month"] = test["pickup_datetime"].dt.month
test["day"] = test["pickup_datetime"].dt.day
test["day_of_week"] = test["pickup_datetime"].dt.dayofweek
test["hour"] = test["pickup_datetime"].dt.hour
test = test.drop(["pickup_datetime"], axis=1)



## === cell 15
test_keys = test["key"].copy()




## === cell 16
def euc_distance(lat1, long1, lat2, long2):
    return ((lat1 - lat2) ** 2 + (long1 - long2) ** 2) ** 0.5


df["travel_distance"] = euc_distance(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"],
)
test["travel_distance"] = euc_distance(
    test["pickup_latitude"],
    test["pickup_longitude"],
    test["dropoff_latitude"],
    test["dropoff_longitude"],
)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(np.float64))
    lon1 = np.radians(lon1.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0 * c  # Earth radius in km


df["haversine_km"] = haversine_km(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"],
)
test["haversine_km"] = haversine_km(
    test["pickup_latitude"],
    test["pickup_longitude"],
    test["dropoff_latitude"],
    test["dropoff_longitude"],
)



## === cell 17
df.describe()



## === cell 18
test.describe()



## === cell 19
print(df.isnull().sum())
print(test.isnull().sum())



## === cell 20
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor



## === cell 21
y = df["fare_amount"].astype(np.float32)
X = df.drop(["fare_amount", "key"], axis=1)

X_test_full = test.drop(["key"], axis=1)
X_test_full = X_test_full.reindex(columns=X.columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_full)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=test.index)



## === cell 22
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)



## === cell 23
model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32, 8),
    activation="relu",
    solver="adam",
    alpha=0.0,  # closer to the original (no explicit L2 penalty in Keras code)
    batch_size=256,
    learning_rate_init=0.001,
    max_iter=20,  # match the original epochs=20 as closely as possible
    shuffle=True,
    random_state=42,
    early_stopping=False,  # do not introduce early stopping
    n_iter_no_change=1000,  # irrelevant when early_stopping=False, but keeps behavior stable
    verbose=True,
)

model.fit(X_train, y_train)

train_pred = model.predict(X_train).reshape(-1)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

val_pred = model.predict(X_val).reshape(-1)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

print("Train RMSE: {:0.2f}".format(train_rmse))
print("Test RMSE: {:0.2f}".format(val_rmse))
print("------------------------")



## === cell 24
pred = model.predict(X_test_scaled).reshape(-1)
pred.shape



## === cell 25
pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

pred = np.clip(pred, 0.0, 100.0)

submission = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)
submission["key"] = test_keys.values
submission["fare_amount"] = pred

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
