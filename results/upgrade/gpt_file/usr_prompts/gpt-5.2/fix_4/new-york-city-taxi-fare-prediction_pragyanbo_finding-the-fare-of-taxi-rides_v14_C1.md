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

4.54269

# 6. Current score

6.24054

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1035.73284) has done: 'I remove the TensorFlow Estimator section that fails under TF 2.18 (no `tf.estimator`) and the protobuf `MessageFactory` error, because it prevents the notebook from completing and producing a valid submission. I keep your existing feature engineering and the scikit-learn RandomForest/LinearRegression core logic, but make one minimal, score-improving change: parse `pickup_datetime` once (vectorized) and add `month` and `weekday` time features, which typically reduces RMSE without changing the modeling approach. I also set `random_state` for deterministic splits/models and ensure the final output is a valid `submission.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 6.24054) has done: 'Your current RMSE (~1035) indicates the model is producing extreme/invalid fares for a large fraction of the test set, which usually comes from dirty coordinates (0/0, out-of-NYC bounds) and/or unbounded predictions. To move the score toward the target with minimal core-logic change, I (1) add the standard NYC coordinate/passenger/fare cleaning filter on the *training sample* (same model, same features), (2) ensure test rows with invalid/missing time features are handled safely, and (3) clip predictions to a realistic fare range derived from the cleaned training set (a small post-processing step that dramatically reduces catastrophic RMSE without changing the model/training approach). I keep your RandomForest and LinearRegression pipelines intact and still write `submission.csv` and `linear_reg.csv` with `key,fare_amount`. These changes are directly aimed at eliminating outlier-driven errors that dominate RMSE and should bring the score much closer to the target.'
- What this solution (achieved 6.24054) has done: 'Your score (RMSE 6.24054, lower is better) is still far from the target 4.54269, so we should improve performance with minimal logic changes. The simplest high-impact fix while preserving your approach is to correct the distance calculation: your current function computes miles but mistakenly multiplies by 0.6213712 twice, which weakens the key feature and hurts both models. I replace it with a standard haversine-in-km then convert once to miles (same single “distance” feature, same models/training), and keep your existing cleaning + time features + clipping unchanged. This should move RMSE materially downward toward the target without altering the modeling approach.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")



## === cell 1
train_path = "../input/train.csv"
test_path = "../input/test.csv"

train_df = pd.read_csv(train_path, nrows=200000)
train_df.shape



## === cell 2
test_df = pd.read_csv(test_path)
test_df.shape



## === cell 3
train_df.head(5)



## === cell 4
train_df.isnull().sum()



## === cell 5
train_df.dropna(inplace=True)



## === cell 6
train_df.describe()



## === cell 7
train_df = train_df[train_df["fare_amount"] > 0].copy()
train_df.shape




## === cell 8
def distance(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)

    r = 6371.0088  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    )
    km = 2.0 * r * np.arcsin(np.sqrt(a))
    miles = km * 0.6213712
    return miles




## === cell 9
train_df["distance"] = distance(
    train_df.pickup_latitude,
    train_df.pickup_longitude,
    train_df.dropoff_latitude,
    train_df.dropoff_longitude,
)

test_df["distance"] = distance(
    test_df.pickup_latitude,
    test_df.pickup_longitude,
    test_df.dropoff_latitude,
    test_df.dropoff_longitude,
)



## === cell 10
nyc_lat_min, nyc_lat_max = 40.5, 41.0
nyc_lon_min, nyc_lon_max = -74.5, -73.0

train_df = train_df[
    (train_df["pickup_latitude"].between(nyc_lat_min, nyc_lat_max))
    & (train_df["dropoff_latitude"].between(nyc_lat_min, nyc_lat_max))
    & (train_df["pickup_longitude"].between(nyc_lon_min, nyc_lon_max))
    & (train_df["dropoff_longitude"].between(nyc_lon_min, nyc_lon_max))
].copy()

train_df = train_df[train_df["distance"] < 15].copy()

train_df = train_df[
    (train_df["passenger_count"] >= 1) & (train_df["passenger_count"] < 10)
].copy()

train_df = train_df[
    (train_df["fare_amount"] >= 2.5) & (train_df["fare_amount"] <= 250)
].copy()

train_df.describe()



## === cell 11
train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"], errors="coerce", utc=True
)
test_df["pickup_datetime"] = pd.to_datetime(
    test_df["pickup_datetime"], errors="coerce", utc=True
)

train_df = train_df.dropna(subset=["pickup_datetime"]).copy()

for df in (train_df, test_df):
    df["hour"] = df["pickup_datetime"].dt.hour.astype("float32")
    df["year"] = df["pickup_datetime"].dt.year.astype("float32")
    df["month"] = df["pickup_datetime"].dt.month.astype("float32")
    df["weekday"] = df["pickup_datetime"].dt.weekday.astype("float32")

time_cols = ["hour", "year", "month", "weekday"]
test_df[time_cols] = test_df[time_cols].fillna(0.0)



## === cell 12
feat_cols_s = ["distance", "passenger_count", "hour", "year", "month", "weekday"]
X = train_df[feat_cols_s]
y = train_df["fare_amount"]



## === cell 13
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=42
)



## === cell 14
from sklearn.ensemble import RandomForestRegressor

r_reg = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
r_reg.fit(X_train, y_train)



## === cell 15
y_pred_final = r_reg.predict(test_df[feat_cols_s])

fare_low = float(np.percentile(y, 0.5))
fare_high = float(np.percentile(y, 99.5))
y_pred_final = np.clip(y_pred_final, fare_low, fare_high)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": y_pred_final})
submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 16
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model_l = Pipeline(
    steps=[("standard_scaler", StandardScaler()), ("lin_reg", LinearRegression())]
)
model_l.fit(X_train, y_train)

y_pred_lr = model_l.predict(test_df[feat_cols_s])
y_pred_lr = np.clip(y_pred_lr, fare_low, fare_high)

submission_lr = pd.DataFrame({"key": test_df["key"], "fare_amount": y_pred_lr})
submission_lr.to_csv("linear_reg.csv", index=False)



## === cell 17
print("Wrote submission.csv (RandomForest) and linear_reg.csv (LinearRegression).")
print(
    f"Clipped predictions to [{fare_low:.2f}, {fare_high:.2f}] based on cleaned train fares."
)
