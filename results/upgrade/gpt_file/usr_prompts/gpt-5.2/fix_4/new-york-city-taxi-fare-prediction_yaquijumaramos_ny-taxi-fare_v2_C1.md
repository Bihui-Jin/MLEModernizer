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

3.9

# 3. Installed packages

geopandas==0.14.4
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

4.19963

# 6. Current score

7.14307

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.50764) has done: 'I fix the submission generation so it always creates a valid `.csv` with the correct `key` values and filename extension, since your current code overwrites `key` with row indices and writes a non-CSV filename. To move RMSE toward the target, I keep your XGBoost setup but add the standard NYC taxi distance feature (haversine) using the same lat/long inputs; this is minimal feature engineering that doesn’t change the model type or training approach. I also ensure train/test columns align exactly (same feature set, same order) and clip negative fare predictions to 0 for stability (a safe post-process for a nonnegative target). These changes are directly tied to producing a valid submission and improving score without altering the core modeling logic.'
- What this solution (achieved 6.56179) has done: 'Your current RMSE (7.50764) is worse than the target (4.19963), so we need a modest, safe improvement without changing your core model/training approach. The biggest low-risk gain here is adding the standard datetime-derived features (hour, day-of-week, month, year) from `pickup_datetime`, because fare strongly depends on time patterns and this preserves your XGBoost pipeline. I also apply the same geographic bounds filtering to the test set (with row-preserving clipping rather than dropping) to avoid out-of-distribution coordinates at prediction time. Finally, I fix a performance bug: you are using `early_stopping_rounds` (disallowed by your requirements), so I remove it and keep `num_boost_round` unchanged to preserve the intended training procedure while making it compliant.'
- What this solution (achieved 7.14307) has done: 'Your current RMSE (6.56179) is worse than the target (4.19963), so we should make a small, low-risk improvement without changing the model type or training loop. The biggest gain for NYC taxi fares while preserving your pipeline is adding a few standard location-based features derived from the same inputs: straight-line deltas, manhattan distance in degrees, and a simple bearing. I also make the geographic bounds handling consistent by filtering out-of-bounds rows from the training sample (as you already do) and clipping the test rows (row-preserving) exactly as you already intended, while ensuring no NaNs remain after datetime parsing. These changes keep XGBoost and the same training procedure intact, but typically move RMSE materially toward your target on this competition.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
df_train = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=400000
)
df_train.head()



## === cell 2
df_train.shape



## === cell 3
df_test = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
df_test.head()



## === cell 4
df_train.info()
df_test.info()



## === cell 5
df_train.describe()



## === cell 6
df_test.describe()



## === cell 7
df_train.drop(df_train[df_train["passenger_count"] > 5].index, axis=0, inplace=True)
df_train.drop(df_train[df_train["passenger_count"] == 0].index, axis=0, inplace=True)



## === cell 8
df_train.drop(df_train[df_train["fare_amount"] < 3].index, axis=0, inplace=True)
df_train.drop(df_train[df_train["fare_amount"] > 200].index, axis=0, inplace=True)



## === cell 9
df_train.dropna(inplace=True)

df_train.drop(
    df_train.index[
        (df_train.pickup_longitude < -75.0)
        | (df_train.pickup_longitude > -72.0)
        | (df_train.pickup_latitude < 40.0)
        | (df_train.pickup_latitude > 42.0)
    ],
    inplace=True,
)

df_train.drop(
    df_train.index[
        (df_train.dropoff_longitude < -75.0)
        | (df_train.dropoff_longitude > -72.0)
        | (df_train.dropoff_latitude < 40.0)
        | (df_train.dropoff_latitude > 42.0)
    ],
    inplace=True,
)



## === cell 10
df_train.describe()




## === cell 11
def haversine_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c  # km


def bearing_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(y, x))


def add_datetime_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["pickup_year"] = dt.dt.year.astype("float32")
    df["pickup_month"] = dt.dt.month.astype("float32")
    df["pickup_day"] = dt.dt.day.astype("float32")
    df["pickup_hour"] = dt.dt.hour.astype("float32")
    df["pickup_dayofweek"] = dt.dt.dayofweek.astype("float32")
    return df


def add_geo_features(df):
    df["abs_lon_diff"] = (
        (df["dropoff_longitude"] - df["pickup_longitude"]).abs().astype("float32")
    )
    df["abs_lat_diff"] = (
        (df["dropoff_latitude"] - df["pickup_latitude"]).abs().astype("float32")
    )
    df["manhattan_deg"] = (df["abs_lon_diff"] + df["abs_lat_diff"]).astype("float32")
    df["bearing"] = bearing_np(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    ).astype("float32")
    return df


df_train = add_datetime_features(df_train)
df_test = add_datetime_features(df_test)

df_train.dropna(inplace=True)

df_train["haversine_km"] = haversine_np(
    df_train["pickup_longitude"].values,
    df_train["pickup_latitude"].values,
    df_train["dropoff_longitude"].values,
    df_train["dropoff_latitude"].values,
).astype("float32")

df_test["haversine_km"] = haversine_np(
    df_test["pickup_longitude"].values,
    df_test["pickup_latitude"].values,
    df_test["dropoff_longitude"].values,
    df_test["dropoff_latitude"].values,
).astype("float32")

df_train = add_geo_features(df_train)
df_test = add_geo_features(df_test)

for col, lo, hi in [
    ("pickup_longitude", -75.0, -72.0),
    ("dropoff_longitude", -75.0, -72.0),
    ("pickup_latitude", 40.0, 42.0),
    ("dropoff_latitude", 40.0, 42.0),
]:
    df_test[col] = df_test[col].clip(lo, hi)

from sklearn.model_selection import train_test_split

X = df_train.drop(["fare_amount", "pickup_datetime", "key"], axis=1)
y = df_train["fare_amount"]

X_test = df_test.drop(["pickup_datetime", "key"], axis=1)

X_test = X_test[X.columns]

train_x, valid_x, train_y, valid_y = train_test_split(
    X, y, test_size=0.2, random_state=12
)



## === cell 12
train_x.dtypes



## === cell 13
import xgboost as xgb

dtrain = xgb.DMatrix(train_x, label=train_y)
dvalid = xgb.DMatrix(valid_x, label=valid_y)
dtest = xgb.DMatrix(X_test)
watchlist = [(dtrain, "train"), (dvalid, "valid")]

params = {
    "objective": "reg:squarederror",
    "min_child_weight": 1,
    "learning_rate": 0.005,
    "colsample_bytree": 0.7,
    "max_depth": 10,
    "subsample": 0.7,
    "n_jobs": -1,
    "booster": "gbtree",
    "eval_metric": "rmse",
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=700,
    evals=watchlist,
    maximize=False,
    verbose_eval=50,
)



## === cell 14
prediction = model.predict(dtest)

prediction = np.clip(prediction, 0.0, None)



## === cell 15
submission = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)
submission.head()



## === cell 16
output = pd.DataFrame({"key": df_test["key"].values, "fare_amount": prediction})
output.head()



## === cell 17
output.to_csv("my_submission.csv", index=False)
print("Wrote my_submission.csv with shape:", output.shape)
print("Columns:", list(output.columns))
print("Any NA fare_amount:", output["fare_amount"].isna().any())
print("Any duplicated keys:", output["key"].duplicated().any())
