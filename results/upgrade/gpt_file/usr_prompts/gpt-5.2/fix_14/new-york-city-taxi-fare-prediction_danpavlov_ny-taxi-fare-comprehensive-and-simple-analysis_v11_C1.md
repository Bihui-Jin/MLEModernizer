# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
geopy==2.4.1
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
seaborn==0.12.2
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

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from geopy.distance import great_circle  # kept to preserve original intent
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)



## === cell 1
print(os.listdir("../input"))



## === cell 2
test = pd.read_csv("../input/test.csv")



## === cell 3
test.dtypes



## === cell 4
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}




## === cell 5
def read_train_random_sample(
    path: str,
    n_sample: int = 500000,
    chunksize: int = 250000,
    seed: int = 42,
    dtype: dict = None,
):
    rng = np.random.RandomState(seed)
    reservoir = None
    seen = 0

    for chunk in pd.read_csv(path, chunksize=chunksize, dtype=dtype):
        chunk = chunk.reset_index(drop=True)
        if reservoir is None:
            reservoir = chunk.iloc[0:0].copy()

        for i in range(len(chunk)):
            seen += 1
            row = chunk.iloc[[i]]
            if len(reservoir) < n_sample:
                reservoir = pd.concat([reservoir, row], ignore_index=True)
            else:
                j = rng.randint(0, seen)
                if j < n_sample:
                    reservoir.iloc[j] = row.iloc[0]

    reservoir = reservoir.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return reservoir


train = read_train_random_sample(
    "../input/train.csv", n_sample=500000, chunksize=250000, seed=42, dtype=types
)



## === cell 6
train.head()



## === cell 7
train.describe()



## === cell 8
try:
    sns.distplot(train["fare_amount"])
    plt.show()
except Exception:
    pass



## === cell 9
try:
    sns.distplot(train["passenger_count"])
    plt.show()
except Exception:
    pass



## === cell 10
train.isnull().sum()



## === cell 11
train.dropna(inplace=True)

train = train[(train["fare_amount"] > 0) & (train["fare_amount"] < 250)]

min_lon, max_lon = -74.5, -72.8
min_lat, max_lat = 40.5, 41.8

train = train[
    (train["pickup_longitude"] >= min_lon)
    & (train["pickup_longitude"] <= max_lon)
    & (train["dropoff_longitude"] >= min_lon)
    & (train["dropoff_longitude"] <= max_lon)
    & (train["pickup_latitude"] >= min_lat)
    & (train["pickup_latitude"] <= max_lat)
    & (train["dropoff_latitude"] >= min_lat)
    & (train["dropoff_latitude"] <= max_lat)
]

train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]

for c in ["pickup_longitude", "dropoff_longitude"]:
    test[c] = pd.to_numeric(test[c], errors="coerce").astype("float32")
    test[c] = (
        test[c].fillna((min_lon + max_lon) / 2.0).clip(lower=min_lon, upper=max_lon)
    )

for c in ["pickup_latitude", "dropoff_latitude"]:
    test[c] = pd.to_numeric(test[c], errors="coerce").astype("float32")
    test[c] = (
        test[c].fillna((min_lat + max_lat) / 2.0).clip(lower=min_lat, upper=max_lat)
    )

test["passenger_count"] = pd.to_numeric(test["passenger_count"], errors="coerce")
test["passenger_count"] = (
    test["passenger_count"].fillna(1).clip(lower=1, upper=6).astype("uint8")
)



## === cell 12
train.describe()




## === cell 13
def dist_calc(df: pd.DataFrame) -> None:
    lat1 = np.deg2rad(df["pickup_latitude"].astype("float64").to_numpy())
    lon1 = np.deg2rad(df["pickup_longitude"].astype("float64").to_numpy())
    lat2 = np.deg2rad(df["dropoff_latitude"].astype("float64").to_numpy())
    lon2 = np.deg2rad(df["dropoff_longitude"].astype("float64").to_numpy())

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    R_km = 6371.0
    df["distance"] = (R_km * c).astype("float32")




## === cell 14
def extra_geo_features(df: pd.DataFrame) -> None:
    df["manhattan"] = (
        (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
        + (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
    ).astype("float32")

    jfk_lat, jfk_lon = 40.6413, -73.7781
    lga_lat, lga_lon = 40.7769, -73.8740

    p_lat = df["pickup_latitude"].astype("float32")
    p_lon = df["pickup_longitude"].astype("float32")
    d_lat = df["dropoff_latitude"].astype("float32")
    d_lon = df["dropoff_longitude"].astype("float32")

    pjfk = (p_lat - jfk_lat) ** 2 + (p_lon - jfk_lon) ** 2
    djfk = (d_lat - jfk_lat) ** 2 + (d_lon - jfk_lon) ** 2
    plga = (p_lat - lga_lat) ** 2 + (p_lon - lga_lon) ** 2
    dlga = (d_lat - lga_lat) ** 2 + (d_lon - lga_lon) ** 2

    near_airport = (pjfk < 0.0025) | (djfk < 0.0025) | (plga < 0.0016) | (dlga < 0.0016)
    df["near_airport"] = near_airport.astype("uint8")




## === cell 15
def add_bearing_and_deltas(df: pd.DataFrame) -> None:
    dlon = (
        df["dropoff_longitude"].astype("float64")
        - df["pickup_longitude"].astype("float64")
    ).to_numpy()
    dlat = (
        df["dropoff_latitude"].astype("float64")
        - df["pickup_latitude"].astype("float64")
    ).to_numpy()

    df["delta_lon"] = dlon.astype("float32")
    df["delta_lat"] = dlat.astype("float32")
    df["abs_dlon"] = np.abs(dlon).astype("float32")
    df["abs_dlat"] = np.abs(dlat).astype("float32")

    lat1 = np.deg2rad(df["pickup_latitude"].astype("float64").to_numpy())
    lat2 = np.deg2rad(df["dropoff_latitude"].astype("float64").to_numpy())
    dlon_rad = np.deg2rad(dlon)
    y = np.sin(dlon_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon_rad)
    df["bearing"] = np.arctan2(y, x).astype("float32")




## === cell 16
def add_interaction_features(df: pd.DataFrame) -> None:
    df["pickup_lat_lon"] = (df["pickup_latitude"] * df["pickup_longitude"]).astype(
        "float32"
    )
    df["dropoff_lat_lon"] = (df["dropoff_latitude"] * df["dropoff_longitude"]).astype(
        "float32"
    )
    df["lat_sum"] = (df["pickup_latitude"] + df["dropoff_latitude"]).astype("float32")
    df["lon_sum"] = (df["pickup_longitude"] + df["dropoff_longitude"]).astype("float32")




## === cell 17
dist_calc(train)
dist_calc(test)
extra_geo_features(train)
extra_geo_features(test)
add_bearing_and_deltas(train)
add_bearing_and_deltas(test)
add_interaction_features(train)
add_interaction_features(test)



## === cell 18
train["pickup_datetime"] = (
    train["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
)
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], errors="coerce", utc=False
)



## === cell 19
test["pickup_datetime"] = (
    test["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], errors="coerce", utc=False
)



## === cell 20
train.dropna(subset=["pickup_datetime"], inplace=True)
test["pickup_datetime"] = test["pickup_datetime"].fillna(
    pd.Timestamp("2015-01-01 00:00:00")
)

train["year"] = train.pickup_datetime.dt.year.astype("int16")
train["hour"] = train.pickup_datetime.dt.hour.astype("int8")
train["month"] = train.pickup_datetime.dt.month.astype("int8")
train["dayofweek"] = train.pickup_datetime.dt.dayofweek.astype("int8")

test["year"] = test.pickup_datetime.dt.year.astype("int16")
test["hour"] = test.pickup_datetime.dt.hour.astype("int8")
test["month"] = test.pickup_datetime.dt.month.astype("int8")
test["dayofweek"] = test.pickup_datetime.dt.dayofweek.astype("int8")



## === cell 21
test.head()



## === cell 22
try:
    plt.figure(figsize=(15, 8))
    sns.heatmap(
        train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
    )
    plt.show()
except Exception:
    pass



## === cell 23
X = train.drop(
    [
        "key",
        "fare_amount",
        "pickup_datetime",
    ],
    axis=1,
)
y = train["fare_amount"]



## === cell 24
X.head()



## === cell 25
y.head()



## === cell 26
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 27
test_pred = test.drop(
    [
        "key",
        "pickup_datetime",
    ],
    axis=1,
)



## === cell 28
fill_values = X_train.median(numeric_only=True)
X_train = X_train.fillna(fill_values)
X_test = X_test.fillna(fill_values)
test_pred = test_pred.fillna(fill_values)



## === cell 29
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_test, y_test))



## === cell 30
y_pred = lm.predict(X)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y))
lrmse



## === cell 31
LinearPredictions = lm.predict(test_pred).astype("float32")
LinearPredictions



## === cell 32
LinearPredictions.size



## === cell 33
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)
linear_submission.head()




## === cell 34
def XGBoost(X_train, X_test, y_train, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.03,
        "max_depth": 7,
        "min_child_weight": 5.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "lambda": 2.0,
        "alpha": 0.0,
        "gamma": 0.05,
        "max_delta_step": 1.0,
        "seed": 42,
        "tree_method": "hist",
        "verbosity": 0,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    return booster




## === cell 35
xgbm = XGBoost(X_train, X_test, y_train, y_test)

dvalid = xgb.DMatrix(X_test)
best_iter = int(getattr(xgbm, "best_iteration", 0) or 0)
valid_pred = xgbm.predict(dvalid, iteration_range=(0, best_iter + 1))
valid_rmse = float(np.sqrt(metrics.mean_squared_error(y_test, valid_pred)))
print("Holdout RMSE (XGB):", valid_rmse, "| best_iteration:", best_iter)

dtest_full = xgb.DMatrix(test_pred)
XGBPredictions = xgbm.predict(dtest_full, iteration_range=(0, best_iter + 1))



## === cell 36
XGBPredictions



## === cell 37
fare_min = 0.0
fare_max = float(min(200.0, y_train.quantile(0.999)))

XGBPredictions_final = np.clip(XGBPredictions, fare_min, fare_max).astype("float32")
XGBPredictions_final



## === cell 38
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions_final},
    columns=["key", "fare_amount"],
)
XGB_submission.head()



## === cell 39
submission = XGB_submission



## === cell 40
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
