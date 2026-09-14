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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from geopy.distance import great_circle  # calculate distances
from sklearn import metrics  # evaluating models
from sklearn.model_selection import train_test_split  # set splitting and validation
from sklearn.linear_model import LinearRegression
import xgboost as xgb  # XGBoost regressor
import matplotlib.pyplot as plt  # plotting
import seaborn as sns  # plotting
from math import sin, cos, sqrt, atan2, radians

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
INPUT_DIR = "/kaggle/input"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "../input"

print("Input dir:", INPUT_DIR)
print(os.listdir(INPUT_DIR)[:50])



## === cell 2
test_path = os.path.join(INPUT_DIR, "test.csv")
train_path = os.path.join(INPUT_DIR, "train.csv")
sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

test = pd.read_csv(test_path)



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
TRAIN_NROWS = 1_000_000
CHUNK_SIZE = 250_000  # keeps runtime/memory within Kaggle limits

rng = np.random.RandomState(RANDOM_STATE)

reservoir = None
seen = 0

for chunk in pd.read_csv(train_path, dtype=types, chunksize=CHUNK_SIZE):
    chunk = chunk.dropna()
    n = len(chunk)
    if n == 0:
        continue

    if reservoir is None:
        if n >= TRAIN_NROWS:
            reservoir = chunk.sample(n=TRAIN_NROWS, random_state=rng).reset_index(
                drop=True
            )
            seen = n
            continue
        else:
            reservoir = chunk.copy().reset_index(drop=True)
            seen = n
            continue

    if len(reservoir) < TRAIN_NROWS:
        need = TRAIN_NROWS - len(reservoir)
        take = min(need, n)
        if take > 0:
            reservoir = pd.concat(
                [reservoir, chunk.sample(n=take, random_state=rng)], ignore_index=True
            )
        seen += n
        continue

    idx = np.arange(seen, seen + n, dtype=np.int64) + 1  # 1..seen+n (offset by seen)
    j = rng.randint(
        1, idx + 1
    )  # vectorized randint upper bound per-element isn't supported
    for t in range(n):
        seen += 1
        j = rng.randint(1, seen + 1)
        if j <= TRAIN_NROWS:
            reservoir.iloc[j - 1] = chunk.iloc[t]

train = reservoir.reset_index(drop=True)
print("Loaded uniform train sample:", train.shape)



## === cell 6
train.head()



## === cell 7
train.describe()



## === cell 8
try:
    sns.histplot(train["fare_amount"], kde=True)
    plt.show()
except Exception as e:
    print("Plotting skipped:", repr(e))



## === cell 9
try:
    sns.histplot(train["passenger_count"], kde=True)
    plt.show()
except Exception as e:
    print("Plotting skipped:", repr(e))



## === cell 10
train.isnull().sum()



## === cell 11
train.dropna(inplace=True)



## === cell 12
train["pickup_datetime"] = (
    train["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
)
train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"], errors="coerce")
train = train.dropna(subset=["pickup_datetime"]).copy()

if len(train) > TRAIN_NROWS:
    train = train.sample(n=TRAIN_NROWS, random_state=rng).reset_index(drop=True)

print("After datetime parse & downsample:", train.shape)



## === cell 13
train = train[train["fare_amount"] > 0]
train = train[train["fare_amount"] < 250]  # common filter in this competition
train = train[(train["pickup_longitude"] != 0) & (train["pickup_latitude"] != 0)]
train = train[(train["dropoff_longitude"] != 0) & (train["dropoff_latitude"] != 0)]
train = train[(train["pickup_longitude"] > -80) & (train["pickup_longitude"] < -70)]
train = train[(train["dropoff_longitude"] > -80) & (train["dropoff_longitude"] < -70)]
train = train[(train["pickup_latitude"] > 35) & (train["pickup_latitude"] < 45)]
train = train[(train["dropoff_latitude"] > 35) & (train["dropoff_latitude"] < 45)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]



## === cell 14
for col, lo, hi in [
    ("pickup_longitude", -80.0, -70.0),
    ("dropoff_longitude", -80.0, -70.0),
    ("pickup_latitude", 35.0, 45.0),
    ("dropoff_latitude", 35.0, 45.0),
]:
    test.loc[(test[col] < lo) | (test[col] > hi) | (test[col] == 0), col] = np.nan

test["passenger_count"] = test["passenger_count"].clip(lower=1, upper=9)



## === cell 15
train.describe()




## === cell 16
def dist_calc(df):
    for i, row in df.iterrows():
        df.at[i, "distance"] = great_circle(
            (row["pickup_latitude"], row["pickup_longitude"]),
            (row["dropoff_latitude"], row["dropoff_longitude"]),
        ).km




## === cell 17
def quick_dist_calc(df):
    R = 6373.0
    if "distance" not in df.columns:
        df["distance"] = np.nan

    for i, row in df.iterrows():
        lat1 = radians(row["pickup_latitude"])
        lon1 = radians(row["pickup_longitude"])
        lat2 = radians(row["dropoff_latitude"])
        lon2 = radians(row["dropoff_longitude"])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c
        df.at[i, "distance"] = distance




## === cell 18
def quick_dist_calc_vectorized(df):
    R = 6373.0  # keep same radius constant as your original quick_dist_calc
    lat1 = np.radians(df["pickup_latitude"].astype("float64").values)
    lon1 = np.radians(df["pickup_longitude"].astype("float64").values)
    lat2 = np.radians(df["dropoff_latitude"].astype("float64").values)
    lon2 = np.radians(df["dropoff_longitude"].astype("float64").values)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    df["distance"] = (R * c).astype("float32")
    return df


coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
train_meds = train[coord_cols].median(numeric_only=True)
test[coord_cols] = test[coord_cols].fillna(train_meds)

quick_dist_calc_vectorized(train)
quick_dist_calc_vectorized(test)



## === cell 19
test["pickup_datetime"] = (
    test["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
)
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"], errors="coerce")



## === cell 20
train["hour"] = train.pickup_datetime.dt.hour
train["weekday"] = train.pickup_datetime.dt.weekday
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year

test["hour"] = test.pickup_datetime.dt.hour
test["weekday"] = test.pickup_datetime.dt.weekday
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year

test[["hour", "weekday", "month", "year"]] = (
    test[["hour", "weekday", "month", "year"]]
    .fillna({"hour": 0, "weekday": 0, "month": 1, "year": 2010})
    .astype("int16")
)




## === cell 21
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    R_earth = 6371
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(
        np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    )
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon / 2.0) ** 2
    )

    return 2 * R_earth * np.arcsin(np.sqrt(a))




## === cell 22
def add_airport_dist(dataset):
    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)

    pickup_lat = dataset["pickup_latitude"]
    dropoff_lat = dataset["dropoff_latitude"]
    pickup_lon = dataset["pickup_longitude"]
    dropoff_lon = dataset["dropoff_longitude"]

    pickup_jfk = sphere_dist(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1])
    dropoff_jfk = sphere_dist(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon)
    pickup_ewr = sphere_dist(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])
    dropoff_ewr = sphere_dist(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon)
    pickup_lga = sphere_dist(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1])
    dropoff_lga = sphere_dist(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon)

    dataset["jfk_dist"] = pd.concat([pickup_jfk, dropoff_jfk], axis=1).min(axis=1)
    dataset["ewr_dist"] = pd.concat([pickup_ewr, dropoff_ewr], axis=1).min(axis=1)
    dataset["lga_dist"] = pd.concat([pickup_lga, dropoff_lga], axis=1).min(axis=1)

    return dataset




## === cell 23
train = add_airport_dist(train)
test = add_airport_dist(test)



## === cell 24
train.head()



## === cell 25
test.head()



## === cell 26
try:
    plt.figure(figsize=(15, 8))
    sns.heatmap(
        train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
    )
    plt.show()
except Exception as e:
    print("Heatmap skipped:", repr(e))



## === cell 27
for df in (train, test):
    df["abs_lon_diff"] = (
        (df["pickup_longitude"] - df["dropoff_longitude"]).abs().astype("float32")
    )
    df["abs_lat_diff"] = (
        (df["pickup_latitude"] - df["dropoff_latitude"]).abs().astype("float32")
    )
    df["log1p_distance"] = np.log1p(df["distance"].astype("float64")).astype("float32")

train = train[train["distance"] > 0.0]
train = train[train["distance"] < 200.0]



## === cell 28
X = train.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = train["fare_amount"]



## === cell 29
X.head()



## === cell 30
y.head()



## === cell 31
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)



## === cell 32
test_pred = test.drop(["key", "pickup_datetime"], axis=1)
test_pred = test_pred.reindex(columns=X.columns)



## === cell 33
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_test, y_test))



## === cell 34
y_pred = lm.predict(X)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y))
lrmse



## === cell 35
LinearPredictions = lm.predict(test_pred)
LinearPredictions = np.clip(LinearPredictions, 0.0, None)
LinearPredictions



## === cell 36
LinearPredictions.size



## === cell 37
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)



## === cell 38
linear_submission.head()




## === cell 39
def XGBoost(X_train, X_test, y_train, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": RANDOM_STATE,
        "eta": 0.08,
        "max_depth": 8,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    }
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        early_stopping_rounds=50,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        verbose_eval=False,
    )
    return booster




## === cell 40
xgbm = XGBoost(X_train, X_test, y_train, y_test)

dtest_full = xgb.DMatrix(test_pred)

best_iter = getattr(xgbm, "best_iteration", None)
if best_iter is None:
    XGBPredictions = xgbm.predict(dtest_full)
else:
    XGBPredictions = xgbm.predict(dtest_full, iteration_range=(0, best_iter + 1))



## === cell 41
XGBPredictions



## === cell 42
XGBPredictions = np.clip(XGBPredictions, 0.0, None)
XGBPredictions



## === cell 43
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)
XGB_submission.head()



## === cell 44
submission = (
    XGB_submission
    if "XGB_submission" in globals() and XGB_submission is not None
    else linear_submission
)



## === cell 45
out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote submission:", out_path)
print(submission.head())
print("Submission shape:", submission.shape)
assert submission.shape[1] == 2 and list(submission.columns) == ["key", "fare_amount"]
assert submission.shape[0] == test.shape[0]
