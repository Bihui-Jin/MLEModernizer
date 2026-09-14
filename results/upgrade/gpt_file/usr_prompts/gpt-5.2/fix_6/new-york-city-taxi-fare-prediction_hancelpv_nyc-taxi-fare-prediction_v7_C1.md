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
import pandas as pd  # data processing, CSV file I/O
import math
import os

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

print("Listing ../input:")
try:
    print(os.listdir("../input"))
except FileNotFoundError:
    print("WARNING: ../input not found. Available root:", os.listdir("/"))



## === cell 1
train_cols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_cols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}
test_dtypes = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

train = pd.read_csv(
    "../input/train.csv", nrows=1_000_000, usecols=train_cols, dtype=train_dtypes
)
test = pd.read_csv("../input/test.csv", usecols=test_cols, dtype=test_dtypes)
samp = pd.read_csv("../input/sample_submission.csv")

print(
    "train shape:", train.shape, "test shape:", test.shape, "sample shape:", samp.shape
)



## === cell 2
train = train.dropna(how="any", axis="rows")
print("train shape after dropna:", train.shape)



## === cell 3
test.shape




## === cell 4
def clean_train_rows(df):
    mask = (
        (df["fare_amount"] > 0)
        & (df["fare_amount"] <= 250)
        & (df["passenger_count"] >= 1)
        & (df["passenger_count"] <= 6)
        & (df["pickup_longitude"].between(-74.5, -72.8))
        & (df["dropoff_longitude"].between(-74.5, -72.8))
        & (df["pickup_latitude"].between(40.5, 41.8))
        & (df["dropoff_latitude"].between(40.5, 41.8))
        & ~(
            (df["pickup_longitude"] == df["dropoff_longitude"])
            & (df["pickup_latitude"] == df["dropoff_latitude"])
        )
    )
    return df.loc[mask].reset_index(drop=True)


train = clean_train_rows(train)
print("train shape after clean_train_rows:", train.shape)



## === cell 5
y = train["fare_amount"].to_numpy()
n_train = len(train)
n_test = len(test)
test_id = test["key"]

all_data = pd.concat(
    (
        train.drop(columns=["fare_amount"], errors="ignore"),
        test.drop(columns=["key"], errors="ignore"),
    ),
    axis=0,
    ignore_index=True,
)

print("all_data shape:", all_data.shape, "n_train:", n_train, "n_test:", n_test)




## === cell 6
def week_num(day):
    """
    given the day of the month, return the week number of the month
    """
    if day <= 7:
        return "first"
    if (day > 7) and (day <= 14):
        return "second"
    if (day > 14) and (day <= 21):
        return "third"
    if (day > 21) and (day <= 28):
        return "fourth"
    return "fifth"




## === cell 7
def add_time_features(data):
    data = data.copy()
    dt = pd.to_datetime(data["pickup_datetime"], errors="coerce", cache=True)

    data["hour"] = dt.dt.hour.astype("Int16").astype("string")
    data["day_of_week"] = dt.dt.day_name().astype("string")
    dom = dt.dt.day.astype("Int16")
    data["week_of_month"] = dom.map(week_num).astype("string")
    data["month"] = dt.dt.month.astype("Int16").astype("string")
    data["year"] = dt.dt.year.astype("Int16").astype("string")

    return data.drop(columns=["pickup_datetime"], errors="ignore")




## === cell 8
def add_geo_features(data):
    data = data.copy()
    p_long = data["pickup_longitude"].to_numpy()
    d_long = data["dropoff_longitude"].to_numpy()
    p_lat = data["pickup_latitude"].to_numpy()
    d_lat = data["dropoff_latitude"].to_numpy()

    abs_diff_long = np.abs(d_long - p_long)
    abs_diff_lat = np.abs(d_lat - p_lat)

    data["abs_diff_longitude"] = abs_diff_long
    data["abs_diff_latitude"] = abs_diff_lat
    data["manhattan_distance"] = abs_diff_long + abs_diff_lat

    squared_long = abs_diff_long * abs_diff_long
    squared_lat = abs_diff_lat * abs_diff_lat
    data["squared_long"] = squared_long
    data["squared_lat"] = squared_lat
    data["euclid_disance"] = np.sqrt(squared_long + squared_lat)

    return data




## === cell 9
all_data = add_time_features(all_data)
all_data = add_geo_features(all_data)

drop_cols = []
for c in ["key"]:
    if c in all_data.columns:
        drop_cols.append(c)
if drop_cols:
    all_data = all_data.drop(columns=drop_cols)

print("columns after feature eng:", all_data.columns.tolist())



## === cell 10
features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "euclid_disance",
    "manhattan_distance",
]

missing = [c for c in features if c not in all_data.columns]
if missing:
    raise KeyError(f"Missing engineered feature columns: {missing}")

all_data = all_data[features]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

cat_cols = ["hour", "day_of_week", "week_of_month", "month", "year"]
num_cols = [c for c in features if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=True), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    sparse_threshold=0.3,
)

X_all = all_data  # keep as DataFrame for ColumnTransformer column selection
X_train = X_all.iloc[:n_train]
X_test = X_all.iloc[n_train:]

print(
    "Prepared split: X_train shape:",
    X_train.shape,
    "X_test shape:",
    X_test.shape,
    "y shape:",
    y.shape,
)



## === cell 11
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    print("Applied sklearnex patch for speed.")
except Exception as e:
    print("sklearnex patch not applied (continuing):", repr(e))

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42, n_jobs=1)



## === cell 12
params = {
    "bootstrap": [True],
    "max_depth": [80, 90, 100, 110],
    "max_features": [2, 3],
    "min_samples_leaf": [3, 4, 5],
    "min_samples_split": [8, 10, 12],
    "n_estimators": [100, 200, 300, 1000],
}



## === cell 13
from sklearn.model_selection import RandomizedSearchCV

preprocess.fit(X_train)
X_train_tr = preprocess.transform(X_train)
X_test_tr = preprocess.transform(X_test)

print("Transformed shapes:", X_train_tr.shape, X_test_tr.shape)

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=params,
    n_iter=12,  # keep same as provided
    scoring="neg_root_mean_squared_error",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True,
)

search.fit(X_train_tr, y)

print("Best CV RMSE:", -float(search.best_score_))
print("Best params:", search.best_params_)

best_model = search.best_estimator_



## === cell 14
test_pred = best_model.predict(X_test_tr)

test_pred = np.maximum(test_pred, 0)

print(
    "Pred stats:",
    float(np.min(test_pred)),
    float(np.mean(test_pred)),
    float(np.max(test_pred)),
)



## === cell 15
sub = pd.DataFrame({"key": test_id.values, "fare_amount": test_pred})
assert sub.shape[0] == test.shape[0], "Submission rows do not match test rows"
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
