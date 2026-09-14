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

bayesian-optimization==3.1.0
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
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from bayes_opt import BayesianOptimization



## === cell 1
usecols_train = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_map = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
train_path = Path("../input/train.csv")
chunks = pd.read_csv(
    train_path,
    usecols=usecols_train,
    parse_dates=["pickup_datetime"],
    dtype=dtype_map,
    low_memory=False,
    chunksize=500_000,
)
filtered_chunks = []
for chunk in chunks:
    mask = (chunk["fare_amount"] > 0.05) & (chunk["passenger_count"] > 0)
    mask &= chunk["pickup_longitude"].between(-75, -73)
    mask &= chunk["dropoff_longitude"].between(-75, -73)
    mask &= chunk["pickup_latitude"].between(40, 42)
    mask &= chunk["dropoff_latitude"].between(40, 42)
    mask &= chunk["passenger_count"].between(0, 8)
    mask &= chunk["fare_amount"].between(0, 250)
    filtered = chunk.loc[mask].sample(frac=0.10, random_state=42)
    filtered_chunks.append(filtered)
df_train = pd.concat(filtered_chunks, ignore_index=True)



## === cell 2
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_path = Path("../input/test.csv")
df_test = pd.read_csv(
    test_path,
    usecols=usecols_test,
    parse_dates=["pickup_datetime"],
    low_memory=True,
)




## === cell 3
def haversine_miles(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in miles."""
    p = np.pi / 180.0
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


df_train["distance_miles"] = haversine_miles(
    df_train["pickup_latitude"],
    df_train["pickup_longitude"],
    df_train["dropoff_latitude"],
    df_train["dropoff_longitude"],
)
df_train["distance_squared"] = df_train["distance_miles"] ** 2

df_test["distance_miles"] = haversine_miles(
    df_test["pickup_latitude"],
    df_test["pickup_longitude"],
    df_test["dropoff_latitude"],
    df_test["dropoff_longitude"],
)
df_test["distance_squared"] = df_test["distance_miles"] ** 2

for df in (df_train, df_test):
    df["year"] = df["pickup_datetime"].dt.year
    df["hour"] = df["pickup_datetime"].dt.hour
    df["month"] = df["pickup_datetime"].dt.month
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek



## === cell 4
features = [
    "year",
    "hour",
    "month",
    "dayofweek",
    "distance_miles",
    "distance_squared",
    "passenger_count",
]
X = df_train[features]
y = df_train["fare_amount"]
X_test = df_test[features]

subset_frac = 0.20
X_sub, _, y_sub, _ = train_test_split(
    X, y, train_size=subset_frac, random_state=42, stratify=None
)
X_train_bo, X_val_bo, y_train_bo, y_val_bo = train_test_split(
    X_sub, y_sub, test_size=0.2, random_state=42
)



## === cell 5
dtrain_bo = xgb.DMatrix(X_train_bo.astype(np.float32), label=y_train_bo)
dval_bo = xgb.DMatrix(X_val_bo.astype(np.float32), label=y_val_bo)


def xgb_eva(max_depth, gamma, colsample_bytree):
    params = {
        "eval_metric": "rmse",
        "objective": "reg:squarederror",
        "max_depth": int(max_depth),
        "subsample": 0.8,
        "eta": 0.1,
        "gamma": gamma,
        "colsample_bytree": colsample_bytree,
        "tree_method": "hist",
        "max_bin": 64,
        "nthread": 4,
    }
    model = xgb.train(
        params,
        dtrain_bo,
        num_boost_round=200,
        evals=[(dval_bo, "eval")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    return -float(model.best_score)




## === cell 6
xgb_bo = BayesianOptimization(
    f=xgb_eva,
    pbounds={
        "max_depth": (3, 8),
        "gamma": (0.0, 1.0),
        "colsample_bytree": (0.3, 0.9),
    },
    random_state=42,
    verbose=0,
)
xgb_bo.maximize(init_points=2, n_iter=3)



## === cell 7
best_params = xgb_bo.max["params"]
best_params["max_depth"] = int(best_params["max_depth"])
best_params["objective"] = "reg:squarederror"
best_params["eval_metric"] = "rmse"
best_params["subsample"] = 0.8
best_params["eta"] = 0.1
best_params["tree_method"] = "hist"
best_params["max_bin"] = 64
best_params["nthread"] = 4



## === cell 8
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

dtrain = xgb.DMatrix(X_train.astype(np.float32), label=y_train_log)
dval = xgb.DMatrix(X_val.astype(np.float32), label=y_val_log)

model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=1000,
    evals=[(dval, "eval")],
    early_stopping_rounds=30,
    verbose_eval=False,
)



## === cell 9
X_test_dm = xgb.DMatrix(X_test.astype(np.float32))
y_pred_log = model.predict(X_test_dm)
y_pred = np.expm1(y_pred_log)  # invert the log1p transform

submission = pd.DataFrame({"key": df_test["key"], "fare_amount": y_pred})
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
