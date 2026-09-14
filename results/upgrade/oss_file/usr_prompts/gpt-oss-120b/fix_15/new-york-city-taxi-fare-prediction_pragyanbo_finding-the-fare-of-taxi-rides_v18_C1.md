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

# 5. Target score

4.05769

# 6. Current score

4.80903

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.69255) has done: 'I fix the XGBoost prediction call that used the removed `ntree_limit` argument by switching to the new `iteration_range` parameter (or omitting it when not needed). This resolves the TypeError, ensures `xgbm_pred` is defined, and allows the script to write a valid `submission.csv`. No other logic is changed, preserving the original model and feature engineering.'
- What this solution (achieved 5.61696) has done: 'I add a few inexpensive feature‑engineering steps that are known to help taxi‑fare models – a log‑distance column and month/day‑of‑week extracted from the pickup timestamp – and I include them in the feature list. I also raise the default XGBoost round limit slightly (to give the model a bit more capacity) while keeping the early‑stopping logic. These minimal changes keep the original pipeline intact but should lower the RMSE toward the target.'
- What this solution (achieved 5.71902) has done: 'I increase the training sample size, add a few inexpensive but predictive features (distance per passenger and weekend flag), train the model on the log‑transformed fare amount (then exponentiate the predictions), and explicitly set a few XGBoost hyper‑parameters while keeping the original pipeline intact. These lightweight changes are expected to lower the RMSE toward the target without overhauling the core logic.'
- What this solution (achieved 5.72152) has done: 'I add cheap cyclical time features (sin/cos of hour, month, and day‑of‑week) and include them in the model, then modestly increase XGBoost capacity (deeper trees, smaller learning rate, more rounds). These changes keep the original pipeline intact while giving the model more useful signal, which should lower the RMSE toward the target.'
- What this solution (achieved 5.71851) has done: 'I increase the training sample size (read more rows) and allow the XGBoost model a slightly larger budget with a longer early‑stopping patience, which should improve its ability to learn from the data and therefore reduce the RMSE toward the target. No core logic or feature engineering is changed.'
- What this solution (achieved 4.9089) has done: 'We cut down memory copies and avoid unnecessary dtype casts, especially when building XGBoost DMatrix objects. By feeding the already‑float32 DataFrames directly to xgboost we eliminate large intermediate `.values.astype(np.float32)` allocations, which markedly reduces RAM pressure and speeds up training/prediction without altering any algorithmic steps or hyper‑parameters. Minor dtype tweaks (storing distances as float32) keep numerical results identical within negligible floating‑point noise, while still preserving the original feature engineering and model logic.'
- What this solution (achieved 4.78037) has done: 'The changes reduce the amount of data read (from 8 M to 2 M rows) and cap the XGBoost thread count, which cuts the most time‑consuming training phase while keeping the same feature engineering, model type, hyper‑parameters and prediction logic. The rest of the pipeline – distance calculation, datetime features, log‑transform, and submission generation – stays unchanged, so the output remains equivalent.'
- What this solution (achieved 4.80903) has done: 'I increase the training sample size, relax the distance filter, give XGBoost a slightly larger round budget with a longer early‑stopping patience, and clip negative predictions to keep them physically plausible. These modest tweaks keep the original pipeline intact while expectedly lowering the RMSE toward the target score.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import operator
import xgboost as xgb

sns.set_style("whitegrid")



## === cell 1
usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
train_path = "../input/train.csv"
test_path = "../input/test.csv"

dtype_dict = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
train_df = pd.read_csv(
    train_path,
    nrows=4_000_000,  # was 2_000_000
    usecols=usecols,
    dtype=dtype_dict,
)



## === cell 2
test_df = pd.read_csv(
    test_path,
    usecols=usecols[0:1] + usecols[2:],
    dtype={k: v for k, v in dtype_dict.items() if k != "fare_amount"},
)



## === cell 3
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)



## === cell 4
train_df.dropna(inplace=True)
train_df = train_df[train_df["fare_amount"] > 0]

train_df = train_df[
    (train_df["passenger_count"] != 0) & (train_df["passenger_count"] < 10)
]




## === cell 5
def distance(lat1, lon1, lat2, lon2):
    lat1_rad = np.radians(lat1.astype(np.float64))
    lon1_rad = np.radians(lon1.astype(np.float64))
    lat2_rad = np.radians(lat2.astype(np.float64))
    lon2_rad = np.radians(lon2.astype(np.float64))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    return (6371.0 * 0.6213712 * 2 * np.arcsin(np.sqrt(a))).astype(np.float32)


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

train_df["log_distance"] = np.log1p(train_df["distance"])
test_df["log_distance"] = np.log1p(test_df["distance"])

train_df["distance_per_passenger"] = train_df["distance"] / (
    train_df["passenger_count"] + 1
)
test_df["distance_per_passenger"] = test_df["distance"] / (
    test_df["passenger_count"] + 1
)



## === cell 6
train_df = train_df[train_df["distance"] < 30]  # was 25



## === cell 7
train_dt = pd.to_datetime(train_df["pickup_datetime"])
test_dt = pd.to_datetime(test_df["pickup_datetime"])

train_df["hour"] = train_dt.dt.hour
train_df["year"] = train_dt.dt.year
train_df["month"] = train_dt.dt.month
train_df["dayofweek"] = train_dt.dt.dayofweek
train_df["is_weekend"] = train_df["dayofweek"].isin([5, 6]).astype(int)

test_df["hour"] = test_dt.dt.hour
test_df["year"] = test_dt.dt.year
test_df["month"] = test_dt.dt.month
test_df["dayofweek"] = test_dt.dt.dayofweek
test_df["is_weekend"] = test_df["dayofweek"].isin([5, 6]).astype(int)

train_df["sin_hour"] = np.sin(2 * np.pi * train_df["hour"] / 24)
train_df["cos_hour"] = np.cos(2 * np.pi * train_df["hour"] / 24)
test_df["sin_hour"] = np.sin(2 * np.pi * test_df["hour"] / 24)
test_df["cos_hour"] = np.cos(2 * np.pi * test_df["hour"] / 24)

train_df["sin_month"] = np.sin(2 * np.pi * (train_df["month"] - 1) / 12)
train_df["cos_month"] = np.cos(2 * np.pi * (train_df["month"] - 1) / 12)
test_df["sin_month"] = np.sin(2 * np.pi * (test_df["month"] - 1) / 12)
test_df["cos_month"] = np.cos(2 * np.pi * (test_df["month"] - 1) / 12)

train_df["sin_dayofweek"] = np.sin(2 * np.pi * train_df["dayofweek"] / 7)
train_df["cos_dayofweek"] = np.cos(2 * np.pi * train_df["dayofweek"] / 7)
test_df["sin_dayofweek"] = np.sin(2 * np.pi * test_df["dayofweek"] / 7)
test_df["cos_dayofweek"] = np.cos(2 * np.pi * test_df["dayofweek"] / 7)



## === cell 8
feat_cols_s = [
    "distance",
    "log_distance",
    "distance_per_passenger",
    "passenger_count",
    "hour",
    "month",
    "dayofweek",
    "year",
    "is_weekend",
    "sin_hour",
    "cos_hour",
    "sin_month",
    "cos_month",
    "sin_dayofweek",
    "cos_dayofweek",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
X = train_df[feat_cols_s].astype(np.float32)
y = np.log1p(train_df["fare_amount"]).astype(np.float32)



## === cell 9
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)




## === cell 10
def XGBoost(X_tr, X_va, y_tr, y_va, num_rounds=4000):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 10,
        "eta": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "lambda": 1.0,
        "tree_method": "hist",
        "max_bin": 256,
        "nthread": 4,
        "verbosity": 0,
    }
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dval, "validation")],
        early_stopping_rounds=150,  # longer patience
        verbose_eval=False,
    )
    return model




## === cell 11
xgbm = XGBoost(X_train, X_val, y_train, y_val)

del X_train, X_val, y_train, y_val, X, y, train_df
import gc

gc.collect()



## === cell 12
if hasattr(xgbm, "best_iteration") and xgbm.best_iteration is not None:
    best_iter = int(xgbm.best_iteration) + 1
    pred_log = xgbm.predict(
        xgb.DMatrix(test_df[feat_cols_s].astype(np.float32)),
        iteration_range=(0, best_iter),
    )
else:
    pred_log = xgbm.predict(xgb.DMatrix(test_df[feat_cols_s].astype(np.float32)))
xgbm_pred = np.expm1(pred_log)
xgbm_pred = np.clip(xgbm_pred, a_min=0, a_max=None)



## === cell 13
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": xgbm_pred})
submission.to_csv("submission.csv", index=False)
print(
    "Submission written to submission.csv with shape:",
    submission.shape,
)



## === cell 14
importance = xgbm.get_score(importance_type="weight")
importance = sorted(importance.items(), key=operator.itemgetter(1))
df_imp = pd.DataFrame(importance, columns=["feature", "score"])
plt.figure(figsize=(10, 6))
df_imp.plot(kind="barh", x="feature", y="score", legend=False)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
