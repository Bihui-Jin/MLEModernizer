# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

cudf-polars-cu12==25.6.0
geopandas==0.14.4
lightgbm==4.6.0
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
polars==1.25.0
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

# 5. Target score

3.76015

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.50243) has done: 'The fix updates LightGBM training to use the correct callback syntax for early stopping, which removes the `TypeError` and allows the model (`bst`) to be created. This enables downstream prediction, submission file generation, and feature‑importance plotting. No other logic is changed, preserving the original feature engineering and model configuration while keeping the pipeline functional.'
- What this solution (achieved 5.61173) has done: 'I add a simple rule‑based feature (`calculated_value`) to both train and test, and modestly increase model capacity (more leaves and a slightly higher learning rate) while keeping the rest of the pipeline unchanged. These changes should improve the predictive power and move the RMSE closer to the target without altering the core logic.'
- What this solution (achieved 5.52834) has done: 'I add a simple interaction feature (`passenger_hav`) that multiplies passenger count by the haversine distance, and I modestly increase model capacity (more leaves, a slightly lower learning rate, and more boosting rounds) while keeping early stopping. These changes keep the original pipeline intact but give the model a richer signal, which should lower the RMSE toward the target.'
- What this solution (achieved 5.52877) has done: 'I modestly adjust the LightGBM training configuration to allow the model more capacity and a longer early‑stopping window, which should lower the validation RMSE and move the score closer to the target. The core feature engineering and data handling remain unchanged.'
- What this solution (achieved 5.60724) has done: 'We drop the low‑quality engineered columns `distance` and `calculated_value` that are superseded by the haversine features added later. Removing these noisy features should lower the validation RMSE, moving the score toward the target while keeping the rest of the pipeline unchanged. The change is limited to the preprocessing cell so the core model logic remains intact.'
- What this solution (achieved 5.51898) has done: 'I fixed the column‑not‑found error by reordering the preprocessing steps so the `distance` column is still present when `is_short_distance` is computed, and I stopped dropping the useful `distance` and `calculated_value` features. I also changed LightGBM to run on CPU (ensuring it works in the environment) which keeps the core model unchanged while allowing the script to execute successfully. These minimal adjustments keep the original logic intact and should lower the RMSE toward the target.'

# 9. Code solution

## === cell 0
import polars as pl
import numpy as np

train_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")
train_df = train_df.sample(n=3000000, seed=42)  # keep reduced size for speed
test_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")

np.random.seed(42)

train_df = train_df.drop_nulls()  # remove rows with any nulls before preprocessing




## === cell 1
train = train_df
test = test_df




## === cell 2
import numpy as np
import polars as pl


def preprocess(df):
    df = df.with_columns(
        pl.col("pickup_datetime")
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S UTC")
        .alias("pickup_datetime")
    )
    df = df.with_columns(
        [
            df["pickup_datetime"].dt.year().alias("pickup_year"),
            df["pickup_datetime"].dt.month().alias("pickup_month"),
            df["pickup_datetime"].dt.day().alias("pickup_day"),
            df["pickup_datetime"].dt.hour().alias("pickup_hour"),
            df["pickup_datetime"].dt.minute().alias("pickup_minute"),
            df["pickup_datetime"].dt.second().alias("pickup_second"),
            df["pickup_datetime"].dt.weekday().alias("pickup_weekday"),
        ]
    )
    df = df.with_columns(
        [
            (df["pickup_longitude"] - df["dropoff_longitude"])
            .abs()
            .alias("abs_longitude"),
            (df["pickup_latitude"] - df["dropoff_latitude"])
            .abs()
            .alias("abs_latitude"),
        ]
    )
    return df


def distance(df):
    longitude = 85.393
    latitude = 111.034
    df = df.with_columns(
        (df["abs_longitude"] * longitude + df["abs_latitude"] * latitude).alias(
            "distance"
        )
    )
    return df


def drop_encoding(df):
    drop_columns = ["key", "pickup_datetime", "pickup_minute", "pickup_second"]
    df = df.drop(drop_columns)
    return df


def cycling_encoding(df):
    df = df.with_columns(
        [
            (pl.col("pickup_month") * 2 * np.pi / 12).sin().alias("pickup_month_sin"),
            (pl.col("pickup_month") * 2 * np.pi / 12).cos().alias("pickup_month_cos"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).sin().alias("pickup_hour_sin"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).cos().alias("pickup_hour_cos"),
        ]
    )
    return df


def one_hot_encoding(df):
    one_hot_cols = ["pickup_year", "pickup_day"]
    return df.to_dummies(one_hot_cols)


def max_min_scaling(df):
    return df.filter(
        (pl.col("pickup_longitude") >= 70) & (pl.col("pickup_longitude") <= 90)
    )


def is_central(df):
    left_latitude = 40.737676
    right_latitude = 40.791716
    left_longitude = -74.015781
    right_longitude = -73.945521
    df = df.with_columns(
        (
            (df["pickup_latitude"] >= left_latitude)
            & (df["pickup_latitude"] <= right_latitude)
            & (df["pickup_longitude"] >= left_longitude)
            & (df["pickup_longitude"] <= right_longitude)
        )
        .cast(pl.Int8)
        .alias("is_pickup_central")
    )
    df = df.with_columns(
        (
            (df["dropoff_latitude"] >= left_latitude)
            & (df["dropoff_latitude"] <= right_latitude)
            & (df["dropoff_longitude"] >= left_longitude)
            & (df["dropoff_longitude"] <= right_longitude)
        )
        .cast(pl.Int8)
        .alias("is_dropoff_central")
    )
    return df


def diff_central(df):
    base_longitude = 85.393
    base_latitude = 111.034
    central_latitude = 40.764696
    central_longitude = -73.98065
    df = df.with_columns(
        (df["pickup_longitude"] - central_longitude)
        .abs()
        .alias("central_pickup_abs_longitude"),
        (df["pickup_latitude"] - central_latitude)
        .abs()
        .alias("central_pickup_abs_latitude"),
    )
    df = df.with_columns(
        (
            df["central_pickup_abs_longitude"] * base_longitude
            + df["central_pickup_abs_latitude"] * base_latitude
        ).alias("central_pickup_distance")
    )
    df = df.drop(["central_pickup_abs_longitude", "central_pickup_abs_latitude"])
    df = df.with_columns(
        (df["dropoff_longitude"] - central_longitude)
        .abs()
        .alias("central_dropoff_abs_longitude"),
        (df["dropoff_latitude"] - central_latitude)
        .abs()
        .alias("central_dropoff_abs_latitude"),
    )
    df = df.with_columns(
        (
            df["central_dropoff_abs_longitude"] * base_longitude
            + df["central_dropoff_abs_latitude"] * base_latitude
        ).alias("central_dropoff_distance")
    )
    df = df.drop(["central_dropoff_abs_longitude", "central_dropoff_abs_latitude"])
    return df


def is_short_distance(df):
    return df.with_columns(
        (df["distance"] < 0.32).cast(pl.Int8).alias("is_short_distance")
    )


def drop_outliner(df):
    df = df.filter((pl.col("pickup_latitude") >= 40) & (pl.col("pickup_latitude") < 41))
    df = df.filter(
        (pl.col("pickup_longitude") >= -74) & (pl.col("pickup_longitude") < -73)
    )
    df = df.filter(
        (pl.col("dropoff_longitude") >= -74) & (pl.col("dropoff_longitude") < -73)
    )
    df = df.filter(
        (pl.col("dropoff_latitude") >= 40) & (pl.col("dropoff_latitude") < 41)
    )
    df = df.filter((pl.col("passenger_count") >= 1) & (pl.col("passenger_count") < 6))
    df = df.filter(pl.col("fare_amount") < 10000)
    return df


def rule_base(df):
    is_weekday = df["pickup_weekday"].is_in([0, 1, 2, 3, 4])
    is_weekend = df["pickup_weekday"].is_in([5, 6])
    df = df.with_columns(
        pl.when(is_weekday & df["pickup_hour"].is_in([16, 17, 18, 19, 20]))
        .then(9 * df["distance"] / 0.32)
        .when(
            is_weekday & df["pickup_hour"].is_in([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6])
        )
        .then(df["distance"] / 0.32)
        .when(is_weekday)
        .then(0.5 * df["distance"] / 0.32)
        .when(
            is_weekend & df["pickup_hour"].is_in([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6])
        )
        .then(60 * df["distance"] / 19.2)
        .otherwise(30 * df["distance"] / 19.2)
        .alias("calculated_value")
    )
    return df


train = preprocess(train)
train = distance(train)
train = rule_base(train)
train = is_short_distance(train)
train = drop_encoding(train)
train = cycling_encoding(train)
train = one_hot_encoding(train)
train = is_central(train)

test = preprocess(test)
test_key = test["key"]
test = distance(test)
test = rule_base(test)
test = is_short_distance(test)
test = drop_encoding(test)
test = cycling_encoding(test)
test = one_hot_encoding(test)
test = is_central(test)

train = drop_outliner(train)

train = train.fill_null(0)
test = test.fill_null(0)




## === cell 3
import polars as pld
import pandas as pd

float64_cols = [
    col for col, dtype in zip(train.columns, train.dtypes) if dtype == pl.Float64
]
train = train.with_columns([pl.col(col).cast(pl.Float32) for col in float64_cols])

float64_cols_test = [
    col for col, dtype in zip(test.columns, test.dtypes) if dtype == pl.Float64
]
test = test.with_columns([pl.col(col).cast(pl.Float32) for col in float64_cols_test])




## === cell 4
import warnings

warnings.simplefilter("ignore")
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = train.drop(["fare_amount"]).to_pandas()
y = train["fare_amount"].to_pandas()
test_pd = test.to_pandas()


def add_haversine(df):
    lat1 = np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    dlat = lat2 - lat1
    dlon = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["haversine_distance"] = 6371.0 * c
    df["passenger_hav"] = df["haversine_distance"] * df["passenger_count"]
    return df


X = add_haversine(X)
test_pd = add_haversine(test_pd)

X = X.fillna(0)
test_pd = test_pd.fillna(0)

y_log = np.log1p(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_log, test_size=0.2, random_state=0
)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "regression",
    "metric": "rmse",
    "device": "cpu",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,  # slightly higher to speed learning
    "num_leaves": 2047,  # more capacity
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "lambda_l2": 0.1,
    "max_bin": 63,
    "verbose": -1,
}

bst = lgb.train(
    params,
    train_data,
    num_boost_round=5000,
    valid_sets=[valid_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=150, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

y_pred_log = bst.predict(X_val, num_iteration=bst.best_iteration)
y_pred = np.expm1(y_pred_log)
y_val_original = np.expm1(y_val)

rmse = np.sqrt(mean_squared_error(y_val_original, y_pred))
print(f"rmse:{rmse}")

model_name = "lgbm"




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3340066911.py in <cell line: 0>()
     72 y_val_original = np.expm1(y_val)
     73 
---> 74 rmse = np.sqrt(mean_squared_error(y_val_original, y_pred))
     75 print(f"rmse:{rmse}")
     76 

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_regression.py in mean_squared_error(y_true, y_pred, sample_weight, multioutput, squared)
    440     0.825...
    441     """
--> 442     y_type, y_true, y_pred, multioutput = _check_reg_targets(
    443         y_true, y_pred, multioutput
    444     )

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_regression.py in _check_reg_targets(y_true, y_pred, multioutput, dtype)
     99     """
    100     check_consistent_length(y_true, y_pred)
--> 101     y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
    102     y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)
    103 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    919 
    920         if force_all_finite:
--> 921             _assert_all_finite(
    922                 array,
    923                 input_name=input_name,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    159                 "#estimators-that-handle-nan-values"
    160             )
--> 161         raise ValueError(msg_err)
    162 
    163 

ValueError: Input contains NaN.

## === cell 5
sub_pred_log = bst.predict(test_pd, num_iteration=bst.best_iteration)
sub_pred = np.expm1(sub_pred_log)

exp_num = "123467"
submission = pd.DataFrame({"key": test_key.to_numpy(), "fare_amount": sub_pred})
submission.to_csv(f"submission_{model_name}_{exp_num}.csv", index=False)
print("Submission saved as:", f"submission_{model_name}_{exp_num}.csv")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2471281167.py in <cell line: 0>()
      4 exp_num = "123467"
      5 submission = pd.DataFrame({"key": test_key.to_numpy(), "fare_amount": sub_pred})
----> 6 submission.to_csv(f"submission_{model_name}_{exp_num}.csv", index=False)
      7 print("Submission saved as:", f"submission_{model_name}_{exp_num}.csv")
      8 

NameError: name 'model_name' is not defined

## === cell 6
import matplotlib.pyplot as plt

feature_importance_df = pd.DataFrame()
feature_importance_df["feature"] = X.columns
feature_importance_df["importance"] = bst.feature_importance()
feature_importance_df = feature_importance_df.sort_values(
    by="importance", ascending=False
).head(15)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["feature"], feature_importance_df["importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.show()
