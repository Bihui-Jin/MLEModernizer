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

3.76365

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.77851) has done: 'The fix adds proper LightGBM early‑stopping via callbacks (instead of the unsupported `early_stopping_rounds` argument) and ensures the test set is converted to pandas with the same column order as the training data before prediction. This resolves the runtime errors, guarantees a valid CSV submission, and keeps the original model logic unchanged.'
- What this solution (achieved 6.42287) has done: 'The changes reduce the amount of data loaded and processed by reading a fixed smaller subset of the training file (5 million rows) instead of the full 20 million random sample. This cuts I/O, memory usage, and LightGBM training time while keeping all preprocessing, feature engineering, and model logic unchanged. The rest of the pipeline (feature creation, casting, training, and submission) remains identical, preserving deterministic behavior and result semantics.'
- What this solution (achieved 5.42437) has done: 'Implemented a NaN‑filtering step before the train/validation split to prevent `mean_squared_error` from receiving invalid values, which resolves the runtime error in cell 2 and ensures `model_name` is defined for the submission step. No core logic was altered, preserving the original feature engineering and LightGBM configuration, while guaranteeing a valid CSV submission file is written.'
- What this solution (achieved 6.26862) has done: 'Implemented a switch to train the LightGBM model on the original fare amount instead of the log‑transformed target. This removes the `log1p`/`expm1` conversions, allowing the model to optimize directly for RMSE on the natural scale, which should bring the validation error closer to the target value. Updated variable names and the model identifier to reflect the change, while preserving all preprocessing, feature engineering, and training hyper‑parameters.'

# 9. Code solution

## === cell 0
import polars as pl
import numpy as np

sample_size = 8_000_000  # was 5_000_000

train = pl.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    n_rows=sample_size,
).lazy()
test = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv").lazy()

train = train.drop_nulls()


def preprocess(df):
    return df.with_columns(
        pl.col("pickup_datetime")
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S UTC")
        .alias("pickup_datetime"),
        pl.col("pickup_datetime").dt.year().alias("pickup_year"),
        pl.col("pickup_datetime").dt.month().alias("pickup_month"),
        pl.col("pickup_datetime").dt.day().alias("pickup_day"),
        pl.col("pickup_datetime").dt.hour().alias("pickup_hour"),
        pl.col("pickup_datetime").dt.minute().alias("pickup_minute"),
        pl.col("pickup_datetime").dt.second().alias("pickup_second"),
        pl.col("pickup_datetime").dt.weekday().alias("pickup_weekday"),
        (pl.col("pickup_longitude") - pl.col("dropoff_longitude"))
        .abs()
        .alias("abs_longitude"),
        (pl.col("pickup_latitude") - pl.col("dropoff_latitude"))
        .abs()
        .alias("abs_latitude"),
    )


def distance(df):
    longitude = 85.393
    latitude = 111.034
    return df.with_columns(
        (
            (pl.col("abs_longitude") * longitude) + (pl.col("abs_latitude") * latitude)
        ).alias("distance")
    )


def haversine_distance_df(df):
    """Add haversine distance to a *collected* Polars DataFrame."""
    R = 6371.0  # Earth radius in kilometres
    lon1 = np.radians(df["pickup_longitude"].to_numpy())
    lat1 = np.radians(df["pickup_latitude"].to_numpy())
    lon2 = np.radians(df["dropoff_longitude"].to_numpy())
    lat2 = np.radians(df["dropoff_latitude"].to_numpy())

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    return df.with_columns(pl.Series("haversine_distance", distance))


def drop_encoding(df):
    return df.drop(["key", "pickup_datetime", "pickup_minute", "pickup_second"])


def cycling_encoding(df):
    return df.with_columns(
        (pl.col("pickup_month") * 2 * np.pi / 12).sin().alias("pickup_month_sin"),
        (pl.col("pickup_month") * 2 * np.pi / 12).cos().alias("pickup_month_cos"),
        (pl.col("pickup_hour") * 2 * np.pi / 24).sin().alias("pickup_hour_sin"),
        (pl.col("pickup_hour") * 2 * np.pi / 24).cos().alias("pickup_hour_cos"),
    )


def one_hot_encoding(df):
    return df.to_dummies(["pickup_year", "pickup_day"])


def max_min_scaling(df):
    return df.filter(
        (pl.col("pickup_longitude") >= 70) & (pl.col("pickup_longitude") <= 90)
    )


def is_central(df):
    left_lat, right_lat = 40.737676, 40.791716
    left_lon, right_lon = -74.015781, -73.945521
    return df.with_columns(
        (
            (pl.col("pickup_latitude").is_between(left_lat, right_lat))
            & (pl.col("pickup_longitude").is_between(left_lon, right_lon))
        )
        .cast(pl.Int8)
        .alias("is_pickup_central"),
        (
            (pl.col("dropoff_latitude").is_between(left_lat, right_lat))
            & (pl.col("dropoff_longitude").is_between(left_lon, right_lon))
        )
        .cast(pl.Int8)
        .alias("is_dropoff_central"),
    )


def diff_central(df):
    base_longitude = 85.393
    base_latitude = 111.034
    central_latitude = 40.764696
    central_longitude = -73.98065
    df = df.with_columns(
        (pl.col("pickup_longitude") - central_longitude)
        .abs()
        .alias("central_pickup_abs_longitude"),
        (pl.col("pickup_latitude") - central_latitude)
        .abs()
        .alias("central_pickup_abs_latitude"),
    )
    df = df.with_columns(
        (
            pl.col("central_pickup_abs_longitude") * base_longitude
            + pl.col("central_pickup_abs_latitude") * base_latitude
        ).alias("central_pickup_distance")
    )
    df = df.drop(["central_pickup_abs_longitude", "central_pickup_abs_latitude"])
    df = df.with_columns(
        (pl.col("dropoff_longitude") - central_longitude)
        .abs()
        .alias("central_dropoff_abs_longitude"),
        (pl.col("dropoff_latitude") - central_latitude)
        .abs()
        .alias("central_dropoff_abs_latitude"),
    )
    df = df.with_columns(
        (
            pl.col("central_dropoff_abs_longitude") * base_longitude
            + pl.col("central_dropoff_abs_latitude") * base_latitude
        ).alias("central_dropoff_distance")
    )
    return df.drop(["central_dropoff_abs_longitude", "central_dropoff_abs_latitude"])


def is_short_distance(df):
    return df.with_columns(
        (pl.col("distance") < 0.32).cast(pl.Int8).alias("is_short_distance")
    )


def drop_outliner(df):
    return (
        df.filter(pl.col("pickup_latitude").is_between(40, 41))
        .filter(pl.col("pickup_longitude").is_between(-74, -73))
        .filter(pl.col("dropoff_longitude").is_between(-74, -73))
        .filter(pl.col("dropoff_latitude").is_between(40, 41))
        .filter(pl.col("passenger_count").is_between(1, 6))
        .filter(pl.col("fare_amount") < 10000)
    )


def rule_base(df):
    is_weekday = df["pickup_weekday"].is_in([0, 1, 2, 3, 4])
    is_weekend = df["pickup_weekday"].is_in([5, 6])
    return df.with_columns(
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


train = preprocess(train)
train = distance(train)
train = drop_encoding(train)
train = cycling_encoding(train)
train = one_hot_encoding(train)
train = is_central(train)
train = drop_outliner(train)
train = is_short_distance(train)
train = rule_base(train)

test = preprocess(test)
test = distance(test)
test = drop_encoding(test)
test = cycling_encoding(test)
test = one_hot_encoding(test)
test = is_central(test)
test = is_short_distance(test)
test = rule_base(test)

train = train.collect()
test = test.collect()
test_key = test["key"].collect().to_series()

train = haversine_distance_df(train)
test = haversine_distance_df(test)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1912962829.py in <cell line: 0>()
    185 train = drop_encoding(train)
    186 train = cycling_encoding(train)
--> 187 train = one_hot_encoding(train)
    188 train = is_central(train)
    189 train = drop_outliner(train)

/tmp/ipykernel_55/1912962829.py in one_hot_encoding(df)
     77 
     78 def one_hot_encoding(df):
---> 79     return df.to_dummies(["pickup_year", "pickup_day"])
     80 
     81 

AttributeError: 'LazyFrame' object has no attribute 'to_dummies'

## === cell 1
def cast_numeric(df):
    float64_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Float64]
    df = df.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols])
    int64_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Int64]
    df = df.with_columns([pl.col(c).cast(pl.Int32) for c in int64_cols])
    return df


train = cast_numeric(train)
test = cast_numeric(test)




## === cell 2
import warnings

warnings.simplefilter("ignore")
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

y = train["fare_amount"].to_numpy()
X = train.drop("fare_amount").to_numpy()

mask = ~np.isnan(y)
X = X[mask]
y = y[mask]

y_log = np.log1p(y)

X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.2, random_state=0
)

train_data = lgb.Dataset(X_train, label=y_train_log)
val_data = lgb.Dataset(X_val, label=y_val_log, reference=train_data)

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 511,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

callbacks = [
    lgb.early_stopping(stopping_rounds=200, verbose=False),
    lgb.log_evaluation(period=0),
]

bst = lgb.train(
    params,
    train_data,
    num_boost_round=5000,
    valid_sets=[val_data],
    callbacks=callbacks,
)

y_val_pred_log = bst.predict(X_val, num_iteration=bst.best_iteration)
y_val_pred = np.expm1(y_val_pred_log)

rmse = np.sqrt(mean_squared_error(np.expm1(y_val_log), y_val_pred))
print(f"rmse:{rmse}")

model_name = "lgbm_log_target"




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2972190190.py in <cell line: 0>()
      6 from sklearn.metrics import mean_squared_error
      7 
----> 8 y = train["fare_amount"].to_numpy()
      9 X = train.drop("fare_amount").to_numpy()
     10 

/usr/local/lib/python3.11/dist-packages/polars/lazyframe/frame.py in __getitem__(self, item)
    687                 "\n\nUse `select()` or `filter()` instead."
    688             )
--> 689             raise TypeError(msg)
    690         return LazyPolarsSlice(self).apply(item)
    691 

TypeError: 'LazyFrame' object is not subscriptable (aside from slicing)

Use `select()` or `filter()` instead.

## === cell 3
import pandas as pd

feature_names = train.drop("fare_amount").columns
X_test = test.select(feature_names).to_numpy()

sub_pred_log = bst.predict(X_test)
sub_pred = np.expm1(sub_pred_log)

exp_num = "01"
submission = pd.DataFrame({"key": test_key.to_pandas(), "fare_amount": sub_pred})
submission_path = f"submission_{model_name}_{exp_num}.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/727513609.py in <cell line: 0>()
      2 
      3 feature_names = train.drop("fare_amount").columns
----> 4 X_test = test.select(feature_names).to_numpy()
      5 
      6 sub_pred_log = bst.predict(X_test)

AttributeError: 'LazyFrame' object has no attribute 'to_numpy'

## === cell 4
import matplotlib.pyplot as plt

feature_importance_df = pd.DataFrame(
    {"feature": feature_names, "importance": bst.feature_importance()}
)
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

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2309706907.py in <cell line: 0>()
      2 
      3 feature_importance_df = pd.DataFrame(
----> 4     {"feature": feature_names, "importance": bst.feature_importance()}
      5 )
      6 feature_importance_df = feature_importance_df.sort_values(

NameError: name 'bst' is not defined
