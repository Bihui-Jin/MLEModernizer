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

3.84929

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 5.983) has done: 'I fix the LightGBM `train` call (remove the unsupported `early_stopping_rounds` argument and use the proper callback), change the device to CPU for safety, and convert the Polars test set (and its key column) to pandas before predicting. These changes resolve the runtime errors, allow the model to train and produce predictions, and ensure a correctly‑named CSV submission is written.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import polars as pl

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
train_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")
test_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")

np.random.seed(0)  # reproducibility
num_rows = train_df.height
sample_size = min(10_000_000, num_rows)  # reduced from 20 M
random_indices = np.random.choice(num_rows, size=sample_size, replace=False)
train_df = train_df[random_indices]

train_df = train_df.drop_nulls()




## === cell 2
train = train_df
test = test_df




## === cell 3
def preprocess(df):
    df = df.with_columns(
        pl.col("pickup_datetime")
        .str.replace_all(" UTC", "")
        .str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S.%f", strict=False)
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
    longitude = 85.393
    latitude = 111.034
    df = df.with_columns(
        (pl.col("abs_longitude") * longitude + pl.col("abs_latitude") * latitude).alias(
            "distance"
        )
    )
    return df


def drop_encoding(df):
    return df.drop(["key", "pickup_datetime", "pickup_minute", "pickup_second"])


def cycling_encoding(df):
    return df.with_columns(
        [
            (pl.col("pickup_month") * 2 * np.pi / 12).sin().alias("pickup_month_sin"),
            (pl.col("pickup_month") * 2 * np.pi / 12).cos().alias("pickup_month_cos"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).sin().alias("pickup_hour_sin"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).cos().alias("pickup_hour_cos"),
        ]
    )


def one_hot_encoding(df):
    return df.to_dummies(["pickup_year", "pickup_day"])


def is_central(df):
    left_latitude = 40.737676
    right_latitude = 40.791716
    left_longitude = -74.015781
    right_longitude = -73.945521
    df = df.with_columns(
        (
            (pl.col("pickup_latitude") >= left_latitude)
            & (pl.col("pickup_latitude") <= right_latitude)
            & (pl.col("pickup_longitude") >= left_longitude)
            & (pl.col("pickup_longitude") <= right_longitude)
        )
        .cast(pl.Int8)
        .alias("is_pickup_central")
    )
    df = df.with_columns(
        (
            (pl.col("dropoff_latitude") >= left_latitude)
            & (pl.col("dropoff_latitude") <= right_latitude)
            & (pl.col("dropoff_longitude") >= left_longitude)
            & (pl.col("dropoff_longitude") <= right_longitude)
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
        (
            (pl.col("pickup_longitude") - central_longitude).abs() * base_longitude
            + (pl.col("pickup_latitude") - central_latitude).abs() * base_latitude
        ).alias("central_pickup_distance")
    )
    df = df.with_columns(
        (
            (pl.col("dropoff_longitude") - central_longitude).abs() * base_longitude
            + (pl.col("dropoff_latitude") - central_latitude).abs() * base_latitude
        ).alias("central_dropoff_distance")
    )
    return df


def is_short_distance(df):
    return df.with_columns(
        (pl.col("distance") < 0.32).cast(pl.Int8).alias("is_short_distance")
    )


def drop_outliner(df):
    return df.filter(
        (pl.col("pickup_latitude") >= 40)
        & (pl.col("pickup_latitude") < 41)
        & (pl.col("pickup_longitude") >= -74)
        & (pl.col("pickup_longitude") < -73)
        & (pl.col("dropoff_longitude") >= -74)
        & (pl.col("dropoff_longitude") < -73)
        & (pl.col("dropoff_latitude") >= 40)
        & (pl.col("dropoff_latitude") < 41)
        & (pl.col("passenger_count") >= 1)
        & (pl.col("passenger_count") < 6)
        & (pl.col("fare_amount") < 10000)
    )


def rule_base(df):
    is_weekday = pl.col("pickup_weekday").is_in([0, 1, 2, 3, 4])
    is_weekend = pl.col("pickup_weekday").is_in([5, 6])
    df = df.with_columns(
        pl.when(is_weekday & pl.col("pickup_hour").is_in([16, 17, 18, 19, 20]))
        .then(9 * pl.col("distance") / 0.32)
        .when(
            is_weekday
            & pl.col("pickup_hour").is_in([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6])
        )
        .then(pl.col("distance") / 0.32)
        .when(is_weekday)
        .then(0.5 * pl.col("distance") / 0.32)
        .when(
            is_weekend
            & pl.col("pickup_hour").is_in([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6])
        )
        .then(60 * pl.col("distance") / 19.2)
        .otherwise(30 * pl.col("distance") / 19.2)
        .alias("calculated_value")
    )
    return df


train = preprocess(train)
train = drop_encoding(train)
train = cycling_encoding(train)
train = one_hot_encoding(train)
train = is_central(train)
train = diff_central(train)
train = drop_outliner(train)
train = is_short_distance(train)
train = rule_base(train)

test = preprocess(test)
test_key = test["key"]  # keep for submission
test = drop_encoding(test)
test = cycling_encoding(test)
test = one_hot_encoding(test)
test = is_central(test)
test = diff_central(test)
test = is_short_distance(test)
test = rule_base(test)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/137971751.py in <cell line: 0>()
    144 
    145 
--> 146 train = preprocess(train)
    147 train = drop_encoding(train)
    148 train = cycling_encoding(train)

/tmp/ipykernel_55/137971751.py in preprocess(df)
      4         pl.col("pickup_datetime")
      5         .str.replace_all(" UTC", "")
----> 6         .str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S.%f", strict=False)
      7         .alias("pickup_datetime"),
      8         pl.col("pickup_datetime").dt.year().alias("pickup_year"),

TypeError: ExprStringNameSpace.strptime() got an unexpected keyword argument 'fmt'

## === cell 4
float64_cols = [c for c, dt in zip(train.columns, train.dtypes) if dt == pl.Float64]
train = train.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols])

float64_cols_test = [c for c, dt in zip(test.columns, test.dtypes) if dt == pl.Float64]
test = test.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols_test])




## === cell 5
import warnings

warnings.simplefilter("ignore")
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_np = train.drop(["fare_amount"]).to_numpy()
y_np = train["fare_amount"].to_numpy()

X_train, X_val, y_train, y_val = train_test_split(
    X_np, y_np, test_size=0.2, random_state=0
)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "regression",
    "metric": "rmse",
    "device": "cpu",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 255,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "num_threads": -1,
}

bst = lgb.train(
    params,
    train_data,
    num_boost_round=5000,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
)

y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"rmse:{rmse}")

model_name = "lgbm"




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3696803182.py in <cell line: 0>()
     31 }
     32 
---> 33 bst = lgb.train(
     34     params,
     35     train_data,

/usr/local/lib/python3.11/dist-packages/lightgbm/engine.py in train(params, train_set, num_boost_round, valid_sets, valid_names, feval, init_model, keep_training_booster, callbacks)
    295     # construct booster
    296     try:
--> 297         booster = Booster(params=params, train_set=train_set)
    298         if is_valid_contain_train:
    299             booster.set_train_data_name(train_data_name)

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in __init__(self, params, train_set, model_file, model_str)
   3654                 )
   3655             # construct booster object
-> 3656             train_set.construct()
   3657             # copy the parameters from train_set
   3658             params.update(train_set.get_params())

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in construct(self)
   2588             else:
   2589                 # create train
-> 2590                 self._lazy_init(
   2591                     data=self.data,
   2592                     label=self.label,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _lazy_init(self, data, label, reference, weight, group, init_score, predictor, feature_name, categorical_feature, params, position)
   2185             self.__init_from_csc(data, params_str, ref_dataset)
   2186         elif isinstance(data, np.ndarray):
-> 2187             self.__init_from_np2d(data, params_str, ref_dataset)
   2188         elif _is_pyarrow_table(data):
   2189             self.__init_from_pyarrow_table(data, params_str, ref_dataset)

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in __init_from_np2d(self, mat, params_str, ref_dataset)
   2314 
   2315         self._handle = ctypes.c_void_p()
-> 2316         data, layout = _np2d_to_np1d(mat)
   2317         ptr_data, type_ptr_data, _ = _c_float_array(data)
   2318         _safe_call(

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _np2d_to_np1d(mat)
    202         layout = _C_API_IS_ROW_MAJOR
    203     # ensure dtype and order, copies if either do not match
--> 204     data = np.asarray(mat, dtype=dtype, order=order)
    205     # flatten array without copying
    206     return data.ravel(order=order), layout

ValueError: could not convert string to float: '2012-10-28 01:29:53.0000004'

## === cell 6
test_np = test.to_numpy()
sub_pred = bst.predict(test_np, num_iteration=bst.best_iteration)

test_key_pd = test_key.to_pandas()
submission = pd.DataFrame({"key": test_key_pd, "fare_amount": sub_pred})
submission_file = f"submission_{model_name}_rule_base.csv"
submission.to_csv(submission_file, index=False)
print(f"Submission saved to {submission_file}")




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2169577830.py in <cell line: 0>()
      1 test_np = test.to_numpy()
----> 2 sub_pred = bst.predict(test_np, num_iteration=bst.best_iteration)
      3 
      4 test_key_pd = test_key.to_pandas()
      5 submission = pd.DataFrame({"key": test_key_pd, "fare_amount": sub_pred})

NameError: name 'bst' is not defined

## === cell 7
import matplotlib.pyplot as plt

feature_names = train.drop(["fare_amount"]).columns
feature_importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": bst.feature_importance(importance_type="gain"),
    }
)
feature_importance_df = feature_importance_df.sort_values(
    by="importance", ascending=False
).head(15)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["feature"], feature_importance_df["importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Feature Importances")
plt.gca().invert_yaxis()
plt.show()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/474290918.py in <cell line: 0>()
      5     {
      6         "feature": feature_names,
----> 7         "importance": bst.feature_importance(importance_type="gain"),
      8     }
      9 )

NameError: name 'bst' is not defined
