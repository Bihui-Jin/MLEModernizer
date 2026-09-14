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

3.82404

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd, polars as pl

train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

train_df = pl.read_csv(train_path)
test_df = pl.read_csv(test_path)

np.random.seed(0)
train_df = train_df.sample(n=2_000_000, seed=0)
train_df = train_df.drop_nulls()

test_key = test_df["key"].to_pandas()

train = train_df
test = test_df




## === cell 1
def preprocess(df):
    df = df.with_columns(
        pl.col("pickup_datetime")
        .str.replace_all(" UTC", "")
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)  # fixed signature
        .alias("pickup_datetime")
    )
    df = df.with_columns(
        [
            pl.col("pickup_datetime").dt.year().alias("pickup_year"),
            pl.col("pickup_datetime").dt.month().alias("pickup_month"),
            pl.col("pickup_datetime").dt.day().alias("pickup_day"),
            pl.col("pickup_datetime").dt.hour().alias("pickup_hour"),
            pl.col("pickup_datetime").dt.minute().alias("pickup_minute"),
            pl.col("pickup_datetime").dt.second().alias("pickup_second"),
            pl.col("pickup_datetime").dt.weekday().alias("pickup_weekday"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("pickup_longitude") - pl.col("dropoff_longitude"))
            .abs()
            .alias("abs_longitude"),
            (pl.col("pickup_latitude") - pl.col("dropoff_latitude"))
            .abs()
            .alias("abs_latitude"),
        ]
    )
    return df


def distance(df):
    longitude = 85.393
    latitude = 111.034
    return df.with_columns(
        (pl.col("abs_longitude") * longitude + pl.col("abs_latitude") * latitude).alias(
            "distance"
        )
    )


def haversine_distance(df):
    R = 6371.0  # km

    def calc(row):
        lat1 = row["pickup_latitude"] * np.pi / 180.0
        lat2 = row["dropoff_latitude"] * np.pi / 180.0
        dlat = (row["dropoff_latitude"] - row["pickup_latitude"]) * np.pi / 180.0
        dlon = (row["dropoff_longitude"] - row["pickup_longitude"]) * np.pi / 180.0
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    return df.with_columns(
        pl.struct(
            [
                pl.col("pickup_latitude"),
                pl.col("dropoff_latitude"),
                pl.col("pickup_longitude"),
                pl.col("dropoff_longitude"),
            ]
        )
        .apply(calc)
        .alias("haversine_distance")
    )


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
    left_lat, right_lat = 40.737676, 40.791716
    left_lon, right_lon = -74.015781, -73.945521
    df = df.with_columns(
        (
            (pl.col("pickup_latitude") >= left_lat)
            & (pl.col("pickup_latitude") <= right_lat)
            & (pl.col("pickup_longitude") >= left_lon)
            & (pl.col("pickup_longitude") <= right_lon)
        )
        .cast(pl.Int8)
        .alias("is_pickup_central")
    )
    df = df.with_columns(
        (
            (pl.col("dropoff_latitude") >= left_lat)
            & (pl.col("dropoff_latitude") <= right_lat)
            & (pl.col("dropoff_longitude") >= left_lon)
            & (pl.col("dropoff_longitude") <= right_lon)
        )
        .cast(pl.Int8)
        .alias("is_dropoff_central")
    )
    return df


def drop_outliner(df):
    return (
        df.filter(pl.col("pickup_latitude") >= 40)
        .filter(pl.col("pickup_latitude") < 41)
        .filter(pl.col("pickup_longitude") >= -74)
        .filter(pl.col("pickup_longitude") < -73)
        .filter(pl.col("dropoff_longitude") >= -74)
        .filter(pl.col("dropoff_longitude") < -73)
        .filter(pl.col("dropoff_latitude") >= 40)
        .filter(pl.col("dropoff_latitude") < 41)
        .filter(pl.col("passenger_count") >= 1)
        .filter(pl.col("passenger_count") < 6)
        .filter(pl.col("fare_amount") < 10000)
    )




## === cell 2
train = preprocess(train)
train = distance(train)
train = haversine_distance(train)
train = drop_encoding(train)
train = cycling_encoding(train)
train = one_hot_encoding(train)
train = is_central(train)
train = drop_outliner(train)

test = preprocess(test)
test = distance(test)
test = haversine_distance(test)
test = drop_encoding(test)
test = cycling_encoding(test)
test = one_hot_encoding(test)
test = is_central(test)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1519820777.py in <cell line: 0>()
      1 train = preprocess(train)
      2 train = distance(train)
----> 3 train = haversine_distance(train)
      4 train = drop_encoding(train)
      5 train = cycling_encoding(train)

/tmp/ipykernel_55/3460744737.py in haversine_distance(df)
     61             ]
     62         )
---> 63         .apply(calc)
     64         .alias("haversine_distance")
     65     )

AttributeError: 'Expr' object has no attribute 'apply'

## === cell 3
train_features = train.drop(["fare_amount"]).columns
missing_in_test = set(train_features) - set(test.columns)
for col in missing_in_test:
    test = test.with_columns(pl.lit(0.0).cast(pl.Float32).alias(col))

test = test.select(train_features)



## === cell 4
float64_cols_train = [c for c, t in zip(train.columns, train.dtypes) if t == pl.Float64]
if float64_cols_train:
    train = train.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols_train])

float64_cols_test = [c for c, t in zip(test.columns, test.dtypes) if t == pl.Float64]
if float64_cols_test:
    test = test.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols_test])



## === cell 5
import warnings, lightgbm as lgb, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

warnings.simplefilter("ignore")

X = train.drop(["fare_amount"]).to_pandas()
y = train["fare_amount"].to_pandas()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

model = lgb.LGBMRegressor(
    objective="regression",
    metric="rmse",
    device="cpu",
    boosting_type="gbdt",
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    n_estimators=5000,
    verbose=-1,
)

model.fit(X_train, y_train)

y_pred = model.predict(
    X_val,
    num_iteration=getattr(model, "best_iteration_", None) or model.n_estimators,
)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"rmse:{rmse}")

bst = model  # keep name used later
model_name = "lgbm"



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3552308443.py in <cell line: 0>()
     25 )
     26 
---> 27 model.fit(X_train, y_train)
     28 
     29 y_pred = model.predict(

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in fit(self, X, y, sample_weight, init_score, eval_set, eval_names, eval_sample_weight, eval_init_score, eval_metric, feature_name, categorical_feature, callbacks, init_model)
   1396     ) -> "LGBMRegressor":
   1397         """Docstring is inherited from the LGBMModel."""
-> 1398         super().fit(
   1399             X,
   1400             y,

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in fit(self, X, y, sample_weight, init_score, group, eval_set, eval_names, eval_sample_weight, eval_class_weight, eval_init_score, eval_group, eval_metric, feature_name, categorical_feature, callbacks, init_model)
   1047         callbacks.append(record_evaluation(evals_result))
   1048 
-> 1049         self._Booster = train(
   1050             params=params,
   1051             train_set=train_set,

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
   2121             categorical_feature = reference.categorical_feature
   2122         if isinstance(data, pd_DataFrame):
-> 2123             data, feature_name, categorical_feature, self.pandas_categorical = _data_from_pandas(
   2124                 data=data,
   2125                 feature_name=feature_name,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _data_from_pandas(data, feature_name, categorical_feature, pandas_categorical)
    866 
    867     return (
--> 868         _pandas_to_numpy(data, target_dtype=target_dtype),
    869         feature_name,
    870         categorical_feature,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _pandas_to_numpy(data, target_dtype)
    812     target_dtype: "np.typing.DTypeLike",
    813 ) -> np.ndarray:
--> 814     _check_for_bad_pandas_dtypes(data.dtypes)
    815     try:
    816         # most common case (no nullable dtypes)

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _check_for_bad_pandas_dtypes(pandas_dtypes_series)
    803     ]
    804     if bad_pandas_dtypes:
--> 805         raise ValueError(
    806             f"pandas dtypes must be int, float or bool.\nFields with bad pandas dtypes: {', '.join(bad_pandas_dtypes)}"
    807         )

ValueError: pandas dtypes must be int, float or bool.
Fields with bad pandas dtypes: key: object, pickup_datetime: datetime64[us]

## === cell 6
test_pd = test.to_pandas()
sub_pred = bst.predict(
    test_pd,
    num_iteration=getattr(bst, "best_iteration_", None) or bst.n_estimators,
)
submission = pd.DataFrame({"key": test_key, "fare_amount": sub_pred})
submission_path = f"submission_{model_name}_distance.csv"
submission.to_csv(submission_path, index=False)
print(f"submission written to {submission_path}")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1335477055.py in <cell line: 0>()
      1 test_pd = test.to_pandas()
----> 2 sub_pred = bst.predict(
      3     test_pd,
      4     num_iteration=getattr(bst, "best_iteration_", None) or bst.n_estimators,
      5 )

NameError: name 'bst' is not defined

## === cell 7
import matplotlib.pyplot as plt

feature_importance_df = pd.DataFrame(
    {"feature": X.columns, "importance": bst.feature_importance()}
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
/tmp/ipykernel_55/2586354983.py in <cell line: 0>()
      2 
      3 feature_importance_df = pd.DataFrame(
----> 4     {"feature": X.columns, "importance": bst.feature_importance()}
      5 )
      6 feature_importance_df = feature_importance_df.sort_values(

NameError: name 'bst' is not defined
