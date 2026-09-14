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

3.10

# 3. Installed packages

folium==0.20.0
geopandas==0.14.4
haversine==2.9.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

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

4.46748

# 6. Current score

7.59113

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 439.94995) has done: 'I fix the TensorFlow/protobuf import crash by switching all Keras/TensorFlow imports to the already-installed `tf_keras` package (this avoids the `MessageFactory.GetPrototype` error in this environment). Then I fix the feature mismatch causing the `KeyError` by creating `fare_to_dist_ratio` for the test set too (since it exists in train and is used in `feature_cols`). Finally, I keep the same model/training logic but ensure the pipeline runs end-to-end and always writes a valid `/kaggle/working/submission.csv` with columns `key,fare_amount`.'
- What this solution (achieved 558.74241) has done: 'I fix the protobuf/Keras crash by removing the standalone `keras` stack and using TensorFlow’s built-in `tf.keras` consistently, which is the most stable combination in this Kaggle environment. Then I address the extremely bad score by fixing the core feature logic bug: `fare_to_dist_ratio` currently leaks the label into training (making the model learn an unusable feature) and is set to a constant in test, causing a catastrophic train/test feature mismatch; I set it to the same deterministic, label-free value for both train and test (distance-based) while keeping the rest of the feature engineering and model/training loop unchanged. Finally, I ensure the script always writes `/kaggle/working/submission.csv` with the exact required columns `key,fare_amount`.'
- What this solution (achieved 607.09942) has done: 'I fix the immediate runtime crash caused by the TensorFlow/protobuf incompatibility by switching all Keras usage to the already-installed `tf_keras` package (which avoids the `MessageFactory.GetPrototype` error in this environment). Then I fix the main logic issue causing the huge RMSE: the model is currently training on raw longitude/latitude values that are on a very different scale than the engineered distance features, and with only 10 epochs it tends to underfit badly; I keep the exact same model and training loop but apply the already-fitted `StandardScaler` to all numeric features as intended (the code already does this, so we just ensure datetime/tuple columns never leak into the numeric matrix). Finally, I ensure the submission is always written as `/kaggle/working/submission.csv` with columns exactly `key,fare_amount` and aligned row-for-row with the test file.'
- What this solution (achieved 775.63797) has done: 'I fix the TensorFlow/protobuf crash by switching all Keras imports to `tf_keras`, which is already installed and avoids the `MessageFactory.GetPrototype` error in this environment. Then I fix the `haversine_km` implementation so it can accept scalar airport coordinates (currently it calls `.astype()` on Python floats) while keeping the same distance feature logic. Finally, I ensure train/test feature columns always match (including `fare_to_dist_ratio` and airport distance columns), so scaling/training runs and a valid `/kaggle/working/submission.csv` with `key,fare_amount` is always written.'
- What this solution (achieved 756.28) has done: 'I fix the current hard crash (`MessageFactory.GetPrototype`) by avoiding importing TensorFlow/Keras at module import time and instead using `tf_keras` only (and forcing the pure-Python protobuf implementation via an environment variable set before importing TF-related modules). Then I fix the main logic issue driving the extreme RMSE: the engineered `fare_to_dist_ratio` is currently the inverse distance (not fare-related anymore), which is OK, but the pipeline still accidentally includes non-numeric tuple features (`loc1`, `loc2`) in intermediate steps unless explicitly excluded; I make the drop/feature column selection stricter so only numeric model inputs are used consistently. Finally, I keep the same model/training loop and ensure a valid `/kaggle/working/submission.csv` with columns `key,fare_amount` is always written and aligned to the test rows.'
- What this solution (achieved 702.45722) has done: 'I fix the immediate crash (`MessageFactory.GetPrototype`) by avoiding `tf_keras` imports and using the stable `tensorflow.keras` API after forcing the pure-Python protobuf implementation before importing TensorFlow. Then I keep the same model/training loop and feature engineering, but remove the non-essential plotting/folium cells that can waste time or fail in the Kaggle runtime (this is score-neutral and improves stability under the 600s limit). Finally, I keep the exact same submission writing logic but add a small alignment/sanity check so `key` and predictions always match row-for-row and the output is guaranteed to be a valid `submission.csv`.'
- What this solution (achieved 672.45667) has done: 'I fix the immediate TensorFlow/protobuf crash by switching all model-related imports from `tensorflow.keras` to the already-installed `tf_keras`, which is the stable option in this Kaggle environment. Then I make the TensorFlow import order deterministic (set protobuf env vars before importing TF stacks) to prevent the `MessageFactory.GetPrototype` error. Finally, I keep the same feature engineering, scaling, model architecture, and training loop, and ensure the script always writes a valid `/kaggle/working/submission.csv` with columns `key,fare_amount`.'
- What this solution (achieved 808.55128) has done: 'I fix the immediate crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by ensuring TensorFlow is not imported at all (it’s not actually required for your `tf_keras` training code), and by enforcing the protobuf environment variables before any protobuf/TensorFlow-related imports. Then I keep your exact feature engineering, scaling, model architecture, and training loop unchanged, so the evaluation semantics remain the same while the notebook runs end-to-end. Finally, I make sure the submission file is always written as `/kaggle/working/submission.csv` with the required columns `key,fare_amount` and correctly aligned to the test rows.'
- What this solution (achieved 10.10503) has done: 'I fix the runtime crash happening before any training by removing the `tf_keras` dependency (it is still triggering the protobuf `MessageFactory.GetPrototype` issue in this environment) and using scikit-learn’s `HistGradientBoostingRegressor` instead so the notebook runs end-to-end reliably. I keep all your existing feature engineering, cleaning, scaling, and train/validation split logic unchanged, and only swap the model/train/predict blocks to a CPU-stable regressor that matches the RMSE objective. This should drastically reduce the RMSE from the current catastrophic level by producing non-degenerate predictions without changing the data pipeline semantics. Finally, I ensure `/kaggle/working/submission.csv` is always written with the exact required columns `key,fare_amount` aligned to the test rows.'
- What this solution (achieved 7.74742) has done: 'Your current RMSE (10.105) is worse than the target (4.467), so we should improve it with the smallest, safest changes that don’t alter the overall pipeline structure. The biggest low-risk gain here is to make the model more robust by (1) using a stronger-but-still-same-family HistGradientBoosting configuration (more trees, smaller learning rate, leaf control) and (2) adding the standard `log1p` / `expm1` target transform, which is a common way to reduce the impact of large-fare outliers under RMSE without changing the core training approach. I also remove the (unnecessary and potentially harmful) `StandardScaler` for tree models, since scaling provides no benefit for histogram-based trees and can slightly hurt due to altered binning; everything else (data loading, cleaning, feature engineering, split, and submission writing) stays the same. These minimal changes are aimed at moving RMSE downward toward the target band without introducing new dependencies or changing the feature set.'
- What this solution (achieved 6.54909) has done: 'Your current RMSE (7.747) is still substantially worse than the target (4.467), so we should improve it with the smallest changes that directly affect generalization under RMSE without changing your overall approach (same features, same train/val split, same HistGradientBoostingRegressor). The biggest low-risk issue is that the model is being trained on a small 100k slice without any handling of obvious heavy-tailed target noise; we keep your `log1p/expm1` setup but add a tiny amount of robustification by setting a modest `l2_regularization` and slightly increasing `min_samples_leaf` to reduce overfitting on the small sample. Additionally, we ensure the train and test undergo the same outlier filtering logic for passenger_count/coords (clipping to valid ranges for test instead of dropping rows) so extreme test rows don’t explode predictions. These changes are minimal, keep the same model family and training flow, and are aimed at moving RMSE down toward the target band.'
- What this solution (achieved 6.52248) has done: 'Your current RMSE (6.549) is worse than the target (4.467), so we should make small, low-risk changes that improve generalization without changing your feature set or model family. The biggest issue is that the training data cleaning is much stricter than test handling (train drops out-of-bounds NYC rows, but test is only clipped to world bounds), creating a distribution mismatch; we apply the same NYC bounding-box clipping to test to better match train semantics. Next, we add one very standard, minimal feature for this competition—`manhattan_distance` (|Δlat|+|Δlon|)—which preserves your core approach (handcrafted distance features + HistGradientBoosting) but typically reduces RMSE noticeably. Finally, we slightly increase `max_iter` to reduce underfitting on a 100k sample while keeping everything else (log1p target, model type, training flow, submission format) unchanged.'
- What this solution (achieved 7.59113) has done: 'We make two minimal, score-relevant changes to move RMSE down toward the 4.467 target without changing your overall approach (same feature engineering + HistGradientBoosting + log1p target). First, we sample more training rows (still safely within time) because 100k is typically underpowered here and is the main reason you’re stuck around ~6.5 RMSE. Second, we remove `max_depth` while keeping the rest of your model settings the same; with histogram GBDT this often improves generalization on this task by letting the learner use `max_leaf_nodes` as the primary complexity control (a small, safe tuning change). Everything else (cleaning rules, features, train/val split, target transform, submission writing) stays the same.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

import seaborn as sns
import math
from math import sqrt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import HistGradientBoostingRegressor

np.random.seed(42)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:50]:
        print(os.path.join(dirname, filename))



## === cell 1
train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=500000,
    parse_dates=["pickup_datetime"],
)
test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 2
print(train.shape)
print(test.shape)



## === cell 3
train.head()



## === cell 4
train.dtypes



## === cell 5
train.describe()



## === cell 6
print(train.isnull().sum())



## === cell 7
train = train.dropna(how="any", axis="rows")



## === cell 8
print("Old size: %d" % len(train))



## === cell 9
train = train.drop(train[train.fare_amount < 2.5].index, axis=0)
train = train.drop(train[train.fare_amount > 300].index, axis=0)



## === cell 10
train = train.drop(train[train["passenger_count"] > 6].index, axis=0)
train = train.drop(train[train["passenger_count"] < 0].index, axis=0)



## === cell 11
train = train.drop(train[train["pickup_latitude"] < -90].index, axis=0)
train = train.drop(train[train["pickup_latitude"] > 90].index, axis=0)



## === cell 12
train = train.drop(train[train["pickup_longitude"] < -180].index, axis=0)
train = train.drop(train[train["pickup_longitude"] > 180].index, axis=0)



## === cell 13
train = train.drop(train[train["dropoff_latitude"] < -90].index, axis=0)
train = train.drop(train[train["dropoff_latitude"] > 90].index, axis=0)

train = train.drop(train[train["dropoff_longitude"] < -180].index, axis=0)
train = train.drop(train[train["dropoff_longitude"] > 180].index, axis=0)




## === cell 14
def select_outside_boundingbox(df, BB):
    filter_df = df.loc[
        (df["pickup_longitude"] < BB[0])
        | (df["pickup_longitude"] > BB[1])
        | (df["pickup_latitude"] < BB[2])
        | (df["pickup_latitude"] > BB[3])
        | (df["dropoff_longitude"] < BB[0])
        | (df["dropoff_longitude"] > BB[1])
        | (df["dropoff_latitude"] < BB[2])
        | (df["dropoff_latitude"] > BB[3])
    ]
    return filter_df


NYC_BB = (-74.5, -72.8, 40.5, 41.8)



## === cell 15
outliers = select_outside_boundingbox(train, NYC_BB)
outliers.head()



## === cell 16
train = train.drop(outliers.index, axis=0)



## === cell 17
print("New size: %d" % len(train))



## === cell 18
pass



## === cell 19
test.dtypes



## === cell 20
train["loc1"] = train[["pickup_latitude", "pickup_longitude"]].apply(tuple, axis=1)
train["loc2"] = train[["dropoff_latitude", "dropoff_longitude"]].apply(tuple, axis=1)




## === cell 21
def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1 = np.radians(np.asarray(lon1, dtype=np.float64))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2 = np.radians(np.asarray(lon2, dtype=np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


train["H_Distance"] = haversine_km(
    train["pickup_latitude"].values,
    train["pickup_longitude"].values,
    train["dropoff_latitude"].values,
    train["dropoff_longitude"].values,
).astype(np.float32)




## === cell 22
def chebyshev(pickup_long, dropoff_long, pickup_lat, dropoff_lat):
    return np.maximum(
        np.absolute(pickup_long - dropoff_long), np.absolute(pickup_lat - dropoff_lat)
    )


train["Chebyshev"] = chebyshev(
    train["pickup_longitude"],
    train["dropoff_longitude"],
    train["pickup_latitude"],
    train["dropoff_latitude"],
)



## === cell 23
train.head()



## === cell 24
iso_week = train["pickup_datetime"].dt.isocalendar().week.astype(np.int16)

train["hour"] = train.pickup_datetime.dt.hour
train["day_of_week"] = train.pickup_datetime.dt.weekday
train["day_of_month"] = train.pickup_datetime.dt.day
train["week"] = iso_week
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year - 2000

train["minute"] = train["pickup_datetime"].dt.minute
train["second"] = train["pickup_datetime"].dt.second
train["dayofyear"] = train["pickup_datetime"].dt.dayofyear



## === cell 25
train.head()



## === cell 26
train["manhattan_distance"] = (
    np.abs(train["pickup_latitude"] - train["dropoff_latitude"])
    + np.abs(train["pickup_longitude"] - train["dropoff_longitude"])
).astype(np.float32)

train["fare_to_dist_ratio"] = 1.0 / (train["H_Distance"] + 0.0001)



## === cell 27
train = train.drop(train[train["loc1"] == train["loc2"]].index, axis=0)




## === cell 28
def add_distances_from_airport(dataset):
    jfk_coords = (40.639722, -73.778889)
    ewr_coords = (40.6925, -74.168611)
    lga_coords = (40.77725, -73.872611)

    dataset["pickup_jfk_distance"] = haversine_km(
        dataset["pickup_latitude"].values,
        dataset["pickup_longitude"].values,
        jfk_coords[0],
        jfk_coords[1],
    ).astype(np.float32)
    dataset["dropof_jfk_distance"] = haversine_km(
        dataset["dropoff_latitude"].values,
        dataset["dropoff_longitude"].values,
        jfk_coords[0],
        jfk_coords[1],
    ).astype(np.float32)

    dataset["pickup_ewr_distance"] = haversine_km(
        dataset["pickup_latitude"].values,
        dataset["pickup_longitude"].values,
        ewr_coords[0],
        ewr_coords[1],
    ).astype(np.float32)
    dataset["dropof_ewr_distance"] = haversine_km(
        dataset["dropoff_latitude"].values,
        dataset["dropoff_longitude"].values,
        ewr_coords[0],
        ewr_coords[1],
    ).astype(np.float32)

    dataset["pickup_lga_distance"] = haversine_km(
        dataset["pickup_latitude"].values,
        dataset["pickup_longitude"].values,
        lga_coords[0],
        lga_coords[1],
    ).astype(np.float32)
    dataset["dropof_lga_distance"] = haversine_km(
        dataset["dropoff_latitude"].values,
        dataset["dropoff_longitude"].values,
        lga_coords[0],
        lga_coords[1],
    ).astype(np.float32)

    return dataset


train = add_distances_from_airport(train)



## === cell 29
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])

test["passenger_count"] = pd.to_numeric(
    test["passenger_count"], errors="coerce"
).fillna(1)
test["passenger_count"] = test["passenger_count"].clip(lower=0, upper=6)

for col, lo, hi in [
    ("pickup_latitude", -90, 90),
    ("dropoff_latitude", -90, 90),
    ("pickup_longitude", -180, 180),
    ("dropoff_longitude", -180, 180),
]:
    test[col] = pd.to_numeric(test[col], errors="coerce")
    test[col] = test[col].clip(lower=lo, upper=hi)

test["pickup_longitude"] = test["pickup_longitude"].clip(
    lower=NYC_BB[0], upper=NYC_BB[1]
)
test["dropoff_longitude"] = test["dropoff_longitude"].clip(
    lower=NYC_BB[0], upper=NYC_BB[1]
)
test["pickup_latitude"] = test["pickup_latitude"].clip(lower=NYC_BB[2], upper=NYC_BB[3])
test["dropoff_latitude"] = test["dropoff_latitude"].clip(
    lower=NYC_BB[2], upper=NYC_BB[3]
)

test["loc1"] = test[["pickup_latitude", "pickup_longitude"]].apply(tuple, axis=1)
test["loc2"] = test[["dropoff_latitude", "dropoff_longitude"]].apply(tuple, axis=1)

test["H_Distance"] = haversine_km(
    test["pickup_latitude"].values,
    test["pickup_longitude"].values,
    test["dropoff_latitude"].values,
    test["dropoff_longitude"].values,
).astype(np.float32)

test["Chebyshev"] = chebyshev(
    test["pickup_longitude"],
    test["dropoff_longitude"],
    test["pickup_latitude"],
    test["dropoff_latitude"],
)

test_iso_week = test["pickup_datetime"].dt.isocalendar().week.astype(np.int16)

test["hour"] = test.pickup_datetime.dt.hour
test["day_of_week"] = test.pickup_datetime.dt.weekday
test["day_of_month"] = test.pickup_datetime.dt.day
test["week"] = test_iso_week
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year - 2000

test["minute"] = test["pickup_datetime"].dt.minute
test["second"] = test["pickup_datetime"].dt.second
test["dayofyear"] = test["pickup_datetime"].dt.dayofyear

test = add_distances_from_airport(test)

test["manhattan_distance"] = (
    np.abs(test["pickup_latitude"] - test["dropoff_latitude"])
    + np.abs(test["pickup_longitude"] - test["dropoff_longitude"])
).astype(np.float32)

test["fare_to_dist_ratio"] = 1.0 / (test["H_Distance"] + 0.0001)




## === cell 30
def downcast(df):
    df_int = df.select_dtypes(include=["int64", "int32", "int16", "int8", "int"])
    if len(df_int.columns) > 0:
        df[df_int.columns] = df_int.apply(pd.to_numeric, downcast="unsigned")

    df_float = df.select_dtypes(include=["float64", "float32", "float16", "float"])
    if len(df_float.columns) > 0:
        df[df_float.columns] = df_float.apply(pd.to_numeric, downcast="float")

    return df


downcast(train)
downcast(test)
train.dtypes



## === cell 31
pass



## === cell 32
pass



## === cell 33
pass



## === cell 34
drop_cols = ["key", "fare_amount", "pickup_datetime", "loc1", "loc2"]
feature_cols = [c for c in train.columns if c not in drop_cols]

numeric_feature_cols = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train[c]):
        numeric_feature_cols.append(c)
feature_cols = numeric_feature_cols

missing_in_test = [c for c in feature_cols if c not in test.columns]
for c in missing_in_test:
    test[c] = 0.0

X = train[feature_cols].copy()
y = train["fare_amount"].astype(np.float32).copy()
X_test = test[feature_cols].copy()

X = X.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")

X = X.fillna(0.0)
X_test = X_test.fillna(0.0)

X_matrix = X.values
X_test_matrix = X_test.values

y_log = np.log1p(y.values)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_matrix, y_log, test_size=0.2, random_state=42
)

print("Num features:", X_tr.shape[1])
print("First 10 feature cols:", feature_cols[:10])



## === cell 35
model = HistGradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.05,
    max_iter=1000,
    max_leaf_nodes=63,
    min_samples_leaf=35,
    l2_regularization=0.1,
    random_state=42,
)

model.fit(X_tr, y_tr)

val_pred_log = model.predict(X_val).reshape(-1)
val_pred = np.expm1(val_pred_log)
val_pred = np.clip(val_pred, 0.0, None)

rmse = mean_squared_error(np.expm1(y_val), val_pred, squared=False)
print("Validation RMSE:", rmse)



## === cell 36
test_pred_log = model.predict(X_test_matrix).reshape(-1)
test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, 0.0, None)

assert len(test_pred) == len(test), "Prediction length mismatch vs test rows"

submission = pd.DataFrame(
    {"key": test["key"].astype(str).values, "fare_amount": test_pred.astype(np.float32)}
)
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head())
print("Submission shape:", submission.shape)
