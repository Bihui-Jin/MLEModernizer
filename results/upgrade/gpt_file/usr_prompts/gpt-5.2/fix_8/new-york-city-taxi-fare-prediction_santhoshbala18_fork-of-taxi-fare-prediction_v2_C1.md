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

3.70513

# 6. Current score

5.72589

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.41453) has done: 'I fix the LightGBM API break by removing the deprecated `silent` argument and correcting misspelled parameter keys so training runs in LightGBM 4.6.0. I also make the pipeline deterministic and prevent a common logic bug where the airport longitude/latitude bounds are incorrectly signed (NYC longitudes are negative), applying the same feature logic to both train and test. Finally, I keep the core modeling/training approach the same, but ensure the script runs end-to-end and always writes a valid `submission.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 5.41453) has done: 'We need to move RMSE down from 5.41 toward 3.705, so the smallest meaningful gain is to fix a feature mismatch/omission that currently hurts generalization: you compute `is_night`, `is_airport`, `is_surge` but then drop the datetime-derived columns and never actually use `pickup_datetime` itself, so the flags are the only time signal and they are kept—however, your current feature set also includes raw lat/lon with no sanity filtering on test and no clipping of extreme distances, which can cause the model to extrapolate poorly on the leaderboard. I keep the same LightGBM training approach and parameters, but (1) apply the same passenger_count bounds and bounding-box filtering to *test* rows via safe clipping (not dropping rows, to keep submission row count), and (2) add a single robustification step: clip `distance_km` to a reasonable max based on the filtered training distribution to reduce the impact of outliers. These are minimal, metric-aligned changes that typically reduce RMSE for this competition without changing the core modeling logic. The script still run end-to-end and write `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.36821) has done: 'Your current RMSE (5.41) is far above the target (3.705), so the smallest meaningful improvement is to fix likely train/test distribution issues without changing the model or training loop. I add a minimal but very impactful training filter: remove extreme/unrealistic fares and very long trips from the training sample (a standard NYC Taxi Fare cleanup), while keeping your existing bounding-box + passenger filtering and the same LightGBM setup. I also ensure the same `distance_km` clipping logic remains consistent and non-leaky by computing caps from the already-filtered training data only. This should reduce outlier-driven error and move RMSE down toward the target while preserving your core approach.'
- What this solution (achieved 5.7214) has done: 'Your RMSE (5.37) is substantially worse than the 3.705 target (lower is better), so we should make the smallest changes that improve generalization without changing your model/training loop. The biggest issue is that you compute time-derived flags but then drop the underlying time features (`hour/day/month/weekday`), which removes useful signal; we keep those columns as numeric features while still dropping the raw datetime string. We also add one standard, minimal NYC taxi cleanup filter that removes “zero-distance but non-trivial fare” rows (often meter errors) from the *training* sample only, which tends to reduce RMSE without changing modeling semantics. Everything else (LightGBM, objective, boosting rounds, distance feature, bounding box, and submission writing) stays the same.'
- What this solution (achieved 5.72806) has done: 'Your RMSE (5.7214) is still far from the target (3.70513, lower is better), so we should make a small but high-impact correction that keeps the same LightGBM training approach and features. The biggest remaining issue is that `distance_km` is computed on raw (possibly out-of-bounds) test coordinates, then you later clip test coordinates—this creates an inconsistent feature (distance doesn’t match the final lat/lon). I recompute `distance_km` after the test-set clipping (and keep train untouched) so the model sees consistent geometry at inference. This is minimal, preserves your core logic, and typically reduces leaderboard error without changing the model, loss, or training loop.'
- What this solution (achieved 5.72806) has done: 'Your current RMSE (5.72806, lower is better) is far above the 3.70513 target, so the smallest high-impact move is to fix a remaining train/test mismatch that hurts generalization. Right now you clip test coordinates to the bounding box, but you never clip training coordinates; that makes the model learn on raw coordinates (including residual edge/outlier values that survive the filter) while predicting on clipped coordinates, which can degrade accuracy. I apply the same safe clipping (not row-dropping) to the already-filtered training data before computing `distance_km` caps and training, keeping your LightGBM setup, features, and training loop unchanged. This keeps the core logic identical while reducing distribution shift, which should move RMSE down toward the target band.'
- What this solution (achieved 5.72589) has done: 'Your current RMSE (5.72806, lower is better) is well above the target (3.70513), so we should make a small, metric-aligned improvement without changing the model or training loop. The biggest remaining issue is that the model can output negative fares and very small fares for short trips, which are heavily penalized by RMSE; we add a minimal, standard post-processing step to clip predictions to a realistic lower bound (and a conservative upper bound) for both validation and submission. This keeps identical training semantics and features, but improves evaluation by preventing impossible predictions. We also keep the submission format unchanged and ensure it still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)

print(os.listdir("../input"))



## === cell 1
df = pd.read_csv("../input/train.csv", nrows=10**6)
test_set = pd.read_csv("../input/test.csv")




## === cell 2
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


df["distance_km"] = distance(
    df.pickup_latitude,
    df.pickup_longitude,
    df.dropoff_latitude,
    df.dropoff_longitude,
)



## === cell 3
BB = (-75, -73, 40, 41.5)


def select_within_boundingbox(df_in, BB):
    return (
        (df_in.pickup_longitude >= BB[0])
        & (df_in.pickup_longitude <= BB[1])
        & (df_in.pickup_latitude >= BB[2])
        & (df_in.pickup_latitude <= BB[3])
        & (df_in.dropoff_longitude >= BB[0])
        & (df_in.dropoff_longitude <= BB[1])
        & (df_in.dropoff_latitude >= BB[2])
        & (df_in.dropoff_latitude <= BB[3])
    )


print("Old size: %d" % len(df))
df = df[select_within_boundingbox(df, BB)]
df = df[(df.passenger_count > 0) & (df.passenger_count < 10)]
print("New size: %d" % len(df))

df["passenger_count"] = df["passenger_count"].clip(lower=1, upper=9)
for col, lo, hi in [
    ("pickup_longitude", BB[0], BB[1]),
    ("dropoff_longitude", BB[0], BB[1]),
    ("pickup_latitude", BB[2], BB[3]),
    ("dropoff_latitude", BB[2], BB[3]),
]:
    df[col] = df[col].clip(lower=lo, upper=hi)

df["distance_km"] = distance(
    df.pickup_latitude,
    df.pickup_longitude,
    df.dropoff_latitude,
    df.dropoff_longitude,
)

test_set["passenger_count"] = test_set["passenger_count"].clip(lower=1, upper=9)
for col, lo, hi in [
    ("pickup_longitude", BB[0], BB[1]),
    ("dropoff_longitude", BB[0], BB[1]),
    ("pickup_latitude", BB[2], BB[3]),
    ("dropoff_latitude", BB[2], BB[3]),
]:
    test_set[col] = test_set[col].clip(lower=lo, upper=hi)

test_set["distance_km"] = distance(
    test_set.pickup_latitude,
    test_set.pickup_longitude,
    test_set.dropoff_latitude,
    test_set.dropoff_longitude,
)




## === cell 4
def add_datetime_info(dataset):
    dataset["pickup_datetime"] = pd.to_datetime(dataset["pickup_datetime"])

    dataset["hour"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["weekday"] = dataset.pickup_datetime.dt.weekday

    return dataset


df = add_datetime_info(df)
test_set = add_datetime_info(test_set)




## === cell 5
def add_custom_flags(dataset):
    dataset["is_night"] = np.where(
        (
            ((dataset["hour"] >= 20) & (dataset["hour"] <= 23))
            | ((dataset["hour"] >= 0) & (dataset["hour"] < 6))
        ),
        1,
        0,
    )

    dataset["is_airport"] = np.where(
        (
            (dataset["dropoff_longitude"] >= -73.79)
            & (dataset["dropoff_longitude"] <= -73.75)
            & (dataset["dropoff_latitude"] >= 40.63)
            & (dataset["dropoff_latitude"] <= 40.66)
        ),
        1,
        0,
    )

    dataset["is_surge"] = np.where(
        (
            ((dataset["hour"] >= 16) & (dataset["hour"] < 20))
            & ((dataset["weekday"] != 5) & (dataset["weekday"] != 6))
        ),
        1,
        0,
    )
    return dataset


df = add_custom_flags(df)
test_set = add_custom_flags(test_set)



## === cell 6
df = df.drop(df[df["fare_amount"] < 0].index, axis=0)
df = df[df["fare_amount"] <= 250].copy()  # typical competition cleanup
df = df[
    df["distance_km"] <= 200
].copy()  # remove extreme trips within BB that are likely noise
df = df[~((df["distance_km"] < 0.05) & (df["fare_amount"] > 3.0))].copy()

dist_cap = float(df["distance_km"].quantile(0.999))
df["distance_km"] = df["distance_km"].clip(lower=0.0, upper=dist_cap)
test_set["distance_km"] = test_set["distance_km"].clip(lower=0.0, upper=dist_cap)



## === cell 7
try:
    plt.scatter(df["is_night"], df["fare_amount"], c="r", s=2, alpha=0.3)
    plt.show()
except Exception as e:
    print("Plot skipped due to:", repr(e))



## === cell 8
from sklearn.model_selection import train_test_split

train = df.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
target = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    train, target, test_size=0.1, random_state=42
)



## === cell 9
import lightgbm as lgbm

params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_alpha": 1.0,
    "reg_lambda": 0.001,
    "min_split_gain": 0.5,
    "min_child_weight": 1,
    "min_child_samples": 10,
    "verbose": -1,
    "seed": 42,
    "feature_fraction_seed": 42,
    "bagging_seed": 42,
}

train_set = lgbm.Dataset(X_train, label=y_train)

model = lgbm.train(params, train_set=train_set, num_boost_round=300)



## === cell 10
predictions = model.predict(X_test, num_iteration=model.current_iteration())



## === cell 11
import math
from sklearn.metrics import mean_squared_error

predictions_pp = np.clip(predictions, 2.5, 250.0)

rmse = math.sqrt(mean_squared_error(y_test, predictions_pp))
print(rmse)



## === cell 12
test_set_features = test_set.drop(["key", "pickup_datetime"], axis=1)
test_set_key = test_set["key"]

y_pred_final = model.predict(test_set_features, num_iteration=model.current_iteration())

y_pred_final_pp = np.clip(y_pred_final, 2.5, 250.0)

submission = pd.DataFrame(
    {"key": test_set_key, "fare_amount": y_pred_final_pp},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)
print("Submitted: submission.csv")



## === cell 13
submission.shape
