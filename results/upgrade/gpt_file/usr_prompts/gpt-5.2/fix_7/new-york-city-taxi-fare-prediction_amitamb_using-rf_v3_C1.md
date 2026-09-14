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

3.74165

# 6. Current score

4.58724

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 14.29987) has done: 'You’re currently training on only 10,000 rows from a 55M-row dataset, which is the main reason your RMSE (4.78) is far from the target (3.74). To move RMSE closer to the target without changing the model or features, I (1) increase the training sample size within a safe runtime budget, (2) filter obvious outliers/invalid coordinates that hurt RMSE, and (3) make the train/validation split deterministic so results are stable. I also remove notebook-only magics so the script runs end-to-end as a .py and still writes a valid `RFSubmission.csv` with the required columns.'
- What this solution (achieved 6.9381) has done: 'The timeout is dominated by two hotspots: the Python-loop reservoir sampling over the full 55M-row CSV (effectively O(N) with heavy per-row Python overhead), and fitting a very large RandomForest (500 trees, depth 25) on ~1M rows. To preserve core logic while making it finish, I replace the reservoir sampler with a provably equivalent uniform sampling strategy using `skiprows` (still reads exactly `n_train` rows, uniformly without replacement, but avoids scanning the whole file), and I ensure the sampled training set is materialized as contiguous NumPy arrays once (no repeated conversions/copies). I also apply safe runtime tweaks: faster CSV parsing settings, avoiding unnecessary DataFrame copies, and using `sklearnex` patch as you already do. Model, features, cleaning rules, split, and training approach remain the same.'
- What this solution (achieved 4.58724) has done: 'Your RMSE is far above target, so we should improve accuracy without changing the core model/features/training loop. The biggest low-risk gain is fixing the train/test distribution mismatch created by your cleaning: you filter many bad/edge rows in train but only clip coordinates in test (and you also keep zero-distance rides in test), which makes the model extrapolate and hurts RMSE. I (1) apply the *same* feature creation and validity filters to test (dropping impossible/zero-distance rows and then re-aligning to the sample_submission keys), and (2) add a minimal training-weighting step by sampling the training subset in a deterministic way but keeping more informative medium-distance trips (still RandomForest, same features, same loss/fit loop). This keeps the core logic intact and should move RMSE downward toward the 3.74 target while still finishing within constraints and producing a valid submission CSV with all required keys.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

print(os.listdir("../input"))



## === cell 1
n_train = 1_000_000  # keep identical sample size

train_path = "../input/train.csv"
test_path = "../input/test.csv"
sample_sub_path = "../input/sample_submission.csv"

dtype_map = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float64,
    "pickup_latitude": np.float64,
    "dropoff_longitude": np.float64,
    "dropoff_latitude": np.float64,
    "passenger_count": np.int8,
}

train_usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]




## === cell 2
def _count_data_rows_fast(path, buf_size=8 * 1024 * 1024):
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            n += b.count(b"\n")
    return max(n - 1, 0)


def uniform_sample_csv_skiprows(
    path,
    n_samples,
    usecols,
    dtype_map,
    parse_dates,
    seed=42,
):
    total_rows = _count_data_rows_fast(path)  # number of data rows (excluding header)
    if n_samples >= total_rows:
        return pd.read_csv(
            path,
            usecols=usecols,
            dtype=dtype_map,
            parse_dates=parse_dates,
            engine="c",
            low_memory=False,
        )

    rng = np.random.RandomState(seed)
    n_skip = total_rows - n_samples

    skip = rng.choice(
        np.arange(1, total_rows + 1, dtype=np.int64), size=n_skip, replace=False
    )
    skip = np.sort(skip)

    return pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype_map,
        parse_dates=parse_dates,
        skiprows=skip,
        engine="c",
        low_memory=False,
    )




## === cell 3
df = uniform_sample_csv_skiprows(
    train_path,
    n_samples=n_train,
    usecols=train_usecols,
    dtype_map=dtype_map,
    parse_dates=["pickup_datetime"],
    seed=42,
)

df_test = pd.read_csv(
    test_path,
    usecols=test_usecols,
    parse_dates=["pickup_datetime"],
    dtype=dtype_map,
    engine="c",
    low_memory=False,
)

sample_sub = pd.read_csv(sample_sub_path, usecols=["key"])
print(
    "Loaded train shape:",
    df.shape,
    "test shape:",
    df_test.shape,
    "sample_sub shape:",
    sample_sub.shape,
)




## === cell 4
def add_travel_vector_features(dfin):
    dfin["abs_diff_longitude"] = (
        dfin["dropoff_longitude"] - dfin["pickup_longitude"]
    ).abs()
    dfin["abs_diff_latitude"] = (
        dfin["dropoff_latitude"] - dfin["pickup_latitude"]
    ).abs()


add_travel_vector_features(df)
add_travel_vector_features(df_test)

print(df.isnull().sum())
print("Old size: %d" % len(df))
df = df.dropna(how="any", axis="rows")
print("New size: %d" % len(df))



## === cell 5
lon_min, lon_max = -74.5, -72.8
lat_min, lat_max = 40.5, 41.8

df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 250)]
df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 6)]
df = df[
    (df["pickup_longitude"].between(lon_min, lon_max))
    & (df["dropoff_longitude"].between(lon_min, lon_max))
    & (df["pickup_latitude"].between(lat_min, lat_max))
    & (df["dropoff_latitude"].between(lat_min, lat_max))
]
df = df[(df["abs_diff_longitude"] > 0) | (df["abs_diff_latitude"] > 0)]

df_test = df_test.dropna(how="any", axis="rows")
df_test = df_test[(df_test["passenger_count"] >= 1) & (df_test["passenger_count"] <= 6)]
df_test = df_test[
    (df_test["pickup_longitude"].between(lon_min, lon_max))
    & (df_test["dropoff_longitude"].between(lon_min, lon_max))
    & (df_test["pickup_latitude"].between(lat_min, lat_max))
    & (df_test["dropoff_latitude"].between(lat_min, lat_max))
]
df_test = df_test[
    (df_test["abs_diff_longitude"] > 0) | (df_test["abs_diff_latitude"] > 0)
]

print("After cleaning size: %d" % len(df))
print("After test filtering size: %d" % len(df_test))



## === cell 6
min_year = df["pickup_datetime"].dt.year.min()

dt_train = df["pickup_datetime"].dt
df["pickup_year"] = (dt_train.year - min_year).astype(np.int16)
df["pickup_hour"] = dt_train.hour.astype(np.int8)
df["pickup_day"] = dt_train.dayofyear.astype(np.int16)

dt_test = df_test["pickup_datetime"].dt
df_test["pickup_year"] = (dt_test.year - min_year).astype(np.int16)
df_test["pickup_hour"] = dt_test.hour.astype(np.int8)
df_test["pickup_day"] = dt_test.dayofyear.astype(np.int16)



## === cell 7
from sklearn.model_selection import train_test_split

df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
len(df_val)



## === cell 8
FEATURE_COLS = [
    "abs_diff_longitude",
    "abs_diff_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "pickup_longitude",
    "pickup_latitude",
    "passenger_count",
    "pickup_year",
    "pickup_hour",
    "pickup_day",
]


def get_input_matrix(dfin):
    return np.asarray(dfin[FEATURE_COLS].to_numpy(copy=False))


x_train, x_val = get_input_matrix(df_train), get_input_matrix(df_val)
y_train = np.asarray(df_train.fare_amount.to_numpy(copy=False))
y_val = np.asarray(df_val.fare_amount.to_numpy(copy=False))



## === cell 9
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(
    max_depth=25,
    n_estimators=500,
    oob_score=True,
    n_jobs=-1,
    min_samples_split=10,
    verbose=1,
    random_state=42,
)



## === cell 10
reg.fit(x_train, y_train)



## === cell 11
reg.oob_score_



## === cell 12
from sklearn.metrics import r2_score

y_pred = reg.predict(x_val)



## === cell 13
score = r2_score(y_val, y_pred)
score



## === cell 14
pass



## === cell 15
reg.score(x_val, y_val)



## === cell 16
from sklearn.metrics import mean_squared_error

mean_squared_error(y_val, y_pred)



## === cell 17
x_test = get_input_matrix(df_test)

pred_test = reg.predict(x_test)
pred_test = np.clip(pred_test, 0, None)

fallback_fare = float(np.median(y_train))
pred_map = pd.DataFrame({"key": df_test["key"].values, "fare_amount": pred_test})

RFSubmission = sample_sub.merge(pred_map, on="key", how="left")
RFSubmission["fare_amount"] = (
    RFSubmission["fare_amount"].fillna(fallback_fare).astype(np.float32)
)

RFSubmission.to_csv("RFSubmission.csv", index=False)
print(
    "Wrote RFSubmission.csv with shape:",
    RFSubmission.shape,
    "missing filled:",
    int(RFSubmission["fare_amount"].isna().sum()),
)
print("Fallback fare used:", fallback_fare)
