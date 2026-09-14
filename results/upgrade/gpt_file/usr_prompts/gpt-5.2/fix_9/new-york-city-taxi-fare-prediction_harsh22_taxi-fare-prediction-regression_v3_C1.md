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

3.37198

# 6. Current score

4.66538

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 16.9774) has done: 'I fix the pandas boolean/axis bugs that stop your cleaning steps from running under modern pandas (e.g., `any(1)` and mixing DataFrames with `|`). Then I ensure the same cleaning/feature-engineering pipeline leaves no NaNs in the final training and test matrices by dropping remaining NaN rows in train and imputing test NaNs with training medians (minimal, score-safe). Finally, I keep your RandomForest approach intact but make it deterministic and stable, and I write a valid `submission_1.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 5.81457) has done: 'The timeout is dominated by fitting a 300-tree `RandomForestRegressor` on ~1M rows (very expensive) and by some unnecessary pandas overhead (row-wise null checks, repeated `.copy()`, non-vectorized datetime parsing defaults). To preserve core logic and accuracy, the key speed fix is to use Intel-accelerated scikit-learn (available via `scikit-learn-intelex`) for the same RandomForest algorithm, plus small-but-safe vectorizations for filtering and feature engineering. I also reduce pandas copies and compute masks using NumPy arrays to avoid costly `DataFrame.isnull().any(axis=1)` scans. All paths, features, model type, hyperparameters, and training/prediction semantics remain the same.'
- What this solution (achieved 5.29874) has done: 'Your current gap to target is large (RMSE 5.81457 vs 3.37198, lower is better), so we need a modest accuracy boost while keeping your RandomForest + existing cleaning/feature pipeline intact. The biggest safe improvement with minimal semantic change is to train on more data without increasing runtime too much: we raise `nrows` (still a simple subset read, same features/filters), and we add `max_features="sqrt"` which typically reduces RF variance and improves generalization on this competition with negligible risk. We also ensure the train/test filtering remains consistent by applying the same basic coordinate/passenger validity mask to the test set (this avoids extreme out-of-domain test rows that can blow up predictions and RMSE). Finally, we keep the exact submission schema and write `submission_1.csv` as before.'
- What this solution (achieved 5.36486) has done: 'We’re still far from the target (5.29874 vs 3.37198, lower is better), so we need a modest accuracy boost without changing your RandomForest + feature pipeline. The smallest high-impact fix is to add one well-known, cheap feature for this competition: straight-line distance in degrees (Euclidean on lat/lon deltas), which complements your Haversine and Manhattan features and often reduces RMSE noticeably. I also apply the missing longitude bound check on the training mask (you already had it for test), which removes extreme outliers that RandomForest otherwise fits poorly and can hurt generalization. Everything else (data subset size, cleaning logic, model type/hyperparams, training loop) stays the same, and it still writes `submission_1.csv` with `key,fare_amount`.'
- What this solution (achieved 4.66538) has done: 'Your current RMSE (5.36486, lower is better) is still far above the target (3.37198), so we should make a small, score-oriented improvement without changing the overall RandomForest + feature-engineering approach. The highest-impact minimal tweak here is to use a more appropriate validity filter for NYC taxi data: restrict lat/lon to a NYC bounding box (instead of global -180..180 / -90..90), which removes extreme-coordinate outliers that the RF otherwise fits and generalizes poorly from. To keep train/test semantics consistent, the same NYC mask is applied to test, and invalid test rows are filled with training medians (same idea you already use). Everything else (features, model type, estimator count, submission format/path) remains the same.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import os

INPUT_DIR = "/kaggle/input" if os.path.isdir("/kaggle/input") else "../input"
print("Using INPUT_DIR:", INPUT_DIR)
print(os.listdir(INPUT_DIR))

np.random.seed(42)

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception as e:
    print("sklearnex patch not applied:", repr(e))



## === cell 1
usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_train = {
    "key": "string",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}
dtype_test = {
    "key": "string",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

TRAIN_NROWS = 2_000_000

train = pd.read_csv(
    f"{INPUT_DIR}/train.csv",
    nrows=TRAIN_NROWS,
    usecols=usecols_train,
    dtype=dtype_train,
    parse_dates=["pickup_datetime"],
)
test = pd.read_csv(
    f"{INPUT_DIR}/test.csv",
    usecols=usecols_test,
    dtype=dtype_test,
    parse_dates=["pickup_datetime"],
)



## === cell 2
_ = train.shape



## === cell 3
_ = test.shape



## === cell 4
_ = None



## === cell 5
_ = None



## === cell 6
_ = None



## === cell 7
fare = train["fare_amount"].to_numpy()
pc = train["passenger_count"].to_numpy()
plat = train["pickup_latitude"].to_numpy()
plon = train["pickup_longitude"].to_numpy()
dlat = train["dropoff_latitude"].to_numpy()
dlon = train["dropoff_longitude"].to_numpy()

notnull_mask = ~(
    pd.isna(fare)
    | pd.isna(train["pickup_datetime"].to_numpy())
    | pd.isna(plat)
    | pd.isna(plon)
    | pd.isna(dlat)
    | pd.isna(dlon)
    | pd.isna(pc)
)

NYC_LAT_MIN, NYC_LAT_MAX = 40.0, 42.0
NYC_LON_MIN, NYC_LON_MAX = -75.0, -72.0

mask = (
    notnull_mask
    & (fare >= 0)
    & (pc != 208)
    & (plat >= NYC_LAT_MIN)
    & (plat <= NYC_LAT_MAX)
    & (plon >= NYC_LON_MIN)
    & (plon <= NYC_LON_MAX)
    & (dlat >= NYC_LAT_MIN)
    & (dlat <= NYC_LAT_MAX)
    & (dlon >= NYC_LON_MIN)
    & (dlon <= NYC_LON_MAX)
)
train = train.loc[mask]



## === cell 8
tpc = test["passenger_count"].to_numpy()
tplat = test["pickup_latitude"].to_numpy()
tplon = test["pickup_longitude"].to_numpy()
tdlat = test["dropoff_latitude"].to_numpy()
tdlon = test["dropoff_longitude"].to_numpy()

test_notnull = ~(
    pd.isna(test["pickup_datetime"].to_numpy())
    | pd.isna(tplat)
    | pd.isna(tplon)
    | pd.isna(tdlat)
    | pd.isna(tdlon)
    | pd.isna(tpc)
)

test_mask = (
    test_notnull
    & (tpc != 208)
    & (tplat >= NYC_LAT_MIN)
    & (tplat <= NYC_LAT_MAX)
    & (tplon >= NYC_LON_MIN)
    & (tplon <= NYC_LON_MAX)
    & (tdlat >= NYC_LAT_MIN)
    & (tdlat <= NYC_LAT_MAX)
    & (tdlon >= NYC_LON_MIN)
    & (tdlon <= NYC_LON_MAX)
)
invalid_test_mask = ~test_mask



## === cell 9
_ = train.shape



## === cell 10
_ = None



## === cell 11
from collections import Counter

_ = None



## === cell 12
_ = train.shape



## === cell 13
_ = None



## === cell 14
_ = None



## === cell 15
_ = None



## === cell 16
_ = None



## === cell 17
_ = None



## === cell 18
_ = None



## === cell 19
_ = None



## === cell 20
_ = None



## === cell 21
_ = None



## === cell 22
_ = train.shape



## === cell 23
_ = None



## === cell 24
_ = None



## === cell 25
_ = None



## === cell 26
_ = None



## === cell 27
_ = None



## === cell 28
_ = None



## === cell 29
_ = None



## === cell 30
_ = None



## === cell 31
_ = train.dtypes



## === cell 32
test_key = test["key"].copy()

train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"], errors="coerce")
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"], errors="coerce")



## === cell 33
_ = train.dtypes




## === cell 34
def _add_haversine(df, lat1, lon1, lat2, lon2, out_col="H_Distance"):
    r = 6371.0
    lat1v = df[lat1].to_numpy()
    lon1v = df[lon1].to_numpy()
    lat2v = df[lat2].to_numpy()
    lon2v = df[lon2].to_numpy()

    phi1 = np.radians(lat1v)
    phi2 = np.radians(lat2v)
    dphi = np.radians(lat2v - lat1v)
    dlambda = np.radians(lon2v - lon1v)

    a = (np.sin(dphi / 2.0) ** 2) + (
        np.cos(phi1) * np.cos(phi2) * (np.sin(dlambda / 2.0) ** 2)
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    df[out_col] = (r * c).astype("float32", copy=False)




## === cell 35
_add_haversine(
    train,
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
)
_add_haversine(
    test, "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)



## === cell 36
_ = None



## === cell 37
for i in (train, test):
    dt = i["pickup_datetime"].dt
    i["Year"] = dt.year.astype("int16", copy=False)
    i["Month"] = dt.month.astype("int8", copy=False)
    i["Date"] = dt.day.astype("int8", copy=False)
    i["Day of Week"] = dt.dayofweek.astype("int8", copy=False)
    i["Hour"] = dt.hour.astype("int8", copy=False)



## === cell 38
for i in (train, test):
    plon = i["pickup_longitude"].to_numpy()
    dlon = i["dropoff_longitude"].to_numpy()
    plat = i["pickup_latitude"].to_numpy()
    dlat = i["dropoff_latitude"].to_numpy()

    abs_lon = np.abs(plon - dlon).astype("float32", copy=False)
    abs_lat = np.abs(plat - dlat).astype("float32", copy=False)
    i["abs_lon_diff"] = abs_lon
    i["abs_lat_diff"] = abs_lat
    i["manhattan_dist"] = (abs_lon + abs_lat).astype("float32", copy=False)

    i["euclidean_dist"] = np.sqrt(abs_lon * abs_lon + abs_lat * abs_lat).astype(
        "float32", copy=False
    )



## === cell 39
_ = None



## === cell 40
mask_bad1 = (
    (train["pickup_latitude"].eq(0) & train["pickup_longitude"].eq(0))
    & (~train["dropoff_latitude"].eq(0) & ~train["dropoff_longitude"].eq(0))
    & train["fare_amount"].eq(0)
)
train = train.loc[~mask_bad1]



## === cell 41
_ = train.shape



## === cell 42
_ = None



## === cell 43
mask_bad2 = (
    (~train["pickup_latitude"].eq(0) & ~train["pickup_longitude"].eq(0))
    & (train["dropoff_latitude"].eq(0) & train["dropoff_longitude"].eq(0))
    & train["fare_amount"].eq(0)
)
train = train.loc[~mask_bad2]



## === cell 44
high_distance_idx = train.index[
    (train["H_Distance"] > 200) & (train["fare_amount"] != 0)
]



## === cell 45
_ = None



## === cell 46
train.loc[high_distance_idx, "H_Distance"] = (
    (train.loc[high_distance_idx, "fare_amount"] - 2.50) / 1.56
).astype("float32", copy=False)



## === cell 47
_ = None



## === cell 48
_ = None



## === cell 49
_ = None



## === cell 50
train = train.loc[~((train["H_Distance"] == 0) & (train["fare_amount"] == 0))]



## === cell 51
_ = None



## === cell 52
_ = None



## === cell 53
rush_hour_mask = (
    (train["Hour"].between(6, 20, inclusive="both"))
    & (train["Day of Week"].between(1, 5, inclusive="both"))
    & (train["H_Distance"].eq(0))
    & (train["fare_amount"].lt(2.5))
)
train = train.loc[~rush_hour_mask]



## === cell 54
_ = None



## === cell 55
_ = None



## === cell 56
_ = None



## === cell 57
scenario_3_mask = (train["H_Distance"] != 0) & (train["fare_amount"] == 0)



## === cell 58
_ = int(scenario_3_mask.sum())



## === cell 59
_ = None



## === cell 60
train.loc[scenario_3_mask, "fare_amount"] = (
    (train.loc[scenario_3_mask, "H_Distance"] * 1.56) + 2.50
).astype("float32", copy=False)



## === cell 61
_ = None



## === cell 62
_ = None



## === cell 63
_ = None



## === cell 64
scenario_4_mask = (train["H_Distance"] == 0) & (train["fare_amount"] != 0)



## === cell 65
_ = int(scenario_4_mask.sum())



## === cell 66
_ = None



## === cell 67
_ = None



## === cell 68
scenario_4_sub_mask = scenario_4_mask & (train["fare_amount"] > 3.0)



## === cell 69
scenario_4_sub_mask = scenario_4_mask & (train["fare_amount"] > 3.0)



## === cell 70
_ = int(scenario_4_sub_mask.sum())



## === cell 71
train.loc[scenario_4_sub_mask, "H_Distance"] = (
    (train.loc[scenario_4_sub_mask, "fare_amount"] - 2.50) / 1.56
).astype("float32", copy=False)



## === cell 72
_ = None



## === cell 73
_ = train.columns



## === cell 74
_ = test.columns



## === cell 75
train = train.drop(["key", "pickup_datetime"], axis=1)
test = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 76
x_train = train.iloc[:, train.columns != "fare_amount"]
y_train = train["fare_amount"].to_numpy()
x_test = test

xtr_np = x_train.to_numpy(copy=False)
train_mask = ~np.isnan(xtr_np).any(axis=1) & ~pd.isna(y_train)
x_train = x_train.loc[train_mask]
y_train = y_train[train_mask]

train_medians = x_train.median(numeric_only=True)
x_train = x_train.fillna(train_medians)
x_test = x_test.fillna(train_medians)

if invalid_test_mask.any():
    for col, med in train_medians.items():
        if col in x_test.columns:
            x_test.loc[invalid_test_mask, col] = med



## === cell 77
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=300, random_state=42, n_jobs=-1, max_features="sqrt"
)

Xtr = np.ascontiguousarray(x_train.to_numpy(dtype=np.float32, copy=False))
ytr = np.ascontiguousarray(y_train.astype(np.float32, copy=False))
Xte = np.ascontiguousarray(x_test.to_numpy(dtype=np.float32, copy=False))

rf.fit(Xtr, ytr)
rf_predict = rf.predict(Xte)



## === cell 78
submission = pd.DataFrame({"key": test_key, "fare_amount": rf_predict})
submission.to_csv("submission_1.csv", index=False)
_ = submission.head(20)
