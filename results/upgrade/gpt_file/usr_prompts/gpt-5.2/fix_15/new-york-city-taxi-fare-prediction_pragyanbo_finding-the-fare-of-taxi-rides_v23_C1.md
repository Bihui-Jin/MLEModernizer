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

4.38819

# 6. Current score

6.51964

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.89776) has done: 'I fix the XGBoost prediction call to be compatible with xgboost==2.0.3: `best_ntree_limit` is no longer available, so we use `iteration_range` (or fall back safely) based on `best_iteration` from early stopping. I also make the script runnable as a .py (Kaggle script) by removing the Jupyter-only `%matplotlib inline` magic while keeping the same imports and core logic. Finally, I ensure a submission file is always written with the required columns and a `.csv` suffix, even if early stopping metadata differs.'
- What this solution (achieved 5.89866) has done: 'To move RMSE down toward the 4.38819 target (lower is better) with minimal disruption, I keep your exact feature set and XGBoost training approach but fix two high-impact issues: the train/validation split be made deterministic and the XGBoost parameters be updated to sensible defaults for xgboost==2.0.3 (while keeping the same training API and early stopping). I also make the split and training consistent by setting seeds in both `train_test_split` and XGBoost, which typically improves generalization and reduces score variance. Finally, I add a tiny, safe clamp of predictions to a non-negative range (fares can’t be negative), which usually helps RMSE without changing your core logic.'
- What this solution (achieved 5.45466) has done: 'I keep your exact feature set (distance + passenger_count) and the same XGBoost training API, but fix two issues that commonly hurt RMSE without changing core logic: (1) ensure `distance` is finite in both train and test before filtering/training (so bad rows don’t silently distort the model), and (2) add a tiny, safe post-processing clamp of passenger_count into the valid range you already intended (1–9) for both train and test to reduce train/test mismatch. I also make the validation evaluation stronger by evaluating on both train and valid (still using the same early stopping loop) to stabilize `best_iteration` selection. Finally, I guarantee the submission uses the full unfiltered `test.csv` key order by filtering only for feature computation but mapping predictions back to the original key order (so you never drop test rows and accidentally create an invalid submission).'
- What this solution (achieved 5.89372) has done: 'You’re currently worse than the target RMSE (5.45466 vs 4.38819, lower is better), so the smallest safe way to move toward the target is to improve generalization without changing your core modeling approach (XGBoost on distance + passenger_count). The biggest issue hurting RMSE here is that you’re filtering the test set (NYC bounds), then filling missing predictions with a fallback—this creates a train/test distribution mismatch and can add avoidable error; we keep filtering/cleaning for training, but compute features for the full test set and predict for every row directly. We also add one very small, standard outlier cleanup on training fares (drop extreme fare_amount values) while keeping your same features and training loop; this typically reduces RMSE noticeably on this competition without changing semantics. Finally, we keep your xgboost==2.0.3 `iteration_range` handling and non-negative clamp, and ensure the submission row order matches `test.csv` exactly.'
- What this solution (achieved 6.08864) has done: 'We keep your exact feature set (distance + passenger_count) and the same XGBoost training API/early-stopping loop, but make two small, high-impact data-cleaning tweaks that usually move RMSE down for this competition: (1) remove zero-distance rides from training (they disproportionately include errors/fraud and add noise), and (2) add a gentle upper bound on distance (keep most data but drop extreme long trips that your simple feature set can’t model well). We also apply the same “finite distance” cleanup to the test features before prediction to avoid NaN/inf artifacts. These changes are minimal, don’t alter the modeling approach, and should move your score toward the 4.38819 target (lower is better) from 5.89372.'
- What this solution (achieved 6.01425) has done: 'Your current RMSE (6.08864) is much worse than the target (4.38819), so we should make the smallest changes that reliably improve generalization without changing your feature set (distance + passenger_count) or the XGBoost training loop. The biggest issue here is that you train on only 100k rows while the dataset is huge; increasing the training sample (still via `nrows`, same pipeline) typically yields a large RMSE drop for this competition. I also make the train/valid split slightly more robust by shuffling deterministically (already seeded) and keeping the same early stopping semantics. Finally, I keep your existing cleaning/clipping/post-processing and ensure the submission format stays identical.'
- What this solution (achieved 5.9464) has done: 'To move RMSE down toward your 4.38819 target (lower is better) without changing the core approach (XGBoost on `distance` + `passenger_count` with the same training loop), the smallest high-impact fix is to add a few time-based features extracted from `pickup_datetime` (hour/day-of-week/month). This keeps the model class and training procedure identical, but gives the model information that strongly explains fare variation beyond distance. I compute these features for both train and test, drop only rows with invalid datetimes (minimal), and keep your existing cleaning, split, early stopping, and submission writing unchanged. This should reduce the gap substantially versus relying on distance alone.'
- What this solution (achieved 6.5464) has done: 'You’re currently worse than the target (RMSE 5.9464 vs 4.38819; lower is better), so we should make the smallest changes that usually improve RMSE without changing your model class or training loop. The biggest low-risk gain for NYC taxi fare is adding a simple location feature (`haversine` + Manhattan distance proxy) and a basic “airline vs. street-grid” interaction, while keeping the same XGBoost training API and early stopping. I also make the time features slightly richer (year is unnecessary; but weekend flag is cheap and helpful) without altering the training approach. Finally, I keep your cleaning and submission-writing semantics intact and ensure the added features are computed consistently for train and test.'
- What this solution (achieved 6.64012) has done: 'Your current RMSE (6.5464) is worse than the target (4.38819), so we should make small, reliable improvements without changing the core approach (XGBoost regression with the same feature set family and training loop). The biggest low-risk issue is that the training set still contains some corrupted/outlier coordinate pairs (even within NYC bounds) that create extremely noisy labels for a simple distance/time model; adding a minimal “coordinate sanity” filter and a realistic fare-per-mile cap usually reduces RMSE substantially on this competition. I also add one tiny, standard feature that doesn’t change the modeling approach (straight-line displacement in degrees) and align the training sample to be temporally random but stable by using a fixed random sample from the loaded 2M rows. All changes keep the same XGBoost API, early stopping, and submission format, and still finish quickly.'
- What this solution (achieved 6.59332) has done: 'Your current RMSE (6.64012) is worse than the target (4.38819), so we should make a small, reliable improvement without changing the model type (XGBoost), training API/early-stopping loop, or the general feature family. The biggest low-risk issue is that the training sample is heavily shuffled but still “uniform over all years”; NYC taxi fares have strong long-term drift, and a random split can hurt generalization on this competition. I keep the same feature columns and XGBoost loop, but switch to a time-aware split (train on earlier rides, validate on later rides) while still shuffling within the train fold; this typically reduces overfitting and improves public LB RMSE for this dataset. I also add one minimal, standard cleaning step specific to this competition (remove rare but very harmful `passenger_count` outliers and enforce `passenger_count` integer casting) to reduce noisy labels without altering the approach.'
- What this solution (achieved 6.74637) has done: 'You’re worse than the target (6.59332 vs 4.38819 RMSE; lower is better), so the smallest reliable improvement is to reduce label noise from the known “dirty” rows in this dataset while keeping your exact XGBoost approach and feature set. I add two minimal, competition-standard cleaning steps: remove rows with missing/invalid coordinates and filter out clearly implausible “fare per mile” outliers using a slightly tighter, more realistic range. I also make the time-aware split strictly consistent with your filtered training data (so the model doesn’t train/validate on subtly different distributions due to NaT handling). The training loop, model type, early stopping, features, and submission format remain the same.'
- What this solution (achieved 6.84825) has done: 'Your current RMSE (6.74637) is worse than the target (4.38819), so we should make the smallest changes that reliably reduce error without changing the model type or training loop. The biggest likely issue is that the NYC coordinate filters are too tight and are dropping a lot of valid trips, creating a train/test distribution mismatch; I relax the bounds slightly to the standard competition ranges while keeping the same filtering approach. I also add one minimal, competition-standard cleanup that’s strongly beneficial for this dataset: removing rows where pickup and dropoff coordinates are identical (these are often bad records) while keeping the rest of your features and XGBoost setup unchanged. Finally, I keep your existing feature set, early stopping, prediction handling, and submission format, ensuring the script still runs end-to-end and writes a valid CSV.'
- What this solution (achieved 6.51964) has done: 'Your RMSE (6.84825) is worse than the 4.38819 target, so we should make a small, reliable improvement without changing the model type or training loop. The biggest likely source of avoidable error here is that the model is trying to explain a very non-linear target (`fare_amount`) directly with relatively simple features; switching to a log1p target transform and then expm1 at prediction time keeps the same XGBoost reg:squarederror setup but typically reduces RMSE on this competition. I also add one minimal, standard feature that doesn’t change the approach (the raw bearing angle between pickup/dropoff) to help the tree splits capture directional structure. Everything else (data loading size, cleaning, time-aware split, early stopping, submission format) stays the same and still writes a valid `XGboost_regression.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=2_000_000)
train_df = train_df.reset_index(drop=True)



## === cell 2
train_df.shape



## === cell 3
test_df_full = pd.read_csv("../input/test.csv")
test_df = test_df_full.copy()



## === cell 4
test_df.shape



## === cell 5
train_df.head(5)



## === cell 6
train_df.isnull().sum()



## === cell 7
train_df.dropna(inplace=True)



## === cell 8
train_df.describe()



## === cell 9
train_df = train_df[train_df["fare_amount"] > 0]



## === cell 10
train_df.shape




## === cell 11
def distance(lat1, lon1, lat2, lon2):
    a = (
        0.5
        - np.cos((lat2 - lat1) * 0.017453292519943295) / 2
        + np.cos(lat1 * 0.017453292519943295)
        * np.cos(lat2 * 0.017453292519943295)
        * (1 - np.cos((lon2 - lon1) * 0.017453292519943295))
        / 2
    )
    res = 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
    return res




## === cell 12
def filter_nyc_coords(df):
    lon_min, lon_max = -74.5, -72.8
    lat_min, lat_max = 40.0, 41.8

    coord_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]

    df = df.copy()
    m = np.ones(len(df), dtype=bool)
    m &= df["pickup_longitude"].between(lon_min, lon_max)
    m &= df["dropoff_longitude"].between(lon_min, lon_max)
    m &= df["pickup_latitude"].between(lat_min, lat_max)
    m &= df["dropoff_latitude"].between(lat_min, lat_max)
    m &= df[coord_cols].abs().sum(axis=1) > 0
    return df.loc[m]


train_df = filter_nyc_coords(train_df)



## === cell 13
train_df["distance"] = distance(
    train_df.pickup_latitude,
    train_df.pickup_longitude,
    train_df.dropoff_latitude,
    train_df.dropoff_longitude,
)

test_df_full["distance"] = distance(
    test_df_full.pickup_latitude,
    test_df_full.pickup_longitude,
    test_df_full.dropoff_latitude,
    test_df_full.dropoff_longitude,
)

train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=[
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "distance",
    ]
)
test_df_full = test_df_full.replace([np.inf, -np.inf], np.nan)
test_df_full["distance"] = test_df_full["distance"].fillna(0.0)




## === cell 14
def coord_sanity_filter(df):
    df = df.copy()
    lon_min, lon_max = -74.5, -72.8
    lat_min, lat_max = 40.0, 41.8
    m = np.ones(len(df), dtype=bool)
    m &= df["pickup_longitude"].between(lon_min, lon_max)
    m &= df["dropoff_longitude"].between(lon_min, lon_max)
    m &= df["pickup_latitude"].between(lat_min, lat_max)
    m &= df["dropoff_latitude"].between(lat_min, lat_max)
    return df.loc[m]


train_df = coord_sanity_filter(train_df)



## === cell 15
same_coord = (train_df["pickup_latitude"] == train_df["dropoff_latitude"]) & (
    train_df["pickup_longitude"] == train_df["dropoff_longitude"]
)
train_df = train_df.loc[~same_coord].copy()



## === cell 16
train_df = train_df[train_df["distance"] > 0].copy()



## === cell 17
train_df = train_df[train_df["distance"] < 30].copy()



## === cell 18
train_df.describe()



## === cell 19
train_df = train_df[train_df["fare_amount"].between(2.5, 200.0)].copy()



## === cell 20
train_df["passenger_count"] = pd.to_numeric(
    train_df["passenger_count"], errors="coerce"
)
test_df_full["passenger_count"] = pd.to_numeric(
    test_df_full["passenger_count"], errors="coerce"
)

train_df = train_df.dropna(subset=["passenger_count"]).copy()
test_df_full["passenger_count"] = test_df_full["passenger_count"].fillna(1)

train_df["passenger_count"] = np.round(train_df["passenger_count"]).astype("int16")
test_df_full["passenger_count"] = np.round(test_df_full["passenger_count"]).astype(
    "int16"
)

train_df = train_df[
    (train_df["passenger_count"] >= 1) & (train_df["passenger_count"] <= 6)
].copy()
test_df_full["passenger_count"] = test_df_full["passenger_count"].clip(lower=1, upper=6)



## === cell 21
fare_per_mile = train_df["fare_amount"] / train_df["distance"]
train_df = train_df[fare_per_mile.between(1.0, 35.0)].copy()




## === cell 22
def add_geo_features(df):
    df = df.copy()
    df["manhattan_deg"] = (
        (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
        + (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    ).astype("float32")
    eps = np.float32(1e-6)
    df["dist_manh_ratio"] = (
        df["distance"].astype("float32") / (df["manhattan_deg"] + eps)
    ).astype("float32")

    df["euclid_deg"] = np.sqrt(
        (df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
        + (df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
    ).astype("float32")
    return df


train_df = add_geo_features(train_df)
test_df_full = add_geo_features(test_df_full)




## === cell 23
def add_bearing_feature(df):
    df = df.copy()
    lat1 = np.deg2rad(df["pickup_latitude"].astype("float64").values)
    lon1 = np.deg2rad(df["pickup_longitude"].astype("float64").values)
    lat2 = np.deg2rad(df["dropoff_latitude"].astype("float64").values)
    lon2 = np.deg2rad(df["dropoff_longitude"].astype("float64").values)
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(y, x)  # [-pi, pi]
    df["bearing"] = bearing.astype("float32")
    return df


train_df = add_bearing_feature(train_df)
test_df_full = add_bearing_feature(test_df_full)




## === cell 24
def add_time_features(df, col="pickup_datetime"):
    df = df.copy()
    dt = pd.to_datetime(df[col], errors="coerce", utc=True)
    df["pickup_hour"] = dt.dt.hour.astype("float32")
    df["pickup_dow"] = dt.dt.dayofweek.astype("float32")
    df["pickup_month"] = dt.dt.month.astype("float32")
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype("float32")
    return df


train_df = add_time_features(train_df, "pickup_datetime")
test_df_full = add_time_features(test_df_full, "pickup_datetime")

train_df = train_df.dropna(
    subset=["pickup_hour", "pickup_dow", "pickup_month", "is_weekend"]
).copy()



## === cell 25
feat_cols_s = [
    "distance",
    "manhattan_deg",
    "dist_manh_ratio",
    "euclid_deg",
    "bearing",
    "passenger_count",
    "pickup_hour",
    "pickup_dow",
    "pickup_month",
    "is_weekend",
]

X = train_df[feat_cols_s]
y = train_df["fare_amount"]



## === cell 26
from sklearn.model_selection import train_test_split

train_df_sorted = train_df.copy()
train_df_sorted["_pickup_dt"] = pd.to_datetime(
    train_df_sorted["pickup_datetime"], errors="coerce", utc=True
)
train_df_sorted = (
    train_df_sorted.dropna(subset=["_pickup_dt"])
    .sort_values("_pickup_dt")
    .reset_index(drop=True)
)

X_sorted = train_df_sorted[feat_cols_s]

y_sorted = np.log1p(train_df_sorted["fare_amount"].astype("float32"))

cut = int(len(train_df_sorted) * 0.9)
X_train, X_test = X_sorted.iloc[:cut], X_sorted.iloc[cut:]
y_train, y_test = y_sorted.iloc[:cut], y_sorted.iloc[cut:]

perm = np.random.RandomState(RANDOM_STATE).permutation(len(X_train))
X_train = X_train.iloc[perm].reset_index(drop=True)
y_train = y_train.iloc[perm].reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)



## === cell 27
import xgboost as xgb




## === cell 28
def XGBoost(X_train, X_test, y_train, y_test, num_rounds=500):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": RANDOM_STATE,
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        early_stopping_rounds=20,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        verbose_eval=False,
    )
    return booster




## === cell 29
xgbm = XGBoost(X_train, X_test, y_train, y_test)

test_feat_full = test_df_full[feat_cols_s].copy()
test_feat_full = test_feat_full.replace([np.inf, -np.inf], np.nan).fillna(0.0)

dtest_submit = xgb.DMatrix(test_feat_full)

best_iter = getattr(xgbm, "best_iteration", None)
if best_iter is not None:
    pred_log = xgbm.predict(dtest_submit, iteration_range=(0, best_iter + 1))
else:
    pred_log = xgbm.predict(dtest_submit)

xgbm_pred_full = np.expm1(pred_log)

xgbm_pred_full = np.maximum(xgbm_pred_full, 0.0)

submission = pd.DataFrame(
    {"key": test_df_full["key"], "fare_amount": xgbm_pred_full},
    columns=["key", "fare_amount"],
)

submission.to_csv("XGboost_regression.csv", index=False)

print(submission.head())
print("Wrote submission to: XGboost_regression.csv")
