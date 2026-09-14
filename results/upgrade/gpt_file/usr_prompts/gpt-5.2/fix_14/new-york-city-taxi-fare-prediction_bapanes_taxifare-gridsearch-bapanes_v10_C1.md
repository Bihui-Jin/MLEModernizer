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

No external packages required in the script and installed.

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

3.830643716740832

# 6. Current score

11.7589

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.21833) has done: 'I fix the environment-breaking TensorFlow/Keras imports by removing the incompatible scikit-learn/Keras wrapper grid-search section and switching to `tf.keras` only, which is enough to train and predict with the same neural-network core logic. I also fix the file-path issues by reading the provided `../input/new-york-city-taxi-fare-prediction/train.csv` and `test.csv` instead of non-existent `my-taxi-fare-data/*` files. Because your code expects a `distance` feature that is not in the raw CSVs, I minimally add it (haversine distance) during preprocessing for both train and test. Finally, I ensure normalization uses the training-set mean/std (not test-set stats) and always writes a valid `submission_file.csv` with columns `key,fare_amount`.'
- What this solution (achieved 1066.87501) has done: 'I fix the TensorFlow import crash (`MessageFactory` / protobuf mismatch) by avoiding TensorFlow entirely and replacing it with a small NumPy-based linear regression that preserves your existing feature engineering (including the haversine `distance`) and normalization pipeline. This change removes the environment-breaking dependency while keeping the same supervised regression semantics and produces a valid `submission_file.csv` with `key,fare_amount`. I also add a minimal, standard NYC Taxi cleanup filter on the training shards (valid coordinate bounds and fare/passenger sanity) to improve RMSE toward the target without changing the feature set or introducing extra data. Finally, I keep the same file paths and ensure test keys stay aligned with predictions.'
- What this solution (achieved 1066.87503) has done: 'Your huge RMSE is most likely coming from a feature/normalization mismatch bug: you compute normalized features for training, but you accidentally fit the ridge regression on the *unnormalized* features. I make the minimal fix by training on `mynp_train_norm_*` (the already-normalized matrices) while keeping the exact same ridge model, same features, and same prediction pipeline. I also add a tiny numerical-stability tweak to the normalizer (cast mean/std to float64 during normalization) to avoid dtype drift, without changing semantics. This should move RMSE dramatically down toward the target by making train/test feature scales consistent.'
- What this solution (achieved 5.96502) has done: 'I fix the training shard reader so it doesn’t drop the CSV header when using `skiprows`, which is the root cause of the missing columns `KeyError` and the cascade of `NameError`s. I do this by reading with `header=0` and `skiprows=range(1, skip+1)` so pandas still reads the header while skipping only data rows. I also make the shard loading robust to occasional empty/over-filtered shards by skipping them and concatenating whatever valid shards are available (without changing the model or features). Finally, I keep the same ridge-regression training/prediction pipeline and ensure a valid `submission_file.csv` is always written with the required columns and row count.'
- What this solution (achieved 1066.87599) has done: 'Your current RMSE (5.965) is worse than the target (3.83), so we should improve accuracy with the smallest changes that keep your ridge + haversine core intact. The biggest score drag in your pipeline is that you apply the NYC “sanity filter” to the test set, which drops rows and forces you to fill them with a constant fallback—this typically hurts RMSE a lot. I keep the same model and features but stop filtering out test rows (only drop NaNs, then compute distance for all remaining rows), ensuring every test row gets a model-based prediction. I also make train-set mean/std truly global across all rows (not “mean of shard means”), which is a small correctness fix that usually improves calibration without changing the approach.'
- What this solution (achieved 1066.87599) has done: 'I fix the immediate runtime error in `_add_distance_feature` where `_haversine_np` is called with the wrong number of arguments, which currently prevents any training shards from loading and cascades into `NameError`s later. I keep the ridge-regression + haversine-distance core logic identical, only removing the accidental duplicate/invalid haversine call so features are computed consistently. I also preserve the existing shard-loading, global mean/std normalization, and submission writing flow, ensuring a valid `submission_file.csv` is produced end-to-end in the Kaggle environment. No score-target calibration is possible yet because no submission is being generated; this patch is primarily correctness/stability and should also restore reasonable RMSE by making training/test preprocessing consistent.'
- What this solution (achieved 1066.87599) has done: 'Your RMSE is exploding because test-time NaNs are filled using `global_mean` that is in *feature units*, but `global_mean` is computed on the *raw training features* while your training actually uses a *sanity-filtered subset* and then normalizes—this mismatch can create huge feature outliers after normalization and ruin predictions. I keep your ridge + haversine + normalization core unchanged, but compute `global_mean/global_std` from the exact concatenated training matrix you actually train on (after filtering and shuffling), so train/test scaling is consistent. I also add a minimal guard to replace any remaining non-finite test values *after* distance computation with the corresponding training means before normalization. These are small correctness fixes that should move RMSE sharply down toward your 3.83 target without changing the model or feature set.'
- What this solution (achieved 10.84802) has done: 'Your RMSE is catastrophically high because the model is producing some extreme fare predictions (then clipped only at 0), which typically happens when a few test rows have corrupted coordinates/passenger_count that create huge “distance” values and explode a linear model. To move the score down toward your 3.83 target with minimal changes and without altering the model/training, I keep the exact ridge-regression + normalization pipeline but add the same basic NYC coordinate/passenger sanity filter *as a capping transform* for test features (winsorize/clip to training-valid bounds rather than dropping rows). I also clip the computed `distance` to a reasonable upper bound derived from the training distribution (e.g., 99.9th percentile) to prevent outliers from dominating predictions, while still predicting every test row. These changes preserve your core logic and should greatly reduce outlier-driven errors that cause RMSE ~1000.'
- What this solution (achieved 10.81671) has done: 'Your current RMSE (10.85) is far above the target (3.83), so we need a modest accuracy boost without changing the ridge+distance core. The biggest gain with minimal risk is to add the standard time-derived features (hour/day-of-week/month/year) from `pickup_datetime`, since fare depends strongly on time patterns and these are already available in both train/test. To keep semantics consistent, we compute these features for both train shards and test, include them in the same normalization, and keep the same closed-form ridge training/prediction. We also add a tiny guard to ensure parsed datetimes that fail become neutral (filled with training means), avoiding NaN-induced prediction blowups.'
- What this solution (achieved 11.7589) has done: 'I fix the runtime error that prevents shard loading by correcting `_add_distance_feature` to call `_haversine_np` with the proper 4-argument signature (it currently passes 5 args). Then I keep the rest of your ridge+normalization+log1p target pipeline the same so it runs end-to-end, ensuring `global_mean/std`, `DIST_MAX`, and all downstream variables are defined. Finally, I make sure the script always writes a valid `submission_file.csv` with exactly the required `key,fare_amount` columns and the full test row count.'

# 9. Code solution

## === cell 0
print("hello moto")



## === cell 1
import os
import numpy as np
import pandas as pd

np.random.seed(42)

TRAIN_PATH = "../input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "../input/new-york-city-taxi-fare-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"

for p in [TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Expected file not found: {p}")

print("Using NumPy-only training to avoid TF/protobuf runtime crash.")
print(
    "train exists:",
    os.path.exists(TRAIN_PATH),
    "test exists:",
    os.path.exists(TEST_PATH),
)




## === cell 2
def _haversine_np(lon1, lat1, lon2, lat2):
    """
    Vectorized haversine distance in kilometers.
    """
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)

    rlon1 = np.radians(lon1)
    rlat1 = np.radians(lat1)
    rlon2 = np.radians(lon2)
    rlat2 = np.radians(lat2)

    dlon = rlon2 - rlon1
    dlat = rlat2 - rlat1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0  # km
    return R * c


def _add_distance_feature(df):
    """
    Adds the 'distance' column expected by the original code.

    Bugfix: _haversine_np takes 4 args; previous code passed 5 which crashed.
    """
    df = df.copy()
    df["distance"] = _haversine_np(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    )
    return df


def _add_time_features(df):
    df = df.copy()
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["pickup_hour"] = dt.dt.hour.astype("float32")
    df["pickup_dayofweek"] = dt.dt.dayofweek.astype("float32")
    df["pickup_month"] = dt.dt.month.astype("float32")
    df["pickup_year"] = dt.dt.year.astype("float32")
    return df


def data_to_np(input_file, nrows=900000):
    """
    Reads a slice of the training file, builds the same feature set the original code expects,
    and returns (X, y).
    """
    df = pd.read_csv(input_file, sep=",", nrows=nrows)

    df = df.dropna(
        subset=[
            "fare_amount",
            "pickup_datetime",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]
    )

    df = _add_distance_feature(df)
    df = _add_time_features(df)

    header_names = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
        "pickup_year",
    ]

    df_train = df[header_names].astype(np.float32)
    np_df_train = df_train.values

    df_label = df["fare_amount"].astype(np.float32)
    np_df_label = df_label.values

    return np_df_train, np_df_label




## === cell 3
def global_mean_per_column(mynp_train_list):
    total_sum = None
    total_n = 0
    for X in mynp_train_list:
        X64 = np.asarray(X, dtype=np.float64)
        if total_sum is None:
            total_sum = X64.sum(axis=0)
        else:
            total_sum += X64.sum(axis=0)
        total_n += X64.shape[0]
    mean = total_sum / max(total_n, 1)
    return mean.astype(np.float64)




## === cell 4
def global_std_per_column(mynp_train_list, global_mean):
    total_ss = None
    total_n = 0
    mu = np.asarray(global_mean, dtype=np.float64)
    for X in mynp_train_list:
        X64 = np.asarray(X, dtype=np.float64)
        diff = X64 - mu
        ss = (diff * diff).sum(axis=0)
        if total_ss is None:
            total_ss = ss
        else:
            total_ss += ss
        total_n += X64.shape[0]
    var = total_ss / max(total_n, 1)
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1.0, std)
    return std.astype(np.float64)




## === cell 5
def norm_mynp_train(mynp_train, mean, std):
    mean64 = np.asarray(mean, dtype=np.float64)
    std64 = np.asarray(std, dtype=np.float64)
    X64 = np.asarray(mynp_train, dtype=np.float64)
    mynp_train_norm = (X64 - mean64) / std64
    return mynp_train_norm.astype(np.float32)




## === cell 6
def _basic_nyc_sanity_filter(df, has_fare=True):
    """
    Remove clearly invalid rows that hurt RMSE.
    Keeps the same features/target, only cleans corrupted/outlier points.
    """
    cond = (
        df["pickup_longitude"].between(-75, -72)
        & df["dropoff_longitude"].between(-75, -72)
        & df["pickup_latitude"].between(40, 42)
        & df["dropoff_latitude"].between(40, 42)
        & df["passenger_count"].between(1, 6)
    )
    if has_fare:
        cond = cond & df["fare_amount"].between(2.5, 250.0)
    return df.loc[cond].copy()


def load_train_shards(train_path, shard_rows=300000, n_shards=3):
    """
    Read multiple shards while keeping the header intact and skipping only data rows.
    """
    mynp_trains, mynp_labels = [], []
    header_names = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
        "pickup_year",
    ]

    for i in range(n_shards):
        skip = i * shard_rows
        if skip > 0:
            df = pd.read_csv(
                train_path,
                nrows=shard_rows,
                header=0,
                skiprows=range(1, skip + 1),
            )
        else:
            df = pd.read_csv(train_path, nrows=shard_rows)

        needed_cols = [
            "fare_amount",
            "pickup_datetime",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]
        if not set(needed_cols).issubset(df.columns):
            continue

        df = df.dropna(subset=needed_cols)
        df = _basic_nyc_sanity_filter(df, has_fare=True)
        if df.empty:
            continue

        df = _add_distance_feature(df)
        df = _add_time_features(df)

        X = df[header_names].astype(np.float32).values
        y = df["fare_amount"].astype(np.float32).values
        if X.shape[0] == 0:
            continue

        mynp_trains.append(X)
        mynp_labels.append(y)

    if len(mynp_trains) == 0:
        raise RuntimeError(
            "No usable training shards loaded (all empty or failed parsing)."
        )
    return mynp_trains, mynp_labels




## === cell 7
mynp_train_list, mynp_label_list = load_train_shards(
    TRAIN_PATH, shard_rows=300000, n_shards=3
)

print([m.shape for m in mynp_train_list], [l.shape for l in mynp_label_list])




## === cell 8
def _shuffle_pair(X, y):
    order = np.argsort(np.random.random(y.shape))
    return X[order], y[order]


mynp_train_list_shuf = []
mynp_label_list_shuf = []
for X, y in zip(mynp_train_list, mynp_label_list):
    Xs, ys = _shuffle_pair(X, y)
    mynp_train_list_shuf.append(Xs)
    mynp_label_list_shuf.append(ys)

mynp_train_list = mynp_train_list_shuf
mynp_label_list = mynp_label_list_shuf



## === cell 9
mynp_train_list = [
    np.where(np.isfinite(X), X, np.nan).astype(np.float32) for X in mynp_train_list
]

global_mean = global_mean_per_column(
    [np.where(np.isfinite(X), X, 0.0).astype(np.float32) for X in mynp_train_list]
)

feat_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "pickup_year",
]
feat_index = {c: i for i, c in enumerate(feat_cols)}

mynp_train_list_filled = []
for X in mynp_train_list:
    X64 = X.astype(np.float64)
    mask = ~np.isfinite(X64)
    if mask.any():
        X64[mask] = np.take(global_mean, np.where(mask)[1])
    mynp_train_list_filled.append(X64.astype(np.float32))
mynp_train_list = mynp_train_list_filled

global_mean = global_mean_per_column(mynp_train_list)
global_std = global_std_per_column(mynp_train_list, global_mean)
print("global_mean:", global_mean)
print("global_std :", global_std)

_train_dist = np.concatenate(
    [X[:, feat_index["distance"]].astype(np.float64) for X in mynp_train_list], axis=0
)
DIST_MAX = float(np.nanpercentile(_train_dist, 99.9))
if not np.isfinite(DIST_MAX) or DIST_MAX <= 0:
    DIST_MAX = 200.0  # safe fallback (km)
print("DIST_MAX (train p99.9):", DIST_MAX)

mynp_train_list = [
    np.concatenate(
        [
            X[:, : feat_index["distance"]],
            np.clip(
                X[:, feat_index["distance"] : feat_index["distance"] + 1], 0.0, DIST_MAX
            ),
            X[:, feat_index["distance"] + 1 :],
        ],
        axis=1,
    ).astype(np.float32)
    for X in mynp_train_list
]



## === cell 10
mynp_train_norm_list = [
    norm_mynp_train(X, global_mean, global_std) for X in mynp_train_list
]
mynp_train_concat = np.concatenate(mynp_train_norm_list, axis=0)
mynp_label_concat = np.concatenate(mynp_label_list, axis=0)

print(mynp_train_concat.shape, mynp_label_concat.shape)

mynp_label_concat_trans = np.log1p(
    np.clip(mynp_label_concat.astype(np.float64), 0.0, None)
).astype(np.float64)




## === cell 11
def fit_ridge_regression_closed_form(X, y, l2=1.0):
    """
    Fit ridge regression with intercept using a closed-form solve:
    w = (X^T X + l2*I)^(-1) X^T y, with intercept unregularized.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

    Xb = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)

    XtX = Xb.T @ Xb
    reg = np.eye(XtX.shape[0], dtype=np.float64) * float(l2)
    reg[0, 0] = 0.0  # don't regularize intercept

    Xty = Xb.T @ y
    w = np.linalg.solve(XtX + reg, Xty)  # (d+1, 1)
    return w.reshape(-1)


def predict_ridge_regression(X, w):
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    Xb = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)
    return (Xb @ w).astype(np.float64)




## === cell 12
w = fit_ridge_regression_closed_form(mynp_train_concat, mynp_label_concat_trans, l2=1.0)
print("Trained ridge weights shape:", w.shape)



## === cell 13
df_test_full = pd.read_csv(TEST_PATH, sep=",")

needed_test_cols = [
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

for c in needed_test_cols:
    if c != "pickup_datetime":
        df_test_full[c] = pd.to_numeric(df_test_full[c], errors="coerce")
        df_test_full[c] = df_test_full[c].fillna(global_mean[feat_index[c]])

df_test_full["pickup_longitude"] = df_test_full["pickup_longitude"].clip(-75.0, -72.0)
df_test_full["dropoff_longitude"] = df_test_full["dropoff_longitude"].clip(-75.0, -72.0)
df_test_full["pickup_latitude"] = df_test_full["pickup_latitude"].clip(40.0, 42.0)
df_test_full["dropoff_latitude"] = df_test_full["dropoff_latitude"].clip(40.0, 42.0)
df_test_full["passenger_count"] = df_test_full["passenger_count"].clip(1.0, 6.0)

df_test_full = _add_distance_feature(df_test_full)
df_test_full = _add_time_features(df_test_full)

for c in ["distance", "pickup_hour", "pickup_dayofweek", "pickup_month", "pickup_year"]:
    df_test_full[c] = pd.to_numeric(df_test_full[c], errors="coerce")
    df_test_full[c] = df_test_full[c].fillna(global_mean[feat_index[c]])

df_test_full["distance"] = df_test_full["distance"].clip(0.0, DIST_MAX)

df_test = df_test_full[["key"] + feat_cols].copy()
df_test_fn = df_test[feat_cols].astype(np.float32)

mynp_test = df_test_fn.values
mynp_test = (
    mynp_test.astype(np.float64) - np.asarray(global_mean, dtype=np.float64)
) / np.asarray(global_std, dtype=np.float64)
mynp_test = mynp_test.astype(np.float32)

print("Test matrix shape:", mynp_test.shape)
print("Original test rows:", len(df_test_full), "Used test rows:", len(df_test_full))



## === cell 14
test_predictions_trans = predict_ridge_regression(mynp_test, w).reshape(-1)
test_predictions = np.expm1(test_predictions_trans)
test_predictions = np.clip(test_predictions, 0.0, None)

print(test_predictions[:5], test_predictions.shape)



## === cell 15
df_output = pd.DataFrame(
    {"key": df_test["key"].values, "fare_amount": test_predictions}
)
df_output = df_output[["key", "fare_amount"]]

if df_output.shape[0] != df_test_full.shape[0]:
    raise RuntimeError("Submission row count mismatch with test set.")

print(df_output.head())

out_path = "submission_file.csv"
df_output.to_csv(out_path, index=False)

print("Wrote:", out_path, "rows:", len(df_output))
print(df_output.dtypes)
