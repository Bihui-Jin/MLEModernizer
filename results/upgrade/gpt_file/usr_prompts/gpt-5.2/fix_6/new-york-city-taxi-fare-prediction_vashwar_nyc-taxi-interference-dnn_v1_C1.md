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

4.13939939564504

# 6. Current score

36.71466

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.67607) has done: 'I fix the TensorFlow/Keras 3 compatibility errors by replacing the deprecated `tf1.layers.dense` calls with `tf1.keras.layers.Dense` while keeping the same architecture (same number of layers/neurons, ReLU, and He initialization). I also remove the TF1 protobuf-related import crash by avoiding importing TensorFlow at the very start and by using TF1-compat mode only after data loading. Finally, I ensure the graph is built successfully so `init` and `y_test` exist, and the script always writes a valid `submission.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 9.96593) has done: 'I fix the TensorFlow import crash (`MessageFactory`/protobuf mismatch) by avoiding TensorFlow entirely and switching the training/inference to a lightweight NumPy linear regression that uses the exact same features and scaling already computed. This keeps the overall semantics (regression on the engineered/scaled inputs) while making the notebook stable in the Kaggle Python 3.7 environment and ensuring a `submission.csv` is always written. To move RMSE down toward your target, I also compute `mu`/`sigma` from the actual loaded training sample instead of using the hardcoded arrays (which likely mismatch your sampled train distribution), and use a proper train/valid shuffle split to prevent ordering bias. The rest of the pipeline (date features, feature list, clipping non-negative fares, submission format) stays the same.'
- What this solution (achieved 1066.88915) has done: 'Your current RMSE (9.96593) is far above the target (4.1394), so we need a real but still minimal modeling improvement without changing the overall approach (feature engineering + scaling + simple regression). The biggest win for this competition with your exact setup is to add a single strong engineered feature: great-circle (Haversine) distance between pickup/dropoff, while keeping the same linear ridge closed-form model. I also apply the same basic coordinate bounding-box cleaning used widely for this dataset (NYC-area filtering) to reduce label noise in the sampled training rows, which typically drops RMSE substantially. Everything else (data paths, date features, scaling, ridge solver, clipping, and submission format) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 36.71466) has done: 'Your RMSE is catastrophically worse than the target, which strongly suggests a correctness bug rather than a modeling gap. The biggest issue is that your `y_test` predictions are being clipped at 0, but taxi fares in this dataset should never be near 0 and the competition expects realistic positive fares; more importantly, your model can produce negative values due to linear extrapolation, and clipping to 0 creates huge errors for many cases. I keep your exact pipeline (same features, haversine, scaling, ridge closed-form) and make only minimal, metric-aligned post-processing: clip predictions to a realistic fare range (e.g., [2.5, 250]) instead of [0, inf), matching the training filter and the known minimum fare behavior. I also ensure `passenger_count` is constrained to a reasonable range in both train/test (1–6) to reduce extreme outliers without changing the model.'

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)

print("Listing ../input:")
print(os.listdir("../input"))



## === cell 1
DATA_DIR = "../input/new-york-city-taxi-fare-prediction"
print("Data dir exists:", os.path.exists(DATA_DIR))
print("Files:", [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])



## === cell 2
df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print(df_test.shape)
print(df_test.columns.tolist())




## === cell 3
def add_datepart(df, fldname, drop=True):
    """
    Create datetime-derived columns; compatible with pandas versions where .dt.week is removed.
    """
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(
            fld, infer_datetime_format=True, errors="coerce"
        )

    targ_pre = re.sub("[Dd]ate$", "", fldname)

    df[targ_pre + "Year"] = fld.dt.year
    df[targ_pre + "Month"] = fld.dt.month
    try:
        df[targ_pre + "Week"] = fld.dt.isocalendar().week.astype(np.int16)
    except Exception:
        df[targ_pre + "Week"] = fld.dt.strftime("%V").astype(np.int16)

    df[targ_pre + "Day"] = fld.dt.day
    df[targ_pre + "Dayofweek"] = fld.dt.dayofweek
    df[targ_pre + "Dayofyear"] = fld.dt.dayofyear
    df[targ_pre + "hour"] = fld.dt.hour

    df[targ_pre + "Is_month_end"] = fld.dt.is_month_end.astype(np.int8)
    df[targ_pre + "Is_month_start"] = fld.dt.is_month_start.astype(np.int8)
    df[targ_pre + "Is_quarter_end"] = fld.dt.is_quarter_end.astype(np.int8)
    df[targ_pre + "Is_quarter_start"] = fld.dt.is_quarter_start.astype(np.int8)
    df[targ_pre + "Is_year_end"] = fld.dt.is_year_end.astype(np.int8)
    df[targ_pre + "Is_year_start"] = fld.dt.is_year_start.astype(np.int8)

    try:
        elapsed = (fld.view("int64") // 10**9).astype(np.int64)
    except Exception:
        elapsed = (fld.astype("int64") // 10**9).astype(np.int64)
    df[targ_pre + "Elapsed"] = elapsed

    if drop:
        df.drop(fldname, axis=1, inplace=True)




## === cell 4
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(np.float64))
    lat1 = np.radians(lat1.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    r = 6371.0
    return (r * c).astype(np.float32)


add_datepart(df_test, "pickup_datetime", drop=True)

feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "pickup_datetimeYear",
    "pickup_datetimeMonth",
    "pickup_datetimeWeek",
    "pickup_datetimeDay",
    "pickup_datetimeDayofweek",
    "pickup_datetimeDayofyear",
    "pickup_datetimehour",
    "pickup_datetimeElapsed",
    "haversine_km",
]

df_test["haversine_km"] = haversine_km(
    df_test["pickup_longitude"].values,
    df_test["pickup_latitude"].values,
    df_test["dropoff_longitude"].values,
    df_test["dropoff_latitude"].values,
)

df_test["passenger_count"] = df_test["passenger_count"].clip(1, 6)

missing = [c for c in feature_cols if c not in df_test.columns]
print("Missing test columns:", missing)
print(df_test[feature_cols].head())



## === cell 5
train_path = os.path.join(DATA_DIR, "train.csv")

N_TRAIN = 600000
usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

df_train = pd.read_csv(train_path, usecols=usecols, nrows=N_TRAIN)
df_train = df_train.dropna()

df_train = df_train[df_train["passenger_count"].between(1, 6)]

df_train = df_train[(df_train["fare_amount"] > 0) & (df_train["fare_amount"] < 250)]

nyc_bounds = (
    (df_train["pickup_longitude"].between(-75, -72))
    & (df_train["dropoff_longitude"].between(-75, -72))
    & (df_train["pickup_latitude"].between(40, 42))
    & (df_train["dropoff_latitude"].between(40, 42))
)
df_train = df_train[nyc_bounds]

add_datepart(df_train, "pickup_datetime", drop=True)

df_train["haversine_km"] = haversine_km(
    df_train["pickup_longitude"].values,
    df_train["pickup_latitude"].values,
    df_train["dropoff_longitude"].values,
    df_train["dropoff_latitude"].values,
)

for c in feature_cols:
    if c not in df_train.columns:
        df_train[c] = 0

X_all_unscl = df_train[feature_cols].astype(np.float32).values
y_all = df_train["fare_amount"].astype(np.float32).values.reshape(-1, 1)

mu = X_all_unscl.mean(axis=0).astype(np.float32)
sigma = X_all_unscl.std(axis=0).astype(np.float32)
sigma = np.where(sigma < 1e-6, 1.0, sigma).astype(np.float32)

X_all = (X_all_unscl - mu) / sigma

rng = np.random.RandomState(0)
idx = np.arange(X_all.shape[0])
rng.shuffle(idx)

split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

X_train, X_valid = X_all[tr_idx], X_all[va_idx]
y_train, y_valid = y_all[tr_idx], y_all[va_idx]

print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
print("mu shape:", mu.shape, "sigma shape:", sigma.shape)




## === cell 6
def fit_ridge_closed_form(X, y, alpha=1.0):
    """
    Closed-form ridge regression with intercept.
    X: (n, d) float32/float64
    y: (n, 1)
    Returns w (d,), b (scalar)
    """
    X = X.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)

    n, d = X.shape
    X1 = np.concatenate([X, np.ones((n, 1), dtype=np.float64)], axis=1)  # add bias
    I = np.eye(d + 1, dtype=np.float64)
    I[-1, -1] = 0.0  # don't regularize intercept

    A = X1.T @ X1 + alpha * I
    B = X1.T @ y
    w_full = np.linalg.solve(A, B)  # (d+1,1)
    w = w_full[:-1, 0]
    b = w_full[-1, 0]
    return w.astype(np.float32), np.float32(b)


def predict_linear(X, w, b):
    return (X.astype(np.float32, copy=False) @ w.reshape(-1, 1) + b).astype(np.float32)


w, b = fit_ridge_closed_form(X_train, y_train, alpha=1.0)

y_valid_pred = predict_linear(X_valid, w, b)
valid_rmse = float(np.sqrt(np.mean((y_valid_pred - y_valid) ** 2)))
print(f"valid_rmse={valid_rmse:.5f}")



## === cell 7
x_test_unscl = df_test[feature_cols].astype(np.float32).values
x_test = (x_test_unscl - mu) / sigma
print("x_test:", x_test.shape)

y_test = predict_linear(x_test, w, b).reshape(-1)

y_test = np.clip(y_test.astype(np.float32), 2.5, 250.0)

print(y_test[:5], y_test.shape)



## === cell 8
my_submission = pd.DataFrame(
    {"key": df_test["key"].astype(str).values, "fare_amount": y_test}
)
my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())
print("Exists submission.csv:", os.path.exists("submission.csv"))
