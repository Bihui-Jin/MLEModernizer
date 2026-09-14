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

3.14

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

4.736901452993294

# 6. Current score

5.36737

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 360.22067) has done: 'Your huge RMSE is coming from a feature mismatch between train and test: in training you include `pickup_datetime` (as raw strings) but in test you drop it and also shift columns (`iloc[:,2:]`), so the model is trained on a different set/order of features than it sees at inference. I make train/test use the exact same feature columns in the same order, and I convert `pickup_datetime` into numeric time features (year/month/day/hour/weekday) in both sets so scikit-learn can learn from it without breaking. I also ensure the `key` column is preserved for submission while never used as a feature, and keep the rest of your filtering, haversine feature, and VotingRegressor core logic intact. These minimal fixes should move RMSE dramatically down toward your target (lower is better) without changing your modeling approach.'
- What this solution (achieved 360.22256) has done: 'Your current score is far worse than the target (lower-is-better), so we should improve performance while keeping your core model/feature logic intact. The biggest safe gain here is to ensure the datetime-derived features are numeric (not pandas nullable Int dtypes that can cause dtype/object promotion in some sklearn paths) and that train/test feature matrices have identical dtypes and no NaNs after datetime parsing. I also make the BaggingRegressor parameter compatible with sklearn 1.2.2 (`base_estimator` instead of `estimator`) to avoid silent incompatibilities, and I keep your filtering, haversine feature, and VotingRegressor approach unchanged. These are minimal, execution-safe changes that should substantially reduce RMSE toward your target by removing feature/dtype issues without changing the modeling approach.'
- What this solution (achieved 14.60597) has done: 'Your RMSE is orders of magnitude too high for this competition, which almost always indicates a train/test feature matrix mismatch or severe out-of-distribution predictions. I keep your exact model ensemble and your existing filters/features, but I (1) enforce identical numeric dtypes for all feature columns in both train and test, (2) remove any remaining non-finite rows after feature engineering, and (3) add a small, competition-appropriate prediction sanity clip (upper bound) to prevent a few extreme predictions from dominating RMSE. These are minimal changes that preserve your core logic and typically move RMSE dramatically toward the target by eliminating dtype/object promotion issues and outlier blow-ups. The submission format and paths remain unchanged and the script still writes `submission.csv`.'
- What this solution (achieved 14.31252) has done: 'Your current RMSE (14.61) is still far above the target (4.74, lower-is-better), so we should improve without changing your ensemble/models. The biggest issue left is that the SVR inside Bagging is being trained on unscaled features; SVR is very sensitive to feature scaling, and this can easily dominate the VotingRegressor and hurt generalization. I wrap the SVR in a `Pipeline(StandardScaler -> SVR)` (same model, same training loop), and I also make the imputation consistent by using the training-feature medians computed from `X` (instead of `np.nanmedian(X.to_numpy())` which depends on column order and can be brittle). Everything else (filters, haversine, time features, VotingRegressor, submission format/path) stays the same.'
- What this solution (achieved 5.42208) has done: 'Your current RMSE (14.31, lower-is-better) is still far above the target (4.74), and the most likely remaining issue is train/test distribution mismatch from filtering only the training set while leaving the test set unfiltered (especially for out-of-range coordinates and distance). I keep your exact model ensemble, training loop, and feature set, but apply the same geographic and distance sanity filters to the test features in a submission-safe way (do not drop rows; instead mark invalid rows and fall back to a conservative prediction). I also stop clipping at 100 (too low for some real fares) and instead use a wider, still-safe clip to reduce large error from underprediction on higher fares. These are minimal changes aimed at reducing extreme errors without changing the modeling approach.'
- What this solution (achieved 5.34558) has done: 'Your current RMSE (5.422) is worse than the target (4.737, lower-is-better), so we should make a small improvement without changing your ensemble or feature set. The most likely easy gain is to add a simple, competition-standard post-processing calibration: fit a linear correction `y ≈ a*pred + b` on a small held-out validation split from your training data, then apply it to test predictions; this often reduces RMSE by correcting systematic bias/scale. I also preserve your existing “invalid test rows fallback” logic by applying the calibration only to valid rows, and keep the same clipping behavior. Everything else (filters, haversine, time features, model definitions, training loop) stays intact and the script still writes `submission.csv`.'
- What this solution (achieved 5.36906) has done: 'The timeout is dominated by fitting the `VotingRegressor` because it includes an RBF `SVR` inside a `BaggingRegressor` (10 estimators), which is very expensive; additionally, `VotingRegressor(n_jobs=-1)` can oversubscribe threads and slow everything down. The fastest safe wins without changing core logic are: enable Intel scikit-learn acceleration (if available), reduce redundant pandas conversions/copies by computing masks once and using NumPy arrays for filtering, and prevent thread oversubscription by capping BLAS/OpenMP threads while still using model-level parallelism. These changes are provably equivalent (same data, same models, same training loops) and only reduce overhead/parallel contention. The rest of the pipeline (feature engineering, calibration, prediction, submission) remains identical in semantics.'
- What this solution (achieved 5.36906) has done: 'Your current RMSE (5.369) is worse than the target (4.737, lower-is-better), so the goal is a small, safe improvement without changing your ensemble or feature set. The biggest issue is that your “calibration” is fit on the same data the model was trained on, which tends to overfit and can worsen test RMSE; I keep the exact idea but fit the linear correction on a small validation split instead. I also make the calibration robust by clipping the learned slope/intercept to reasonable bounds so it can’t harm performance if the split is noisy. Everything else (data loading, filters, haversine/time features, models, prediction fallback, clipping, and submission writing) stays the same.'
- What this solution (achieved 5.36906) has done: 'Your current RMSE (5.369) is worse than the target (4.737, lower-is-better), so we should make a small, low-risk improvement without changing the model ensemble or feature set. The main issue is that you fit the calibration on a validation split but keep the model trained on *all* data, which makes the calibration slightly misaligned; I instead split first, fit the VotingRegressor on the train split, learn calibration on the val split, then refit the same VotingRegressor on the full data for final test predictions (same core logic, just correct ordering). I also make the calibration use a robust closed-form least-squares fit (with fallback) and apply it to all test rows (not only “valid” rows), while still using your conservative fallback for invalid rows—this typically reduces systematic bias and improves RMSE modestly. All I/O paths and the submission format remain unchanged, and it still write `submission.csv`.'
- What this solution (achieved 5.37194) has done: 'Your current RMSE (5.369) is still above the target (4.737, lower-is-better), so we want a small, low-risk improvement without changing your ensemble or feature set. The main safe gain is to make the calibration less noisy by using a larger validation split and to fit the linear correction with a standard `LinearRegression` (same semantic goal, fewer numerical edge cases than the manual closed-form), then apply it exactly as before. I also make the “invalid-row fallback” value more consistent with the training distribution by using the **median of the model predictions on the training set** (rather than median of y), which typically reduces error when the model has a systematic bias (still conservative and label-leak-free). Everything else (data reading, filters, haversine/time features, models, training loop, clipping, and submission writing) remains the same.'
- What this solution (achieved 5.36737) has done: 'Your current RMSE (5.37194) is worse than the target (4.7369, lower-is-better), so we want a small, low-risk improvement without changing your ensemble, features, or training loop. The biggest safe lever is to stop throwing away higher fares in training (you currently cap `fare_amount < 100`), because Kaggle’s test set includes fares above 100 and this truncation causes systematic underprediction that hurts RMSE. I widen that training fare filter to a conservative upper bound and widen the final prediction clip accordingly, keeping the rest of your filtering/feature engineering/calibration/VotingRegressor intact. This should reduce bias on higher-fare trips and move RMSE closer to the target.'

# 9. Code solution

## === cell 0
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)



## === cell 1
TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"



## === cell 2
train_usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
train_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}
df_train = pd.read_csv(
    TRAIN_PATH,
    nrows=2_000_000,
    usecols=train_usecols,
    dtype=train_dtypes,
    parse_dates=["pickup_datetime"],
)



## === cell 3
pass



## === cell 4
pass



## === cell 5
pass



## === cell 6
pass



## === cell 7
pass



## === cell 8
df_train = df_train.dropna()

m = (df_train["fare_amount"] > 1) & (df_train["fare_amount"] < 300)

m &= (df_train["passenger_count"] >= 1) & (df_train["passenger_count"] <= 6)
m &= (
    (df_train["pickup_longitude"] > -75)
    & (df_train["pickup_longitude"] < -72)
    & (df_train["dropoff_longitude"] > -75)
    & (df_train["dropoff_longitude"] < -72)
    & (df_train["pickup_latitude"] > 40)
    & (df_train["pickup_latitude"] < 42)
    & (df_train["dropoff_latitude"] > 40)
    & (df_train["dropoff_latitude"] < 42)
)
df_train = df_train.loc[m]




## === cell 9
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # kilometers




## === cell 10
df_train["distance_km"] = haversine(
    df_train["pickup_latitude"],
    df_train["pickup_longitude"],
    df_train["dropoff_latitude"],
    df_train["dropoff_longitude"],
)
df_train = df_train[(df_train["distance_km"] > 0.1) & (df_train["distance_km"] < 30)]




## === cell 11
def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df = df.copy()
    df["pickup_year"] = dt.dt.year.astype(np.int16)
    df["pickup_month"] = dt.dt.month.astype(np.int8)
    df["pickup_day"] = dt.dt.day.astype(np.int8)
    df["pickup_hour"] = dt.dt.hour.astype(np.int8)
    df["pickup_weekday"] = dt.dt.weekday.astype(np.int8)
    return df


df_train = add_time_features(df_train)
df_train = df_train.dropna(
    subset=[
        "pickup_year",
        "pickup_month",
        "pickup_day",
        "pickup_hour",
        "pickup_weekday",
    ]
)



## === cell 12
feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance_km",
    "pickup_year",
    "pickup_month",
    "pickup_day",
    "pickup_hour",
    "pickup_weekday",
]

X = df_train[feature_cols]
y = df_train["fare_amount"]



## === cell 13
X = X.astype(np.float32, copy=False)
y = y.astype(np.float32, copy=False)

X_np = X.to_numpy(dtype=np.float32, copy=False)
y_np = y.to_numpy(dtype=np.float32, copy=False)

mask_finite = np.isfinite(X_np).all(axis=1) & np.isfinite(y_np)
if not mask_finite.all():
    X_np = X_np[mask_finite]
    y_np = y_np[mask_finite]
    X = X.iloc[np.flatnonzero(mask_finite)]
    y = y.iloc[np.flatnonzero(mask_finite)]

train_feature_medians = X.median(axis=0)
train_target_median = float(np.median(y_np))



## === cell 14
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=2,
    bootstrap=True,
    max_samples=5000,
    n_jobs=1,
)



## === cell 15
from sklearn.linear_model import LinearRegression

lr = LinearRegression()



## === cell 16
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

svr_scaled = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svr", SVR(kernel="rbf")),
    ]
)

br = BaggingRegressor(
    n_estimators=10,
    base_estimator=svr_scaled,
    max_samples=5000,
    bootstrap=True,
    random_state=2,
)



## === cell 17
from sklearn.ensemble import VotingRegressor

vr = VotingRegressor([("lr", lr), ("rf", rf), ("svr", br)], verbose=False, n_jobs=1)



## === cell 18
from sklearn.model_selection import train_test_split

idx = np.arange(len(X), dtype=np.int64)
idx_tr, idx_val = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)

X_tr = X.iloc[idx_tr]
y_tr = y.iloc[idx_tr]
X_val = X.iloc[idx_val]
y_val = y.iloc[idx_val].to_numpy(dtype=np.float32, copy=False)

vr.fit(X_tr, y_tr)

pred_val = vr.predict(X_val).astype(np.float32, copy=False)
m = np.isfinite(pred_val) & np.isfinite(y_val)
pred_val = pred_val[m]
y_val2 = y_val[m]

if pred_val.size >= 500:
    cal = LinearRegression()
    cal.fit(pred_val.reshape(-1, 1).astype(np.float64), y_val2.astype(np.float64))
    a = float(cal.coef_.ravel()[0])
    b = float(cal.intercept_)
    a = float(np.clip(a, 0.5, 1.5))
    b = float(np.clip(b, -20.0, 20.0))
else:
    a, b = 1.0, 0.0

print(f"Calibration (val-based): y ≈ {a:.6f} * pred + {b:.6f}")

vr.fit(X, y)

pred_train = vr.predict(X).astype(np.float32, copy=False)
pred_train = pred_train[np.isfinite(pred_train)]
train_pred_median = (
    float(np.median(pred_train)) if pred_train.size else train_target_median
)



## === cell 19
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_dtypes = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}
df_test = pd.read_csv(
    TEST_PATH,
    usecols=test_usecols,
    dtype=test_dtypes,
    parse_dates=["pickup_datetime"],
)



## === cell 20
key = df_test["key"].copy()



## === cell 21
df_test["distance_km"] = haversine(
    df_test["pickup_latitude"],
    df_test["pickup_longitude"],
    df_test["dropoff_latitude"],
    df_test["dropoff_longitude"],
)

df_test = add_time_features(df_test)

for c in ["pickup_year", "pickup_month", "pickup_day", "pickup_hour", "pickup_weekday"]:
    if df_test[c].isna().any():
        df_test[c] = (
            df_test[c]
            .fillna(df_test[c].mode(dropna=True).iloc[0])
            .astype(df_test[c].dtype)
        )

valid_geo = (
    (df_test["pickup_longitude"] > -75)
    & (df_test["pickup_longitude"] < -72)
    & (df_test["dropoff_longitude"] > -75)
    & (df_test["dropoff_longitude"] < -72)
    & (df_test["pickup_latitude"] > 40)
    & (df_test["pickup_latitude"] < 42)
    & (df_test["dropoff_latitude"] > 40)
    & (df_test["dropoff_latitude"] < 42)
)
valid_dist = (df_test["distance_km"] > 0.1) & (df_test["distance_km"] < 30)
test_valid_mask = (valid_geo & valid_dist).to_numpy()



## === cell 22
X_test = df_test[feature_cols].astype(np.float32, copy=False)

X_test_np = X_test.to_numpy(dtype=np.float32, copy=False)
bad = ~np.isfinite(X_test_np)
if bad.any():
    X_test = X_test.mask(~np.isfinite(X_test), np.nan)
    X_test = X_test.fillna(train_feature_medians)



## === cell 23
y_pred = vr.predict(X_test)
y_pred = np.asarray(y_pred, dtype=np.float32)

if np.isfinite(a) and np.isfinite(b):
    y_pred = (a * y_pred + b).astype(np.float32)

y_pred[~test_valid_mask] = np.float32(train_pred_median)



## === cell 24
y_pred = np.clip(y_pred, 0.0, 1000.0)



## === cell 25
results = pd.DataFrame({"key": key, "fare_amount": y_pred})
print(results.head())
print("Submission rows:", len(results))



## === cell 26
results.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
