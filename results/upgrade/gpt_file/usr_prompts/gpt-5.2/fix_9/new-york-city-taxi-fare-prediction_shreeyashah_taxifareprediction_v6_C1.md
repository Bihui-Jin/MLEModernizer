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

4.743604326040083

# 6. Current score

5.41497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 360.21916) has done: 'Your score is extremely worse than the target (RMSE 361 vs 4.74), which strongly suggests the model is being trained/predicted on the wrong features (your `X.iloc[:,1:]` drops `pickup_datetime` but keeps it as an unparsed string/object, and the train/test columns don’t match). I keep your exact core approach (same haversine feature, same three base models, same VotingRegressor), but make the minimal fix: consistently extract numeric datetime features (year/month/day/hour/weekday) for both train and test, drop the raw datetime string, and ensure the exact same feature columns/order are used for fit and predict. I also stop discarding columns by position (which is fragile) and instead select by name so train/test alignment is correct, while keeping all your existing filters and model hyperparameters unchanged. This should move RMSE dramatically down toward the target without changing the modeling logic.'
- What this solution (achieved 360.21865) has done: 'Your current RMSE is far above the target, so we need a small but high-impact fix that improves correctness without changing your modeling approach. The biggest likely issue is feature dtype mismatch and unhandled NaNs in test after datetime parsing (coerce→NaT) which can cascade into non-numeric columns or missing values at predict time and yield nonsense predictions. I keep your exact feature set and ensemble, but (1) enforce numeric dtypes for all feature columns in both train and test, (2) drop rows with NaNs in the *used* feature columns (train) and impute remaining test NaNs with train medians (test), and (3) clip negative predictions to 0 to avoid RMSE blowups from pathological outputs, while keeping everything else the same. These changes are minimal, metric-aligned, and should move RMSE dramatically down toward your target band.'
- What this solution (achieved 6.14375) has done: 'Your RMSE is wildly off-target, which usually happens when the model is trained on one feature distribution but predicts on a mismatched/dirty test feature matrix (e.g., unrealistic coordinates/distances in test, NaNs/inf, or wrong column order/dtypes). I keep your exact feature set and ensemble models unchanged, but apply the same coordinate + distance sanity filters to the test set as you already do for train, then fill any dropped/invalid test rows with a safe fallback prediction (the train median fare) so the submission stays complete and aligned by `key`. I also enforce identical feature column order and numeric dtypes right before fitting/predicting, and guard against inf values, which can silently destabilize predictions. These are minimal correctness fixes that should dramatically reduce RMSE toward your target without changing your core modeling approach.'
- What this solution (achieved 5.43685) has done: 'Your current RMSE (6.14) is still worse than the target (4.74), so we should nudge performance upward with minimal, safe changes that don’t alter your core modeling approach (same features, same three models, same VotingRegressor). The biggest low-risk improvement is to make the ensemble more robust by scaling the features for the SVR branch only (SVR is very sensitive to feature scales), while leaving the LR and RF branches unchanged. This keeps the “same three base models + voting” logic intact but fixes a common correctness/performance issue for SVR in mixed-scale tabular data. I also keep your existing train/test cleaning and fallback logic, just ensuring the SVR sees the exact same column order and a stable numeric matrix.'
- What this solution (achieved 5.41633) has done: 'Your current RMSE (5.43685) is worse than the target (4.7436), so we should make a small, safe change that improves accuracy without changing your model family/ensemble logic. The biggest likely remaining issue is that you’re training on raw coordinates that include a lot of outliers/noise (even after your broad bounding-box filter), which can hurt all three regressors; we add the same standard NYC Taxi competition “cleaning” constraint: remove near-zero coordinate rows (0,0) and constrain latitude/longitude to realistic NYC bounds more tightly. This keeps your exact feature set, models, and training loop intact, but improves data quality so the learned mapping generalizes better to test. We also ensure passenger_count is integer-like (still numeric) and keep everything else unchanged, including the fallback logic and submission format.'
- What this solution (achieved 5.43558) has done: 'Your current RMSE (5.41633) is worse than the target (4.7436), so we should make a small change that improves accuracy without changing your model/feature logic. The lowest-risk gain here is to use a slightly larger training sample (still far from the full 55M) so the same ensemble generalizes better, while keeping all filters, features, and models identical. I also fix a small inefficiency/bug-risk by using the already-defined `X` (instead of reindexing `X[feature_cols]`) and make `BaggingRegressor` deterministic with `random_state`, which stabilizes the outcome without changing the training approach. Everything else (cleaning rules, haversine, datetime features, three base models + VotingRegressor, fallback for invalid test rows, and submission format) remains the same.'
- What this solution (achieved 5.37759) has done: 'Your current RMSE (5.43558) is worse than the target (4.7436), so we should make a small, low-risk improvement that keeps the same feature set and the same three-model VotingRegressor core. The biggest likely remaining gap comes from non-linear effects in distance and coordinate interactions that LinearRegression can’t capture; we add a minimal polynomial expansion *only* to the LR branch via a pipeline, leaving the RF and SVR branches unchanged. This preserves the ensemble/training approach while usually giving a modest RMSE lift on this competition. Everything else (data filters, haversine, datetime features, test validity mask + fallback, submission schema/path) remains identical.'
- What this solution (achieved 5.41497) has done: 'Your current RMSE (5.3776) is worse than the target (4.7436), so we should make a small, safe improvement without changing your model families, ensemble structure, or overall training approach. The biggest remaining low-risk win is to align the `RandomForestRegressor` with `n_jobs=-1` (so it can fully utilize CPU) and modestly increase its trees while keeping depth and subsampling behavior the same; this typically improves stability/accuracy without changing the core logic. To avoid any accidental train/test feature mismatch inside the pipelines, we also explicitly enforce the exact `feature_cols` order and float dtype right before fitting and predicting. These changes should move RMSE down toward the target band while keeping runtime under the 600s budget.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import warnings

warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns




## === cell 2
def add_datetime_features(df, col="pickup_datetime"):
    dt = pd.to_datetime(df[col], errors="coerce", utc=True)
    df["pickup_year"] = dt.dt.year.astype("float32")
    df["pickup_month"] = dt.dt.month.astype("float32")
    df["pickup_day"] = dt.dt.day.astype("float32")
    df["pickup_hour"] = dt.dt.hour.astype("float32")
    df["pickup_weekday"] = dt.dt.weekday.astype("float32")
    return df




## === cell 3
df_train = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=3_000_000,
    parse_dates=["pickup_datetime"],
)
df_train.head()



## === cell 4
df_train = df_train.drop(columns=["key"])



## === cell 5
df_train = add_datetime_features(df_train, "pickup_datetime")
df_train = df_train.drop(columns=["pickup_datetime"])



## === cell 6
df_train.info()



## === cell 7
df_train.shape



## === cell 8
df_train.describe()



## === cell 9
df_train.isna().sum()



## === cell 10
df_train = df_train.dropna()



## === cell 11
df_train.isna().sum()



## === cell 12
df_train = df_train[(df_train["fare_amount"] > 1) & (df_train["fare_amount"] < 100)]



## === cell 13
df_train = df_train[
    (df_train["passenger_count"] >= 1) & (df_train["passenger_count"] <= 6)
]



## === cell 14
df_train = df_train[
    (df_train["pickup_longitude"] > -74.5)
    & (df_train["pickup_longitude"] < -72.8)
    & (df_train["dropoff_longitude"] > -74.5)
    & (df_train["dropoff_longitude"] < -72.8)
    & (df_train["pickup_latitude"] > 40.5)
    & (df_train["pickup_latitude"] < 41.9)
    & (df_train["dropoff_latitude"] > 40.5)
    & (df_train["dropoff_latitude"] < 41.9)
]

df_train = df_train[
    (df_train["pickup_longitude"] != 0)
    & (df_train["pickup_latitude"] != 0)
    & (df_train["dropoff_longitude"] != 0)
    & (df_train["dropoff_latitude"] != 0)
]




## === cell 15
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # kilometers




## === cell 16
df_train["distance_km"] = haversine(
    df_train["pickup_latitude"],
    df_train["pickup_longitude"],
    df_train["dropoff_latitude"],
    df_train["dropoff_longitude"],
)

df_train = df_train[(df_train["distance_km"] > 0.1) & (df_train["distance_km"] < 30)]



## === cell 17
target_col = "fare_amount"
feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "pickup_year",
    "pickup_month",
    "pickup_day",
    "pickup_hour",
    "pickup_weekday",
    "distance_km",
]

for c in feature_cols + [target_col]:
    df_train[c] = pd.to_numeric(df_train[c], errors="coerce")

df_train["passenger_count"] = df_train["passenger_count"].round().clip(1, 6)

df_train = df_train.replace([np.inf, -np.inf], np.nan).dropna(
    subset=feature_cols + [target_col]
)

X = df_train[feature_cols]
y = df_train[target_col]

fallback_fare = float(y.median())



## === cell 18
X.head(), y.head()



## === cell 19
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=250,
    max_depth=10,
    random_state=2,
    bootstrap=True,
    max_samples=5000,
    n_jobs=-1,
)



## === cell 20
from sklearn.linear_model import LinearRegression

lr = LinearRegression()



## === cell 21
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor

sr = SVR(kernel="rbf")

br = BaggingRegressor(
    n_estimators=10,
    estimator=sr,
    max_samples=5000,
    bootstrap=True,
    random_state=2,
)



## === cell 22
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

lr_poly = Pipeline(
    steps=[
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("lr", lr),
    ]
)

br_scaled = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("br", br),
    ]
)

vr = VotingRegressor(
    [("lr", lr_poly), ("rf", rf), ("svr", br_scaled)], verbose=True, n_jobs=-1
)

X_fit = X.reindex(columns=feature_cols).astype("float32")
y_fit = y.astype("float32")
vr.fit(X_fit, y_fit)



## === cell 23
df_test = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)
df_test.head()



## === cell 24
key = df_test["key"].copy()

df_test = add_datetime_features(df_test, "pickup_datetime")
df_test = df_test.drop(columns=["pickup_datetime", "key"])



## === cell 25
df_test["distance_km"] = haversine(
    df_test["pickup_latitude"],
    df_test["pickup_longitude"],
    df_test["dropoff_latitude"],
    df_test["dropoff_longitude"],
)



## === cell 26
for c in feature_cols:
    df_test[c] = pd.to_numeric(df_test[c], errors="coerce")

df_test["passenger_count"] = df_test["passenger_count"].round().clip(1, 6)

df_test = df_test.replace([np.inf, -np.inf], np.nan)

valid_mask = (
    df_test["pickup_longitude"].between(-74.5, -72.8)
    & df_test["dropoff_longitude"].between(-74.5, -72.8)
    & df_test["pickup_latitude"].between(40.5, 41.9)
    & df_test["dropoff_latitude"].between(40.5, 41.9)
    & df_test["passenger_count"].between(1, 6)
    & df_test["distance_km"].between(0.1, 30)
    & (df_test["pickup_longitude"] != 0)
    & (df_test["pickup_latitude"] != 0)
    & (df_test["dropoff_longitude"] != 0)
    & (df_test["dropoff_latitude"] != 0)
)

df_test_valid = df_test.loc[valid_mask, feature_cols].copy()

train_medians = X_fit.median(numeric_only=True)
df_test_valid[feature_cols] = df_test_valid[feature_cols].fillna(train_medians)

X_test_valid = df_test_valid.reindex(columns=feature_cols).astype("float32")



## === cell 27
y_pred_full = np.full(shape=(len(df_test),), fill_value=fallback_fare, dtype=np.float64)

y_pred_valid = vr.predict(X_test_valid)
y_pred_valid = np.clip(y_pred_valid, 0.0, None)

y_pred_full[valid_mask.to_numpy()] = y_pred_valid



## === cell 28
results = pd.DataFrame({"key": key, "fare_amount": y_pred_full})
print(results.head())
print("Submission rows:", len(results))
print(
    "Valid predicted rows:",
    int(valid_mask.sum()),
    "Invalid/fallback rows:",
    int((~valid_mask).sum()),
)



## === cell 29
results.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
