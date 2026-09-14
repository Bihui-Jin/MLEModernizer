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

4.203419222523976

# 6. Current score

10.20341

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1096.2251) has done: 'I remove the unavailable `feather` dependency and stop trying to write/read a feather file, loading the training data directly instead. I fix the Pandas `.str.split()` API error by using keyword arguments so datetime parsing works on the current Kaggle/Pandas version. I also make the notebook run end-to-end by ensuring `df` is defined before it’s used, fixing indentation/cell-order issues, and keeping feature names consistent (`diff_lon`/`diff_long`) between train and test. Finally, I ensure a valid submission CSV with exactly `key,fare_amount` is written (keeping the same core linear regression and random forest modeling logic).'
- What this solution (achieved 1096.2251) has done: 'I fix the runtime error by making the RandomForest use the exact same feature names at train and test time (use `diff_lon` consistently, and drop `diff_long` from the predict call). This is a correctness/stability fix that also prevents the model from silently misaligning features, which should move the RMSE drastically toward the target (your current score is dominated by this failure/mismatch). I also ensure the test feature columns used in prediction are numeric and free of NaNs in the required columns to avoid downstream validation issues. The model choices and overall approach (LR + RF, same hyperparameters) remain unchanged.'
- What this solution (achieved 1096.2251) has done: 'Your RMSE (1096) indicates the submission is badly misaligned rather than just “a bit weak,” and the most likely cause in this script is that you drop NaNs in test (`test_rf = test.dropna(...)` / `test_lr = test.dropna(...)`) and then write predictions only for those remaining rows, producing a submission with fewer rows than `sample_submission` and/or missing keys. Kaggle then scores many rows as missing/wrong, exploding RMSE. I keep your exact feature engineering and models, but change prediction to always cover *all* test rows by filling missing feature values using medians computed from the training data, and ensure we write predictions for every `key` in the original test order. This is a minimal, semantics-preserving fix that should move RMSE drastically toward the target band without changing the core modeling approach. I also clip negative fares to 0 in predictions (a safe post-process for this metric) to avoid extreme errors.'
- What this solution (achieved 5.6533) has done: 'Your RMSE (1096) is far from the target (4.20), so the submission is likely being scored with huge errors due to a major correctness issue rather than normal model weakness. The smallest high-impact fix is to remove train/test distribution mismatch caused by filtering/cleaning only on train: we should apply the same NYC bounding-box + distance cap preprocessing to the test set and also recompute `distance` after coercing coordinates to numeric (so invalid strings don’t propagate NaNs/infs). This keeps your exact feature engineering and the same two models (LinearRegression + RandomForestRegressor) but makes train/test features consistent and prevents extreme/unphysical distances from dominating predictions. Finally, we keep the “predict for all keys” behavior and clip predictions to a reasonable range to prevent occasional outliers from exploding RMSE.'
- What this solution (achieved 5.6533) has done: 'Your current RMSE (5.6533) is worse than the target (4.2034), so we should make a small, legitimate improvement without changing the overall approach (same feature engineering, same LR + RF models). The biggest safe gain here is to *blend* the two existing model predictions (instead of submitting only the RF), because LR tends to reduce variance/outliers while RF captures nonlinearity; a simple weighted average often improves RMSE with minimal risk. To avoid occasional huge errors, we also clip the predictions to a slightly tighter, still-reasonable fare range based on the cleaned training distribution (quantile-based cap), which is a metric-aligned post-process. Everything else (data loading size, cleaning rules, features, models, training calls) stays the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved 5.67954) has done: 'To move RMSE down toward your 4.20 target with minimal risk, I keep your exact data cleaning, features, and the same two models (LinearRegression + RandomForestRegressor), but tune only the post-model combination. Specifically, I choose the blend weight using a small holdout split from your already-cleaned training sample (no new model, no CV loops), then refit both models on all cleaned data and predict test with that calibrated weight. I also replace the fixed [0,200] clipping with a training-quantile based cap applied consistently to LR, RF, and the blend to reduce outlier-driven RMSE without changing the modeling approach. The script still run end-to-end and write a valid `submission.csv` with `key,fare_amount` in the original test order.'
- What this solution (achieved 5.15312) has done: 'Your current RMSE (5.6795) is still above the target (4.2034), so we should make one small, low-risk improvement that keeps your exact feature set and the same two models. The biggest likely issue remaining is that you clean/filter the training set (NYC bounding box + distance cap) but the *test* set keeps out-of-NYC rows as NaN and long-distance rows as NaN, then gets median-imputed—this can produce systematically wrong predictions for those rows. I keep your existing logic, but for test I apply the same *row filtering to the feature construction* (set out-of-NYC / distance>30 to NaN as you do now) and then additionally add two simple, metric-safe features already implied by your existing columns: `abs_lon`/`abs_lat` (same as your current diffs but without changing model types), and include them only in the RF (still the same RF model, just a minimal feature list extension). This often reduces RMSE toward ~4.x without changing the training approach, and we still predict for all test keys and write a valid `submission.csv`.'
- What this solution (achieved 5.60186) has done: 'Your RMSE (5.153) is still worse than the target (4.203), so we should make a small, low-risk improvement without changing your feature engineering or the LR+RF training approach. The biggest correctness/performance leak remaining is that you compute `year`/`hour` but never actually use them in either model, which usually improves this competition at low risk because fare patterns depend on time. I minimally add `year` and `hour` to both `lr_features` and `rf_features`, and ensure train/test medians are computed and applied for these columns so we still predict for every test key in order. Everything else (data cleaning, haversine distance, RF hyperparameters, blending via a single holdout) stays the same.'
- What this solution (achieved 5.58021) has done: 'We should move your RMSE down toward the 4.203 target with a minimal, metric-aligned fix: your datetime features are currently too weak (only `year` and `hour`), and the RF in particular benefits from simple cyclical time features without changing the modeling approach. I keep your exact models (LinearRegression + RandomForestRegressor), training flow (holdout-selected blend), and existing features, and only add `month`, `weekday`, and sine/cosine transforms for `hour` to both train and test. This tends to improve taxi-fare RMSE materially while staying within your core logic, and it preserves the “predict for all test keys in order” behavior and valid `submission.csv` output.'
- What this solution (achieved 8.69831) has done: 'I fix the immediate runtime error by ensuring `in_nyc` and `too_far` exist in the training dataframe `df` before they’re referenced in `lr_features`/`rf_features` (right now they only exist in `test`, causing the `KeyError`). I do this in a minimal way that preserves your current semantics: since you already filtered train to NYC and distance<=30, `in_nyc=1` and `too_far=0` for all remaining train rows. I also add a small safety check to ensure all required feature columns exist in `test` before filling/selection, so the pipeline reliably writes `submission.csv` end-to-end.'
- What this solution (achieved 8.69831) has done: 'Your current RMSE (8.698) is far worse than the target (4.203), so the most likely issue is train/test feature distribution mismatch rather than the models themselves. I keep your exact feature set and the same LR + RF + holdout-blend training flow, but apply the same NYC bounding-box and distance<=30 “flagging” logic to the training features (compute `in_nyc` and `too_far` from coordinates/distance instead of constant 1/0). I also ensure all coordinate columns are coerced to numeric in train (as you already do for test) before computing distance/diffs, preventing silent object dtypes/NaNs from degrading training. These are minimal semantic fixes that typically move RMSE substantially toward the target without changing the modeling approach.'
- What this solution (achieved 8.69831) has done: 'Your RMSE (8.698) is far above the 4.203 target, so this looks like a correctness issue in the train/validation blending step rather than just weak modeling. The smallest high-impact fix is to keep the holdout evaluation aligned row-by-row: instead of intersecting/sorting by `key` (which can reorder and mismatch LR vs RF rows), we use the exact same holdout mask and drop rows only where *either* model’s features are missing, preserving identical ordering for both predictions. This keeps your exact feature engineering, the same LinearRegression + RandomForestRegressor, and the same “choose blend weight on a holdout” logic, but makes the blend-weight selection meaningful (and typically reduces Kaggle RMSE materially). Submission generation remains unchanged (predict for all test keys in original order, write `submission.csv`).'
- What this solution (achieved 10.20341) has done: 'Your RMSE (8.698) is much worse than the 4.203 target, and the biggest likely cause is a train/test feature mismatch: you filter training data to NYC-only and distance<=30, but for test you keep all rows and only add flags—so the model never sees out-of-NYC / too-far patterns during training, leading to systematically wrong predictions on those test rows. I keep your exact models, features, and blending logic, but (1) stop hard-dropping those rows in train and instead create `in_nyc`/`too_far` flags before filtering, and (2) apply a very light “keep-but-downweight-extremes” approach by keeping all rows with reasonable fare and nonzero coords while letting the flags handle outliers. This is a minimal semantic fix that typically moves RMSE substantially toward your target without changing architecture/training approach. I also ensure `distance` is recomputed after numeric coercion and that medians are computed on the full (flagged) training set for consistent imputation.'
- What this solution (achieved 10.20341) has done: 'Your current RMSE (10.203) is much worse than the 4.203 target, which strongly suggests a correctness/distribution issue rather than “just model weakness.” I make the smallest high-impact fix: stop training only on NYC/<=30mi-ish cleaned rows while still predicting on all test rows; instead, keep those rows but encode them via your existing `in_nyc` and `too_far` flags so the models can learn how to handle them. To avoid label noise exploding RMSE, I keep only minimal target sanity filtering (fare range + nonzero coords + passenger_count>0) and keep the rest of your core pipeline unchanged (same features, same LR+RF models, same holdout blending, same clipping, same output schema). This should materially reduce systematic errors on the flagged test rows and move RMSE toward the target band without changing architecture or training approach.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.linear_model import LinearRegression

plt.style.use("seaborn-whitegrid")

TRAIN_PATH = "../input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "../input/new-york-city-taxi-fare-prediction/test.csv"



## === cell 1
df = pd.read_csv(TRAIN_PATH, nrows=1_000_000, low_memory=True)
print(df.head())
print(df.shape)



## === cell 2
for col in [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "passenger_count",
    "fare_amount",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[df.passenger_count > 0]
df = df[df.dropoff_latitude != 0]
df = df[df.pickup_longitude != 0]
df = df[df.pickup_latitude != 0]
df = df[df.dropoff_longitude != 0]

df = df[df.fare_amount > 2.5]
df = df[df.fare_amount < 100]

df = df.dropna()

dt_parsed = pd.to_datetime(
    df["pickup_datetime"], errors="coerce", utc=True
).dt.tz_convert(None)
df["year"] = dt_parsed.dt.year
df["hour"] = dt_parsed.dt.hour
df["month"] = dt_parsed.dt.month
df["weekday"] = dt_parsed.dt.weekday
df["hour_sin"] = np.sin(2.0 * np.pi * (df["hour"].astype(float) / 24.0))
df["hour_cos"] = np.cos(2.0 * np.pi * (df["hour"].astype(float) / 24.0))

df = df.dropna()



## === cell 3
df[["year", "hour", "month", "weekday", "hour_sin", "hour_cos"]] = df[
    ["year", "hour", "month", "weekday", "hour_sin", "hour_cos"]
].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=["year", "hour", "month", "weekday", "hour_sin", "hour_cos"])
print(df.head())




## === cell 4
def select_within_newYork(df_in, BB):
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


NYC = (-74.5, -72.8, 40.5, 41.8)

print("Train shape before NYC filter (kept):", df.shape)




## === cell 5
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    miles = 6367 * c * 0.62137
    return miles


df["distance"] = haversine_np(
    df.pickup_longitude, df.pickup_latitude, df.dropoff_longitude, df.dropoff_latitude
)

print(df[["fare_amount", "distance"]].head())



## === cell 6
print("Co-relation b/w Fare and Distance")
try:
    print(st.pearsonr(df.distance, df.fare_amount))
except Exception as e:
    print("Pearsonr skipped:", repr(e))
print(df["distance"].corr(df["fare_amount"], method="pearson"))

print("Train shape before distance<=30 filter (kept):", df.shape)



## === cell 7
try:
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    con = (
        (df.distance < 30)
        & (df.distance > 0.5)
        & (df.fare_amount > 0)
        & (df.fare_amount < 200)
    )
    axs[0].scatter(df[con].distance, df[con].fare_amount, alpha=0.3)
    axs[0].set_xlabel("Distance")
    axs[0].set_ylabel("Fare")
    axs[0].set_title("Distance vs Fare")
    plt.close(fig)
except Exception as e:
    print("Plotting skipped:", repr(e))



## === cell 8
df["diff_lon"] = (df.dropoff_longitude - df.pickup_longitude).abs()
df["diff_lat"] = (df.dropoff_latitude - df.pickup_latitude).abs()

df["abs_lon"] = df.dropoff_longitude - df.pickup_longitude
df["abs_lat"] = df.dropoff_latitude - df.pickup_latitude

df["in_nyc"] = select_within_newYork(df, NYC).astype(int)
df["too_far"] = (df["distance"] > 30).fillna(False).astype(int)

df = df.dropna(
    subset=[
        "distance",
        "diff_lat",
        "diff_lon",
        "abs_lat",
        "abs_lon",
        "passenger_count",
        "fare_amount",
        "year",
        "hour",
        "month",
        "weekday",
        "hour_sin",
        "hour_cos",
        "in_nyc",
        "too_far",
    ]
).copy()
print(
    df[
        ["diff_lon", "diff_lat", "abs_lon", "abs_lat", "distance", "in_nyc", "too_far"]
    ].head()
)
print("Final train shape:", df.shape)



## === cell 9
test = pd.read_csv(TEST_PATH, low_memory=True)
print(test.head())

for col in [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "passenger_count",
]:
    if col in test.columns:
        test[col] = pd.to_numeric(test[col], errors="coerce")

test["diff_lat"] = (test.dropoff_latitude - test.pickup_latitude).abs()
test["diff_lon"] = (test.dropoff_longitude - test.pickup_longitude).abs()

test["abs_lon"] = test.dropoff_longitude - test.pickup_longitude
test["abs_lat"] = test.dropoff_latitude - test.pickup_latitude

test["distance"] = haversine_np(
    test.pickup_longitude,
    test.pickup_latitude,
    test.dropoff_longitude,
    test.dropoff_latitude,
)

dt_parsed_t = pd.to_datetime(
    test["pickup_datetime"], errors="coerce", utc=True
).dt.tz_convert(None)
test["year"] = dt_parsed_t.dt.year
test["hour"] = dt_parsed_t.dt.hour
test["month"] = dt_parsed_t.dt.month
test["weekday"] = dt_parsed_t.dt.weekday
test["hour_sin"] = np.sin(2.0 * np.pi * (test["hour"].astype(float) / 24.0))
test["hour_cos"] = np.cos(2.0 * np.pi * (test["hour"].astype(float) / 24.0))

test_id = test["key"].values
print(test.describe(include="all").transpose().head(20))

for col in [
    "distance",
    "diff_lat",
    "diff_lon",
    "abs_lat",
    "abs_lon",
    "year",
    "hour",
    "month",
    "weekday",
    "hour_sin",
    "hour_cos",
]:
    if col in test.columns:
        test[col] = pd.to_numeric(test[col], errors="coerce")

in_nyc = select_within_newYork(test, NYC)
test["in_nyc"] = in_nyc.astype(int)

too_far = test["distance"] > 30
test["too_far"] = too_far.fillna(False).astype(int)




## === cell 10
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


fare_lo = float(np.nanquantile(df["fare_amount"].values, 0.001))
fare_hi = float(np.nanquantile(df["fare_amount"].values, 0.995))
fare_lo = max(0.0, min(5.0, fare_lo))
fare_hi = max(50.0, min(250.0, fare_hi))
print("Train-based clip range:", (fare_lo, fare_hi))

n = len(df)
rng = np.random.RandomState(42)
perm = rng.permutation(n)
holdout_size = int(0.15 * n)
va_idx = perm[:holdout_size]
tr_idx = perm[holdout_size:]

df_tr = df.iloc[tr_idx].copy()
df_va = df.iloc[va_idx].copy()

lr = LinearRegression()

lr_features = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "distance",
    "passenger_count",
    "year",
    "hour",
    "month",
    "weekday",
    "hour_sin",
    "hour_cos",
    "in_nyc",
    "too_far",
]

rf_features = [
    "distance",
    "diff_lat",
    "diff_lon",
    "abs_lat",
    "abs_lon",
    "passenger_count",
    "year",
    "hour",
    "month",
    "weekday",
    "hour_sin",
    "hour_cos",
    "in_nyc",
    "too_far",
]

for col in set(lr_features + rf_features):
    if col not in test.columns:
        test[col] = np.nan

df_tr_lr = df_tr.dropna(subset=lr_features + ["fare_amount"]).copy()
df_tr_rf = df_tr.dropna(subset=rf_features + ["fare_amount"]).copy()

needed_for_both = list(dict.fromkeys(lr_features + rf_features + ["fare_amount"]))
df_va_common = df_va.dropna(subset=needed_for_both).copy()
y_va = df_va_common["fare_amount"].values.astype(float)

lr_fill_tr = df_tr_lr[lr_features].median(numeric_only=True)
rf_fill_tr = df_tr_rf[rf_features].median(numeric_only=True)

X_va_lr = df_va_common[lr_features].fillna(lr_fill_tr)
X_va_rf = df_va_common[rf_features].fillna(rf_fill_tr)

lr.fit(df_tr_lr[lr_features].fillna(lr_fill_tr), df_tr_lr["fare_amount"])

random_forest = rf(
    n_estimators=20,
    max_depth=20,
    max_features=None,
    oob_score=True,
    bootstrap=True,
    verbose=0,
    n_jobs=-1,
    random_state=42,
)
random_forest.fit(df_tr_rf[rf_features].fillna(rf_fill_tr), df_tr_rf["fare_amount"])

pred_va_lr = lr.predict(X_va_lr)
pred_va_rf = random_forest.predict(X_va_rf)

pred_va_lr = np.clip(pred_va_lr, fare_lo, fare_hi)
pred_va_rf = np.clip(pred_va_rf, fare_lo, fare_hi)

weights = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
rmses = []
for w in weights:
    pred_bl = w * pred_va_rf + (1.0 - w) * pred_va_lr
    pred_bl = np.clip(pred_bl, fare_lo, fare_hi)
    rmses.append(rmse(y_va, pred_bl))
best_i = int(np.argmin(rmses))
blend_w_rf = float(weights[best_i])
blend_w_lr = 1.0 - blend_w_rf
print("Holdout RMSEs:", dict(zip(weights.tolist(), [round(x, 6) for x in rmses])))
print(
    "Chosen blend weights rf/lr:",
    blend_w_rf,
    blend_w_lr,
    "best_holdout_rmse:",
    rmses[best_i],
)

df_lr = df.dropna(subset=lr_features + ["fare_amount"]).copy()
df_rf = df.dropna(subset=rf_features + ["fare_amount"]).copy()

lr_fill = df_lr[lr_features].median(numeric_only=True)
rf_fill = df_rf[rf_features].median(numeric_only=True)

test_lr = test.copy()
test_lr[lr_features] = test_lr[lr_features].fillna(lr_fill)
test_rf = test.copy()
test_rf[rf_features] = test_rf[rf_features].fillna(rf_fill)

lr.fit(df_lr[lr_features], df_lr["fare_amount"])
random_forest.fit(df_rf[rf_features], df_rf["fare_amount"])

preds_lr = lr.predict(test_lr[lr_features])
preds_rf = random_forest.predict(test_rf[rf_features])

preds_lr = np.clip(preds_lr, fare_lo, fare_hi)
preds_rf = np.clip(preds_rf, fare_lo, fare_hi)

preds_blend = blend_w_rf * preds_rf + blend_w_lr * preds_lr
preds_blend = np.clip(preds_blend, fare_lo, fare_hi)

sub_lr = pd.DataFrame({"key": test_id, "fare_amount": preds_lr})
sub_lr.to_csv("output_lr.csv", index=False)
print("Wrote output_lr.csv", sub_lr.shape, sub_lr.head())

sub = pd.DataFrame({"key": test_id, "fare_amount": preds_blend})
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv", sub.shape, sub.head())
print("Submission columns:", list(sub.columns))

try:
    sample_sub = pd.read_csv(
        "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"
    )
    print("Sample submission shape:", sample_sub.shape)
    print("Our submission shape:", sub.shape)
    print(
        "Key match (all):", np.array_equal(sample_sub["key"].values, sub["key"].values)
    )
except Exception as e:
    print("Sample submission check skipped:", repr(e))
