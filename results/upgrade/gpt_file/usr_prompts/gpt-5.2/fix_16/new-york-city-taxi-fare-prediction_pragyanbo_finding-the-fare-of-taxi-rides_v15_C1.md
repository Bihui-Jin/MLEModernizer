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

4.58218

# 6. Current score

6.17974

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1033.37678) has done: 'I remove the notebook-only magic and fix the TensorFlow import crash by avoiding `tf.estimator` (not available in TF 2.18 here) and replacing the broken DNNRegressor section with a minimal Keras regressor using the exact same features and MinMax scaling. I also fix a score-hurting logic issue: your linear model was fit on 4 features but the TF part only used 2; the Keras model use all 4 (`distance`, `passenger_count`, `hour`, `year`) to improve RMSE toward the target. Finally, I ensure the script always writes a valid `submission.csv` with required columns and correct row counts.'
- What this solution (achieved 1033.37678) has done: 'You’re hitting a TensorFlow import-time crash caused by an incompatibility between `tensorflow==2.18.0` and `protobuf==6.x` (`MessageFactory.GetPrototype` missing). Since this prevents the pipeline from finishing and writing a valid `submission.csv`, the minimal safe fix is to remove the TensorFlow/Keras block and instead produce the final submission using one of the already-trained scikit-learn models (keeping the same engineered features and train/test processing). To move RMSE strongly toward the target (and far away from the current ~1033), we use the RandomForest predictions as the final `submission.csv` output (the earlier file name had spaces and wasn’t the required name). All other core feature logic and filters are kept intact; the only logic change is which already-computed predictions are used for the final submission.'
- What this solution (achieved 1033.37678) has done: 'Your RMSE is catastrophically high because the `train.csv` you read does not come from the NYC Taxi Fare competition dataset in `/kaggle/input`; it’s likely a different competition’s file, so the model learns nonsense and outputs wildly wrong fares. The smallest change that moves you strongly toward the target is to point `train_path`/`test_path` (and sample submission, for sanity checks) to the actual NYC Taxi Fare folder that exists in your environment, while keeping your exact feature engineering and RandomForest/LinearRegression logic unchanged. I also keep writing `submission.csv` (required) and add a quick key-alignment check against `sample_submission.csv` to prevent silent row-order issues that can hurt score. No model architecture/training loop changes are made.'
- What this solution (achieved 996.13603) has done: 'Your score is catastrophically worse than the target (lower-is-better RMSE), so the smallest safe move is to fix a likely units bug in your `distance()` feature: it currently converts to miles but scales Earth’s radius as if in kilometers, inflating distances and destroying model scale. I change `distance()` to return proper haversine kilometers (keeping the same single “distance” feature) and keep all downstream logic/modeling identical. To keep semantics stable and avoid silent train/test mismatch, I also compute the same datetime-derived columns, fill any missing `hour/year` in test deterministically, and continue writing a valid `submission.csv` aligned to `sample_submission.csv`. No model architecture/training loop changes are made beyond this feature correctness fix, which should move RMSE strongly toward your target band.'
- What this solution (achieved 1030.18312) has done: 'Your RMSE is still extremely far from the target, which usually happens when the model learns from noisy/outlier-heavy training rows or when predictions include impossible fares; the smallest legitimate improvement without changing your model choice is to add a couple of standard NYC Taxi Fare “sanity” filters and to apply the exact same row-order alignment before generating the RF predictions. I keep your exact feature set and RandomForest/LinearRegression training intact, but (1) filter unrealistic lat/lon bounding boxes and extreme fares in train, (2) fix a subtle alignment bug where you reorder `test_df` after already computing `submission_rf`, and (3) clip predictions to a realistic upper bound to reduce RMSE blow-ups from occasional RF spikes. These changes should move RMSE substantially down toward your target without altering the core modeling approach.'
- What this solution (achieved 6.20415) has done: 'Your RMSE is still far above target, so the smallest safe way to move it down is to fix train/test feature mismatch by applying the same geographic bounding-box sanity filter to the test set too (otherwise the model extrapolates wildly on out-of-NYC coordinates). To further reduce blow-ups without changing your model choice or features, I add a minimal fare-dependent clipping based on the 99.5th percentile of the cleaned training fares (instead of a fixed 250), and I also clip distance to the training-filter maximum to keep inference in-distribution. These are legitimate post-processing/sanity steps aligned with RMSE minimization and keep your core logic (features + RandomForest/LinearRegression training) intact. The script still runs end-to-end and writes a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 5.98474) has done: 'You don’t have a valid Kaggle score yet (current_score=None), so the most direct way to move RMSE toward the 4.58218 target is to (1) ensure the submission always contains *all* 9914 test keys (your current code drops test rows via passenger_count filtering, which makes an invalid submission), and (2) keep your existing RandomForest/Linear blend but make it robust by predicting for every test row (no NaNs) and clipping within a train-derived realistic fare range. These changes keep your core model/feature logic intact, but fix the submission-row-count issue and prevent extreme/NaN outputs that can otherwise explode RMSE. I also keep your sample_submission key-order alignment, but apply it before feature extraction to avoid any subtle mismatch. The script always write `submission.csv` with the required columns and correct row count.'
- What this solution (achieved 6.17974) has done: 'Your current RMSE (5.98474) is worse than the target (4.58218), so we should make a small, legitimate improvement without changing the model types or feature set. The biggest likely gain with minimal risk is to remove a distribution shift you introduced: you clip `distance` at 15km, but then you *train only on* rides with distance < 15km; this biases the model downward for longer trips present in the test set. I keep the same single `distance` feature, but instead of dropping those rows I just clip both train and test distances to the same cap (still 15km), preserving more training signal while keeping inference in-distribution. Everything else (features, RandomForest + LinearRegression blend, submission alignment) stays the same and it still write a valid `submission.csv`.'
- What this solution (achieved 5.72204) has done: 'Your current RMSE (6.17974, lower-is-better) is still worse than the target (4.58218), so we should make a small, legitimate improvement without changing your feature set or model types. The largest likely gain with minimal risk is to reduce overfitting in the RandomForest by restricting tree growth (your 500 fully-grown trees on 100k rows can overfit noisy labels), while keeping the same training approach and prediction flow. I also set `min_samples_leaf`/`min_samples_split`/`max_features` in the same RF model (no architecture change, still RandomForestRegressor) and keep the LR fallback/blending and clipping exactly as you have it. This should move RMSE downward toward your target while keeping runtime under control and still producing a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 5.87241) has done: 'Your current RMSE (5.72204, lower-is-better) is still above the target (4.58218), so we should make a small, low-risk change that usually improves generalization without changing your core approach. The biggest likely win is adjusting the RandomForest to reduce bias from too-shallow averaging by slightly relaxing `min_samples_leaf`/`min_samples_split` while keeping the same model type, features, and training flow. I also add a minimal, train-derived NYC-bbox fallback for rows outside the bbox: instead of switching to the Linear model (which tends to extrapolate poorly), we use a conservative constant baseline (median fare) for those out-of-distribution rows to avoid large RMSE blow-ups. Everything else (feature engineering, distance cap, filters, RF+LR training, clipping, and submission alignment) stays the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 5.87241) has done: 'Your current RMSE (5.87241, lower-is-better) is still worse than the target (4.58218), so we should make a small improvement that reduces generalization error without changing your feature set or overall approach. The lowest-risk lever in your current setup is to change how you combine models: you already train both RandomForest and LinearRegression, but you only use RF (except for bbox fallback). We keep both models and features identical, and simply blend RF+LR predictions (with a high weight on RF) for in-bbox rows, while keeping the same median baseline for out-of-bbox rows to avoid blow-ups. We also add a minimal non-negativity clamp before the train-derived percentile clipping, which is consistent with the task (fares can’t be negative) and can slightly reduce RMSE from occasional negative LR contributions.'
- What this solution (achieved 6.11719) has done: 'Your current RMSE (5.87241, lower-is-better) is still above the target (4.58218), so we should make a small, low-risk generalization improvement without changing your features or model types. The most impactful minimal lever here is to reduce RandomForest noise/variance by switching it to `bootstrap=False` (still the same RandomForestRegressor, same training flow) and to modestly increase the RF weight in the RF+LR blend since LR tends to underfit this problem. I also make the train/test datetime parsing a bit more robust by filling missing test datetimes consistently (to avoid any hidden NaNs impacting predictions), while keeping the same hour/year feature logic. Everything else (data paths, feature engineering, filters, clipping, key alignment, and writing `submission.csv`) stays the same.'
- What this solution (achieved 6.17974) has done: 'Your current RMSE (6.11719, lower-is-better) is still meaningfully worse than the target (4.58218), so we make the smallest legitimate changes that typically reduce error without changing your core features or model types. The most impactful low-risk fix is to correct the train/validation split to be truly random (your current split can be skewed because `train.csv` is time-ordered), and to use the validation RMSE to set a slightly better RF/LR blend weight (still the same two models, just a data-driven mixing coefficient). We also add a standard NYC taxi cleanup step: removing extreme coordinate zeros/invalid lat/lon before training (test is not dropped; only gets a safe fallback), which reduces label noise/outliers that hurt RMSE. Everything else (paths, features, distance cap, filters, RandomForestRegressor + LinearRegression pipeline, clipping, key alignment, and writing `submission.csv`) stays intact.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

np.random.seed(42)



## === cell 1
BASE_INPUT_DIR = "/kaggle/input"
if not os.path.exists(BASE_INPUT_DIR):
    BASE_INPUT_DIR = "../input"

COMP_DIR = os.path.join(BASE_INPUT_DIR, "new-york-city-taxi-fare-prediction")
if os.path.exists(COMP_DIR):
    INPUT_DIR = COMP_DIR
else:
    INPUT_DIR = BASE_INPUT_DIR

train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")
sample_path = os.path.join(INPUT_DIR, "sample_submission.csv")

print("Using INPUT_DIR:", INPUT_DIR)
print("Train path exists:", os.path.exists(train_path), train_path)
print("Test path exists:", os.path.exists(test_path), test_path)

train_df = pd.read_csv(train_path, nrows=100000)
train_df.shape



## === cell 2
test_df = pd.read_csv(test_path)

sample_sub = None
if os.path.exists(sample_path):
    sample_sub = pd.read_csv(sample_path)
    if "key" in sample_sub.columns and len(sample_sub) == len(test_df):
        if not np.array_equal(sample_sub["key"].values, test_df["key"].values):
            test_df = (
                test_df.set_index("key").loc[sample_sub["key"].values].reset_index()
            )

test_df.shape



## === cell 3
train_df.head(5)



## === cell 4
train_df.isnull().sum()



## === cell 5
train_df.dropna(inplace=True)



## === cell 6
train_df.describe()



## === cell 7
train_df = train_df[train_df["fare_amount"] > 0]
train_df.shape




## === cell 8
def distance(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)

    rlat1 = np.deg2rad(lat1)
    rlon1 = np.deg2rad(lon1)
    rlat2 = np.deg2rad(lat2)
    rlon2 = np.deg2rad(lon2)

    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1

    a = (np.sin(dlat / 2.0) ** 2) + (
        np.cos(rlat1) * np.cos(rlat2) * (np.sin(dlon / 2.0) ** 2)
    )
    c = 2.0 * np.arcsin(np.sqrt(a))

    R_km = 6371.0
    return R_km * c




## === cell 9
train_df["distance"] = distance(
    train_df.pickup_latitude,
    train_df.pickup_longitude,
    train_df.dropoff_latitude,
    train_df.dropoff_longitude,
)

test_df["distance"] = distance(
    test_df.pickup_latitude,
    test_df.pickup_longitude,
    test_df.dropoff_latitude,
    test_df.dropoff_longitude,
)



## === cell 10
DIST_CAP_KM = 15.0
train_df["distance"] = np.clip(
    train_df["distance"].to_numpy(dtype=np.float64), 0.0, DIST_CAP_KM
)
test_df["distance"] = np.clip(
    test_df["distance"].to_numpy(dtype=np.float64), 0.0, DIST_CAP_KM
)

train_df.describe()



## === cell 11
train_df = train_df[
    (train_df["passenger_count"] != 0) & (train_df["passenger_count"] < 10)
]

test_df["passenger_count"] = (
    pd.to_numeric(test_df["passenger_count"], errors="coerce")
    .fillna(1)
    .astype(np.int16)
)
test_df["passenger_count"] = np.clip(test_df["passenger_count"], 1, 9).astype(np.int16)

train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"], errors="coerce", utc=True
)
test_df["pickup_datetime"] = pd.to_datetime(
    test_df["pickup_datetime"], errors="coerce", utc=True
)

train_df = train_df.dropna(subset=["pickup_datetime"])

train_df["hour"] = train_df["pickup_datetime"].dt.hour
train_df["year"] = train_df["pickup_datetime"].dt.year

test_df["pickup_datetime"] = test_df["pickup_datetime"].fillna(
    train_df["pickup_datetime"].median()
)

test_df["hour"] = test_df["pickup_datetime"].dt.hour
test_df["year"] = test_df["pickup_datetime"].dt.year

test_df["hour"] = test_df["hour"].fillna(0).astype(np.int16)
test_df["year"] = (
    test_df["year"].fillna(train_df["year"].mode().iloc[0]).astype(np.int16)
)



## === cell 12
nyc_lon_min, nyc_lon_max = -74.5, -72.8
nyc_lat_min, nyc_lat_max = 40.5, 41.8

geo_mask = (
    train_df["pickup_longitude"].between(nyc_lon_min, nyc_lon_max)
    & train_df["dropoff_longitude"].between(nyc_lon_min, nyc_lon_max)
    & train_df["pickup_latitude"].between(nyc_lat_min, nyc_lat_max)
    & train_df["dropoff_latitude"].between(nyc_lat_min, nyc_lat_max)
)
train_df = train_df[geo_mask]

train_df = train_df[train_df["fare_amount"].between(2.5, 250.0)]

valid_latlon = (
    train_df["pickup_latitude"].between(-90, 90)
    & train_df["dropoff_latitude"].between(-90, 90)
    & train_df["pickup_longitude"].between(-180, 180)
    & train_df["dropoff_longitude"].between(-180, 180)
    & (train_df["pickup_latitude"].abs() > 1e-6)
    & (train_df["dropoff_latitude"].abs() > 1e-6)
    & (train_df["pickup_longitude"].abs() > 1e-6)
    & (train_df["dropoff_longitude"].abs() > 1e-6)
)
train_df = train_df[valid_latlon]

train_df.shape



## === cell 13
test_geo_mask = (
    test_df["pickup_longitude"].between(nyc_lon_min, nyc_lon_max)
    & test_df["dropoff_longitude"].between(nyc_lon_min, nyc_lon_max)
    & test_df["pickup_latitude"].between(nyc_lat_min, nyc_lat_max)
    & test_df["dropoff_latitude"].between(nyc_lat_min, nyc_lat_max)
)

test_df["in_nyc_bbox"] = test_geo_mask



## === cell 14
feat_cols_s = ["distance", "passenger_count", "hour", "year"]
X = train_df[feat_cols_s]
y = train_df["fare_amount"]



## === cell 15
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)



## === cell 16
from sklearn.ensemble import RandomForestRegressor

r_reg = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2,  # keep identical
    min_samples_split=5,  # keep identical
    max_features=0.7,
    bootstrap=False,
)
r_reg.fit(X_train, y_train)

y_pred_rf_all = r_reg.predict(test_df[feat_cols_s])

submission_rf = pd.DataFrame(
    {"key": test_df.key, "fare_amount": y_pred_rf_all}, columns=["key", "fare_amount"]
)
submission_rf.to_csv("Random Forest regression.csv", index=False)



## === cell 17
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model_l = Pipeline(
    (("standard_scaler", StandardScaler()), ("lin_reg", LinearRegression()))
)
model_l.fit(X_train, y_train)

y_pred_lr_all = model_l.predict(test_df[feat_cols_s])

submission_lr = pd.DataFrame(
    {"key": test_df.key, "fare_amount": y_pred_lr_all}, columns=["key", "fare_amount"]
)
submission_lr.to_csv("linear_reg.csv", index=False)



## === cell 18
from sklearn.metrics import mean_squared_error

train_fares = train_df["fare_amount"].to_numpy(dtype=np.float64)
fare_lo = float(np.nanpercentile(train_fares, 0.5))
fare_hi = float(np.nanpercentile(train_fares, 99.5))
fare_lo = max(0.0, min(fare_lo, 10.0))
fare_hi = max(50.0, min(fare_hi, 250.0))

in_mask = test_df["in_nyc_bbox"].to_numpy(dtype=bool)

baseline_fare = float(np.nanmedian(train_fares))

rf_valid = r_reg.predict(X_valid)
lr_valid = model_l.predict(X_valid)

alphas = np.array([0.90, 0.92, 0.94, 0.96], dtype=np.float64)
best_alpha = 0.92
best_rmse = np.inf
for a in alphas:
    blend_v = (a * rf_valid) + ((1.0 - a) * lr_valid)
    rmse = mean_squared_error(y_valid, blend_v, squared=False)
    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = float(a)

alpha_rf = best_alpha
blended_in = (alpha_rf * y_pred_rf_all) + ((1.0 - alpha_rf) * y_pred_lr_all)

final_pred = np.where(in_mask, blended_in, baseline_fare)

final_pred = np.where(np.isfinite(final_pred), final_pred, baseline_fare)
final_pred = np.maximum(final_pred, 0.0)
final_pred = np.clip(final_pred, fare_lo, fare_hi)

submission = pd.DataFrame(
    {"key": test_df["key"].values, "fare_amount": final_pred},
    columns=["key", "fare_amount"],
)

assert len(submission) == len(test_df), "Submission length mismatch."
assert list(submission.columns) == ["key", "fare_amount"], "Wrong submission columns."
if sample_sub is not None:
    assert len(submission) == len(
        sample_sub
    ), "Submission must match sample_submission row count."
    assert np.array_equal(
        submission["key"].values, sample_sub["key"].values
    ), "Key order mismatch vs sample_submission."

submission.to_csv("submission.csv", index=False)

print("Chosen alpha_rf:", alpha_rf, "validation_rmse(blend):", best_rmse)
submission.head()
