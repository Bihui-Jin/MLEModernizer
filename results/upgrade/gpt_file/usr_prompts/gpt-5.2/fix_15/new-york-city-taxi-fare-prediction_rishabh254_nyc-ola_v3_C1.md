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

5.92798

# 6. Current score

16.07138

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 992.59182) has done: 'Your score is extremely high because the linear model is being fit on raw data that still contains many extreme/out-of-range coordinates and implausible fares; these outliers heavily distort the least-squares weights and explode RMSE. I keep your exact core approach (two abs-diff features + intercept, solved by least squares) but add a minimal set of standard NYC Taxi Fare cleaning filters (valid lat/lon ranges, passenger_count bounds, and fare_amount bounds) before fitting. I also remove rounding during validation/prediction (rounding is not part of the metric and can only worsen RMSE) while keeping everything else the same. Finally, I ensure the submission is written as `submission.csv` with the required columns.'
- What this solution (achieved 1008.78579) has done: 'Your current RMSE is far worse than the target, so we should improve it with the smallest changes that preserve your exact modeling core (two abs-diff features + intercept solved by least squares). The biggest remaining issue is that the model is being trained on “worldwide-valid” coordinates; tightening filters to the NYC bounding box and adding a very light datetime feature (hour) dramatically reduce outlier influence without changing the overall approach. I also ensure the same feature engineering is applied consistently to train/valid/test, and clip negative predictions to 0 (fares can’t be negative), which typically reduces RMSE without altering the model fitting. Everything else (OLS via `np.linalg.lstsq`, same train/valid split, same I/O paths, same submission schema) stays intact.'
- What this solution (achieved 1071.02573) has done: 'Your RMSE is still massively above target because the OLS fit is being skewed by remaining outliers and (more importantly) by a few high-leverage trips even inside the broad NYC box. To move the score sharply toward 5.93 without changing your core model (same 3 features + intercept, same OLS via `np.linalg.lstsq`), I (1) tighten the geographic box to the standard NYC bounds, (2) remove obviously bad datetimes and add two tiny, linear-friendly cyclic hour features (sin/cos) while keeping `pickup_hour` too, and (3) add a very light distance-based filter so extreme “short feature / huge fare” and “long feature / tiny fare” points don’t dominate least squares. Everything else (data size, OLS, split, prediction, clipping, submission schema/path) stays the same.'
- What this solution (achieved 1091.95102) has done: 'I fix the shape-mismatch runtime error in the outlier filtering step by ensuring `dist` and `fare` are computed after all boolean filters and are aligned to `data`’s index (using NumPy arrays derived from the same filtered frame). I also make the distance- and fare-per-distance filters index-safe by applying them via a boolean mask on `data` rather than relying on implicit alignment between arrays of different lengths. These changes preserve your core model (same features, same OLS via `np.linalg.lstsq`, same split/training/prediction) while allowing the pipeline to run end-to-end and produce `submission.csv`. With the intended cleaning actually applied (instead of crashing), the score should move sharply down toward the target.'
- What this solution (achieved 7.34495) has done: 'Your RMSE is far above the target, so we should improve it with the smallest changes that keep your exact core model (same engineered features, same OLS via `np.linalg.lstsq`). The biggest remaining score driver is still outliers and high-leverage points: your current `fare_per_dist` filter is using degree-distance, which makes the “per distance” scale inconsistent and lets bad points through while removing good ones. I keep your same filtering idea but compute distance in kilometers (haversine) and use a gentle, more realistic fare-per-km band; this reduces distortion in least squares without changing the modeling approach. I also ensure the test set has the same basic NA handling and clip extreme test coordinates to the same NYC bounds to avoid generating absurd predictions for a few bad test rows, while still writing `submission.csv` in the required format.'
- What this solution (achieved 7.34453) has done: 'We’re still far from the target (7.34 vs 5.93 RMSE; lower is better), so we should make the smallest changes that reduce outlier influence without changing your core approach (same engineered features + OLS via `np.linalg.lstsq`). The biggest remaining lever is that ordinary least squares is very sensitive to the remaining heavy-tailed fare distribution even after your filters, so I keep the exact same features but fit the same linear model with a tiny amount of L2 regularization (ridge via closed form), which stabilizes weights and typically improves RMSE for this competition. I also remove the redundant second OLS computation (which can be numerically unstable) and ensure the exact same test preprocessing is applied consistently, keeping submission schema/path unchanged. These are minimal, metric-aligned changes that should move RMSE down toward the target band.'
- What this solution (achieved 7.00878) has done: 'To move RMSE down toward the 5.93 target while keeping your exact linear/ridge core, the smallest high-impact change is to make the distance signal consistent between filtering and modeling by adding the same haversine_km distance as an additional linear feature (no new model/loop; still closed-form ridge). I also keep your existing cleaning but tighten one obvious remaining outlier source (near-zero “movement” rides) by filtering on `dist_km` directly instead of only on abs-degree diffs, which is still the same outlier-removal intent. Finally, I apply the identical feature engineering (including `dist_km`) to train/val/test and keep the same submission schema/path so it runs end-to-end and writes `submission.csv`. These changes are minimal but should reduce distortion and lower RMSE toward the target band.'
- What this solution (achieved 7.00865) has done: 'We need to reduce RMSE (7.00878 → target 5.92798), so we should make the smallest changes that improve generalization without changing your linear ridge core (same closed-form solve, same feature set). The biggest low-risk win here is to stop applying `add_travel_vector_features()` twice on `test_df` (it overwrites/duplicates columns and can create subtle inconsistencies), and to apply the *exact same* geographic/distance/fare-per-km sanity logic to test only insofar as it affects feature construction (clip coords once, compute features once). Next, we make ridge numerically comparable across feature scales by standardizing features using train statistics (but still a linear ridge model and still closed-form), which typically reduces RMSE materially while preserving semantics. Finally, we keep the intercept unpenalized as you already do, and ensure submission row order/keys remain intact.'
- What this solution (achieved 8.35044) has done: 'Your current RMSE (7.00865) is still above the target (5.92798), so we should make small, low-risk changes that reduce outlier leverage without changing your linear ridge core. The biggest remaining issue is that the ridge penalty (`lam=1.0`) becomes too weak after standardization because the effective strength depends on feature scale and sample size; scaling `lam` by the number of training rows keeps the intended regularization effect and typically lowers RMSE. Second, your NYC bounding box is a bit too tight and can discard many valid airport/outer-borough trips, which hurts generalization; slightly widening it (still NYC-only) usually improves RMSE without changing features or the model. Finally, we keep test row order intact by not dropping NA rows in test (fill them conservatively) so the submission aligns exactly with the provided `key`s.'
- What this solution (achieved 9.5728) has done: 'Your current RMSE (8.35044) is still substantially worse than the target (5.92798), so we should make the smallest changes that reduce error while keeping your exact core (same engineered features + closed-form ridge). The biggest issue is that `lam = n_samples` is far too strong and drives the model toward an intercept-only predictor, which typically *worsens* RMSE; we switch to a small fixed ridge (still the same closed-form solve) to regain useful signal. To avoid changing anything else materially, we keep your cleaning, feature engineering, standardization, intercept handling, and submission logic identical, only adjusting `lam` and keeping everything deterministic. This should move RMSE down toward the target band without altering the approach.'
- What this solution (achieved 9.57357) has done: 'Your current RMSE (9.57) is much worse than the target (5.93), so we should make a small change that improves generalization without changing your core pipeline (same features, same closed-form ridge, same cleaning intent). The most likely culprit is that `lam=1.0` is far too weak given ~millions of rows, making the fit effectively near-OLS and still very sensitive to remaining outliers; we set `lam` proportional to the sample size (but not so huge it collapses to intercept-only). I also make `fare_amount` explicitly float64 during the solve to avoid any dtype surprises, while keeping the same feature engineering, standardization, intercept handling (unpenalized), and submission writing unchanged. This is a minimal, metric-aligned adjustment that should move RMSE down toward your target band.'
- What this solution (achieved 16.21589) has done: 'Your current RMSE (9.57) is far worse than the target (5.93, lower is better), and the biggest single driver is that you’re fitting a very sensitive linear model on a random row sample dominated by short, cheap rides; it underfits longer/airport trips and generalizes poorly to the test distribution. Keeping the exact same features and closed-form ridge solve, we (1) sample the training rows in a stratified way across distance bins to better match the test mix, (2) slightly relax the NYC bounding box so valid outer-borough/airport trips aren’t discarded, and (3) set ridge strength to a stable fixed value in standardized space (instead of scaling by n, which can still behave too close to OLS at 20M rows). These are minimal, metric-aligned changes that reduce outlier leverage while improving distribution coverage, and they still write a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 16.07198) has done: 'Your current RMSE (16.21589) is far above the target (5.92798, lower is better), and the biggest likely cause is that the distance-bin stratified sampling plus a strong ridge penalty is pushing the model toward a biased “average” fit that generalizes poorly to the Kaggle test distribution. I keep your exact feature set and closed-form ridge solve, but (1) remove the distance-bin cap sampling so the model trains on the full cleaned 20M rows you already load, and (2) reduce the ridge strength to a small, stable value in standardized space so it doesn’t collapse toward an intercept-only predictor. Everything else (cleaning rules, haversine distance, standardization with train statistics, unpenalized intercept, clipping negative predictions, and submission format/path) stays the same so the core logic/evaluation semantics are preserved.'
- What this solution (achieved 16.07138) has done: 'Your current RMSE (16.07) is far above the target (5.93, lower is better), and with the same linear ridge model the most likely reason is that `lam=1.0` is effectively “almost OLS” at this scale, so remaining outliers still dominate despite cleaning. I keep your exact feature set, standardization, and closed-form ridge solve, but increase `lam` to a moderate value in standardized space to reduce variance and outlier leverage without collapsing to an intercept-only model. I also make the train/val split deterministic while preserving semantics by stratifying on a coarse `dist_km` bin (same split method, just less distribution shift), which typically improves RMSE for this task. Everything else (data loading, filters, feature engineering, prediction clipping, and writing `submission.csv` with `key,fare_amount`) stays unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

print(os.listdir("../input"))



## === cell 1
USECOLS = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
data = pd.read_csv("../input/train.csv", nrows=20_000_000, usecols=USECOLS)




## === cell 2
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.deg2rad(lon1.astype(np.float64))
    lat1 = np.deg2rad(lat1.astype(np.float64))
    lon2 = np.deg2rad(lon2.astype(np.float64))
    lat2 = np.deg2rad(lat2.astype(np.float64))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (
        (df.dropoff_longitude - df.pickup_longitude).abs().astype("float32")
    )
    df["abs_diff_latitude"] = (
        (df.dropoff_latitude - df.pickup_latitude).abs().astype("float32")
    )

    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["pickup_hour"] = dt.dt.hour.astype("float32")

    hour = df["pickup_hour"].to_numpy(dtype=np.float32)
    angle = (2.0 * np.pi * hour) / 24.0
    df["pickup_hour_sin"] = np.sin(angle).astype("float32")
    df["pickup_hour_cos"] = np.cos(angle).astype("float32")

    df["dist_km"] = haversine_km(
        df["pickup_longitude"].to_numpy(),
        df["pickup_latitude"].to_numpy(),
        df["dropoff_longitude"].to_numpy(),
        df["dropoff_latitude"].to_numpy(),
    ).astype("float32")


add_travel_vector_features(data)



## === cell 3
print(data.isnull().sum())



## === cell 4
print("Old size: %d" % len(data))
data = data.dropna(how="any", axis="rows")
print("New size: %d" % len(data))



## === cell 5
plot = data.iloc[:400000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")



## === cell 6
print("Old size: %d" % len(data))

data = data[
    (data.fare_amount > 0)
    & (data.fare_amount <= 250)
    & (data.passenger_count >= 1)
    & (data.passenger_count <= 6)
    & (data.pickup_longitude >= -74.8)
    & (data.pickup_longitude <= -73.0)
    & (data.dropoff_longitude >= -74.8)
    & (data.dropoff_longitude <= -73.0)
    & (data.pickup_latitude >= 40.0)
    & (data.pickup_latitude <= 41.3)
    & (data.dropoff_latitude >= 40.0)
    & (data.dropoff_latitude <= 41.3)
].copy()

data = data[(data["dist_km"] > 0.2) & (data["dist_km"] < 60.0)].copy()

fare = data["fare_amount"].to_numpy(dtype=np.float32)
dist_km = data["dist_km"].to_numpy(dtype=np.float32)
fare_per_km = fare / (dist_km + 1e-3)
mask_fpk = (fare_per_km > 0.5) & (fare_per_km < 50.0)
data = data.loc[mask_fpk].copy()

print("New size: %d" % len(data))



## === cell 7
from sklearn.model_selection import train_test_split

y = data.fare_amount
X = data.drop("fare_amount", axis=1)

dist_bins = pd.cut(
    X["dist_km"],
    bins=[0.2, 1, 2, 5, 10, 20, 40, 60],
    include_lowest=True,
    labels=False,
)
dist_bins = dist_bins.astype("int32")

train_df, val_df, train_y, val_y = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=dist_bins
)
train_df.dtypes




## === cell 8
def get_feature_matrix_no_intercept(df):
    return np.column_stack(
        (
            df.abs_diff_longitude.to_numpy(dtype=np.float64),
            df.abs_diff_latitude.to_numpy(dtype=np.float64),
            df.dist_km.to_numpy(dtype=np.float64),
            df.pickup_hour.to_numpy(dtype=np.float64),
            df.pickup_hour_sin.to_numpy(dtype=np.float64),
            df.pickup_hour_cos.to_numpy(dtype=np.float64),
        )
    )


def add_intercept(X):
    return np.column_stack((X, np.ones(X.shape[0], dtype=np.float64)))


train_X0 = get_feature_matrix_no_intercept(train_df)
mu = train_X0.mean(axis=0)
sigma = train_X0.std(axis=0)
sigma = np.where(sigma == 0, 1.0, sigma)

train_Xs = (train_X0 - mu) / sigma
train_X = add_intercept(train_Xs)

print(train_X.shape)
print(train_y.shape)



## === cell 9
lam = 50.0

XtX = train_X.T @ train_X
Xty = train_X.T @ train_y.to_numpy(dtype=np.float64)
I = np.eye(XtX.shape[0], dtype=np.float64)
I[-1, -1] = 0.0  # do not penalize intercept term
w = np.linalg.solve(XtX + lam * I, Xty)
print(w)



## === cell 10
test_df = pd.read_csv("../input/test.csv")
test_df.dtypes



## === cell 11
test_df = test_df.copy()
for c in [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]:
    if c in test_df.columns:
        test_df[c] = test_df[c].astype("float64")
        test_df[c] = test_df[c].fillna(
            train_df[c].median() if c in train_df.columns else test_df[c].median()
        )

test_df["pickup_datetime"] = test_df["pickup_datetime"].fillna(
    "2010-01-01 00:00:00 UTC"
)

for col, lo, hi in [
    ("pickup_longitude", -74.8, -73.0),
    ("dropoff_longitude", -74.8, -73.0),
    ("pickup_latitude", 40.0, 41.3),
    ("dropoff_latitude", 40.0, 41.3),
]:
    test_df[col] = test_df[col].clip(lo, hi)

add_travel_vector_features(test_df)

val_X0 = get_feature_matrix_no_intercept(val_df)
val_Xs = (val_X0 - mu) / sigma
val_X = add_intercept(val_Xs)

test_X0 = get_feature_matrix_no_intercept(test_df)
test_Xs = (test_X0 - mu) / sigma
test_X = add_intercept(test_Xs)

test_y_predictions = test_X @ w
val_y_predictions = val_X @ w

test_y_predictions = np.clip(test_y_predictions, 0, None)
val_y_predictions = np.clip(val_y_predictions, 0, None)

from sklearn.metrics import mean_squared_error

print(np.sqrt(mean_squared_error(val_y, val_y_predictions)))

submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": test_y_predictions},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)

print(os.listdir("."))
print(submission.head())
