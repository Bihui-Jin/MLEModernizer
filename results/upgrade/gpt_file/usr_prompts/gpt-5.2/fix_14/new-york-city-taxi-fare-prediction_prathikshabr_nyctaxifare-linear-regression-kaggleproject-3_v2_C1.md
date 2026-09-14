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

3.9

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

5.68914

# 6. Current score

6.64244

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.94064) has done: 'I fix the runtime error caused by the deprecated `normalize` argument in `sklearn.linear_model.LinearRegression` (removed in scikit-learn 1.2+), while keeping the same linear-regression core logic by replacing it with an equivalent `Pipeline` that includes `StandardScaler`. I also ensure train/test feature columns are aligned after one-hot encoding weekdays (robustly adding any missing dummy columns), preventing shape/column mismatch at prediction time. Finally, I make sure the submission is written end-to-end as a valid `.csv` with exactly the required columns `key,fare_amount` and without setting `key` as the index (so the `key` column stays in the file).'
- What this solution (achieved 752.78354) has done: 'Your score is extremely far from the target (936.94 vs 5.69 RMSE), which strongly suggests a correctness issue rather than “model capacity”. The biggest bug is that you normalize `Difference_longitude/latitude` in the test set using the *test set’s own mean/variance*, while the model was trained on features normalized with the *train set’s* statistics—this makes test features inconsistent and can explode RMSE. I make the smallest change to compute those normalization statistics on train once and apply them to both train and test, keeping your exact feature set and LinearRegression+StandardScaler core logic. I also clip negative fare predictions to 0 (valid domain) to avoid outliers hurting RMSE, without changing the model itself.'
- What this solution (achieved 6.51495) has done: 'Your RMSE is still orders of magnitude off the target, which almost always indicates invalid target values and extreme outliers in the training labels/features driving the linear fit to nonsense. I keep your exact feature set and LinearRegression+StandardScaler pipeline, but add the standard NYC Taxi Fare cleaning filters (valid fare range, valid passenger_count, and a NYC bounding box on coordinates) before feature engineering so the model trains on realistic trips. I also apply the same “Difference_* < 5” filter to the test set (setting those rows to the median prediction rather than dropping them) to avoid huge extrapolation errors without changing your model. Finally, I keep the submission schema identical (`key,fare_amount`) and write `Submission.csv` as before.'
- What this solution (achieved 994.16561) has done: 'Your current RMSE (6.51495) is worse than the target (5.68914), so we should make small, low-risk improvements that reduce error without changing the model family or overall approach. The biggest likely remaining issue is the manual “Difference_* normalization” step (using variance and absolute deviation) fighting the `StandardScaler` in your pipeline and distorting those features; replacing it with a standard z-score (using train mean/std, applied to both train/test) keeps the same feature intent but makes the linear fit much more stable. I also avoid penalizing the test set by forcing “Difference_* < 5” rows to a constant fallback (that rule can be wrong for real long trips); instead we keep the mask for safety but make it much less disruptive by using the model’s own predictions everywhere. Finally, I keep the submission schema unchanged and still write `Submission.csv`.'
- What this solution (achieved 994.16797) has done: 'Your current score (994 RMSE) is far worse than the target (5.69), which indicates a correctness bug in the submission rather than “model quality”. The biggest issue is that you apply `pd.get_dummies` to `Weekday` independently for train and test, then concatenate without aligning those dummy column names back into the model feature matrix—this can silently create misordered/missing columns and produce wildly wrong predictions. I make the smallest change: build weekday dummies with a fixed category order (Monday–Sunday) and enforce identical feature columns/order between train and test right before fitting/predicting. I also remove only the test-time use of `test_mask_reasonable` side effects (since you’re not actually using it) and keep everything else (features, LinearRegression+StandardScaler, cleaning, rounding) the same.'
- What this solution (achieved 6.89871) has done: 'Your RMSE exploding to ~994 strongly suggests the submission predictions are numerically unstable/out-of-distribution rather than “just a weak model”. The smallest, core-logic-preserving fix is to (1) make the datetime parsing deterministic and vectorized (your string slicing can produce wrong weekdays/times for some formats), and (2) apply the exact same “reasonable trip” mask you use for training to the test set by replacing only the unreasonable rows’ predictions with a safe central value (median of reasonable predictions), instead of letting the linear model extrapolate wildly on those rows. This keeps your same features, same LinearRegression+StandardScaler pipeline, same cleaning intent, and same submission format, but prevents a handful of extreme test rows from dominating RMSE. I also keep train/test column alignment exactly as you already do.'
- What this solution (achieved 40.97397) has done: 'Your current RMSE (6.89871) is worse than the target (5.68914), so we should make small, low-risk changes that improve generalization without changing the model family or feature set. The biggest score drag in your current pipeline is likely the hard “Difference_* < 5” rule applied to training (it removes legitimate longer trips that exist in test), plus using only a 1% validation split which can mislead you about stability. I keep your exact LinearRegression+StandardScaler pipeline and all engineered features, but relax the training filter on Difference_* to a much more permissive bound (so the model learns from longer trips) while keeping the existing basic cleaning/bounding box. I also adjust the “unreasonable test rows” handling so it only triggers for truly extreme differences (same permissive bound), avoiding unnecessary flattening of predictions.'
- What this solution (achieved 953.70326) has done: 'Your current RMSE (40.97) is much worse than the target (5.689), so we should fix likely correctness/feature issues with minimal, core-logic-preserving changes. The biggest problem is that you train on 10M rows but your cleaning/filters likely drop most of them; additionally, your linear model benefits a lot from the standard NYC “distance > 0 and not absurdly long” filter to prevent extreme/outlier trips from dominating the fit. I add two very small, standard filters (Distance bounds and pickuptime bounds) after you compute Distance (so we don’t change feature engineering), and I also remove the test-time “fallback replacement” (which can flatten legitimate long trips and hurt RMSE) while keeping your existing DIFF_MAX_DEGREES mask computed (no change in semantics besides not overwriting model outputs). Everything else (features, LinearRegression+StandardScaler pipeline, training approach, submission schema/paths) stays the same.'
- What this solution (achieved 39.1346) has done: 'Your current RMSE (~954) is catastrophically worse than the target (~5.69), which almost always means a submission correctness problem rather than “model quality”. The smallest high-impact fix is to stop manually z-scoring `Difference_longitude/latitude` (cell 25), because you already apply `StandardScaler` to *all* features in the pipeline—double-scaling those two features can blow up coefficients and predictions on test. I also make the train/test “reasonable trip” logic consistent by using the existing `test_mask_reasonable` to replace only extreme/unreasonable test predictions with a safe in-distribution fallback (median prediction on reasonable rows), preventing a handful of outliers from dominating RMSE. Everything else (data cleaning, feature engineering, LinearRegression+StandardScaler pipeline, training loop, submission schema/path) remains the same.'
- What this solution (achieved 6.6018) has done: 'Your current RMSE (39.13) is still far from the target (5.689), so we should make a small correctness-oriented change that typically yields a big RMSE drop for this competition without changing the model family or training loop. The main issue is that your “unreasonable test row” mask is based only on raw coordinate degree deltas (and with a very permissive threshold), so outlier/test-invalid coordinates can still slip through and cause extreme extrapolation; we instead compute the mask from the already-engineered `Distance` feature and apply the same distance bounds to test-time fallback as you do on train. We also ensure `test_data` has no NaNs in engineered features (fill with train medians) so the pipeline never receives NaNs (which can otherwise produce unstable behavior). Everything else (features, LinearRegression + StandardScaler, training split, submission schema/path) stays the same.'
- What this solution (achieved 6.60328) has done: 'Your current RMSE (6.6018) is worse than the target (5.68914), so we make one minimal, high-signal improvement that keeps the same model (LinearRegression + StandardScaler) and the same feature set intent. The main remaining score drag is that `Difference_longitude/latitude` are computed as absolute degree deltas; turning those into real-world deltas in miles (longitude scaled by `cos(latitude)` and both scaled by ~69 miles/degree) usually improves linear fit substantially without changing the “difference” feature concept. We also add a tiny, standard filter to drop near-zero-distance trips in training (these are often noisy/bad rows) while leaving test handling unchanged. Everything else (cleaning, datetime features, airport distances, pipeline, submission schema/filename) stays the same.'
- What this solution (achieved 6.64249) has done: 'We’re already within ~16% of the target RMSE (6.60 vs 5.69), so we should make only low-risk, correctness-aligned tweaks that typically reduce error for this competition without changing your model family or feature set intent. The biggest likely remaining drag is that `Difference_longitude`/`Difference_latitude` now represent axis-aligned miles while you also feed `Distance` (haversine miles); this redundancy plus rounding distances to 2 decimals can slightly hurt a linear model. I keep the same features but stop rounding the continuous distance features (preserves more signal) and tighten the *test-time fallback* mask to only catch truly invalid geometry (Distance==0/NaN or absurdly large), so we don’t overwrite legitimate long trips with a constant. Everything else (cleaning, engineered features, LinearRegression+StandardScaler pipeline, submission schema/filename) remains the same.'
- What this solution (achieved 6.64244) has done: 'We’re currently worse than the target (6.64249 vs 5.68914 RMSE; lower is better), so we make one small, metric-aligned improvement without changing your model family or feature set: stop rounding predictions to 2 decimals, because RMSE is computed on raw floats and rounding injects avoidable error. We also add a tiny, standard post-processing step to cap extremely large predictions to the max fare seen in your cleaned training set (500), which reduces the impact of rare linear-regression extrapolation outliers while staying consistent with your existing fare cleaning rule. Everything else (cleaning, engineered features, LinearRegression+StandardScaler pipeline, train/test alignment, fallback logic, submission schema and filename) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_data = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=10_000_000
)
train_data.head()



## === cell 2
train_data.shape



## === cell 3
train_data.info()



## === cell 4
test_data = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
test_data.head()



## === cell 5
test_data.info()



## === cell 6
train_data.isna().sum()



## === cell 7
print(f"Before basic cleaning: {len(train_data)}")
train_data = train_data.dropna()

train_data = train_data[
    (train_data["fare_amount"] > 0) & (train_data["fare_amount"] <= 500)
]
train_data = train_data[
    (train_data["passenger_count"] >= 1) & (train_data["passenger_count"] <= 6)
]

train_data = train_data[
    (train_data["pickup_longitude"].between(-75, -72))
    & (train_data["dropoff_longitude"].between(-75, -72))
    & (train_data["pickup_latitude"].between(40, 42))
    & (train_data["dropoff_latitude"].between(40, 42))
]

print(f"After basic cleaning: {len(train_data)}")



## === cell 8
MILES_PER_DEG_LAT = 69.0

train_mid_lat_rad = np.radians(
    (
        train_data["pickup_latitude"].to_numpy(dtype=np.float64)
        + train_data["dropoff_latitude"].to_numpy(dtype=np.float64)
    )
    / 2.0
)
train_dlon_deg = train_data["pickup_longitude"].to_numpy(dtype=np.float64) - train_data[
    "dropoff_longitude"
].to_numpy(dtype=np.float64)
train_dlat_deg = train_data["pickup_latitude"].to_numpy(dtype=np.float64) - train_data[
    "dropoff_latitude"
].to_numpy(dtype=np.float64)

train_data["Difference_longitude"] = (
    np.abs(train_dlon_deg) * np.cos(train_mid_lat_rad) * MILES_PER_DEG_LAT
)
train_data["Difference_latitude"] = np.abs(train_dlat_deg) * MILES_PER_DEG_LAT



## === cell 9
test_mid_lat_rad = np.radians(
    (
        test_data["pickup_latitude"].to_numpy(dtype=np.float64)
        + test_data["dropoff_latitude"].to_numpy(dtype=np.float64)
    )
    / 2.0
)
test_dlon_deg = test_data["pickup_longitude"].to_numpy(dtype=np.float64) - test_data[
    "dropoff_longitude"
].to_numpy(dtype=np.float64)
test_dlat_deg = test_data["pickup_latitude"].to_numpy(dtype=np.float64) - test_data[
    "dropoff_latitude"
].to_numpy(dtype=np.float64)

test_data["Difference_longitude"] = (
    np.abs(test_dlon_deg) * np.cos(test_mid_lat_rad) * MILES_PER_DEG_LAT
)
test_data["Difference_latitude"] = np.abs(test_dlat_deg) * MILES_PER_DEG_LAT



## === cell 10
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")



## === cell 11
try:
    plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")
except Exception as e:
    print("Plot skipped:", e)



## === cell 12
DIFF_MAX_MILES = 20.0 * MILES_PER_DEG_LAT  # ~1380 miles

train_data = train_data[
    (train_data["Difference_longitude"] < DIFF_MAX_MILES)
    & (train_data["Difference_latitude"] < DIFF_MAX_MILES)
]
test_mask_reasonable = (test_data["Difference_longitude"] < DIFF_MAX_MILES) & (
    test_data["Difference_latitude"] < DIFF_MAX_MILES
)



## === cell 13
train_dt = pd.to_datetime(train_data["pickup_datetime"], errors="coerce", utc=True)
test_dt = pd.to_datetime(test_data["pickup_datetime"], errors="coerce", utc=True)

train_dt = train_dt.fillna(pd.Timestamp("2000-01-01", tz="UTC"))
test_dt = test_dt.fillna(pd.Timestamp("2000-01-01", tz="UTC"))

train_data["pickuptime"] = (train_dt.dt.hour * 100 + train_dt.dt.minute).astype(
    np.int32
)
test_data["pickuptime"] = (test_dt.dt.hour * 100 + test_dt.dt.minute).astype(np.int32)

train_data["Weekday"] = train_dt.dt.day_name()
test_data["Weekday"] = test_dt.dt.day_name()

train_data.head()



## === cell 14
train_data.head()



## === cell 15
test_data.head()



## === cell 16
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 17
weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
train_data["Weekday"] = pd.Categorical(train_data["Weekday"], categories=weekday_order)
test_data["Weekday"] = pd.Categorical(test_data["Weekday"], categories=weekday_order)

train_one_hot = pd.get_dummies(train_data["Weekday"], dtype=np.int8)
test_one_hot = pd.get_dummies(test_data["Weekday"], dtype=np.int8)

train_one_hot = train_one_hot.reindex(columns=weekday_order, fill_value=0)
test_one_hot = test_one_hot.reindex(columns=weekday_order, fill_value=0)

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 18
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 19
train_data.head()



## === cell 20
R = 6373.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c

train_data["Distance"] = np.asarray(distance) * 0.621

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1

a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data["Distance"] = np.asarray(distance) * 0.621



## === cell 21
R = 6373.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

lat3 = np.zeros(len(train_data)) + np.radians(40.6413111)
lon3 = np.zeros(len(train_data)) + np.radians(-73.7781391)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff = lon3 - lon2
d_lat_dropoff = lat3 - lat2
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
train_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

lat3 = np.zeros(len(test_data)) + np.radians(40.6413111)
lon3 = np.zeros(len(test_data)) + np.radians(-73.7781391)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff = lon3 - lon2
d_lat_dropoff = lat3 - lat2
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
test_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## === cell 22
pass



## === cell 23
print(f"Before distance/time cleaning: {len(train_data)}")
train_data = train_data[(train_data["Distance"] > 0) & (train_data["Distance"] <= 100)]
train_data = train_data[
    (train_data["pickuptime"] >= 0) & (train_data["pickuptime"] <= 2359)
]

train_data = train_data[train_data["Distance"] >= 0.05]

print(f"After distance/time cleaning: {len(train_data)}")



## === cell 24
test_mask_reasonable = (
    np.isfinite(test_data["Distance"].to_numpy(dtype=np.float64))
    & (test_data["Distance"] > 0)
    & (test_data["Distance"] <= 100)
    & (test_data["pickuptime"] >= 0)
    & (test_data["pickuptime"] <= 2359)
)



## === cell 25
train_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 26
print(train_data.shape)
print(test_data.shape)



## === cell 27
from sklearn.model_selection import train_test_split

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]

test_X = test_data.drop(["key"], axis=1)

test_X = test_X.reindex(columns=X.columns, fill_value=0)

train_medians = X.median(numeric_only=True)
test_X = test_X.fillna(train_medians).fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 28
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lr = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", LinearRegression()),
    ]
)
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 29
pred = lr.predict(test_X)

pred = np.asarray(pred, dtype=np.float64)
reasonable_mask = np.asarray(test_mask_reasonable, dtype=bool)
if reasonable_mask.any():
    fallback = float(np.median(pred[reasonable_mask]))
else:
    fallback = float(np.median(pred))
pred = np.where(reasonable_mask, pred, fallback)

pred = np.clip(pred, 0.0, 500.0)




## === cell 30
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 31
Submission = pd.DataFrame({"key": test_data["key"].values, "fare_amount": pred})
Submission = Submission[["key", "fare_amount"]]



## === cell 32
Submission.head()



## === cell 33
Submission.to_csv("Submission.csv", index=False)
print("Wrote submission to Submission.csv")
print(Submission.shape)
print(Submission.head())
