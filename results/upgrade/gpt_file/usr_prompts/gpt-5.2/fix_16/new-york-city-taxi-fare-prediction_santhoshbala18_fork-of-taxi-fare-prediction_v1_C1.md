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

3.55848

# 6. Current score

5.92695

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.05365) has done: 'Your current score is far from the target (9.83 vs 3.56 RMSE; lower is better), and the biggest issue is that the “is_airport” feature is computed with the wrong longitude sign (NYC longitudes are negative), making it essentially noise and hurting performance. I make the same airport/night/surge feature logic consistent and correct for both train and test (minimal change), add a light but standard fare/outlier filter to reduce label noise (still same model/loop), and ensure the train/test feature columns are perfectly aligned before prediction. These changes typically reduce RMSE substantially for this competition without altering the core approach (Haversine + datetime features + XGBRegressor).'
- What this solution (achieved 10.38715) has done: 'Your current RMSE (8.05) is far above the target (3.56), so we should make small, reliable fixes that typically yield a big RMSE drop without changing the core approach (same Haversine + datetime/rule features + XGBRegressor). The main performance issue is that you are *dropping* the datetime-derived columns (`hour/day/month/weekday`) from training and test features, even though they are likely informative; keeping them is a minimal change that preserves the logic and usually improves RMSE substantially. I also make the airport feature slightly more robust by marking both pickup OR dropoff near JFK/LGA/EWR (still the same kind of rule feature), and add a standard distance/coordinate sanity filter to reduce extreme-noise rows in the 1M sample. These changes keep the same model/training loop and ensure the submission remains correctly aligned.'
- What this solution (achieved 5.79387) has done: 'I make two minimal, high-impact fixes that keep your overall approach (same features + XGBRegressor training loop) but reduce RMSE toward the 3.56 target: (1) add a standard “fare ≈ distance” consistency filter (remove rows where fare is implausible given trip distance), and (2) add a minimal XGB configuration tweak (`subsample`/`colsample_bytree`) to reduce overfitting that’s currently causing the very large gap between holdout and Kaggle score. I also ensure we drop rows with invalid datetimes/coords before feature creation so train/test features remain consistent. The output remains a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.82556) has done: 'Your current RMSE (5.79387) is still well above the target (3.55848), so we should make small, reliable changes that improve generalization without changing the core approach (same distance + datetime/rule features + XGBRegressor). The biggest minimal win here is to ensure the model’s objective matches the competition metric by using squared error (RMSE) rather than XGBoost’s default MAE objective, which can materially hurt RMSE on this task. To further stabilize toward the target without changing the feature logic, we switch the train/validation split to be based on `key` ordering (time-like) instead of random, reducing distribution leakage and making the model more robust on the test distribution. Everything else (features, filtering logic, training loop, and submission format) stays the same and still produce a valid `submission.csv`.'
- What this solution (achieved 5.82427) has done: 'We make one small but high-impact generalization change: switch the XGBoost tree method to histogram (`tree_method="hist"`) and use a slightly smaller `max_depth` (10 → 8) while keeping the same model family, features, and training loop. This reduces overfitting on the 1M-row sample and typically lowers Kaggle RMSE for this competition without altering your feature engineering or loss/metric semantics. We also add `min_child_weight` as a mild regularizer (a standard XGBoost parameter) to further stabilize RMSE toward your 3.56 target. Everything else (data loading, filtering, feature creation, split logic, and submission writing) stays the same and still produces `submission.csv`.'
- What this solution (achieved 5.90847) has done: 'Your current RMSE (5.82427) is still well above the target (3.55848), so we should make small, reliable generalization improvements without changing the core approach (same features + XGBRegressor). The largest remaining “minimal change” win for this competition is to train on a modestly larger, cleaner sample so the model learns rare-but-important patterns, while keeping the same feature engineering and model family. I increase the training read from 1,000,000 to 2,000,000 rows (still feasible under the 600s limit with `tree_method="hist"`), and I add `reg_lambda` and `gamma` as mild regularizers to reduce overfitting noise—this tends to lower leaderboard RMSE on NYC Taxi while preserving the same objective/semantics. Everything else (filters, split logic, feature columns alignment, and submission writing) stays the same and still output a valid `submission.csv`.'
- What this solution (achieved 5.8631) has done: 'We make two minimal, high-impact generalization adjustments while preserving your exact pipeline (same sampling, filters, feature set, and XGBRegressor training loop). First, we align the model closer to the competition metric by using `eval_metric="rmse"` (no semantic change to training objective, but it improves boosting behavior) and a slightly smaller `learning_rate` with proportionally more trees to reduce underfitting/overfitting swings on a 2M sample. Second, we add a tiny amount of extra regularization (`reg_alpha`) and slightly reduce `max_depth` (8→7) which often improves leaderboard RMSE for this competition without changing the approach. Everything else—including feature engineering, splitting, and submission format—remains the same and still writes a valid `submission.csv`.'
- What this solution (achieved 6.14989) has done: 'Your current RMSE (5.8631) is still far above the target (3.55848), so we should make small, reliable generalization improvements while keeping the same feature set and XGBRegressor training approach. The biggest remaining “minimal-change” win is to add one standard geospatial feature (`abs_lat_diff`, `abs_lon_diff`, and a simple Manhattan-distance proxy in degrees) that complements your existing Haversine distance without changing the model family or training loop. We also add a very light coordinate sanity clip (still consistent with your bounding-box filtering) to reduce the impact of rare numeric outliers on tree splits. Finally, we keep the exact submission semantics but add a safe post-processing floor (fare can’t be below 2.5) to avoid unrealistic negatives that can inflate RMSE.'
- What this solution (achieved 6.07533) has done: 'We make two small, high-impact generalization tweaks while preserving your exact pipeline (same sampling size, same feature engineering, same XGBRegressor approach/loop, same objective/metric, same submission semantics). First, we add the standard `sqrt(distance_km)` feature (a simple monotonic transform of your existing distance) which often helps trees model the short-trip fare behavior without changing the “core logic” of distance-based learning. Second, we lightly de-noise the 2M training sample by dropping only the most extreme top-tail fares (e.g., >150) which are disproportionately noisy/outlier-prone and tend to worsen public LB RMSE; this is a minimal extension of your existing fare filtering. Everything else (split, model family, training call, column alignment, and `submission.csv` writing) remains the same.'
- What this solution (achieved 5.98831) has done: 'Your current RMSE (6.07533) is still far above the target (3.55848), so we should make small, reliable improvements that preserve your exact pipeline (same feature set + XGBRegressor training). The biggest minimal win here is to stop sorting/splitting by the string `key` (which is not guaranteed time-ordered and can create a non-representative split) and instead split by actual `pickup_datetime` ordering; this keeps evaluation semantics (holdout RMSE) but makes the trained model generalize closer to Kaggle’s test distribution. I also add one standard, minimal feature for this competition—geodesic bearing—computed from the same coordinates you already use; it often materially reduces RMSE without changing the “distance + datetime + rules + XGB” core logic. Finally, I keep train/test feature alignment identical and leave all existing filtering/post-processing intact.'
- What this solution (achieved 5.98831) has done: 'We currently can’t score-match because you don’t have a valid Kaggle score from this exact script yet, so the most direct improvement toward the target is to remove two issues that can materially worsen RMSE while keeping your same feature set and XGBRegressor approach. First, your test-set filtering drops rows (e.g., distance==0, passenger_count out of range), which produce an invalid submission (missing keys) or force unintended key loss; we keep all test rows and instead only apply the stricter filters to training. Second, we keep your model/feature logic identical but add a tiny, standard stability improvement: use the training median fare as a fallback for any remaining NaN/inf feature rows in test by imputing features with training medians (prevents unpredictable predictions). The rest of the pipeline (Haversine + datetime/rule/geo/bearing features, XGB params, train split, prediction floor, submission format) stays the same and reliably write a complete `submission.csv` with all required keys.'
- What this solution (achieved 5.98962) has done: 'We make two minimal changes aimed at reducing RMSE (lower is better) without changing your core pipeline (same feature set + same XGBRegressor training approach). First, we add one very standard, low-risk numeric feature (`distance_km**2`) that helps trees model the convex relationship between distance and fare and usually improves generalization on this competition. Second, we add `base_score` to XGBoost using the training target mean, which often stabilizes boosting for squared-error regression and can shave off some error with negligible semantic change. Everything else (data reading size, filters, split by pickup_datetime, model family, submission format) remains the same and still writes a valid `submission.csv`.'
- What this solution (achieved 5.92371) has done: 'Your current RMSE (5.98962; lower is better) is still far above the 3.55848 target, so we should make small, reliable generalization improvements without changing the overall approach (same feature engineering + XGBRegressor). The most impactful minimal fix here is to prevent the model from learning from very common “zero-distance but nonzero fare” GPS artifacts by filtering those rows using a tiny distance threshold (instead of just `> 0`), while still keeping all test rows intact. Next, we add a single low-risk cyclical encoding for hour (`sin/cos`) so the model can represent time-of-day smoothly without changing the datetime feature logic. Finally, we keep the same submission semantics but add a safe upper clip on predictions (rare extreme predictions can inflate RMSE) using a high cap derived from training labels.'
- What this solution (achieved 5.92695) has done: 'You’re still far above the 3.56 RMSE target (5.92; lower is better), so the smallest reliable move is to reduce systematic underprediction/overprediction by adding a single standard NYC-Taxi feature: pickup/dropoff proximity to Manhattan (a binary “in_manhattan” rule), which complements your existing distance/time/airport logic without changing the model family or training loop. This is a minimal feature-engineering addition (same semantics) that typically drops RMSE materially because Manhattan trips dominate and have different fare dynamics. I also add a very light passenger_count cleanup in test (clip to [0,9]) to avoid out-of-range values creating unseen splits, while keeping all test rows intact. Everything else (filters, split-by-datetime, XGB params, prediction floor/cap, and submission writing) remains unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

print("Listing ../input (if exists):")
try:
    print(os.listdir("../input"))
except Exception as e:
    print("Could not list ../input:", repr(e))

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SAMPLE_SUB_PATH = "../input/sample_submission.csv"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/train.csv"
    TEST_PATH = "/kaggle/input/test.csv"
    SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"



## === cell 1
df = pd.read_csv(TRAIN_PATH, nrows=2 * 10**6)
test_set = pd.read_csv(TEST_PATH)




## === cell 2
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


df["distance_km"] = distance(
    df.pickup_latitude, df.pickup_longitude, df.dropoff_latitude, df.dropoff_longitude
)
test_set["distance_km"] = distance(
    test_set.pickup_latitude,
    test_set.pickup_longitude,
    test_set.dropoff_latitude,
    test_set.dropoff_longitude,
)



## === cell 3
BB = (-75, -73, 40, 41.5)


def select_within_boundingbox(d, BB):
    return (
        (d.pickup_longitude >= BB[0])
        & (d.pickup_longitude <= BB[1])
        & (d.pickup_latitude >= BB[2])
        & (d.pickup_latitude <= BB[3])
        & (d.dropoff_longitude >= BB[0])
        & (d.dropoff_longitude <= BB[1])
        & (d.dropoff_latitude >= BB[2])
        & (d.dropoff_latitude <= BB[3])
    )


print("Old size: %d" % len(df))

df = df[select_within_boundingbox(df, BB)].copy()

df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce")
test_set["passenger_count"] = pd.to_numeric(
    test_set["passenger_count"], errors="coerce"
)

df["passenger_count"] = df["passenger_count"].fillna(0)
test_set["passenger_count"] = test_set["passenger_count"].fillna(0)

test_set["passenger_count"] = test_set["passenger_count"].clip(lower=0, upper=9)

df = df[(df.passenger_count > 0) & (df.passenger_count < 10)].copy()
df = df[(df["distance_km"] >= 0.05) & (df["distance_km"] < 200)].copy()

coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
df = df.dropna(subset=coord_cols + ["fare_amount", "distance_km"]).copy()

for c in ["pickup_longitude", "dropoff_longitude"]:
    df[c] = df[c].clip(BB[0], BB[1])
    test_set[c] = test_set[c].clip(BB[0], BB[1])
for c in ["pickup_latitude", "dropoff_latitude"]:
    df[c] = df[c].clip(BB[2], BB[3])
    test_set[c] = test_set[c].clip(BB[2], BB[3])

print("New size: %d" % len(df))




## === cell 4
def add_datetime_info(dataset):
    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], utc=True, errors="coerce"
    )
    dataset["hour"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["weekday"] = dataset.pickup_datetime.dt.weekday

    h = dataset["hour"].astype("float32")
    dataset["hour_sin"] = np.sin(2.0 * np.pi * h / 24.0).astype("float32")
    dataset["hour_cos"] = np.cos(2.0 * np.pi * h / 24.0).astype("float32")
    return dataset


df = add_datetime_info(df)
test_set = add_datetime_info(test_set)

df = df.dropna(subset=["pickup_datetime", "hour", "day", "month", "weekday"]).copy()




## === cell 5
def add_rule_features(d):
    d["is_night"] = np.where(
        (
            ((d["hour"] >= 20) & (d["hour"] <= 23))
            | ((d["hour"] >= 0) & (d["hour"] < 6))
        ),
        1,
        0,
    )

    def in_box(lon, lat, lon_min, lon_max, lat_min, lat_max):
        return (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)

    jfk = in_box(
        d["pickup_longitude"], d["pickup_latitude"], -73.83, -73.74, 40.62, 40.67
    ) | in_box(
        d["dropoff_longitude"], d["dropoff_latitude"], -73.83, -73.74, 40.62, 40.67
    )
    lga = in_box(
        d["pickup_longitude"], d["pickup_latitude"], -73.90, -73.85, 40.76, 40.78
    ) | in_box(
        d["dropoff_longitude"], d["dropoff_latitude"], -73.90, -73.85, 40.76, 40.78
    )
    ewr = in_box(
        d["pickup_longitude"], d["pickup_latitude"], -74.20, -74.15, 40.67, 40.71
    ) | in_box(
        d["dropoff_longitude"], d["dropoff_latitude"], -74.20, -74.15, 40.67, 40.71
    )

    d["is_airport"] = np.where(jfk | lga | ewr, 1, 0)

    d["is_surge"] = np.where(
        (
            ((d["hour"] >= 16) & (d["hour"] < 20))
            & ((d["weekday"] != 5) & (d["weekday"] != 6))
        ),
        1,
        0,
    )
    return d


df = add_rule_features(df)
test_set = add_rule_features(test_set)




## === cell 6
def add_geo_deltas(d):
    d["abs_lat_diff"] = (d["pickup_latitude"] - d["dropoff_latitude"]).abs()
    d["abs_lon_diff"] = (d["pickup_longitude"] - d["dropoff_longitude"]).abs()
    d["manhattan_deg"] = d["abs_lat_diff"] + d["abs_lon_diff"]
    return d


df = add_geo_deltas(df)
test_set = add_geo_deltas(test_set)



## === cell 7
df["sqrt_distance_km"] = np.sqrt(df["distance_km"].clip(lower=0))
test_set["sqrt_distance_km"] = np.sqrt(test_set["distance_km"].clip(lower=0))




## === cell 8
def add_bearing(d):
    lat1 = np.deg2rad(d["pickup_latitude"].values)
    lon1 = np.deg2rad(d["pickup_longitude"].values)
    lat2 = np.deg2rad(d["dropoff_latitude"].values)
    lon2 = np.deg2rad(d["dropoff_longitude"].values)

    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(y, x)  # radians in [-pi, pi]
    d["bearing_rad"] = bearing
    d["abs_bearing_rad"] = np.abs(bearing)
    return d


df = add_bearing(df)
test_set = add_bearing(test_set)



## === cell 9
df["distance_km_sq"] = (df["distance_km"] ** 2).astype(np.float32)
test_set["distance_km_sq"] = (test_set["distance_km"] ** 2).astype(np.float32)




## === cell 10
def add_manhattan_feature(d):
    def in_box(lon, lat, lon_min, lon_max, lat_min, lat_max):
        return (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)

    man = in_box(
        d["pickup_longitude"], d["pickup_latitude"], -74.03, -73.93, 40.70, 40.88
    ) | in_box(
        d["dropoff_longitude"], d["dropoff_latitude"], -74.03, -73.93, 40.70, 40.88
    )
    d["in_manhattan"] = np.where(man, 1, 0).astype(np.int8)
    return d


df = add_manhattan_feature(df)
test_set = add_manhattan_feature(test_set)



## === cell 11
df = df.drop(df[df["fare_amount"] < 0].index, axis=0)
df = df[(df["fare_amount"] >= 2.5) & (df["fare_amount"] <= 150)].copy()

min_fare_by_dist = 2.5 + 0.5 * df["distance_km"]  # at least some cost with distance
max_fare_by_dist = 2.5 + 20.0 * df["distance_km"]  # very generous upper bound
df = df[
    (df["fare_amount"] >= min_fare_by_dist) & (df["fare_amount"] <= max_fare_by_dist)
].copy()



## === cell 12
try:
    plt.scatter(df["is_night"], df["fare_amount"], c="r", s=1, alpha=0.3)
    plt.show()
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 13
X = df.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = df["fare_amount"]

order = np.argsort(df["pickup_datetime"].astype("int64").values)
X_ord = X.iloc[order].reset_index(drop=True)
y_ord = y.iloc[order].reset_index(drop=True)

split_idx = int(len(X_ord) * 0.9)
X_train, X_test = X_ord.iloc[:split_idx], X_ord.iloc[split_idx:]
y_train, y_test = y_ord.iloc[:split_idx], y_ord.iloc[split_idx:]



## === cell 14
from xgboost import XGBRegressor

base_score = float(y_train.mean())

regressor = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    max_depth=7,
    learning_rate=0.05,
    n_estimators=600,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=2.0,
    reg_alpha=0.1,
    gamma=0.1,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    base_score=base_score,
)
regressor.fit(X_train, y_train)



## === cell 15
predictions = regressor.predict(X_test)



## === cell 16
import math
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(y_test, predictions))
print("Holdout RMSE:", rmse)



## === cell 17
test_set_features = test_set.drop(["key", "pickup_datetime"], axis=1)
test_set_features = test_set_features.reindex(columns=X.columns)

train_medians = X.median(numeric_only=True)
test_set_features = test_set_features.replace([np.inf, -np.inf], np.nan)
test_set_features = test_set_features.fillna(train_medians)

test_set_key = test_set["key"]
y_pred_final = regressor.predict(test_set_features)

y_pred_final = np.maximum(y_pred_final, 2.5)

pred_cap = float(np.nanpercentile(y_train.values, 99.9))
y_pred_final = np.minimum(y_pred_final, pred_cap)

submission = pd.DataFrame(
    {"key": test_set_key, "fare_amount": y_pred_final}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv")

print("Submission shape:", submission.shape)
print(submission.head())
