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

3.41764

# 6. Current score

5.89711

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.34196) has done: 'Your current score is far from the target (8.14 vs 3.42, lower is better), so we can make a few minimal, metric-aligned fixes without changing the core model approach (still a RandomForest on the same feature set). The biggest issue is that you train/evaluate on noisy/outlier fares and impossible geos; adding standard NYC Taxi competition cleaning (fare range, passenger_count range, and NYC-ish coordinate bounds) typically reduces RMSE substantially. Second, your “busyness” merges are currently wrong because you merge without specifying keys, which can silently misalign/join on unintended columns; fixing the merge keys stabilizes those features and improves generalization. Finally, we make the RandomForest deterministic (random_state) and clip negative predictions to 0 (fares can’t be negative), both of which usually nudge RMSE down.'
- What this solution (achieved 5.4051) has done: 'We keep your exact feature set and RandomForest setup, but fix two issues that typically cause a large RMSE gap here: (1) stabilize the “busyness” features by ensuring the grouped tables `a` and `b` have the expected column names after the `agg(["mean","count"])` (your current renaming can silently misalign depending on pandas behavior), and (2) avoid `minute` being computed but unused by keeping everything else identical while making the train/test preprocessing fully consistent and deterministic. Additionally, we ensure busyness merges always succeed by explicitly enforcing the merge keys’ dtypes and filling missing busyness with the training median (instead of 1) to better match the training distribution without changing the model. These are minimal changes aimed at reducing the current RMSE (9.34) toward the target (3.42) without altering the core modeling approach.'
- What this solution (achieved 5.89711) has done: 'Your current RMSE (5.4051) is still worse than the target (3.41764), so we should make small, competition-standard improvements without changing the core approach (same features, same RandomForest, same training loop). The biggest remaining gap is usually from residual outliers/dirty rows and a mismatch between training sampling and the test distribution; we tighten the cleaning to remove extreme distances/fare-per-km anomalies while keeping the same engineered features. We also train on more rows (still feasible under the time limit with your 20-tree RandomForest) to reduce variance and better match the leaderboard distribution. Finally, we ensure strict feature validity (clip passenger_count, handle any remaining NaNs in both train/test busyness consistently) and keep the submission format unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)



## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=2_000_000)




## === cell 2
def haversine_np(lon1, lat1, lon2, lat2):
    """
    https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
    Calculate the great circle distance between two points on the earth (specified in decimal degrees)
    All args must be of equal length.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km




## === cell 3
train_df["distance"] = haversine_np(
    train_df["pickup_longitude"],
    train_df["pickup_latitude"],
    train_df["dropoff_longitude"],
    train_df["dropoff_latitude"],
)



## === cell 4
train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"], errors="coerce"
)



## === cell 5
train_df["year"] = train_df["pickup_datetime"].dt.year
train_df["month"] = train_df["pickup_datetime"].dt.month
train_df["day"] = train_df["pickup_datetime"].dt.day
train_df["hour"] = train_df["pickup_datetime"].dt.hour
train_df["minute"] = train_df["pickup_datetime"].dt.minute



## === cell 6
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 7
new_york_lat = 40
new_york_long = -74
train_df.describe()



## === cell 8
fare_cond = (train_df["fare_amount"] > 0) & (train_df["fare_amount"] <= 200)
passenger_cond = (train_df["passenger_count"] >= 1) & (train_df["passenger_count"] <= 6)
dist_cond = (train_df["distance"] > 0) & (train_df["distance"] <= 100)

geo_cond = (
    train_df["pickup_longitude"].between(-74.3, -73.7)
    & train_df["dropoff_longitude"].between(-74.3, -73.7)
    & train_df["pickup_latitude"].between(40.5, 41.0)
    & train_df["dropoff_latitude"].between(40.5, 41.0)
)

fare_per_km = train_df["fare_amount"] / (train_df["distance"] + 1e-6)
fpk_cond = (fare_per_km >= 1.0) & (fare_per_km <= 50.0)

print("Old size: %d" % len(train_df))
train_df = train_df[fare_cond & passenger_cond & dist_cond & geo_cond & fpk_cond].copy()
print("New size: %d" % len(train_df))



## === cell 9
train_df.describe()



## === cell 10
for col in {
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
}:
    train_df["rough" + col] = train_df[col].round(2)



## === cell 11
a = train_df.groupby(["roughpickup_latitude", "roughpickup_longitude"])[
    ["pickup_latitude", "pickup_longitude"]
].agg(["mean", "count"])
a.columns = ["_".join(c) for c in a.columns.to_flat_index()]
a.head()
a.sort_values(
    ["pickup_latitude_count", "pickup_longitude_count"], ascending=False
).head()



## === cell 12
b = train_df.groupby(["roughdropoff_latitude", "roughdropoff_longitude"])[
    ["dropoff_latitude", "dropoff_longitude"]
].agg(["mean", "count"])
b.columns = ["_".join(c) for c in b.columns.to_flat_index()]
b.head()
b.sort_values(
    ["dropoff_latitude_count", "dropoff_longitude_count"], ascending=False
).head(n=20)



## === cell 13
b[b["dropoff_latitude_mean"] < 40.7].sort_values(
    ["dropoff_latitude_count", "dropoff_longitude_count"], ascending=False
).head()



## === cell 14
b["dropoff_latitude_count"].plot.hist()
plt.title("occurance of counts of rough dropoff locations")
plt.yscale("log")



## === cell 15
a["pickup_latitude_count"].plot.hist()
plt.title("occurance of counts of rough dropoff locations")
plt.yscale("log")



## === cell 16
a[a["pickup_latitude_count"] > 1000].shape
a.shape



## === cell 17
a = (
    a[["pickup_latitude_count"]]
    .reset_index()
    .rename(columns={"pickup_latitude_count": "pickup_busyness"})
)
b = (
    b[["dropoff_latitude_count"]]
    .reset_index()
    .rename(columns={"dropoff_latitude_count": "dropoff_busyness"})
)



## === cell 18
for c in [
    "roughpickup_latitude",
    "roughpickup_longitude",
    "roughdropoff_latitude",
    "roughdropoff_longitude",
]:
    train_df[c] = train_df[c].astype(np.float64)



## === cell 19
train_df = pd.merge(
    train_df, a, how="left", on=["roughpickup_latitude", "roughpickup_longitude"]
)
train_df = pd.merge(
    train_df, b, how="left", on=["roughdropoff_latitude", "roughdropoff_longitude"]
)



## === cell 20
pickup_med = float(train_df["pickup_busyness"].median())
dropoff_med = float(train_df["dropoff_busyness"].median())
train_df["pickup_busyness"] = train_df["pickup_busyness"].fillna(pickup_med)
train_df["dropoff_busyness"] = train_df["dropoff_busyness"].fillna(dropoff_med)

train_df.head()



## === cell 21
X = train_df[
    ["distance", "year", "month", "day", "hour", "pickup_busyness", "dropoff_busyness"]
].values
Y = train_df["fare_amount"].values



## === cell 22
from sklearn.ensemble import RandomForestRegressor



## === cell 23
kwargs = {
    "bootstrap": True,
    "max_depth": None,
    "max_features": 3,
    "min_samples_leaf": 9,
    "min_samples_split": 2,
}

rand_regr = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1, **kwargs)



## === cell 24
rand_regr.fit(X, Y)



## === cell 25
y_pred = rand_regr.predict(X)
print(
    "chi squared  rand forest with date %s"
    % (np.sum((Y - y_pred) ** 2.0) / len(Y)) ** 0.5
)



## === cell 26
rand_regr.score(X, Y)



## === cell 27
test_df = pd.read_csv("../input/test.csv")



## === cell 28
test_df["distance"] = haversine_np(
    test_df["pickup_longitude"],
    test_df["pickup_latitude"],
    test_df["dropoff_longitude"],
    test_df["dropoff_latitude"],
)



## === cell 29
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"], errors="coerce")



## === cell 30
test_df["year"] = test_df["pickup_datetime"].dt.year
test_df["month"] = test_df["pickup_datetime"].dt.month
test_df["day"] = test_df["pickup_datetime"].dt.day
test_df["hour"] = test_df["pickup_datetime"].dt.hour
test_df["minute"] = test_df["pickup_datetime"].dt.minute



## === cell 31
for col in {
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
}:
    test_df["rough" + col] = test_df[col].round(2)



## === cell 32
for c in [
    "roughpickup_latitude",
    "roughpickup_longitude",
    "roughdropoff_latitude",
    "roughdropoff_longitude",
]:
    test_df[c] = test_df[c].astype(np.float64)

test_df = pd.merge(
    test_df, a, how="left", on=["roughpickup_latitude", "roughpickup_longitude"]
)
test_df = pd.merge(
    test_df, b, how="left", on=["roughdropoff_latitude", "roughdropoff_longitude"]
)



## === cell 33
pickup_med = float(train_df["pickup_busyness"].median())
dropoff_med = float(train_df["dropoff_busyness"].median())

test_df["pickup_busyness"] = test_df["pickup_busyness"].fillna(pickup_med)
test_df["dropoff_busyness"] = test_df["dropoff_busyness"].fillna(dropoff_med)



## === cell 34
X_to_pred = test_df[
    ["distance", "year", "month", "day", "hour", "pickup_busyness", "dropoff_busyness"]
].values
y_pred = rand_regr.predict(X_to_pred)

y_pred = np.clip(y_pred, 0, None)



## === cell 35
submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": y_pred}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
