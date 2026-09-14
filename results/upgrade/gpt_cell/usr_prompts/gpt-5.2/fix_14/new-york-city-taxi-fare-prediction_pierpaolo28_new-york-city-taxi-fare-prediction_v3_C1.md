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

4.10469

# 6. Current score

5.58744

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.34518) has done: 'Diagnosis: The crash happens because `seaborn.regplot` in seaborn 0.12+ uses keyword-only parameters; passing `'pickup_longitude'` and `'pickup_latitude'` as positional arguments is interpreted differently and results in `data` being assigned twice (once positionally and once via `data=df`).  
Patch summary: Update the two `sns.regplot(...)` calls in cell 12 to pass `x=` and `y=` explicitly while keeping the same plotting behavior (`fit_reg=False`, same axes).  
Updated cells: Only cell 12 is changed; all other logic and outputs remain the same.  
Compatibility notes for cell k+1: `df` is not modified in cell 12, so cell 13 continues to work unchanged.  
Assumptions: seaborn is version 0.12.x (as installed), and the intent is to make the same scatter-style plot without regression fit.'
- What this solution (achieved 5.34069) has done: 'Diagnosis: The crash comes from calling `seaborn.lmplot()` with positional arguments for `x` and `y` while also passing `data=df`; in seaborn 0.12+, `lmplot`’s signature makes the first positional argument `data`, so your first string is interpreted as `data` and then `data=df` is a second value for the same parameter. This leads to `TypeError: lmplot() got multiple values for argument 'data'`. The fix is to pass `x=` and `y=` as keyword arguments to avoid ambiguity and keep the plot semantics unchanged.

Patch summary: Update the `sns.lmplot` call in cell 14 to use explicit keyword arguments `x=` and `y=` while keeping `fit_reg=False` and `data=df` the same.

Updated cells: cell 14 only.

Compatibility notes for cell k+1: `df` is unchanged and still contains `abs_lat_diff` and `abs_lon_diff`, so cell 15 run identically.

Assumptions: seaborn version is 0.12.2 as listed; no other code depends on the return value of `sns.lmplot()` in this cell.'
- What this solution (achieved 5.37712) has done: 'Your current score (5.34069 RMSE) is worse than the target (4.10469), so we should improve generalization with the smallest possible change while keeping the same overall approach (same features and still using `XGBRegressor`). The biggest issue is that `fare_amount` is still present inside `X_train/X_valid` (data leakage into the model via XGBoost using all columns in the DataFrame), which can distort training/validation behavior and harm real test performance. I minimally fix this by removing `fare_amount` from the feature matrices before splitting, keeping your exact feature set for fitting/predicting. I also set a deterministic `random_state` for the split and model to stabilize the outcome without changing the training approach.'
- What this solution (achieved 5.3441) has done: 'Your current RMSE (5.37712) is worse than the target (4.10469), so we should make a small, legitimate improvement without changing the overall approach (still XGBRegressor on the same engineered distance features). The biggest win with minimal risk here is to add the already-parsed `pickup_datetime` signal (hour/day-of-week/month) as additional numeric features, since fare depends strongly on time patterns; this keeps the same model/training loop and just extends the feature columns used for fitting/predicting. I also apply the exact same basic row filtering to the training set coordinates that Kaggle Taxi solutions commonly need (remove impossible lat/lon ranges), which reduces noise and typically improves RMSE without changing the modeling method. Finally, I keep submission formatting identical and ensure the additional features are computed for both train and test so the pipeline runs end-to-end.'
- What this solution (achieved 5.3441) has done: 'You’re currently worse than the target (5.3441 RMSE vs 4.10469), so we should make small, legitimate improvements without changing the core approach (still the same engineered features + `XGBRegressor` fit/predict). The biggest low-risk gain here is correcting the distance formula bug in `minkowski_distance` (it mistakenly uses `abs(y2 - y1)` instead of `abs(y2 - y1) ** p`), which affects your `euclidean` feature (even if it’s not currently used, it’s computed and could be added later). Next, we keep the exact same model type/training loop but set a few standard XGBoost parameters (objective, eval metric, tree method) to stabilize and typically improve RMSE without changing the overall method. Finally, we ensure non-negative fare predictions (a safe post-processing for this competition) and write the submission with a `.csv` filename matching Kaggle expectations.'
- What this solution (achieved 5.3165) has done: 'Your current RMSE (5.3441) is worse than the target (4.10469), so we should improve generalization with the smallest changes while keeping the same XGBRegressor + distance/time features approach. The biggest low-risk gain here is to clean the training data a bit more consistently with the test domain: remove unrealistically large `haversine` trips and non-positive fares after your earlier filters, which otherwise add heavy noise that hurts public LB RMSE. Next, keep the exact same model type and training loop but set a few conservative XGBoost hyperparameters (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`) that usually reduce RMSE on this classic competition without changing the approach. Finally, apply the same coordinate bounds to the test set (not for “score tricks”, but to avoid producing extreme-feature rows that the model never saw) and keep the submission format identical.'
- What this solution (achieved 5.3165) has done: 'Your current RMSE (5.3165) is worse than the target (4.10469), so we should make a small, legitimate improvement without changing the core approach (same engineered features and the same XGBRegressor fit/predict flow). The biggest low-risk issue is that `pickup_datetime` is still present in `X` (as a datetime column) even though it is not in `FEATURES`; this can subtly affect splitting/handling and also risks accidental misuse—dropping it keeps training strictly on the intended numeric features. Next, apply the exact same geographic bounds to the test set as you do to train, but without dropping rows: instead we clip out-of-domain coordinates into the training domain so the model sees feature ranges it was trained on (stabilizes predictions without changing model logic). Finally, we fill any resulting NaNs/infs in the engineered test features (rare but possible) with training medians so the submission is always valid and avoids extreme RMSE from malformed rows.'
- What this solution (achieved 5.3101) has done: 'Your current RMSE (5.3165) is worse than the target (4.10469), so we should make the smallest legitimate improvement while keeping the same XGBRegressor + (distance + time + passenger_count) feature approach. The biggest low-risk issue is a train/test feature mismatch: you compute `haversine` after filtering on coordinates in train, but in test you compute `haversine` before clipping coordinates—this can create out-of-domain distances the model never saw and hurts RMSE. I reorder test preprocessing to clip coordinates first, then compute `abs_*`, `euclidean`, and `haversine`, keeping identical feature definitions but improving consistency. I also apply the same sanity clipping to engineered distance features (e.g., non-negative and capped similarly to train’s `haversine<=100`) to reduce extreme outliers without changing model architecture or training loop.'
- What this solution (achieved 5.28159) has done: 'Your current RMSE (5.3101) is worse than the target (4.10469), so we should make a small improvement without changing the overall approach (same engineered features, same XGBRegressor training loop). The biggest low-risk gain here is to add one more standard, cheap feature derived from existing columns: the straight-line trip distance in kilometers (your already-computed `haversine`), plus a simple interaction term `haversine_per_passenger` to let the model scale distance effects by passenger count. This keeps the model and training semantics identical (still fitting XGBRegressor on numeric engineered features), but typically reduces error on this competition with minimal code changes. We also ensure the new features are computed consistently for train and test and included in `FEATURES`, and keep submission writing unchanged.'
- What this solution (achieved 5.40807) has done: 'Your current RMSE (5.28159) is worse than the target (4.10469), so we should make small, legitimate improvements without changing the core XGBRegressor approach or feature definitions. The biggest low-risk gap is that training still contains label noise/outliers even after basic filters; tightening the geographic + distance + fare sanity filters slightly (while keeping the same logic) typically improves public LB RMSE for this classic competition. Next, we keep the same model and training loop but increase robustness by adding a tiny amount of L2 regularization (`reg_lambda`) and using the same median-imputation strategy for train/valid as you already do for test, preventing rare NaN/inf rows from hurting fit/predict. Finally, we ensure the submission is aligned to `sample_submission.csv` keys to avoid any accidental key ordering mismatch.'
- What this solution (achieved 5.38108) has done: 'Your current RMSE (5.40807) is worse than the target (4.10469), so we should make the smallest legitimate changes that usually reduce error in this competition while keeping the same XGBRegressor + engineered distance/time features approach. The biggest low-risk issue is that we’re training on a random 500k sample that can be skewed (especially in time), so we instead read a slightly larger slice and then subsample deterministically; this typically improves generalization without changing the modeling logic. Next, we add one very standard, minimal feature that strongly improves taxi fare RMSE: the trip direction (bearing) computed from the same lat/lon columns (no new data, same pipeline). Finally, we ensure the exact same feature engineering is applied to both train and test, and keep submission generation identical.'
- What this solution (achieved 5.58744) has done: 'Your current RMSE (5.38108) is still far above the target (4.10469), so we should make small, legitimate generalization improvements without changing the core approach (same engineered numeric features and the same `XGBRegressor` fit/predict flow). The biggest likely issue is residual outliers/noise in the 2M-row slice (e.g., airport/JFK trips and rare coordinate glitches) that aren’t well-handled by the current feature set; we add two very standard, low-risk filters: remove zero-distance “trips” and cap extreme `abs_lat_diff/abs_lon_diff` consistent with NYC geography. To keep semantics identical and stable, we compute `euclidean`/`haversine` right after `abs_*` (before plotting) and ensure those filters are applied after the geographic bounds filter. Finally, we apply the same engineered-feature cleanup (inf/NaN handling) consistently and keep submission creation unchanged.'
- What this solution (achieved 5.58744) has done: 'Your current RMSE (5.58744) is worse than the target (4.10469), so we should make a small, legitimate improvement without changing the core approach (same engineered features and `XGBRegressor` fit/predict flow). The biggest low-risk issue is that we filter train rows using `haversine` that was computed *before* applying the NYC coordinate bounds, so some distance values are inconsistent with the filtered coordinates and can inject noise. I recompute `abs_*`, `euclidean`, `haversine`, and `bearing` immediately after the coordinate filter (and before the final outlier filters), keeping feature definitions identical but making them consistent with the cleaned domain. This is a minimal change that typically improves generalization for this competition while preserving architecture, loss, and training loop, and it still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
raw = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=2_000_000,
    parse_dates=["pickup_datetime"],
).drop(columns="key")

raw = raw.dropna()

df = raw.sample(n=500_000, random_state=42).copy()
del raw

df.head()



## === cell 2
df.describe()



## === cell 3
plt.figure(figsize=(10, 8))
plt.hist(df["fare_amount"])
plt.title("Fare Distribution")



## === cell 4
print(f"Number of negative fares: {len(df[df['fare_amount'] < 0])}")
print(f"Number of fares equal to 0: {len(df[df['fare_amount'] == 0])}")



## === cell 5
df = df[df["fare_amount"].between(left=2.5, right=df["fare_amount"].max())]




## === cell 6
def ecdf(x):
    x = np.sort(x)
    n = len(x)
    y = np.arange(1, n + 1, 1) / n
    return x, y




## === cell 7
x, y = ecdf(df["fare_amount"])
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.ylabel("Percentile")
plt.xlabel("Fare Amount")
plt.title("Fare Amount ECDF")



## === cell 8
df = df[df["fare_amount"].between(left=2.5, right=70)]



## === cell 9
x, y = ecdf(df["fare_amount"])
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.ylabel("Percentile")
plt.xlabel("Fare Amount")
plt.title("Fare Amount ECDF")



## === cell 10
df["passenger_count"].value_counts().plot.bar()
plt.title("Passenger Counts")
plt.xlabel("Passengers Numbers")
plt.ylabel("Frequency")



## === cell 11
df = df.loc[df["passenger_count"] < 6]



## === cell 12
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
axes = axes.flatten()

sns.regplot(
    x="pickup_longitude", y="pickup_latitude", fit_reg=False, data=df, ax=axes[0]
)
sns.regplot(
    x="dropoff_longitude", y="dropoff_latitude", fit_reg=False, data=df, ax=axes[1]
)
axes[0].set_title("Pickup Locations")
axes[1].set_title("Dropoff Locations")



## === cell 13
df["abs_lat_diff"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
df["abs_lon_diff"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()



## === cell 14
sns.lmplot(x="abs_lat_diff", y="abs_lon_diff", fit_reg=False, data=df)
plt.title("Absolute latitude difference vs Absolute longitude difference")



## === cell 15
zero_diff = df[(df["abs_lat_diff"] == 0) & (df["abs_lon_diff"] == 0)]
zero_diff.shape




## === cell 16
def minkowski_distance(x1, x2, y1, y2, p):
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1) ** p)) ** (1 / p)




## === cell 17
df["euclidean"] = minkowski_distance(
    df["pickup_longitude"],
    df["dropoff_longitude"],
    df["pickup_latitude"],
    df["dropoff_latitude"],
    2,
)



## === cell 18
plt.figure(figsize=(10, 8))
plt.hist(df["euclidean"])
plt.title("Euclidean Distance Distribution")
ax = plt.subplot(111)
ax.set_xlim([0, 500])



## === cell 19
plt.figure(figsize=(10, 6))

for p, grouped in df.groupby("passenger_count"):
    sns.kdeplot(grouped["fare_amount"], label=f"{p} passengers")

plt.xlabel("Fare Amount")
plt.ylabel("Density")
plt.title("Distribution of Fare Amount by Number of Passengers")



## === cell 20
df.groupby("passenger_count")["fare_amount"].agg(["mean", "count"])



## === cell 21
df.groupby("passenger_count")["fare_amount"].mean().plot.bar(color="b")
plt.title("Average Fare by Passenger Count")



## === cell 22
R = 6378


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    source: https://stackoverflow.com/a/29546836
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = R * c

    return km




## === cell 23
def bearing_np(lon1, lat1, lon2, lat2):
    """
    Initial bearing from point 1 to point 2 in radians (wrapped to [-pi, pi]).
    Vectorized numpy implementation.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)




## === cell 24
df["haversine"] = haversine_np(
    df["pickup_longitude"],
    df["pickup_latitude"],
    df["dropoff_longitude"],
    df["dropoff_latitude"],
)



## === cell 25
sns.kdeplot(df["haversine"])



## === cell 26
corrs = df.corr(numeric_only=True)
corrs["fare_amount"].plot.bar(color="b")
plt.title("Correlation with Fare Amount")



## === cell 27
df = df[
    df["pickup_longitude"].between(-75, -72)
    & df["dropoff_longitude"].between(-75, -72)
    & df["pickup_latitude"].between(40, 42)
    & df["dropoff_latitude"].between(40, 42)
].copy()

df["abs_lat_diff"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
df["abs_lon_diff"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()

df["euclidean"] = minkowski_distance(
    df["pickup_longitude"],
    df["dropoff_longitude"],
    df["pickup_latitude"],
    df["dropoff_latitude"],
    2,
)

df["haversine"] = haversine_np(
    df["pickup_longitude"],
    df["pickup_latitude"],
    df["dropoff_longitude"],
    df["dropoff_latitude"],
)

df["bearing"] = bearing_np(
    df["pickup_longitude"],
    df["pickup_latitude"],
    df["dropoff_longitude"],
    df["dropoff_latitude"],
).astype(np.float32)

df = df[(df["fare_amount"].between(2.5, 60)) & (df["haversine"].between(0, 60))].copy()

df = df[~((df["abs_lat_diff"] == 0) & (df["abs_lon_diff"] == 0))].copy()

df = df[(df["abs_lat_diff"] <= 1.5) & (df["abs_lon_diff"] <= 2.0)].copy()

df["pickup_hour"] = df["pickup_datetime"].dt.hour.astype(np.int16)
df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek.astype(np.int16)
df["pickup_month"] = df["pickup_datetime"].dt.month.astype(np.int16)

df["haversine_per_passenger"] = df["haversine"] / (
    df["passenger_count"].astype(np.float32) + 1.0
)

df.head()



## === cell 28
test = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)

for col, lo, hi in [
    ("pickup_longitude", -75.0, -72.0),
    ("dropoff_longitude", -75.0, -72.0),
    ("pickup_latitude", 40.0, 42.0),
    ("dropoff_latitude", 40.0, 42.0),
]:
    test[col] = test[col].clip(lo, hi)

test["abs_lat_diff"] = (test["dropoff_latitude"] - test["pickup_latitude"]).abs()
test["abs_lon_diff"] = (test["dropoff_longitude"] - test["pickup_longitude"]).abs()

test_id = list(test.pop("key"))

test["euclidean"] = minkowski_distance(
    test["pickup_longitude"],
    test["dropoff_longitude"],
    test["pickup_latitude"],
    test["dropoff_latitude"],
    2,
)

test["haversine"] = haversine_np(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["dropoff_longitude"],
    test["dropoff_latitude"],
)

test["pickup_hour"] = test["pickup_datetime"].dt.hour.astype(np.int16)
test["pickup_dayofweek"] = test["pickup_datetime"].dt.dayofweek.astype(np.int16)
test["pickup_month"] = test["pickup_datetime"].dt.month.astype(np.int16)

test["haversine"] = test["haversine"].clip(0, 60)
test["euclidean"] = test["euclidean"].clip(lower=0)
test["abs_lat_diff"] = test["abs_lat_diff"].clip(0, 1.5)
test["abs_lon_diff"] = test["abs_lon_diff"].clip(0, 2.0)

test["haversine_per_passenger"] = test["haversine"] / (
    test["passenger_count"].astype(np.float32) + 1.0
)

test["bearing"] = bearing_np(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["dropoff_longitude"],
    test["dropoff_latitude"],
).astype(np.float32)

test.describe()



## === cell 29
from sklearn.model_selection import train_test_split

y = np.array(df["fare_amount"])
X = df.drop(columns=["fare_amount", "pickup_datetime"])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.30, random_state=42
)



## === cell 30
import xgboost as xgb

xgbr = xgb.XGBRegressor(
    random_state=42,
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=2.0,
)

FEATURES = [
    "haversine",
    "haversine_per_passenger",
    "bearing",
    "abs_lat_diff",
    "abs_lon_diff",
    "passenger_count",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
]

train_feature_medians = X_train[FEATURES].median(numeric_only=True)
X_train_features = (
    X_train[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(train_feature_medians)
)
X_valid_features = (
    X_valid[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(train_feature_medians)
)

xgbr.fit(X_train_features, y_train)



## === cell 31
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def metrics(train_pred, valid_pred, y_train, y_valid):
    """Calculate metrics:
    Root mean squared error and mean absolute percentage error"""

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))

    train_ape = abs((y_train - train_pred) / y_train)
    valid_ape = abs((y_valid - valid_pred) / y_valid)

    train_ape[train_ape == np.inf] = 0
    train_ape[train_ape == -np.inf] = 0
    valid_ape[valid_ape == np.inf] = 0
    valid_ape[valid_ape == -np.inf] = 0

    train_mape = 100 * np.mean(train_ape)
    valid_mape = 100 * np.mean(valid_ape)

    return train_rmse, valid_rmse, train_mape, valid_mape


def evaluate(model, features, X_train, X_valid, y_train, y_valid):
    """Mean absolute percentage error"""

    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)

    train_rmse, valid_rmse, train_mape, valid_mape = metrics(
        train_pred, valid_pred, y_train, y_valid
    )

    print(f"Training:   rmse = {round(train_rmse, 2)} \t mape = {round(train_mape, 2)}")
    print(f"Validation: rmse = {round(valid_rmse, 2)} \t mape = {round(valid_mape, 2)}")




## === cell 32
evaluate(
    xgbr,
    FEATURES,
    X_train_features,
    X_valid_features,
    y_train,
    y_valid,
)



## === cell 33
test_features = test[FEATURES].replace([np.inf, -np.inf], np.nan)
test_features = test_features.fillna(train_feature_medians)

preds = xgbr.predict(test_features)
preds = np.clip(preds, 0, None)

sample_sub = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)
sub = pd.DataFrame({"key": test_id, "fare_amount": preds})
sub = sample_sub[["key"]].merge(sub, on="key", how="left")

sub.to_csv("submission.csv", index=False)

sns.histplot(sub["fare_amount"], bins=50)
plt.title("Distribution of Predicted Fare Amount")
