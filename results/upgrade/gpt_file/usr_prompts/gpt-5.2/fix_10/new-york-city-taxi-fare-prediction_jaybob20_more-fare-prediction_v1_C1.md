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
scipy==1.15.3
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

3.70811

# 6. Current score

5.78467

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.49132) has done: 'I fix the pandas datetime API breakage (`.dt.week`/`.dt.weekofyear`) by using ISO calendar week, and correct a typo in the coordinate bounds that was filtering out too much training data. I also fix feature engineering so it uses each dataset’s own `pickup_datetime` (it currently overwrites train with test timestamps), and ensure distance features are created for both train and test before scaling/PCA. To make XGBoost accept the data, I remove non-numeric columns (like `key` and datetime) from the training matrices and update the deprecated XGBoost objective and prediction call. Finally, I ensure a valid `submission.csv` with the required `key,fare_amount` columns is written.'
- What this solution (achieved 4.51377) has done: 'Your current score (4.49132 RMSE) is worse than the target (3.70811), so we should make small, low-risk changes that improve generalization without changing the overall approach. The biggest single issue is that you standardize latitude/longitude before clustering, but then cluster on the already-scaled coordinates, which destroys geographic meaning; clustering should be done on raw lat/lon and then the resulting cluster label can be used alongside scaled features. I keep all models/feature types the same, but (1) compute `cluster` from unscaled raw coordinates and (2) remove early stopping (it’s explicitly disallowed) by training a fixed number of boosting rounds equal to the previous maximum. These changes are minimal and should move RMSE closer to your target while still producing a valid `submission.csv`.'
- What this solution (achieved 4.52538) has done: 'You’re currently worse than the target (4.51377 vs 3.70811 RMSE; lower is better), so we make the smallest changes that typically improve generalization without changing the model type or training loop. The largest issue is that your “distance” features are computed incorrectly: `pdist` is being applied to a 2x2 matrix, so it mixes lat/lon within a point instead of computing pickup→dropoff distances; we replace it with correct vectorized coordinate-distance calculations for each metric. We also fix a silent bug in the multiprocessing loop where `dataset = add_time_features(dataset)` doesn’t persist back into `train/test`, and we ensure we don’t double-call `add_time_features`. Everything else (filters, scaling, PCA, Birch clustering on raw coords, XGBoost objective/rounds) stays the same, and the script still writes a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.38859) has done: 'Your current RMSE (4.52538) is worse than the target (3.70811), so we should make a small, low-risk improvement that doesn’t change the overall modeling approach. The biggest remaining issue is that several geographic features (raw lat/lon and the derived distance/PCA features) are being fit on a random split of individual trips, which causes leakage across identical/similar routes and inflates validation while hurting true generalization; we instead split by `pickup_datetime` order to mimic the real test distribution more closely. We keep the same feature set, same XGBoost training API, and same number of boosting rounds, but we also add a very standard, minimal cleanup: dropping unrealistic coordinate/passenger rows (NYC bounding box + passenger_count>=1) to reduce label noise without changing the model. Finally, we keep the same submission format, but clip negative predictions to 0 since fares can’t be negative (this typically improves RMSE slightly).'
- What this solution (achieved 6.56393) has done: 'Your score got much worse after switching to a time-ordered split because the data was never sorted by `pickup_datetime`, so the “time split” effectively became an arbitrary chunk split that can distort training and learned patterns. To move RMSE back down toward the 3.70811 target with minimal change, I (1) sort the filtered training data by `pickup_datetime` before the split to make the split truly temporal, and (2) slightly soften the coordinate filter by using an NYC bounding box plus passenger/fare filters, rather than restricting to the test min/max long/lat (which can remove too much valid training data and hurt generalization). Everything else (feature engineering, scaling, Birch on raw coords, PCA, XGBoost objective/rounds, clipping, submission format) is kept the same.'
- What this solution (achieved 6.22633) has done: 'We need to move RMSE down toward the 3.70811 target (lower is better), so the smallest safe gains come from fixing a data leakage/bug in your temporal split and ensuring the model can actually use the time features you engineered. Right now you drop `pickup_datetime` before splitting (because it was dropped earlier), so your “time split” is not truly temporal; we keep `pickup_datetime` until after sorting/splitting, then drop it from the model matrices. We also add the engineered time columns (hour/day/week/month/day_of_year) into the model feature set (they’re currently being dropped), which is a minimal change that typically improves RMSE without changing the model type or training loop. Finally, we make sure `cluster` is treated as a numeric feature (int) and we keep the submission format identical.'
- What this solution (achieved 6.19837) has done: 'Your current RMSE (6.22633) is much worse than the target (3.70811), so we should make minimal, legitimate changes that improve generalization without changing the modeling approach. The biggest score-killer in your pipeline is that PCA is fit on **only the training portion** but then applied to test; since the train sample is just 1M rows, this can misrepresent the joint distribution—so we fit PCA on the **combined (train+test) pca_features** to stabilize those components (no label leakage). Next, your temporal split currently trains on the oldest 80% and validates on the newest 20%; since Kaggle test spans years and is not strictly “future”, this can hurt—so we switch back to a standard random split (same holdout size, same model/training loop) to better match test distribution. Finally, we add the very standard fare-specific cleanup of removing trips with near-zero distance but non-trivial fares, which reduces label noise while preserving your core features/model.'
- What this solution (achieved 5.79515) has done: 'Your RMSE (6.19837) is much worse than the 3.70811 target, so we want small, legitimate fixes that usually reduce error without changing the overall model/feature approach. The biggest issue is a train/test distribution mismatch introduced by fitting `StandardScaler` on train only; for these unlabeled coordinate/distance features it’s safe (no target leakage) and typically helps to fit the scaler on the concatenated (train+test) feature matrix so scaling matches test. Next, the current XGBoost configuration is too “aggressive” (very high `eta=0.3` with only 300 rounds), which often underfits this problem; we keep the same XGBoost training API/loop but lower `eta` and increase `num_boost_round` to compensate (no early stopping). Finally, we add a standard NYC Taxi cleanup: remove rare but very harmful outliers where `fare_amount` is extremely high relative to the trip distance, which usually improves RMSE on the leaderboard.'
- What this solution (achieved 5.78467) has done: 'We need to move your RMSE down toward 3.70811 (lower is better) with minimal, safe edits that don’t change the overall XGBoost approach or feature set. The biggest score issue is that you only train on the “train” split and never refit on all available cleaned data, so the final model used for the Kaggle submission is under-trained; we keep the same parameters/num_boost_round but retrain on the full cleaned 1M-row sample after the holdout check. Next, `cluster` is currently treated as an ordinal integer; keeping the same clustering logic, we one-hot encode it (very small change that usually helps trees) and align train/test columns. Finally, we remove rounding to 2 decimals in the submission (it can only worsen RMSE) while keeping the required CSV format identical.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import datetime as dt

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
import xgboost as xgb

np.random.seed(123)



## === cell 1
TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SAMPLE_PATH = "/kaggle/input/sample_submission.csv"

train = pd.read_csv(TRAIN_PATH, nrows=1_000_000)
test = pd.read_csv(TEST_PATH)

print(train.shape, test.shape)
train.head()



## === cell 2
print("Sum of NaN values for each column")
print(train.isnull().sum())

train = train.dropna()
print("Sum of NaN values for each column after dropping NaN")
print(train.isnull().sum())



## === cell 3
pickup_longitude_min = test.pickup_longitude.min()
pickup_longitude_max = test.pickup_longitude.max()
pickup_latitude_min = test.pickup_latitude.min()
pickup_latitude_max = test.pickup_latitude.max()
dropoff_longitude_min = test.dropoff_longitude.min()
dropoff_longitude_max = test.dropoff_longitude.max()
dropoff_latitude_min = test.dropoff_latitude.min()
dropoff_latitude_max = test.dropoff_latitude.max()



## === cell 4
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] < 300)]
train = train.loc[(train["passenger_count"] >= 1) & (train["passenger_count"] <= 8)]

train = train.loc[
    train["pickup_longitude"].between(-74.5, -72.8)
    & train["dropoff_longitude"].between(-74.5, -72.8)
    & train["pickup_latitude"].between(40.5, 41.8)
    & train["dropoff_latitude"].between(40.5, 41.8)
]

train.describe()




## === cell 5
def haversine_np_vec(lon1, lat1, lon2, lat2):
    """
    Vectorized haversine distance (km) between (lon1,lat1) and (lon2,lat2).
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    aa = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(aa))
    km = 6367 * c
    return km


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], utc=True, errors="coerce"
    )
    df["pickup_datetime"] = df["pickup_datetime"].fillna(
        pd.Timestamp("2009-01-01", tz="UTC")
    )

    df["hour_of_day"] = df["pickup_datetime"].dt.hour
    df["day"] = df["pickup_datetime"].dt.day

    iso = df["pickup_datetime"].dt.isocalendar()
    df["week"] = iso.week.astype(np.int16)
    df["week_of_year"] = iso.week.astype(np.int16)

    df["month"] = df["pickup_datetime"].dt.month
    df["day_of_year"] = df["pickup_datetime"].dt.dayofyear
    return df




## === cell 6
dist_types = [
    "chebyshev",
    "euclidean",
    "canberra",
    "sqeuclidean",
    "braycurtis",
    "minkowski",
    "hamming",
    "cityblock",
]


def add_distance_features(df):
    lat1 = df["pickup_latitude"].to_numpy(dtype=np.float64)
    lon1 = df["pickup_longitude"].to_numpy(dtype=np.float64)
    lat2 = df["dropoff_latitude"].to_numpy(dtype=np.float64)
    lon2 = df["dropoff_longitude"].to_numpy(dtype=np.float64)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    abs_dlat = np.abs(dlat)
    abs_dlon = np.abs(dlon)

    df["cityblock"] = abs_dlat + abs_dlon
    df["euclidean"] = np.sqrt(dlat * dlat + dlon * dlon)
    df["sqeuclidean"] = dlat * dlat + dlon * dlon
    df["chebyshev"] = np.maximum(abs_dlat, abs_dlon)

    denom_lat = np.abs(lat1) + np.abs(lat2)
    denom_lon = np.abs(lon1) + np.abs(lon2)
    canberra_lat = np.where(denom_lat == 0, 0.0, abs_dlat / denom_lat)
    canberra_lon = np.where(denom_lon == 0, 0.0, abs_dlon / denom_lon)
    df["canberra"] = canberra_lat + canberra_lon

    denom_bc = np.abs(lat1) + np.abs(lat2) + np.abs(lon1) + np.abs(lon2)
    df["braycurtis"] = np.where(denom_bc == 0, 0.0, (abs_dlat + abs_dlon) / denom_bc)

    df["minkowski"] = df["euclidean"]

    df["hamming"] = (
        (lat1 != lat2).astype(np.float64) + (lon1 != lon2).astype(np.float64)
    ) / 2.0

    df["haversine"] = haversine_np_vec(lon1, lat1, lon2, lat2)

    return df


train = add_distance_features(train)
test = add_distance_features(test)

train = add_time_features(train)
test = add_time_features(test)

train = train.loc[~((train["haversine"] < 0.01) & (train["fare_amount"] > 8.0))].copy()

fare_per_km = train["fare_amount"] / (train["haversine"] + 0.1)
train = train.loc[(fare_per_km < 200)].copy()

train.head(3)



## === cell 7
features = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "haversine",
    "chebyshev",
    "euclidean",
    "canberra",
    "sqeuclidean",
    "braycurtis",
    "minkowski",
    "hamming",
    "cityblock",
]

raw_coords_cols = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
train_coords_raw = train[raw_coords_cols].copy()
test_coords_raw = test[raw_coords_cols].copy()

for df in (train, test):
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    df[features] = df[features].fillna(df[features].median(numeric_only=True))

scaler = StandardScaler()
scaler.fit(pd.concat([train[features], test[features]], axis=0, ignore_index=True))
train[features] = scaler.transform(train[features])
test[features] = scaler.transform(test[features])



## === cell 8
concat_raw = pd.concat([train_coords_raw, test_coords_raw], axis=0, ignore_index=True)

db = Birch(
    branching_factor=50, n_clusters=None, threshold=0.5, compute_labels=True
).fit(concat_raw)
labels = db.labels_

train["cluster"] = labels[: train.shape[0]]
test["cluster"] = labels[train.shape[0] :]

train["cluster"] = train["cluster"].astype(np.int32)
test["cluster"] = test["cluster"].astype(np.int32)



## === cell 9
pca_features = [
    "haversine",
    "chebyshev",
    "euclidean",
    "canberra",
    "braycurtis",
    "minkowski",
    "hamming",
    "cityblock",
]

for df in (train, test):
    df[pca_features] = df[pca_features].replace([np.inf, -np.inf], np.nan)
    df[pca_features] = df[pca_features].fillna(
        df[pca_features].median(numeric_only=True)
    )

pca = PCA(n_components=3, random_state=123)
pca.fit(pd.concat([train[pca_features], test[pca_features]], axis=0, ignore_index=True))

p_result = pca.transform(train[pca_features])
train["pca0"] = p_result[:, 0]
train["pca1"] = p_result[:, 1]
train["pca2"] = p_result[:, 2]

p_result = pca.transform(test[pca_features])
test["pca0"] = p_result[:, 0]
test["pca1"] = p_result[:, 1]
test["pca2"] = p_result[:, 2]



## === cell 10
colormap = plt.cm.RdBu
plt.figure(figsize=(20, 20))
plt.title("Pearson Correlation of Features", y=1.05, size=15)
sns.heatmap(
    train.select_dtypes(include=[np.number]).corr(),
    linewidths=0.1,
    vmax=1.0,
    square=True,
    cmap=colormap,
    linecolor="white",
    annot=False,
)
plt.show()



## === cell 11
good_dist = [
    "pca0",
    "pca1",
    "pca2",
    "cluster",
    "haversine",
    "chebyshev",
    "euclidean",
    "canberra",
    "braycurtis",
    "minkowski",
    "hamming",
    "cityblock",
]

time_cols = ["hour_of_day", "day", "week", "week_of_year", "month", "day_of_year"]

train_features_to_keep = (
    ["fare_amount", "pickup_datetime"]
    + ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    + good_dist
    + time_cols
)
train.drop(columns=train.columns.difference(train_features_to_keep), inplace=True)

test_features_to_keep = (
    ["key", "pickup_datetime"]
    + ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    + good_dist
    + time_cols
)
test.drop(columns=test.columns.difference(test_features_to_keep), inplace=True)

train_cluster_dummies = pd.get_dummies(
    train["cluster"], prefix="cluster", dtype=np.int8
)
test_cluster_dummies = pd.get_dummies(test["cluster"], prefix="cluster", dtype=np.int8)

train = pd.concat([train.drop(columns=["cluster"]), train_cluster_dummies], axis=1)
test = pd.concat([test.drop(columns=["cluster"]), test_cluster_dummies], axis=1)

dummy_cols = sorted(
    [c for c in train.columns if c.startswith("cluster_")]
    + [c for c in test.columns if c.startswith("cluster_")]
)
dummy_cols = sorted(list(set(dummy_cols)))

for c in dummy_cols:
    if c not in train.columns:
        train[c] = 0
    if c not in test.columns:
        test[c] = 0

train = train.reindex(
    columns=[c for c in train.columns if c not in dummy_cols] + dummy_cols
)
test = test.reindex(
    columns=[c for c in test.columns if c not in dummy_cols] + dummy_cols
)

x_pred = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 12
X_all = train.drop(["fare_amount", "pickup_datetime"], axis=1)
y_all = train["fare_amount"]

x_train, x_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=123
)




## === cell 13
def XGBmodel(x_train, x_test, y_train, y_test):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_test = xgb.DMatrix(x_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.1,
        "max_depth": 4,
        "min_child_weight": 3,
        "seed": 123,
    }

    model = xgb.train(
        params=params,
        dtrain=matrix_train,
        num_boost_round=1000,
        evals=[(matrix_test, "test")],
        verbose_eval=False,
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test)

print("Model trained with fixed num_boost_round=1000 (eta=0.1)")



## === cell 14
xgb.plot_importance(model, max_num_features=20)
plt.show()



## === cell 15
full_dtrain = xgb.DMatrix(X_all, label=y_all)

full_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.1,
    "max_depth": 4,
    "min_child_weight": 3,
    "seed": 123,
}

final_model = xgb.train(
    params=full_params,
    dtrain=full_dtrain,
    num_boost_round=1000,
    evals=[],
    verbose_eval=False,
)

dtest = xgb.DMatrix(x_pred)
prediction = final_model.predict(dtest)

prediction = np.clip(prediction, 0.0, None)

submission = pd.DataFrame({"key": test["key"], "fare_amount": prediction})
submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 16
submission
