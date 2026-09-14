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

3.86991

# 6. Current score

5.34962

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.54285) has done: 'Your script likely fails or produces a mis-scored submission because the engineered-distance step uses `pdist` per row with multiprocessing, which is extremely slow and can hang/timeout, and because `early_stopping_rounds` changes training semantics (and can destabilize results) for a fixed-round model. I keep the same overall logic (same features, same PCA/Birch usage, same XGBoost objective) but replace the per-row `pdist` calls with fully vectorized NumPy implementations of the same distance metrics, which makes the notebook finish reliably and yields a valid `sub_fare.csv`. I also remove early stopping while keeping `num_boost_round=300` so training matches the intended fixed training loop and avoids premature stopping variability. Finally, I ensure the submission rows align exactly to the original test `key` order even after any `dropna` operations.'
- What this solution (achieved 4.51336) has done: 'To move RMSE down toward the 3.86991 target (lower is better) with minimal logic changes, I (1) fix a subtle but important bug where you scale coordinates and then cluster on the scaled coordinates mixed with unscaled test-derived bounds—this makes clusters less meaningful and hurts generalization; we compute Birch clusters on the original (unscaled) lat/lon while keeping your scaler for model inputs. (2) Add the missing temporal features you already compute (hour/day/month/etc.) into the final feature set (they’re currently discarded), which is a small, legitimate improvement without changing the model/training loop. (3) Ensure train/test filtering bounds are computed from the same coordinate space and keep submission alignment unchanged.'
- What this solution (achieved 4.5262) has done: 'Your current gap is 4.51336 − 3.86991 = 0.64345 (worse than target; lower RMSE is better), so we need a small, legitimate boost without changing the model/training core. The biggest low-risk gain here is to add `passenger_count` back into the final feature set: you already clean/filter it but you drop it before training, which typically hurts NYC taxi fare RMSE materially. To keep semantics stable and avoid train/test mismatches, we also include `passenger_count` in the scaled feature block (so it’s transformed consistently) and keep the rest of your pipeline (distance features, PCA, Birch, XGBoost params/rounds) unchanged. This should move RMSE down toward the target band while staying within the “minimal changes” constraint.'
- What this solution (achieved 5.36335) has done: 'To move RMSE down toward the 3.86991 target (lower is better) with minimal disruption, I keep your same feature engineering + PCA/Birch + fixed-round XGBoost training loop, but remove one source of unnecessary noise: your train-set filtering bounds currently come only from the small test set, which can over-prune valid training examples and hurt generalization. I switch those geographic bounds to robust quantiles computed from the (already-loaded) 1M training sample, and apply the same bounds to test as a safety check, which typically improves RMSE without changing the core modeling approach. I also add a simple, metric-aligned post-process: predict in log-space is NOT allowed (core logic change), so instead I only clip extreme predictions to a reasonable upper cap derived from training fares to reduce outlier-driven RMSE. Submission writing and key alignment remain identical.'
- What this solution (achieved 5.34962) has done: 'Your current RMSE (5.36335) is worse than the target (3.86991), so we should make a small, legitimate improvement without changing the overall modeling approach. The biggest likely issue is that you fit the scaler on the combined set of raw + engineered features, which standardizes the lat/lon themselves and can distort the cluster/PCA features’ relationship to the target; we keep your engineered features and XGBoost training intact but stop scaling the raw lat/lon and passenger_count (scale only distance-like features). We also add the two existing cluster features (`cluster1`, `cluster2`) into the final model feature set (you already compute them but currently drop them), which is a minimal feature inclusion that often improves RMSE. Finally, we keep your submission alignment logic unchanged and still write `sub_fare.csv`.'

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
TRAIN_PATH_CANDIDATES = [
    "../input/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/train.csv",
]
TEST_PATH_CANDIDATES = [
    "../input/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/data/test.csv",
]


def _first_existing(paths):
    import os

    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


train_path = _first_existing(TRAIN_PATH_CANDIDATES)
test_path = _first_existing(TEST_PATH_CANDIDATES)

train = pd.read_csv(train_path, nrows=1000000)
test = pd.read_csv(test_path)

test_key_order = test[["key"]].copy()



## === cell 2
print("Sum of NaN values for each column")
print(train.isnull().sum())

train = train.dropna()
print("Sum of NaN values for each column after dropping NaN")
print(train.isnull().sum())




## === cell 3
def _robust_bounds(s, lo=0.001, hi=0.999):
    return float(s.quantile(lo)), float(s.quantile(hi))


pickup_longitude_min, pickup_longitude_max = _robust_bounds(train["pickup_longitude"])
pickup_latitude_min, pickup_latitude_max = _robust_bounds(train["pickup_latitude"])
dropoff_longitude_min, dropoff_longitude_max = _robust_bounds(
    train["dropoff_longitude"]
)
dropoff_latitude_min, dropoff_latitude_max = _robust_bounds(train["dropoff_latitude"])



## === cell 4
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] < 300)]
train = train.loc[
    (train["pickup_longitude"] > pickup_longitude_min)
    & (train["pickup_longitude"] < pickup_longitude_max)
]
train = train.loc[
    (train["pickup_latitude"] > pickup_latitude_min)
    & (train["pickup_latitude"] < pickup_latitude_max)
]
train = train.loc[
    (train["dropoff_longitude"] > dropoff_longitude_min)
    & (train["dropoff_longitude"] < dropoff_longitude_max)
]
train = train.loc[
    (train["dropoff_latitude"] > dropoff_latitude_min)
    & (train["dropoff_latitude"] < dropoff_latitude_max)
]
train = train.loc[train["passenger_count"] <= 8]

test = test.loc[
    (test["pickup_longitude"] > pickup_longitude_min)
    & (test["pickup_longitude"] < pickup_longitude_max)
    & (test["pickup_latitude"] > pickup_latitude_min)
    & (test["pickup_latitude"] < pickup_latitude_max)
    & (test["dropoff_longitude"] > dropoff_longitude_min)
    & (test["dropoff_longitude"] < dropoff_longitude_max)
    & (test["dropoff_latitude"] > dropoff_latitude_min)
    & (test["dropoff_latitude"] < dropoff_latitude_max)
    & (test["passenger_count"] <= 8)
].copy()

train.describe()




## === cell 5
def haversine_vec(lon1, lat1, lon2, lat2):
    """
    Vectorized great-circle distance in kilometers.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    aa = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(aa))
    return 6367.0 * c


def pairwise_dists_vec(df):
    """
    Compute distances between pickup and dropoff points using the same metrics as pdist on two points.
    """
    x1 = df["pickup_latitude"].to_numpy(dtype=np.float64)
    y1 = df["pickup_longitude"].to_numpy(dtype=np.float64)
    x2 = df["dropoff_latitude"].to_numpy(dtype=np.float64)
    y2 = df["dropoff_longitude"].to_numpy(dtype=np.float64)

    dx = x1 - x2
    dy = y1 - y2

    euclidean = np.sqrt(dx * dx + dy * dy)
    sqeuclidean = dx * dx + dy * dy
    chebyshev = np.maximum(np.abs(dx), np.abs(dy))
    cityblock = np.abs(dx) + np.abs(dy)

    denom_x = np.abs(x1) + np.abs(x2)
    denom_y = np.abs(y1) + np.abs(y2)
    canberra = np.where(denom_x == 0, 0.0, np.abs(dx) / denom_x) + np.where(
        denom_y == 0, 0.0, np.abs(dy) / denom_y
    )

    denom_bc = np.abs(x1) + np.abs(x2) + np.abs(y1) + np.abs(y2)
    braycurtis = np.where(denom_bc == 0, 0.0, (np.abs(dx) + np.abs(dy)) / denom_bc)

    minkowski = euclidean

    hamming = ((dx != 0).astype(np.float64) + (dy != 0).astype(np.float64)) / 2.0

    return {
        "chebyshev": chebyshev,
        "euclidean": euclidean,
        "canberra": canberra,
        "sqeuclidean": sqeuclidean,
        "braycurtis": braycurtis,
        "minkowski": minkowski,
        "hamming": hamming,
        "cityblock": cityblock,
    }




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

combine = [train, test]
for dataset in combine:
    dataset["haversine"] = haversine_vec(
        dataset["pickup_longitude"].to_numpy(dtype=np.float64),
        dataset["pickup_latitude"].to_numpy(dtype=np.float64),
        dataset["dropoff_longitude"].to_numpy(dtype=np.float64),
        dataset["dropoff_latitude"].to_numpy(dtype=np.float64),
    )

    dmap = pairwise_dists_vec(dataset)
    for dist_type in dist_types:
        dataset[dist_type] = dmap[dist_type]

    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], utc=True, errors="coerce"
    )
    iso_week = dataset["pickup_datetime"].dt.isocalendar().week.astype("int16")
    dataset["hour_of_day"] = dataset["pickup_datetime"].dt.hour.astype("int16")
    dataset["day"] = dataset["pickup_datetime"].dt.day.astype("int16")
    dataset["week"] = iso_week
    dataset["month"] = dataset["pickup_datetime"].dt.month.astype("int16")
    dataset["day_of_year"] = dataset["pickup_datetime"].dt.dayofyear.astype("int16")
    dataset["week_of_year"] = iso_week

train = train.dropna().reset_index(drop=True)
test = test.dropna().reset_index(drop=True)

train.head(3)



## === cell 7
coords = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
train_coords_raw = train[coords].copy()
test_coords_raw = test[coords].copy()

features = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "passenger_count",
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

missing_train = [c for c in features if c not in train.columns]
missing_test = [c for c in features if c not in test.columns]
if missing_train or missing_test:
    raise KeyError(
        f"Missing engineered features. train missing={missing_train}, test missing={missing_test}"
    )

scale_features = [
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
scaler = StandardScaler()
train[scale_features] = scaler.fit_transform(train[scale_features])
test[scale_features] = scaler.transform(test[scale_features])



## === cell 8
concat_raw = pd.concat([train_coords_raw, test_coords_raw], axis=0, ignore_index=True)

db = Birch(
    branching_factor=50, n_clusters=None, threshold=0.01, compute_labels=True
).fit(concat_raw)
labels = db.labels_
train["cluster"] = labels[: train.shape[0]]
test["cluster"] = labels[train.shape[0] :]

db = Birch(
    branching_factor=50, n_clusters=None, threshold=0.01, compute_labels=True
).fit(concat_raw[["pickup_latitude", "pickup_longitude"]])
labels = db.labels_
train["cluster1"] = labels[: train.shape[0]]
test["cluster1"] = labels[train.shape[0] :]

db = Birch(
    branching_factor=50, n_clusters=None, threshold=0.01, compute_labels=True
).fit(concat_raw[["dropoff_latitude", "dropoff_longitude"]])
labels = db.labels_
train["cluster2"] = labels[: train.shape[0]]
test["cluster2"] = labels[train.shape[0] :]



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

train = train.dropna(subset=pca_features).reset_index(drop=True)
test = test.dropna(subset=pca_features).reset_index(drop=True)

pca = PCA(n_components=3, random_state=123)
p_result = pca.fit_transform(train[pca_features].values)
for x in range(p_result.shape[1]):
    train["pca0" + str(x)] = p_result[:, x]

p_result = pca.transform(test[pca_features].values)
for x in range(p_result.shape[1]):
    test["pca0" + str(x)] = p_result[:, x]



## === cell 10
try:
    colormap = plt.cm.RdBu
    plt.figure(figsize=(12, 10))
    plt.title("Pearson Correlation of Features", y=1.05, size=15)
    numeric_corr = train.select_dtypes(include=[np.number]).corr()
    sns.heatmap(
        numeric_corr,
        linewidths=0.1,
        vmax=1.0,
        square=True,
        cmap=colormap,
        linecolor="white",
        annot=False,
    )
    plt.show()
except Exception as e:
    print("Skipping correlation plot due to:", repr(e))



## === cell 11
good_dist = ["pca00", "pca01", "pca02", "cluster", "cluster1", "cluster2"]
time_feats = ["hour_of_day", "day", "month", "day_of_year", "week_of_year"]

train_features_to_keep = (
    ["fare_amount"]
    + [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
        "passenger_count",
    ]
    + good_dist
    + time_feats
)
train.drop(columns=train.columns.difference(train_features_to_keep), inplace=True)

test_features_to_keep = (
    ["key"]
    + [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
        "passenger_count",
    ]
    + good_dist
    + time_feats
)
test.drop(columns=test.columns.difference(test_features_to_keep), inplace=True)

x_pred = test.drop("key", axis=1)



## === cell 12
x_train, x_test, y_train, y_test = train_test_split(
    train.drop("fare_amount", axis=1),
    train["fare_amount"],
    random_state=123,
    test_size=0.2,
)




## === cell 13
def XGBmodel(x_train, x_test, y_train, y_test):
    x_train = x_train.apply(pd.to_numeric, errors="coerce")
    x_test = x_test.apply(pd.to_numeric, errors="coerce")
    if x_train.isnull().any().any() or x_test.isnull().any().any():
        train_mask = ~x_train.isnull().any(axis=1)
        test_mask = ~x_test.isnull().any(axis=1)
        x_train2 = x_train.loc[train_mask]
        y_train2 = y_train.loc[train_mask]
        x_test2 = x_test.loc[test_mask]
        y_test2 = y_test.loc[test_mask]
    else:
        x_train2, y_train2, x_test2, y_test2 = x_train, y_train, x_test, y_test

    matrix_train = xgb.DMatrix(x_train2, label=y_train2)
    matrix_test = xgb.DMatrix(x_test2, label=y_test2)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.3,
        "max_depth": 4,
        "min_child_weight": 3,
        "seed": 123,
    }

    model = xgb.train(
        params=params,
        dtrain=matrix_train,
        num_boost_round=300,
        evals=[(matrix_test, "test")],
        verbose_eval=False,
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test)



## === cell 14
try:
    xgb.plot_importance(model, max_num_features=20)
    plt.show()
except Exception as e:
    print("Skipping importance plot due to:", repr(e))



## === cell 15
x_pred_num = x_pred.apply(pd.to_numeric, errors="coerce")
if x_pred_num.isnull().any().any():
    x_pred_num = x_pred_num.fillna(x_pred_num.median(numeric_only=True))

dtest = xgb.DMatrix(x_pred_num)
prediction = model.predict(dtest)

pred_df = pd.DataFrame({"key": test["key"].values, "fare_amount": prediction})
submission = test_key_order.merge(pred_df, on="key", how="left")

if submission["fare_amount"].isnull().any():
    submission["fare_amount"] = submission["fare_amount"].fillna(
        submission["fare_amount"].median()
    )

fare_cap = float(train["fare_amount"].quantile(0.999))
submission["fare_amount"] = np.clip(
    submission["fare_amount"].to_numpy(dtype=np.float64), 0.0, fare_cap
).round(2)

submission.to_csv("sub_fare.csv", index=False)
print("Wrote submission:", submission.shape, "to sub_fare.csv")



## === cell 16
submission.head()
