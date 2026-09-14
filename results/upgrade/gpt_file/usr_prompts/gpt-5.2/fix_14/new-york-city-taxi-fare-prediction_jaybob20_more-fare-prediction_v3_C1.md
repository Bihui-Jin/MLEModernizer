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

3.6371

# 6. Current score

6.29698

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.47603) has done: 'I fix the datetime feature extraction to work with current pandas (replace deprecated `.dt.week`/`.dt.weekofyear`), correct a latitude/longitude max bug in the filtering bounds, and ensure datetime parsing is applied to each dataset (train/test) instead of mistakenly using `test` for both. I also make the distance-feature generation robust (no leaked pool state, no NaNs from distance calculations), keep scaling consistent, and adjust `drop()` calls to the current pandas API so columns are removed correctly. Finally, I ensure only numeric features (no `key`/`pickup_datetime`) reach XGBoost and update the XGBoost objective to the supported `reg:squarederror`, producing a valid `sub_fare.csv` submission.'
- What this solution (achieved 4.53031) has done: 'Your current RMSE (4.476) is worse than the target (3.6371), so we should make small, metric-aligned improvements that don’t change the overall pipeline. The biggest gain with minimal disruption is to stop rounding predictions to cents before scoring; rounding almost always increases RMSE on this competition. Next, keep your existing early-stopping logic for choosing the number of trees, but retrain once on the full filtered training data using that best number of boosting rounds (same model family/params) so the final model is stronger without changing the approach. These two changes are small, legitimate, and typically move RMSE noticeably toward your target while preserving your core feature engineering and XGBoost setup.'
- What this solution (achieved 4.45607) has done: 'Your current RMSE (4.53031) is worse than the target (3.6371), so we should make the smallest “legal” improvements without changing the overall XGBoost approach. The biggest score drag in your pipeline is that the time features you create (hour/day/month/week) are never used by the model; we keep the same feature-extraction logic but include those already-created columns in the final feature set. In addition, your scaler currently fits only on the filtered train (after using test bounds), which can create a mild train/test mismatch; we keep StandardScaler but fit it on the combined train+test for those distance/coord features to align distributions (this is a common, minimal improvement for this competition and doesn’t change the model family). Everything else (data filtering, PCA, Birch clustering, XGBoost params, early stopping + refit, submission writing) stays the same.'
- What this solution (achieved 4.52098) has done: 'We’re currently worse than the target (4.45607 vs 3.6371 RMSE, lower is better), so we should make the smallest changes that typically improve RMSE without changing the overall XGBoost approach. The main issue is that you scale the coordinate columns and then cluster on those already-scaled columns using a threshold tuned for raw degrees; this can make the `cluster` feature noisy and hurt performance. I keep the same Birch+cluster feature, but fit Birch on the original (unscaled) coordinates while still keeping StandardScaler for the model features. I also add `subsample` and `colsample_bytree` (standard, minimal regularization) to reduce overfitting and usually improve generalization RMSE in this competition.'
- What this solution (achieved 4.55157) has done: 'Your RMSE (4.52098) is worse than the target (3.6371), so we should make a small, reliable improvement that keeps your XGBoost approach and feature engineering intact. The biggest likely score drag is that `cluster` is treated as a numeric continuous feature; making it categorical inside XGBoost (via one-hot encoding) typically improves generalization in this competition without changing the modeling family or training loop. I one-hot encode the `cluster` column for both train/test with aligned columns, then train/predict exactly as before. This is a minimal change, preserves your pipeline, and should move RMSE downward toward the target band.'
- What this solution (achieved 6.29018) has done: 'Your current RMSE (4.55157) is worse than the target (3.6371), so we should make small changes that reliably improve generalization without changing your overall XGBoost + engineered-features pipeline. The single biggest issue is that you’re filtering train coordinate bounds using the *exact* test min/max, which discards many valid training trips and hurts fit; we replace that with a standard NYC bounding box + simple outlier filters (still the same filtering step). Next, we ensure time features can’t be NaN for test by filling invalid datetimes (rather than letting NaNs propagate into XGBoost). Finally, we add a minimal, common-sense engineered feature (straight-line degree deltas and Manhattan-approx distance) which is consistent with your existing distance-feature approach and usually reduces RMSE toward your target.'
- What this solution (achieved 6.31651) has done: 'Your RMSE (6.29018) is far worse than the target (3.6371), so we should make a small change that fixes the most likely regression without changing the overall XGBoost + engineered-features pipeline. The biggest issue is that your `PCA` features are computed on the *scaled* distance columns, but then later you drop those scaled distance columns and keep only `pca0/1/2`; this makes PCA represent the wrong space (and can be much noisier), hurting RMSE. I keep the same features and model, but compute PCA from the original (unscaled) distance features and then keep the rest of your pipeline identical. This is a minimal, metric-aligned correction that typically improves generalization in this competition and should move RMSE down toward the target band.'
- What this solution (achieved 6.31651) has done: 'Your current RMSE (6.31651) is much worse than the target (3.6371), so we should make the smallest changes that typically recover a big chunk of performance without changing your overall XGBoost + engineered-features pipeline. The main regression is that you scale core numeric features in cell 6, but then later (cell 10) you rebuild the modeling table using the now-unscaled coordinate columns while dropping the scaled distance features; this creates a train/test mismatch and discards useful scaled distance signal. I keep your exact feature engineering, clustering, PCA, and XGBoost training approach, but ensure the final training/test tables use the already-scaled versions of the coordinate and delta features (and keep the PCA features as-is). I also explicitly carry the `key` column through unchanged to guarantee correct submission alignment.'
- What this solution (achieved 6.29018) has done: 'Your current RMSE (6.31651) is much worse than the target (3.6371), so we should make a minimal, high-impact fix that doesn’t change your overall XGBoost pipeline. The biggest regression is that your PCA features are computed from the *raw* (unscaled) distance columns, while the rest of your model features are standardized; this makes PCA features dominate/misalign in scale and can badly hurt generalization. We keep the same PCA(3) idea and same distance inputs, but fit/transform PCA on the already-scaled versions of those distance columns so all numeric features live on a comparable scale. Everything else (filters, distance features, Birch clustering on raw coords, XGBoost params/training loop, submission writing) stays the same.'
- What this solution (achieved 6.29698) has done: 'Your current RMSE (6.29018) is much worse than the target (3.6371), so we should make one minimal, high-impact correction that preserves your exact feature engineering and XGBoost setup. The main regression is that your PCA features are built from scaled distances, but you never include any of those underlying scaled distance features in the final model table—so the model only sees PCA + coords + deltas + time, which is typically too weak for this competition. We keep PCA exactly as-is, but also keep the already-scaled distance columns (`haversine`, `euclidean`, etc.) in the final feature set (no new features, no model change), which usually drops RMSE substantially. Everything else (filters, clustering, scaling, training loop/early stopping + refit, and submission writing) remains unchanged.'
- What this solution (achieved 6.29698) has done: 'Your RMSE is much worse than the target, so we make one small, high-impact correction that keeps your exact feature engineering and XGBoost training logic intact: ensure `pickup_datetime`-derived time features never contain NaNs (currently test rows with invalid/NaT datetimes become NaN and then get dropped from `x_pred` implicitly by XGBoost handling/column dtype quirks). We explicitly fill all time feature NaNs to 0 after extraction (for both train and test) and ensure they are numeric integer types before modeling. This is a minimal semantic fix (no new features, no model change) that usually improves generalization RMSE in this competition by stabilizing the time-signal. Everything else (filters, distances, scaling, Birch+PCA, early stopping + refit, submission format) stays the same.'
- What this solution (achieved 6.29698) has done: 'Your current RMSE (6.29698) is far worse than the target (3.6371), so we should make a small, high-impact fix without changing the overall XGBoost pipeline. The main issue is that in cell 5 you compute several `pdist`-based distances on raw latitude/longitude pairs, but `pdist` expects 2D points; with your current `reshape(2,2)` you’re accidentally treating (lat,lon) as the two points (pickup and dropoff), which is incorrect and makes those distance features mostly meaningless. I minimally correct the distance computation to use the intended points `[(pickup_lat, pickup_lon), (dropoff_lat, dropoff_lon)]` while keeping the same feature set, scaling, PCA, clustering, and training procedure. This should materially reduce RMSE (move toward the target) while preserving your core logic and still producing the same submission format.'
- What this solution (achieved 6.29698) has done: 'Your current RMSE (6.29698) is far above the target (3.6371), so we need a small but high-impact correction while keeping your same feature engineering + XGBoost training loop. The biggest issue is that in cell 10 you re-include the *raw/unscaled* distance features (`haversine`, `euclidean`, etc.) after you already standardized them in cell 6, which undoes the scaling and creates a train/test feature-space mismatch that can severely hurt RMSE. I change cell 10 to keep the already-scaled distance columns (same column names, since you overwrote them in-place) and remove only the redundant “raw_dist_feats” naming confusion—no new features, no model change. Everything else (filters, datetime handling, Birch clustering, PCA, early stopping + refit, and submission writing) remains the same.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import datetime as dt

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
import xgboost as xgb

np.random.seed(123)



## === cell 1
train = pd.read_csv("../input/train.csv", nrows=1000000)
test = pd.read_csv("../input/test.csv")



## === cell 2
print("Sum of NaN values for each column")
print(train.isnull().sum())

train = train.dropna()
print("Sum of NaN values for each column after dropping NaN")
print(train.isnull().sum())



## === cell 3
NYC_BOUNDS = {
    "pickup_longitude": (-74.3, -72.9),
    "dropoff_longitude": (-74.3, -72.9),
    "pickup_latitude": (40.5, 41.8),
    "dropoff_latitude": (40.5, 41.8),
}

train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] < 300)]
train = train.loc[(train["passenger_count"] > 0) & (train["passenger_count"] <= 8)]

for col, (lo, hi) in NYC_BOUNDS.items():
    train = train.loc[(train[col] >= lo) & (train[col] <= hi)]

train.describe()




## === cell 4
def haversine_np(a):
    """
    Great-circle distance between pickup and dropoff in km.
    a = [pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude]
    """
    lat1, lon1, lat2, lon2 = a
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    aa = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(aa))
    km = 6367 * c
    return km


def cityblock(a, dist="cityblock"):
    a = np.asarray(a, dtype=np.float64)
    pts = np.array([[a[0], a[1]], [a[2], a[3]]], dtype=np.float64)
    return pdist(pts, dist)[0]




## === cell 5
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
    coords = (
        dataset[
            [
                "pickup_latitude",
                "pickup_longitude",
                "dropoff_latitude",
                "dropoff_longitude",
            ]
        ]
        .astype(np.float64)
        .values
    )

    dataset["haversine"] = np.apply_along_axis(haversine_np, 1, coords)

    for dist_type in dist_types:
        dataset[dist_type] = np.apply_along_axis(
            lambda x: cityblock(x, dist=dist_type), 1, coords
        )

    dataset["abs_dlon"] = (
        dataset["dropoff_longitude"] - dataset["pickup_longitude"]
    ).abs()
    dataset["abs_dlat"] = (
        dataset["dropoff_latitude"] - dataset["pickup_latitude"]
    ).abs()
    dataset["manhattan_deg"] = dataset["abs_dlon"] + dataset["abs_dlat"]

    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], errors="coerce"
    )
    try:
        if getattr(dataset["pickup_datetime"].dt, "tz", None) is not None:
            dataset["pickup_datetime"] = dataset["pickup_datetime"].dt.tz_convert(None)
    except Exception:
        pass

    dataset["hour_of_day"] = dataset["pickup_datetime"].dt.hour
    dataset["day"] = dataset["pickup_datetime"].dt.day
    dataset["month"] = dataset["pickup_datetime"].dt.month
    dataset["day_of_year"] = dataset["pickup_datetime"].dt.dayofyear
    dataset["week_of_year"] = (
        dataset["pickup_datetime"].dt.isocalendar().week.astype("int16")
    )

train = train.dropna(subset=["pickup_datetime"])

time_feats = ["hour_of_day", "day", "month", "day_of_year", "week_of_year"]
for dataset in [train, test]:
    for c in (
        ["haversine"]
        + dist_types
        + ["abs_dlon", "abs_dlat", "manhattan_deg"]
        + time_feats
    ):
        if c in dataset.columns:
            dataset[c] = dataset[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

for dataset in [train, test]:
    for c in time_feats:
        if c in dataset.columns:
            dataset[c] = dataset[c].astype(np.int16)

train.head(3)



## === cell 6
coords_cols = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
train_coords_raw = train[coords_cols].copy()
test_coords_raw = test[coords_cols].copy()

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
    "abs_dlon",
    "abs_dlat",
    "manhattan_deg",
]

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

train_pca_raw = train[pca_features].copy()
test_pca_raw = test[pca_features].copy()
train_pca_raw = train_pca_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
test_pca_raw = test_pca_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

scaler = StandardScaler()
both = pd.concat([train[features], test[features]], axis=0, ignore_index=True)
scaler.fit(both)

train[features] = scaler.transform(train[features])
test[features] = scaler.transform(test[features])

train_pca_scaled = train[pca_features].copy()
test_pca_scaled = test[pca_features].copy()



## === cell 7
concat_raw = pd.concat([train_coords_raw, test_coords_raw], axis=0, ignore_index=True)

db = Birch(
    branching_factor=50, n_clusters=None, threshold=0.5, compute_labels=True
).fit(concat_raw)
labels = db.labels_

train["cluster"] = labels[: train.shape[0]]
test["cluster"] = labels[train.shape[0] :]



## === cell 8
pca = PCA(n_components=3, random_state=123)

p_result = pca.fit_transform(train_pca_scaled.values)
train["pca0"] = p_result[:, 0]
train["pca1"] = p_result[:, 1]
train["pca2"] = p_result[:, 2]

p_result = pca.transform(test_pca_scaled.values)
test["pca0"] = p_result[:, 0]
test["pca1"] = p_result[:, 1]
test["pca2"] = p_result[:, 2]



## === cell 9
colormap = plt.cm.RdBu
plt.figure(figsize=(20, 20))
plt.title("Pearson Correlation of Numeric Features", y=1.05, size=15)
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



## === cell 10
good_dist = ["pca0", "pca1", "pca2", "cluster"]
time_feats = ["hour_of_day", "day", "month", "day_of_year", "week_of_year"]
delta_feats = ["abs_dlon", "abs_dlat", "manhattan_deg"]

dist_feats = ["haversine"] + dist_types

train_features_to_keep = (
    ["fare_amount"]
    + ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    + dist_feats
    + good_dist
    + time_feats
    + delta_feats
)
train.drop(columns=train.columns.difference(train_features_to_keep), inplace=True)

test_features_to_keep = (
    ["key"]
    + ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    + dist_feats
    + good_dist
    + time_feats
    + delta_feats
)
test.drop(columns=test.columns.difference(test_features_to_keep), inplace=True)

for c in (
    time_feats
    + good_dist
    + delta_feats
    + dist_feats
    + ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
):
    if c in train.columns:
        train[c] = train[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if c in test.columns:
        test[c] = test[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

for dataset in [train, test]:
    for c in time_feats:
        if c in dataset.columns:
            dataset[c] = dataset[c].astype(np.int16)

train["cluster"] = train["cluster"].astype("int32")
test["cluster"] = test["cluster"].astype("int32")

train_cluster_ohe = pd.get_dummies(train["cluster"], prefix="cluster", dtype=np.int8)
test_cluster_ohe = pd.get_dummies(test["cluster"], prefix="cluster", dtype=np.int8)

all_cluster_cols = sorted(
    set(train_cluster_ohe.columns).union(set(test_cluster_ohe.columns))
)
train_cluster_ohe = train_cluster_ohe.reindex(columns=all_cluster_cols, fill_value=0)
test_cluster_ohe = test_cluster_ohe.reindex(columns=all_cluster_cols, fill_value=0)

train = pd.concat([train.drop(columns=["cluster"]), train_cluster_ohe], axis=1)
test = pd.concat([test.drop(columns=["cluster"]), test_cluster_ohe], axis=1)

x_pred = test.drop("key", axis=1)



## === cell 11
x_train, x_test, y_train, y_test = train_test_split(
    train.drop("fare_amount", axis=1),
    train["fare_amount"],
    random_state=123,
    test_size=0.2,
)




## === cell 12
def XGBmodel(x_train, x_test, y_train, y_test):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_test = xgb.DMatrix(x_test, label=y_test)
    model = xgb.train(
        params={
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": 0.3,
            "max_depth": 4,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 123,
        },
        dtrain=matrix_train,
        num_boost_round=300,
        early_stopping_rounds=10,
        evals=[(matrix_test, "test")],
        verbose_eval=False,
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test)



## === cell 13
xgb.plot_importance(model, max_num_features=20)
plt.show()



## === cell 14
best_iter = getattr(model, "best_iteration", None)
best_rounds = (best_iter + 1) if best_iter is not None else 300

dall = xgb.DMatrix(train.drop("fare_amount", axis=1), label=train["fare_amount"])
final_model = xgb.train(
    params={
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.3,
        "max_depth": 4,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 123,
    },
    dtrain=dall,
    num_boost_round=best_rounds,
    evals=[(dall, "train")],
    verbose_eval=False,
)



## === cell 15
dtest = xgb.DMatrix(x_pred)

prediction = final_model.predict(dtest)

submission = pd.DataFrame({"key": test["key"], "fare_amount": prediction})
submission.to_csv("sub_fare.csv", index=False)
print("Wrote submission:", submission.shape, "-> sub_fare.csv")
submission.head()



## === cell 16
submission
