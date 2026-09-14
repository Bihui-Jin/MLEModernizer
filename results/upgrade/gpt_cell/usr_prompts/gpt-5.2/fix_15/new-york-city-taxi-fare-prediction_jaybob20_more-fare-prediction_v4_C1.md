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

6.15455

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.7501) has done: 'Diagnosis: Cell 17 crashes because with xgboost==2.0.3 the `Booster` returned by `xgb.train()` no longer exposes `best_ntree_limit` (this attribute existed in older versions and was tied to early stopping). The correct way to limit prediction to the best iteration is to use `iteration_range=(0, model.best_iteration + 1)` when `best_iteration` exists. We change only the prediction call in cell 17 to use `best_iteration` if available, otherwise fall back to a plain predict, preserving the same evaluation semantics from early stopping.

Patch summary: Replace the deprecated `ntree_limit=model.best_ntree_limit` argument with a version-compatible `iteration_range` based on `model.best_iteration` (if present), and keep the rest of the submission logic unchanged.

Updated cells:'
- What this solution (achieved 4.84432) has done: 'Your current gap to the target (4.7501 vs 3.86991, lower is better) is large, so we should make a small, legitimate improvement that doesn’t change your overall approach. The biggest issue is a bug in the geographic filtering bounds: `pickup_longitude_max` is mistakenly taken from `test.pickup_latitude.max()`, which incorrectly filters much of the training data and hurts generalization. I fix that single bound, and also replace the very expensive per-row `pdist` multiprocessing distance features with equivalent vectorized NumPy computations (same feature semantics, far less overhead), keeping the same feature set and XGBoost training loop. This should improve signal quality and stability without changing your model architecture or loss, and it still produce the same submission CSV.'
- What this solution (achieved 5.36626) has done: 'You’re currently far above the target RMSE (4.844 vs 3.869, lower is better), so we make the smallest changes that typically give a real lift without changing your overall model/feature approach. The main improvement is to compute the distance-based features on the original (unscaled) coordinates, then scale only after those features are created; your current code scales lat/lon before Birch clustering and keeps them scaled for the model, which tends to make the model learn on distorted coordinates. We also add a minimal, standard NYC bounding-box cleanup (train-only) to remove obvious outliers that otherwise inflate RMSE, while keeping your existing “test-bounds” filtering intact. Finally, we ensure the submission uses the sample_submission keys ordering to avoid any accidental alignment issues.'
- What this solution (achieved 5.56115) has done: 'Your RMSE is still far above the target (5.366 vs 3.869, lower is better), so we make the smallest changes that typically reduce error without changing your overall XGBoost approach. The main issue is you are scaling latitude/longitude and then feeding those scaled coordinates into Birch, but later you keep only the cluster ID and PCA distances—this makes the clusters much less geographically meaningful. We compute Birch clusters on raw coordinates (as you already prepared), and then also compute PCA on the same raw distance features (not the standardized versions) to keep those engineered features on a consistent, physical scale. Finally, we add a very small, standard cleanup to remove “zero-distance but non-trivial fare” label-noise rows, which commonly hurts RMSE, while keeping all other filtering and training logic unchanged.'
- What this solution (achieved 5.54469) has done: 'Your current RMSE (5.561) is substantially worse than the target (3.869, lower is better), so we should make a small but meaningful signal-quality improvement without changing the model or training loop. The highest-impact minimal fix here is to stop standardizing the raw latitude/longitude inputs: with tree models (XGBoost) scaling those coordinates usually hurts because split thresholds lose their intuitive geometry, while your distance/PCA/cluster features already provide scale-invariant structure. We keep your distance features, PCA, Birch clustering, and XGBoost parameters unchanged, but we only standardize the non-coordinate distance features and keep raw coords for the model. This should reduce error toward the target while preserving core logic and still producing the same valid submission CSV.'
- What this solution (achieved 11.59368) has done: 'Diagnosis: The crash happens in cell 15 when calling `model.predict(dmat_pred)` because XGBoost validates that the prediction `DMatrix` has the exact same feature names as were used during training. Here, the trained model expects engineered features (`haversine`, distance metrics, and `pca00/pca01/pca02`) but `x_pred` currently only contains the reduced set (`pickup/dropoff coords` + `cluster`) due to later feature-selection in cell 11. Since we must not change earlier cells, we fix cell 15 by constructing an inference matrix with the model’s expected feature names, reindexing from available columns and filling any missing expected columns with zeros to satisfy XGBoost’s strict feature validation.

Patch summary: In cell 15, build `x_pred_aligned` by using `model.feature_names` (or booster feature names) and reindex `x_pred` to match them; add missing columns (initialized to 0.0) and order columns identically. Then create `DMatrix` from this aligned frame and run prediction as before, keeping the submission creation logic unchanged.

Updated cells: cell 15 only (buggy cell).

Compatibility notes for cell k+1: There is no cell 16 provided; variables `prediction` and `submission` keep the same types/shapes as before, and `submission.to_csv(...)` behavior is unchanged.

Assumptions: Missing engineered features at inference time cannot be recomputed in this cell without violating the “no new logic” constraint, so we safely fill them with 0.0 solely to satisfy XGBoost’s feature name requirements and unblock execution.'
- What this solution (achieved 6.15455) has done: 'Your current RMSE (11.59) is far worse than the target (3.87, lower is better), and the main cause is that inference-time features don’t match training-time features: the model is trained on the time-based dataset (`train_with_time`) with engineered distance features, but predictions are made from `x_pred` built earlier from a different feature set and many missing columns are filled with zeros (which severely harms accuracy). With minimal changes and without altering the model/training loop, I rebuild the test feature matrix in the same way you build `x_train/x_test` in cell 12 (same scaler/PCA/Birch already fitted) and then predict from that aligned, correctly-engineered matrix. I also keep the submission key ordering exactly as `sample_submission.csv` to avoid any alignment mistakes. These changes should substantially reduce RMSE toward your target while preserving the core logic.'

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



## === cell 1
train = pd.read_csv("../input/train.csv", nrows=1000000)
test = pd.read_csv("../input/test.csv")

sample_sub = pd.read_csv("../input/sample_submission.csv")



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

train = train.loc[
    (train["pickup_longitude"].between(-74.3, -73.7))
    & (train["dropoff_longitude"].between(-74.3, -73.7))
    & (train["pickup_latitude"].between(40.5, 41.0))
    & (train["dropoff_latitude"].between(40.5, 41.0))
]

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
train = train.loc[train["passenger_count"].between(1, 8)]

zero_move = (np.abs(train["pickup_longitude"] - train["dropoff_longitude"]) < 1e-6) & (
    np.abs(train["pickup_latitude"] - train["dropoff_latitude"]) < 1e-6
)
train = train.loc[~(zero_move & (train["fare_amount"] > 5.0))]


def haversine_np_vec(lon1, lat1, lon2, lat2):
    """
    Vectorized haversine distance in km.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


_tmp_dist_km = haversine_np_vec(
    train["pickup_longitude"].astype(float).values,
    train["pickup_latitude"].astype(float).values,
    train["dropoff_longitude"].astype(float).values,
    train["dropoff_latitude"].astype(float).values,
)
_fare_per_km = train["fare_amount"].values / np.maximum(_tmp_dist_km, 0.2)
train = train.loc[(_fare_per_km < 200.0) & (_fare_per_km > 0.0)].copy()

train.describe()



## === cell 5
pass



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
    plon = dataset["pickup_longitude"].astype(float).values
    plat = dataset["pickup_latitude"].astype(float).values
    dlon = dataset["dropoff_longitude"].astype(float).values
    dlat = dataset["dropoff_latitude"].astype(float).values

    dx = dlon - plon
    dy = dlat - plat

    dataset["haversine"] = haversine_np_vec(plon, plat, dlon, dlat)

    abs_dx = np.abs(dx)
    abs_dy = np.abs(dy)

    if "cityblock" in dist_types:
        dataset["cityblock"] = abs_dx + abs_dy
    if "chebyshev" in dist_types:
        dataset["chebyshev"] = np.maximum(abs_dx, abs_dy)
    if "euclidean" in dist_types:
        dataset["euclidean"] = np.sqrt(dx * dx + dy * dy)
    if "sqeuclidean" in dist_types:
        dataset["sqeuclidean"] = dx * dx + dy * dy
    if "canberra" in dist_types:
        denom_x = np.abs(plon) + np.abs(dlon)
        denom_y = np.abs(plat) + np.abs(dlat)
        dataset["canberra"] = (abs_dx / np.where(denom_x == 0, 1.0, denom_x)) + (
            abs_dy / np.where(denom_y == 0, 1.0, denom_y)
        )
    if "braycurtis" in dist_types:
        denom_bc = np.abs(plon) + np.abs(dlon) + np.abs(plat) + np.abs(dlat)
        dataset["braycurtis"] = (abs_dx + abs_dy) / np.where(
            denom_bc == 0, 1.0, denom_bc
        )
    if "minkowski" in dist_types:
        dataset["minkowski"] = np.sqrt(dx * dx + dy * dy)
    if "hamming" in dist_types:
        diff1 = (plon != dlon).astype(float)
        diff2 = (plat != dlat).astype(float)
        dataset["hamming"] = (diff1 + diff2) / 2.0

    dataset["pickup_datetime"] = pd.to_datetime(dataset["pickup_datetime"])
    dataset["hour_of_day"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["week"] = dataset.pickup_datetime.dt.isocalendar().week.astype(int)
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["day_of_year"] = dataset.pickup_datetime.dt.dayofyear
    dataset["week_of_year"] = dataset.pickup_datetime.dt.isocalendar().week.astype(int)

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

coords = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]

train_coords_raw = train[coords].copy()
test_coords_raw = test[coords].copy()

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

scale_only = [
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
train[scale_only] = scaler.fit_transform(train[scale_only])
test[scale_only] = scaler.transform(test[scale_only])



## === cell 8
concat_raw = pd.concat([train_coords_raw, test_coords_raw], ignore_index=True)

db = Birch(
    branching_factor=50, n_clusters=None, threshold=0.5, compute_labels=True
).fit(concat_raw)
labels = db.labels_
train["cluster"] = labels[: train.shape[0]]
test["cluster"] = labels[train.shape[0] :]

db = Birch(
    branching_factor=50, n_clusters=None, threshold=0.5, compute_labels=True
).fit(concat_raw[["pickup_latitude", "pickup_longitude"]])
labels = db.labels_
train["cluster1"] = labels[: train.shape[0]]
test["cluster1"] = labels[train.shape[0] :]

db = Birch(
    branching_factor=50, n_clusters=None, threshold=0.5, compute_labels=True
).fit(concat_raw[["dropoff_latitude", "dropoff_longitude"]])
labels = db.labels_
train["cluster2"] = labels[: train.shape[0]]
test["cluster2"] = labels[train.shape[0] :]



## === cell 9
train_pca_X = train_pca_raw.apply(pd.to_numeric, errors="coerce").replace(
    [np.inf, -np.inf], np.nan
)
test_pca_X = test_pca_raw.apply(pd.to_numeric, errors="coerce").replace(
    [np.inf, -np.inf], np.nan
)

medians = train_pca_X.median()
train_pca_X = train_pca_X.fillna(medians)
test_pca_X = test_pca_X.fillna(medians)

pca = PCA(n_components=3)
p_result = pca.fit_transform(train_pca_X)
for x in range(p_result.shape[1]):
    train["pca0" + str(x)] = p_result[:, x]

p_result = pca.transform(test_pca_X)
for x in range(p_result.shape[1]):
    test["pca0" + str(x)] = p_result[:, x]



## === cell 10
colormap = plt.cm.RdBu
plt.figure(figsize=(20, 20))
plt.title("Pearson Correlation of Features", y=1.05, size=15)

corr_df = train.select_dtypes(include=[np.number]).corr()
sns.heatmap(
    corr_df,
    linewidths=0.1,
    vmax=1.0,
    square=True,
    cmap=colormap,
    linecolor="white",
    annot=True,
)



## === cell 11
good_dist = [
    "pca0",
    "pca1",
    "pca2",
    "cluster",
]
train_features_to_keep = (
    ["fare_amount"]
    + ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    + good_dist
)
train.drop(train.columns.difference(train_features_to_keep), axis=1, inplace=True)

test_features_to_keep = (
    ["key"]
    + ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    + good_dist
)
test.drop(test.columns.difference(test_features_to_keep), axis=1, inplace=True)

x_pred = test.drop("key", axis=1)



## === cell 12
usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]

train_with_time = pd.read_csv("../input/train.csv", nrows=1000000, usecols=usecols)
train_with_time = train_with_time.dropna()

train_with_time["pickup_datetime"] = pd.to_datetime(
    train_with_time["pickup_datetime"], errors="coerce"
)
train_with_time = train_with_time.dropna(subset=["pickup_datetime"])

train_with_time = train_with_time.loc[
    (train_with_time["fare_amount"] > 0) & (train_with_time["fare_amount"] < 300)
]
train_with_time = train_with_time.loc[
    (train_with_time["pickup_longitude"].between(-74.3, -73.7))
    & (train_with_time["dropoff_longitude"].between(-74.3, -73.7))
    & (train_with_time["pickup_latitude"].between(40.5, 41.0))
    & (train_with_time["dropoff_latitude"].between(40.5, 41.0))
].copy()

plon = train_with_time["pickup_longitude"].astype(float).values
plat = train_with_time["pickup_latitude"].astype(float).values
dlon = train_with_time["dropoff_longitude"].astype(float).values
dlat = train_with_time["dropoff_latitude"].astype(float).values

dx = dlon - plon
dy = dlat - plat
abs_dx = np.abs(dx)
abs_dy = np.abs(dy)

train_with_time["haversine"] = haversine_np_vec(plon, plat, dlon, dlat)
train_with_time["cityblock"] = abs_dx + abs_dy
train_with_time["chebyshev"] = np.maximum(abs_dx, abs_dy)
train_with_time["euclidean"] = np.sqrt(dx * dx + dy * dy)
train_with_time["sqeuclidean"] = dx * dx + dy * dy

denom_x = np.abs(plon) + np.abs(dlon)
denom_y = np.abs(plat) + np.abs(dlat)
train_with_time["canberra"] = (abs_dx / np.where(denom_x == 0, 1.0, denom_x)) + (
    abs_dy / np.where(denom_y == 0, 1.0, denom_y)
)

denom_bc = np.abs(plon) + np.abs(dlon) + np.abs(plat) + np.abs(dlat)
train_with_time["braycurtis"] = (abs_dx + abs_dy) / np.where(
    denom_bc == 0, 1.0, denom_bc
)

train_with_time["minkowski"] = np.sqrt(dx * dx + dy * dy)

diff1 = (plon != dlon).astype(float)
diff2 = (plat != dlat).astype(float)
train_with_time["hamming"] = (diff1 + diff2) / 2.0

scale_only = [
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
train_with_time[scale_only] = scaler.transform(train_with_time[scale_only])

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
train_pca_X = (
    train_with_time[pca_features]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
)
try:
    _med = medians
except NameError:
    _med = train_pca_X.median()
train_pca_X = train_pca_X.fillna(_med)

p_result = pca.transform(train_pca_X)
for x in range(p_result.shape[1]):
    train_with_time["pca0" + str(x)] = p_result[:, x]

concat_raw = pd.concat(
    [
        train_with_time[
            [
                "pickup_latitude",
                "pickup_longitude",
                "dropoff_latitude",
                "dropoff_longitude",
            ]
        ],
        test_coords_raw,
    ],
    ignore_index=True,
)
db = Birch(
    branching_factor=50, n_clusters=None, threshold=0.5, compute_labels=True
).fit(concat_raw)
labels = db.labels_
train_with_time["cluster"] = labels[: train_with_time.shape[0]]

train = train_with_time.sort_values("pickup_datetime").reset_index(drop=True)

split_idx = int(train.shape[0] * 0.8)
x_train = train.iloc[:split_idx].drop(["fare_amount", "pickup_datetime"], axis=1)
y_train = train.iloc[:split_idx]["fare_amount"]
x_test = train.iloc[split_idx:].drop(["fare_amount", "pickup_datetime"], axis=1)
y_test = train.iloc[split_idx:]["fare_amount"]




## === cell 13
def XGBmodel(x_train, x_test, y_train, y_test):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_test = xgb.DMatrix(x_test, label=y_test)
    model = xgb.train(
        params={
            "objective": "reg:linear",
            "eval_metric": "rmse",
            "eta": 0.3,
            "max_depth": 4,
            "min_child_weight": 3,
        },
        dtrain=matrix_train,
        num_boost_round=300,
        early_stopping_rounds=10,
        evals=[(matrix_test, "test")],
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test)



## === cell 14
xgb.plot_importance(model)
plt.show()



## === cell 15
test_feat = pd.read_csv("../input/test.csv")
test_feat = test_feat.dropna().copy()
test_feat["pickup_datetime"] = pd.to_datetime(
    test_feat["pickup_datetime"], errors="coerce"
)
test_feat = test_feat.dropna(subset=["pickup_datetime"]).copy()

plon = test_feat["pickup_longitude"].astype(float).values
plat = test_feat["pickup_latitude"].astype(float).values
dlon = test_feat["dropoff_longitude"].astype(float).values
dlat = test_feat["dropoff_latitude"].astype(float).values

dx = dlon - plon
dy = dlat - plat
abs_dx = np.abs(dx)
abs_dy = np.abs(dy)

test_feat["haversine"] = haversine_np_vec(plon, plat, dlon, dlat)
test_feat["cityblock"] = abs_dx + abs_dy
test_feat["chebyshev"] = np.maximum(abs_dx, abs_dy)
test_feat["euclidean"] = np.sqrt(dx * dx + dy * dy)
test_feat["sqeuclidean"] = dx * dx + dy * dy

denom_x = np.abs(plon) + np.abs(dlon)
denom_y = np.abs(plat) + np.abs(dlat)
test_feat["canberra"] = (abs_dx / np.where(denom_x == 0, 1.0, denom_x)) + (
    abs_dy / np.where(denom_y == 0, 1.0, denom_y)
)

denom_bc = np.abs(plon) + np.abs(dlon) + np.abs(plat) + np.abs(dlat)
test_feat["braycurtis"] = (abs_dx + abs_dy) / np.where(denom_bc == 0, 1.0, denom_bc)

test_feat["minkowski"] = np.sqrt(dx * dx + dy * dy)
diff1 = (plon != dlon).astype(float)
diff2 = (plat != dlat).astype(float)
test_feat["hamming"] = (diff1 + diff2) / 2.0

test_feat[scale_only] = scaler.transform(test_feat[scale_only])

test_pca_X2 = (
    test_feat[pca_features]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
)
test_pca_X2 = test_pca_X2.fillna(medians)

p_result = pca.transform(test_pca_X2)
for x in range(p_result.shape[1]):
    test_feat["pca0" + str(x)] = p_result[:, x]

concat_raw2 = pd.concat(
    [
        train_with_time[
            [
                "pickup_latitude",
                "pickup_longitude",
                "dropoff_latitude",
                "dropoff_longitude",
            ]
        ],
        test_feat[
            [
                "pickup_latitude",
                "pickup_longitude",
                "dropoff_latitude",
                "dropoff_longitude",
            ]
        ],
    ],
    ignore_index=True,
)
db2 = Birch(
    branching_factor=50, n_clusters=None, threshold=0.5, compute_labels=True
).fit(concat_raw2)
labels2 = db2.labels_
test_feat["cluster"] = labels2[train_with_time.shape[0] :]

expected_features = getattr(model, "feature_names", None)
if expected_features is None:
    expected_features = model.get_booster().feature_names

x_pred_aligned = test_feat.reindex(columns=expected_features)

dmat_pred = xgb.DMatrix(x_pred_aligned)
if hasattr(model, "best_iteration") and model.best_iteration is not None:
    prediction = model.predict(dmat_pred, iteration_range=(0, model.best_iteration + 1))
else:
    prediction = model.predict(dmat_pred)

pred_df = pd.DataFrame({"key": test_feat["key"].values, "fare_amount": prediction})
submission = sample_sub[["key"]].merge(pred_df, on="key", how="left")
submission["fare_amount"] = submission["fare_amount"].round(2)
submission.to_csv("sub_fare.csv", index=False)

submission
