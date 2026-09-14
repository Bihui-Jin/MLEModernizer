# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

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
train_with_time = pd.read_csv(
    "../input/train.csv", nrows=0
)  # lightweight placeholder to keep structure predictable
train_dt = pd.read_csv(
    "../input/train.csv", nrows=1000000, usecols=["key", "pickup_datetime"]
)
train_dt = train_dt.dropna()
train_dt["pickup_datetime"] = pd.to_datetime(
    train_dt["pickup_datetime"], errors="coerce"
)
train_dt = train_dt.dropna(subset=["pickup_datetime"])
train = train.merge(train_dt[["key", "pickup_datetime"]], on="key", how="left")
train = train.dropna(subset=["pickup_datetime"])
train = train.sort_values("pickup_datetime").reset_index(drop=True)

split_idx = int(train.shape[0] * 0.8)
x_train = train.iloc[:split_idx].drop(["fare_amount", "pickup_datetime"], axis=1)
y_train = train.iloc[:split_idx]["fare_amount"]
x_test = train.iloc[split_idx:].drop(["fare_amount", "pickup_datetime"], axis=1)
y_test = train.iloc[split_idx:]["fare_amount"]




## --- ERROR in cell 12, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1259244026.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     16[0m [0mtrain_dt[0m [0;34m=[0m [0mtrain_dt[0m[0;34m.[0m[0mdropna[0m[0;34m([0m[0msubset[0m[0;34m=[0m[0;34m[[0m[0;34m"pickup_datetime"[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     17[0m [0;31m# Align by key (train has been filtered; keep only remaining keys)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 18[0;31m [0mtrain[0m [0;34m=[0m [0mtrain[0m[0;34m.[0m[0mmerge[0m[0;34m([0m[0mtrain_dt[0m[0;34m[[0m[0;34m[[0m[0;34m"key"[0m[0;34m,[0m [0;34m"pickup_datetime"[0m[0;34m][0m[0;34m][0m[0;34m,[0m [0mon[0m[0;34m=[0m[0;34m"key"[0m[0;34m,[0m [0mhow[0m[0;34m=[0m[0;34m"left"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     19[0m [0mtrain[0m [0;34m=[0m [0mtrain[0m[0;34m.[0m[0mdropna[0m[0;34m([0m[0msubset[0m[0;34m=[0m[0;34m[[0m[0;34m"pickup_datetime"[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m [0mtrain[0m [0;34m=[0m [0mtrain[0m[0;34m.[0m[0msort_values[0m[0;34m([0m[0;34m"pickup_datetime"[0m[0;34m)[0m[0;34m.[0m[0mreset_index[0m[0;34m([0m[0mdrop[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36mmerge[0;34m(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)[0m
[1;32m  10830[0m         [0;32mfrom[0m [0mpandas[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mreshape[0m[0;34m.[0m[0mmerge[0m [0;32mimport[0m [0mmerge[0m[0;34m[0m[0;34m[0m[0m
[1;32m  10831[0m [0;34m[0m[0m
[0;32m> 10832[0;31m         return merge(
[0m[1;32m  10833[0m             [0mself[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m  10834[0m             [0mright[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py[0m in [0;36mmerge[0;34m(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)[0m
[1;32m    168[0m         )
[1;32m    169[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 170[0;31m         op = _MergeOperation(
[0m[1;32m    171[0m             [0mleft_df[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    172[0m             [0mright_df[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py[0m in [0;36m__init__[0;34m(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)[0m
[1;32m    792[0m             [0mleft_drop[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    793[0m             [0mright_drop[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 794[0;31m         ) = self._get_merge_keys()
[0m[1;32m    795[0m [0;34m[0m[0m
[1;32m    796[0m         [0;32mif[0m [0mleft_drop[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py[0m in [0;36m_get_merge_keys[0;34m(self)[0m
[1;32m   1308[0m                         [0;31m#  the latter of which will raise[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1309[0m                         [0mlk[0m [0;34m=[0m [0mcast[0m[0;34m([0m[0mHashable[0m[0;34m,[0m [0mlk[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1310[0;31m                         [0mleft_keys[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mleft[0m[0;34m.[0m[0m_get_label_or_level_values[0m[0;34m([0m[0mlk[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1311[0m                         [0mjoin_names[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mlk[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1312[0m                     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36m_get_label_or_level_values[0;34m(self, key, axis)[0m
[1;32m   1909[0m             [0mvalues[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0maxes[0m[0;34m[[0m[0maxis[0m[0;34m][0m[0;34m.[0m[0mget_level_values[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m.[0m[0m_values[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1910[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1911[0;31m             [0;32mraise[0m [0mKeyError[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1912[0m [0;34m[0m[0m
[1;32m   1913[0m         [0;31m# Check for duplicates[0m[0;34m[0m[0;34m[0m[0m

[0;31mKeyError[0m: 'key'

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
