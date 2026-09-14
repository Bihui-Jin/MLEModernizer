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

from scipy.spatial.distance import pdist, cdist
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
import xgboost as xgb




## === cell 1
def _read_csv_from_known_paths(filename, **kwargs):
    candidates = [
        f"/kaggle/input/{filename}",
        f"/kaggle/input/new-york-city-taxi-fare-prediction/{filename}",
        f"/kaggle/data/{filename}",
        f"/kaggle/data/new-york-city-taxi-fare-prediction/{filename}",
        f"/kaggle/data/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/{filename}",
    ]
    last_err = None
    for p in candidates:
        try:
            return pd.read_csv(p, **kwargs)
        except Exception as e:
            last_err = e
    raise FileNotFoundError(
        f"Could not read {filename} from candidates={candidates}. Last error: {last_err}"
    )


train = _read_csv_from_known_paths("train.csv", nrows=5000000)
test = _read_csv_from_known_paths("test.csv")



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

LON_MARGIN = 0.01
LAT_MARGIN = 0.01
pickup_longitude_min -= LON_MARGIN
pickup_longitude_max += LON_MARGIN
dropoff_longitude_min -= LON_MARGIN
dropoff_longitude_max += LON_MARGIN
pickup_latitude_min -= LAT_MARGIN
pickup_latitude_max += LAT_MARGIN
dropoff_latitude_min -= LAT_MARGIN
dropoff_latitude_max += LAT_MARGIN



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

train = train.loc[(train["passenger_count"] >= 1) & (train["passenger_count"] <= 8)]

train = train.loc[
    (train["pickup_longitude"].between(-75.0, -72.0))
    & (train["dropoff_longitude"].between(-75.0, -72.0))
    & (train["pickup_latitude"].between(40.0, 42.0))
    & (train["dropoff_latitude"].between(40.0, 42.0))
]

train.describe()




## === cell 5
def haversine_np(a):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.
    """
    lon1, lat1, lon2, lat2 = a
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def cityblock(a, dist="cityblock"):
    return pdist(a.reshape(2, 2), dist)[0]




## === cell 6
dist_types = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "kulsinski",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]
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


def add_distance_features(dataset: pd.DataFrame, dist_types):
    coords = dataset[
        ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    ].to_numpy(dtype=np.float64)

    lat1 = coords[:, 0]
    lon1 = coords[:, 1]
    lat2 = coords[:, 2]
    lon2 = coords[:, 3]

    rlon1, rlat1, rlon2, rlat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = rlon2 - rlon1
    dlat = rlat2 - rlat1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    dataset["haversine"] = 6367 * c

    A = np.column_stack([lat1, lon1])
    B = np.column_stack([lat2, lon2])

    for dist_type in dist_types:
        print(dist_type)
        dataset[dist_type] = cdist(A, B, metric=dist_type).diagonal()

    dataset["pickup_datetime"] = pd.to_datetime(dataset["pickup_datetime"])
    dataset["hour_of_day"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["week"] = dataset.pickup_datetime.dt.isocalendar().week.astype(int)
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["day_of_year"] = dataset.pickup_datetime.dt.dayofyear
    dataset["week_of_year"] = dataset.pickup_datetime.dt.isocalendar().week.astype(int)


combine = [train, test]
for dataset in combine:
    add_distance_features(dataset, dist_types)

train.head(3)



## --- ERROR in cell 6, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mMemoryError[0m                               Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3983201188.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     71[0m [0mcombine[0m [0;34m=[0m [0;34m[[0m[0mtrain[0m[0;34m,[0m [0mtest[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m     72[0m [0;32mfor[0m [0mdataset[0m [0;32min[0m [0mcombine[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 73[0;31m     [0madd_distance_features[0m[0;34m([0m[0mdataset[0m[0;34m,[0m [0mdist_types[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     74[0m [0;34m[0m[0m
[1;32m     75[0m [0mtrain[0m[0;34m.[0m[0mhead[0m[0;34m([0m[0;36m3[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3983201188.py[0m in [0;36madd_distance_features[0;34m(dataset, dist_types)[0m
[1;32m     58[0m     [0;32mfor[0m [0mdist_type[0m [0;32min[0m [0mdist_types[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     59[0m         [0mprint[0m[0;34m([0m[0mdist_type[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 60[0;31m         [0mdataset[0m[0;34m[[0m[0mdist_type[0m[0;34m][0m [0;34m=[0m [0mcdist[0m[0;34m([0m[0mA[0m[0;34m,[0m [0mB[0m[0;34m,[0m [0mmetric[0m[0;34m=[0m[0mdist_type[0m[0;34m)[0m[0;34m.[0m[0mdiagonal[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     61[0m [0;34m[0m[0m
[1;32m     62[0m     [0mdataset[0m[0;34m[[0m[0;34m"pickup_datetime"[0m[0;34m][0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mto_datetime[0m[0;34m([0m[0mdataset[0m[0;34m[[0m[0;34m"pickup_datetime"[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/spatial/distance.py[0m in [0;36mcdist[0;34m(XA, XB, metric, out, **kwargs)[0m
[1;32m   3125[0m         [0;32mif[0m [0mmetric_info[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3126[0m             [0mcdist_fn[0m [0;34m=[0m [0mmetric_info[0m[0;34m.[0m[0mcdist_func[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3127[0;31m             [0;32mreturn[0m [0mcdist_fn[0m[0;34m([0m[0mXA[0m[0;34m,[0m [0mXB[0m[0;34m,[0m [0mout[0m[0;34m=[0m[0mout[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3128[0m         [0;32melif[0m [0mmstr[0m[0;34m.[0m[0mstartswith[0m[0;34m([0m[0;34m"test_"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3129[0m             [0mmetric_info[0m [0;34m=[0m [0m_TEST_METRICS[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mmstr[0m[0;34m,[0m [0;32mNone[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mMemoryError[0m: Unable to allocate 173. TiB for an array with shape (4874303, 4874303) and data type float64

## === cell 7
coords = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
concat_coords = pd.concat([train[coords], test[coords]], axis=0)

db = Birch(
    branching_factor=50,
    n_clusters=None,
    threshold=0.5,
    compute_labels=True,
).fit(concat_coords)

labels = db.labels_
train["cluster"] = labels[: train.shape[0]]
test["cluster"] = labels[train.shape[0] :]
