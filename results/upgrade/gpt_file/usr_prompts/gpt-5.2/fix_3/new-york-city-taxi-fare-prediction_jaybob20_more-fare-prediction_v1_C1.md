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

4.51377

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.49132) has done: 'I fix the pandas datetime API breakage (`.dt.week`/`.dt.weekofyear`) by using ISO calendar week, and correct a typo in the coordinate bounds that was filtering out too much training data. I also fix feature engineering so it uses each dataset’s own `pickup_datetime` (it currently overwrites train with test timestamps), and ensure distance features are created for both train and test before scaling/PCA. To make XGBoost accept the data, I remove non-numeric columns (like `key` and datetime) from the training matrices and update the deprecated XGBoost objective and prediction call. Finally, I ensure a valid `submission.csv` with the required `key,fare_amount` columns is written.'
- What this solution (achieved 4.51377) has done: 'Your current score (4.49132 RMSE) is worse than the target (3.70811), so we should make small, low-risk changes that improve generalization without changing the overall approach. The biggest single issue is that you standardize latitude/longitude before clustering, but then cluster on the already-scaled coordinates, which destroys geographic meaning; clustering should be done on raw lat/lon and then the resulting cluster label can be used alongside scaled features. I keep all models/feature types the same, but (1) compute `cluster` from unscaled raw coordinates and (2) remove early stopping (it’s explicitly disallowed) by training a fixed number of boosting rounds equal to the previous maximum. These changes are minimal and should move RMSE closer to your target while still producing a valid `submission.csv`.'

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
train.describe()




## === cell 5
def haversine_np(a):
    """
    Calculate the great circle distance between two points on the earth (decimal degrees).
    All args must be of equal length.
    """
    lon1, lat1, lon2, lat2 = a
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    aa = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(aa))
    km = 6367 * c
    return km


def cityblock(a, dist="cityblock"):
    return pdist(a.reshape(2, 2), dist)[0]




## === cell 6
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




## === cell 7
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

from multiprocessing import Pool

combine = [train, test]
pool = Pool()

for dataset in combine:
    coords = dataset[
        ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    ].values

    dataset["haversine"] = pool.map(haversine_np, coords)

    for dist_type in dist_types:
        dataset[dist_type] = pool.starmap(
            cityblock, [(x, dist_type) for x in coords], chunksize=1000
        )

    dataset = add_time_features(dataset)

pool.close()
pool.join()
del pool

train = add_time_features(train)
test = add_time_features(test)

train.head(3)



## === cell 8
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

n = StandardScaler()
train[features] = n.fit_transform(train[features])
test[features] = n.transform(test[features])



## === cell 9
concat_raw = pd.concat([train_coords_raw, test_coords_raw], axis=0, ignore_index=True)

db = Birch(
    branching_factor=50, n_clusters=None, threshold=0.5, compute_labels=True
).fit(concat_raw)
labels = db.labels_

train["cluster"] = labels[: train.shape[0]]
test["cluster"] = labels[train.shape[0] :]



## === cell 10
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
p_result = pca.fit_transform(train[pca_features])
train["pca0"] = p_result[:, 0]
train["pca1"] = p_result[:, 1]
train["pca2"] = p_result[:, 2]

p_result = pca.transform(test[pca_features])
test["pca0"] = p_result[:, 0]
test["pca1"] = p_result[:, 1]
test["pca2"] = p_result[:, 2]



## === cell 11
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



## === cell 12
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

train_features_to_keep = (
    ["fare_amount"]
    + ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    + good_dist
)
train.drop(columns=train.columns.difference(train_features_to_keep), inplace=True)

test_features_to_keep = (
    ["key"]
    + ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    + good_dist
)
test.drop(columns=test.columns.difference(test_features_to_keep), inplace=True)

x_pred = test.drop("key", axis=1)



## === cell 13
x_train, x_test, y_train, y_test = train_test_split(
    train.drop("fare_amount", axis=1),
    train["fare_amount"],
    random_state=123,
    test_size=0.2,
)




## === cell 14
def XGBmodel(x_train, x_test, y_train, y_test):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_test = xgb.DMatrix(x_test, label=y_test)

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

print("Model trained with fixed num_boost_round=300")



## === cell 15
xgb.plot_importance(model, max_num_features=20)
plt.show()



## === cell 16
dtest = xgb.DMatrix(x_pred)

prediction = model.predict(dtest)

submission = pd.DataFrame({"key": test["key"], "fare_amount": np.round(prediction, 2)})
submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 17
submission
