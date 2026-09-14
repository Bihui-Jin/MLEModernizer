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

4.53031

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.47603) has done: 'I fix the datetime feature extraction to work with current pandas (replace deprecated `.dt.week`/`.dt.weekofyear`), correct a latitude/longitude max bug in the filtering bounds, and ensure datetime parsing is applied to each dataset (train/test) instead of mistakenly using `test` for both. I also make the distance-feature generation robust (no leaked pool state, no NaNs from distance calculations), keep scaling consistent, and adjust `drop()` calls to the current pandas API so columns are removed correctly. Finally, I ensure only numeric features (no `key`/`pickup_datetime`) reach XGBoost and update the XGBoost objective to the supported `reg:squarederror`, producing a valid `sub_fare.csv` submission.'
- What this solution (achieved 4.53031) has done: 'Your current RMSE (4.476) is worse than the target (3.6371), so we should make small, metric-aligned improvements that don’t change the overall pipeline. The biggest gain with minimal disruption is to stop rounding predictions to cents before scoring; rounding almost always increases RMSE on this competition. Next, keep your existing early-stopping logic for choosing the number of trees, but retrain once on the full filtered training data using that best number of boosting rounds (same model family/params) so the final model is stronger without changing the approach. These two changes are small, legitimate, and typically move RMSE noticeably toward your target while preserving your core feature engineering and XGBoost setup.'

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
    return pdist(a.reshape(2, 2), dist)[0]




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

for col in ["haversine"] + dist_types:
    train[col] = train[col].replace([np.inf, -np.inf], np.nan)
    test[col] = test[col].replace([np.inf, -np.inf], np.nan)
train[["haversine"] + dist_types] = train[["haversine"] + dist_types].fillna(0.0)
test[["haversine"] + dist_types] = test[["haversine"] + dist_types].fillna(0.0)

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

n = StandardScaler()
train[features] = n.fit_transform(train[features])
test[features] = n.transform(test[features])



## === cell 8
coords_cols = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
concat = pd.concat([train[coords_cols], test[coords_cols]], axis=0, ignore_index=True)

db = Birch(
    branching_factor=50, n_clusters=None, threshold=0.5, compute_labels=True
).fit(concat)
labels = db.labels_

train["cluster"] = labels[: train.shape[0]]
test["cluster"] = labels[train.shape[0] :]



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

train[pca_features] = train[pca_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
test[pca_features] = test[pca_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

pca = PCA(n_components=3, random_state=123)
p_result = pca.fit_transform(train[pca_features])
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



## === cell 11
good_dist = ["pca0", "pca1", "pca2", "cluster"]

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



## === cell 12
x_train, x_test, y_train, y_test = train_test_split(
    train.drop("fare_amount", axis=1),
    train["fare_amount"],
    random_state=123,
    test_size=0.2,
)




## === cell 13
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



## === cell 14
xgb.plot_importance(model, max_num_features=20)
plt.show()



## === cell 15
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
        "seed": 123,
    },
    dtrain=dall,
    num_boost_round=best_rounds,
    evals=[(dall, "train")],
    verbose_eval=False,
)



## === cell 16
dtest = xgb.DMatrix(x_pred)

prediction = final_model.predict(dtest)

submission = pd.DataFrame({"key": test["key"], "fare_amount": prediction})
submission.to_csv("sub_fare.csv", index=False)
print("Wrote submission:", submission.shape, "-> sub_fare.csv")
submission.head()



## === cell 17
submission
