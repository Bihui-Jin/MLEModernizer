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

4.35657

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 4.35657) has done: 'I correct the data path so the files are found, and drop the raw datetime column after extracting its components (XGBoost requires only numeric features). These fixes unblock the whole pipeline and allow a proper .csv submission to be written.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import Birch
import xgboost as xgb




## === cell 1
base_path = "/kaggle/input/new-york-city-taxi-fare-prediction"
train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")

train = pd.read_csv(train_path, nrows=1_000_000)
test = pd.read_csv(test_path)




## === cell 2
print("NaNs before cleaning")
print(train.isnull().sum())
train = train.dropna()
print("NaNs after dropping")
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
train = train.loc[
    (train["fare_amount"] > 0)
    & (train["fare_amount"] < 300)
    & (train["pickup_longitude"] > pickup_longitude_min)
    & (train["pickup_longitude"] < pickup_longitude_max)
    & (train["pickup_latitude"] > pickup_latitude_min)
    & (train["pickup_latitude"] < pickup_latitude_max)
    & (train["dropoff_longitude"] > dropoff_longitude_min)
    & (train["dropoff_longitude"] < dropoff_longitude_max)
    & (train["dropoff_latitude"] > dropoff_latitude_min)
    & (train["dropoff_latitude"] < dropoff_latitude_max)
    & (train["passenger_count"] <= 8)
]
train.describe()




## === cell 5
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Vectorised haversine distance in kilometres.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km




## === cell 6
for df in [train, test]:
    df["haversine"] = haversine_np(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], utc=True, errors="coerce"
    )
    df["hour_of_day"] = df["pickup_datetime"].dt.hour
    df["day"] = df["pickup_datetime"].dt.day
    df["week"] = df["pickup_datetime"].dt.isocalendar().week.astype(int)
    df["month"] = df["pickup_datetime"].dt.month
    df["day_of_year"] = df["pickup_datetime"].dt.dayofyear
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df.drop(columns=["pickup_datetime"], inplace=True)




## === cell 7
train = train.dropna().reset_index(drop=True)
test = test.dropna().reset_index(drop=True)




## === cell 8
scale_features = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "haversine",
    "hour_of_day",
    "day",
    "week",
    "month",
    "day_of_year",
    "weekday",
]
scaler = StandardScaler()
train[scale_features] = scaler.fit_transform(train[scale_features])
test[scale_features] = scaler.transform(test[scale_features])




## === cell 9
coords = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
concat_coords = pd.concat([train[coords], test[coords]])
birch = Birch(branching_factor=50, threshold=0.5, compute_labels=True).fit(
    concat_coords
)
labels = birch.labels_
train["cluster"] = labels[: len(train)]
test["cluster"] = labels[len(train) :]




## === cell 10
pca_features = ["haversine", "hour_of_day", "day", "week", "month"]
pca = PCA(n_components=3, random_state=42)
pca_train = pca.fit_transform(train[pca_features])
train["pca0"], train["pca1"], train["pca2"] = pca_train.T
pca_test = pca.transform(test[pca_features])
test["pca0"], test["pca1"], test["pca2"] = pca_test.T




## === cell 11
good_numeric = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "haversine",
    "hour_of_day",
    "day",
    "week",
    "month",
    "day_of_year",
    "weekday",
    "cluster",
    "pca0",
    "pca1",
    "pca2",
]
train = train[["fare_amount"] + good_numeric].copy()
test = test[["key"] + good_numeric].copy()




## === cell 12
X = train.drop("fare_amount", axis=1)
y = train["fare_amount"]
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)




## === cell 13
def train_xgb(x_tr, x_va, y_tr, y_va):
    dtrain = xgb.DMatrix(x_tr, label=y_tr)
    dval = xgb.DMatrix(x_va, label=y_va)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.3,
        "max_depth": 4,
        "min_child_weight": 3,
        "seed": 42,
    }
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dval, "validation")],
        early_stopping_rounds=10,
        verbose_eval=False,
    )
    return model


model = train_xgb(x_train, x_val, y_train, y_val)




## === cell 14
xgb.plot_importance(model, max_num_features=10)
plt.tight_layout()
plt.show()




## === cell 15
dtest = xgb.DMatrix(test.drop("key", axis=1))
preds = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
submission = pd.DataFrame({"key": test["key"], "fare_amount": np.round(preds, 2)})
submission_path = "sub_fare.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
