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

4.24009

# 6. Current score

7.19761

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.54255) has done: 'I fix the two failing visualization cells by ensuring `X` and `Y` refer to the same training design matrix at the time of plotting (they currently mismatch because `X` is reassigned later for the multi-feature model). I also make the RandomForest reproducible and use a slightly more sensible set of hyperparameters (still the same model/approach) to move RMSE down toward your target without changing the overall pipeline. Additionally, I harden data loading for the Kaggle path layout you showed (`/kaggle/input/...`) so the notebook runs end-to-end and always writes a valid `submission.csv`. All other core feature engineering and modeling steps remain the same.'
- What this solution (achieved 7.19761) has done: 'You’re currently worse than the target RMSE (6.54 vs 4.24), so we should make small, legitimate improvements that keep the same overall approach (feature engineering + scikit-learn regression, no training-loop changes). The biggest gain with minimal risk is to (1) clean obvious label outliers/non-physical values (negative/huge fares, zero-distance rides) and (2) tighten the NYC geographic bounding instead of the very loose “within 5 degrees of mean” filter, which leaves many bad points that hurt RMSE. I keep the same features and RandomForest model family, but adjust a couple of RF hyperparameters slightly (still a RandomForestRegressor) to generalize better. Finally, I apply the same feature columns consistently and clip negative predictions to 0 to avoid invalid fares that worsen RMSE.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42


def _resolve_path(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


TRAIN_PATH = _resolve_path(
    "/kaggle/input/train.csv",
    "/kaggle/data/train.csv",
    "../input/train.csv",
)
TEST_PATH = _resolve_path(
    "/kaggle/input/test.csv",
    "/kaggle/data/test.csv",
    "../input/test.csv",
)
SAMPLE_SUB_PATH = _resolve_path(
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "../input/sample_submission.csv",
)



## === cell 1
train_df = pd.read_csv(TRAIN_PATH, nrows=2_000_000)




## === cell 2
def haversine_np(lon1, lat1, lon2, lat2):
    """
    https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
    Calculate the great circle distance between two points on the earth (specified in decimal degrees)

    All args must be of equal length.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km




## === cell 3
train_df["distance"] = haversine_np(
    train_df["pickup_longitude"],
    train_df["pickup_latitude"],
    train_df["dropoff_longitude"],
    train_df["dropoff_latitude"],
)



## === cell 4
train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"], errors="coerce"
)



## === cell 5
train_df["year"] = train_df["pickup_datetime"].dt.year
train_df["month"] = train_df["pickup_datetime"].dt.month
train_df["day"] = train_df["pickup_datetime"].dt.day
train_df["hour"] = train_df["pickup_datetime"].dt.hour
train_df["minute"] = train_df["pickup_datetime"].dt.minute



## === cell 6
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 7
new_york_lat = 40
new_york_long = -74
train_df.describe()




## === cell 8
def _nyc_bbox_filter(df):
    lon_min, lon_max = -74.3, -73.7
    lat_min, lat_max = 40.5, 41.0
    return (
        df["pickup_longitude"].between(lon_min, lon_max)
        & df["dropoff_longitude"].between(lon_min, lon_max)
        & df["pickup_latitude"].between(lat_min, lat_max)
        & df["dropoff_latitude"].between(lat_min, lat_max)
    )


cond = _nyc_bbox_filter(train_df)



## === cell 9
print("Old size: %d" % len(train_df))
train_df = train_df[cond]
print("New size: %d" % len(train_df))



## === cell 10
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df["fare_amount"] > 0.0)
    & (train_df["fare_amount"] <= 250.0)
    & (train_df["distance"] > 0.01)
    & (train_df["distance"] <= 200.0)
    & (train_df["passenger_count"] >= 1)
    & (train_df["passenger_count"] <= 6)
]
print("New size: %d" % len(train_df))



## === cell 11
train_df.describe()



## === cell 12
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



## === cell 13
regr = LinearRegression()
regr_quad = LinearRegression()



## === cell 14
X = train_df[["distance"]].values
Y = train_df["fare_amount"].values



## === cell 15
regr.fit(X, Y)



## === cell 16
regr_quad.fit(X**2, Y)



## === cell 17
y_pred = regr.predict(X)
print("chi squared linear %s" % (np.sum((Y - y_pred) ** 2.0) / len(Y)) ** 0.5)
y_pred_quad = regr_quad.predict(X)
print("chi squared quadratic %s" % (np.sum((Y - y_pred_quad) ** 2.0) / len(Y)) ** 0.5)



## === cell 18
regr_more = LinearRegression()
X = train_df[["distance", "year", "month", "day", "hour"]].values
Y = train_df["fare_amount"].values



## === cell 19
regr_more.fit(X, Y)
y_pred = regr_more.predict(X)
print("chi squared linear with date %s" % (np.sum((Y - y_pred) ** 2.0) / len(Y)) ** 0.5)



## === cell 20
from sklearn.ensemble import RandomForestRegressor



## === cell 21
rand_regr = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    min_samples_leaf=3,
    min_samples_split=6,
    max_features="sqrt",
    bootstrap=True,
)



## === cell 22
rand_regr.fit(X, Y)
y_pred = rand_regr.predict(X)
print(
    "chi squared  rand forest with date %s"
    % (np.sum((Y - y_pred) ** 2.0) / len(Y)) ** 0.5
)



## === cell 23
test_df = pd.read_csv(TEST_PATH)



## === cell 24
test_df["distance"] = haversine_np(
    test_df["pickup_longitude"],
    test_df["pickup_latitude"],
    test_df["dropoff_longitude"],
    test_df["dropoff_latitude"],
)



## === cell 25
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"], errors="coerce")



## === cell 26
test_df["year"] = test_df["pickup_datetime"].dt.year
test_df["month"] = test_df["pickup_datetime"].dt.month
test_df["day"] = test_df["pickup_datetime"].dt.day
test_df["hour"] = test_df["pickup_datetime"].dt.hour
test_df["minute"] = test_df["pickup_datetime"].dt.minute



## === cell 27
X_to_pred = test_df[["distance", "year", "month", "day", "hour"]].values
y_pred = rand_regr.predict(X_to_pred)

y_pred = np.clip(y_pred, 0.0, None)



## === cell 28
submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": y_pred}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)



## === cell 29
X_plot = train_df[["distance"]].values.flatten()
Y_plot = train_df["fare_amount"].values

with sns.axes_style("white"):
    sns.jointplot(x=X_plot, y=Y_plot, kind="hex", color="k", bins="log")



## === cell 30
X_plot = train_df[["distance"]].values.flatten()
Y_plot = train_df["fare_amount"].values
mask = (X_plot < 50) & (Y_plot < 100)

with sns.axes_style("white"):
    p = sns.jointplot(x=X_plot[mask], y=Y_plot[mask], kind="hex", color="k", bins="log")

x = np.arange(0, 50)
y = regr.predict(x.reshape(-1, 1))
p.ax_joint.plot(x, y)
