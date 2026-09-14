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

bayesian-optimization==3.1.0
geopandas==0.14.4
lightgbm==4.6.0
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

3.21281

# 6. Current score

5.94241

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.99309) has done: 'You’re currently far worse than the target (RMSE 6.19 vs 3.21, lower is better), so we make a few minimal changes that reliably reduce RMSE without changing the overall modeling approach. The biggest issue is that your `dist()` feature uses Manhattan distance in degrees, which is a poor proxy for real taxi distance; switching to a standard haversine distance (in km) keeps the same feature idea but makes it physically meaningful and typically cuts error substantially. We also ensure the train/test split is reproducible and add a standard XGBoost regression objective/eval metric plus a small learning-rate adjustment paired with more boosting rounds (same algorithm, same training style) to improve generalization toward the target. Finally, we keep the submission format identical and still write `submission.csv`.'
- What this solution (achieved 5.99194) has done: 'We’re far worse than the target (RMSE 5.99 vs 3.21, lower is better), so the smallest reliable gains come from fixing feature issues rather than changing the model type. Your `transform()` currently creates “distance_to_center” using only pickup coordinates (it mistakenly uses NYC center as the “pickup” and the pickup point as the “dropoff”), which corrupts an important feature; we correct that to compute distance from pickup to center. We also clip negative predictions to 0 (fares can’t be negative) to reduce RMSE from outlier predictions without altering the training objective. Finally, we ensure the test read uses the same columns as train (and keep the submission schema unchanged) to avoid subtle train/test feature mismatch.'
- What this solution (achieved 6.95382) has done: 'Your score is still far above the target (RMSE 5.99 vs 3.21, lower is better), so we make small, metric-aligned fixes that typically yield large RMSE drops without changing the core “engineered geo/time features + XGBoost regressor” approach. The biggest remaining issue is that the model is trained on raw coordinates, which encourages spurious splits and hurts generalization; we keep those columns but add standard coordinate sanity filtering and a simple `abs()` version of your lat/long deltas to remove sign ambiguity while preserving the same feature idea. We also train using an explicit validation set with `evals` (same training loop style) so you can see generalization and avoid silent overfitting; prediction and submission format stay identical. Finally, we clip extreme fares in training a bit tighter (still within typical competition practice) to reduce the impact of outliers on RMSE.'
- What this solution (achieved 5.94241) has done: 'You’re far worse than the target (RMSE 6.95 vs 3.21, lower is better), so we should make small, high-impact fixes that keep your “engineered geo/time features + XGBoost regressor” core intact. The biggest remaining issue is that you never remove obviously-bad coordinate cases like identical pickup/dropoff (often bad data) and extreme trip distances, which can heavily inflate RMSE; we add two simple, standard filters on training only. Next, we add a single missing but very common feature consistent with your existing logic: the bearing (direction) between pickup and dropoff, which complements distance without changing the model type. Finally, we keep the same training loop, but add a very light regularization (`reg_lambda`) and enable histogram tree method for speed/stability; submission writing stays identical.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



## === cell 1
df = pd.read_csv(
    "../input/train.csv",
    nrows=2_000_000,
    usecols=[
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)



## === cell 2
df["pickup_datetime"] = df["pickup_datetime"].str.slice(0, 16)
df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"], utc=True, format="%Y-%m-%d %H:%M", errors="coerce"
)



## === cell 3
df.dropna(how="any", axis="rows", inplace=True)

mask = df["pickup_longitude"].between(-75, -73)
mask &= df["dropoff_longitude"].between(-75, -73)
mask &= df["pickup_latitude"].between(40, 42)
mask &= df["dropoff_latitude"].between(40, 42)
mask &= df["passenger_count"].between(1, 8)  # drop 0-passenger noise
mask &= df["fare_amount"].between(
    2.5, 200
)  # slightly tighter to reduce outlier impact on RMSE

mask &= df["pickup_longitude"].between(-180, 180)
mask &= df["dropoff_longitude"].between(-180, 180)
mask &= df["pickup_latitude"].between(-90, 90)
mask &= df["dropoff_latitude"].between(-90, 90)

df = df[mask].copy()




## === cell 4
def dist(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    R = 6371.0
    lat1 = np.radians(pickup_lat)
    lon1 = np.radians(pickup_long)
    lat2 = np.radians(dropoff_lat)
    lon2 = np.radians(dropoff_long)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c


def bearing(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    lat1 = np.radians(pickup_lat)
    lon1 = np.radians(pickup_long)
    lat2 = np.radians(dropoff_lat)
    lon2 = np.radians(dropoff_long)
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.arctan2(y, x)
    return brng  # radians in [-pi, pi]




## === cell 5
def transform(data):
    data["hour"] = data["pickup_datetime"].dt.hour
    data["day"] = data["pickup_datetime"].dt.day
    data["month"] = data["pickup_datetime"].dt.month
    data["year"] = data["pickup_datetime"].dt.year
    data = data.drop("pickup_datetime", axis=1)

    nyc = (-74.0063889, 40.7141667)
    jfk = (-73.7822222222, 40.6441666667)
    ewr = (-74.175, 40.69)
    lgr = (-73.87, 40.77)

    data["distance_to_center"] = dist(
        data["pickup_latitude"], data["pickup_longitude"], nyc[1], nyc[0]
    )

    data["pickup_distance_to_jfk"] = dist(
        jfk[1], jfk[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["dropoff_distance_to_jfk"] = dist(
        jfk[1], jfk[0], data["dropoff_latitude"], data["dropoff_longitude"]
    )
    data["pickup_distance_to_ewr"] = dist(
        ewr[1], ewr[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["dropoff_distance_to_ewr"] = dist(
        ewr[1], ewr[0], data["dropoff_latitude"], data["dropoff_longitude"]
    )
    data["pickup_distance_to_lgr"] = dist(
        lgr[1], lgr[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["dropoff_distance_to_lgr"] = dist(
        lgr[1], lgr[0], data["dropoff_latitude"], data["dropoff_longitude"]
    )

    data["long_dist"] = data["pickup_longitude"] - data["dropoff_longitude"]
    data["lat_dist"] = data["pickup_latitude"] - data["dropoff_latitude"]

    data["abs_long_dist"] = np.abs(data["long_dist"])
    data["abs_lat_dist"] = np.abs(data["lat_dist"])

    data["dist"] = dist(
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["dropoff_latitude"],
        data["dropoff_longitude"],
    )

    data["bearing"] = bearing(
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["dropoff_latitude"],
        data["dropoff_longitude"],
    )
    return data




## === cell 6
df["raw_dist_km"] = dist(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"],
)

mask2 = df["raw_dist_km"].between(
    0.2, 80
)  # remove near-zero and extreme trips (often bad data)
df = df[mask2].copy()
df.drop(columns=["raw_dist_km"], inplace=True)

df = transform(df)



## === cell 7
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



## === cell 8
X = df.drop("fare_amount", axis=1)
y = df["fare_amount"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)



## === cell 9
params = {
    "max_depth": 7,
    "gamma": 0.0,
    "colsample_bytree": 0.3,
    "min_child_weight": 3.0,
    "subsample": 1.0,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.05,
    "seed": 42,
    "reg_lambda": 1.0,
    "tree_method": "hist",
}



## === cell 10
watchlist = [(dtrain, "train"), (dval, "val")]
model2 = xgb.train(
    params, dtrain, num_boost_round=1200, evals=watchlist, verbose_eval=100
)

y_val_pred = model2.predict(dval)
y_train_pred = model2.predict(dtrain)

print(np.sqrt(mean_squared_error(y_val, y_val_pred)))
print(np.sqrt(mean_squared_error(y_train, y_train_pred)))



## === cell 11
import matplotlib.pyplot as plt

fscores = pd.DataFrame(
    {"X": list(model2.get_fscore().keys()), "Y": list(model2.get_fscore().values())}
)
fscores.sort_values(by="Y").plot.bar(x="X")



## === cell 12
test = pd.read_csv(
    "../input/test.csv",
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
).set_index("key")

test["pickup_datetime"] = test["pickup_datetime"].str.slice(0, 16)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], utc=True, format="%Y-%m-%d %H:%M", errors="coerce"
)

test["pickup_datetime"] = test["pickup_datetime"].fillna(
    pd.Timestamp("2010-01-01", tz="UTC")
)

test = transform(test)

dtest = xgb.DMatrix(test)
y_pred_test = model2.predict(dtest)

y_pred_test = np.clip(y_pred_test, 0, None)



## === cell 13
holdout = pd.DataFrame({"key": test.index, "fare_amount": y_pred_test})
holdout.to_csv("submission.csv", index=False)
print("Wrote submission.csv with", len(holdout), "rows")
