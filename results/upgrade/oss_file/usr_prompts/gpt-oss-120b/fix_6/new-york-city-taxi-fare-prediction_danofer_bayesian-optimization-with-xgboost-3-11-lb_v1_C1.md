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

3.60475

# 6. Current score

5.52238

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 12.66823) has done: 'I fixed the key bugs that stopped the pipeline: the training data was being transformed twice, causing a missing `pickup_datetime` column, and the final DMatrix variable name was mismatched. I also added a haversine distance feature (a cheap improvement) and increased the number of boosting rounds to give the model a bit more capacity. All cells now run end‑to‑end and produce a valid `submission.csv` with the required columns.'
- What this solution (achieved 6.83006) has done: 'I add a log‑transform of the target variable so the model works on a more‑stable distribution and then exponentiate the predictions back to the original scale. This small change keeps the core pipeline unchanged while typically lowering RMSE, moving the score closer to the target. The necessary adjustments are made in the data‑splitting, training, validation, final model, and submission steps, and the script now writes a proper `submission.csv`.'
- What this solution (achieved 5.47204) has done: 'I add log‑scaled distance features (log_haversine and log_dist) to give the model a more balanced view of travel length, and tweak the XGBoost hyper‑parameters to a smaller learning rate with more boosting rounds (max_depth = 8, eta = 0.05, num_boost_round = 1000). These minimal adjustments keep the overall pipeline unchanged while improving predictive performance, moving the validation RMSE closer to the target.'
- What this solution (achieved 5.52238) has done: 'I added a few inexpensive cyclical time features (hour sin/cos, day‑of‑week sin/cos) and the raw weekday number to give the model better temporal signals, then kept the rest of the pipeline unchanged. I also increased the boosting rounds from 1000 to 1500 to let the model fully exploit the extra features. These minimal tweaks are expected to lower the validation RMSE toward the target while preserving the original architecture and training logic.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import math




## === cell 1
df = pd.read_csv(
    "../input/train.csv",
    nrows=51234,
    usecols=[1, 2, 3, 4, 5, 6, 7],
    names=[
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    header=0,
)




## === cell 2
df["pickup_datetime"] = df["pickup_datetime"].str.slice(0, 16)
df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"], utc=True, format="%Y-%m-%d %H:%M"
)




## === cell 3
test = pd.read_csv("../input/test.csv")
test = test.set_index("key")
test["pickup_datetime"] = test["pickup_datetime"].str.slice(0, 16)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], utc=True, format="%Y-%m-%d %H:%M"
)




## === cell 4
df.dropna(how="any", axis="rows", inplace=True)

mask = df["pickup_longitude"].between(-75, -72.8)
mask &= df["dropoff_longitude"].between(-75, -72.8)
mask &= df["pickup_latitude"].between(40, 42)
mask &= df["dropoff_latitude"].between(40, 42)
mask &= df["passenger_count"].between(0, 7)
mask &= df["fare_amount"].between(0, 250)

df = df[mask]




## === cell 5
def dist(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    """Manhattan‑style distance."""
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def haversine(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    """Great‑circle distance in kilometres."""
    lat1, lon1, lat2, lon2 = map(
        np.radians, [pickup_lat, pickup_long, dropoff_lat, dropoff_long]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km




## === cell 6
def transform(data):
    data["hour"] = data["pickup_datetime"].dt.hour
    data["day"] = data["pickup_datetime"].dt.day
    data["month"] = data["pickup_datetime"].dt.month
    data["year"] = data["pickup_datetime"].dt.year
    data["dow"] = data["pickup_datetime"].dt.dayofweek  # 0=Mon … 6=Sun

    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["dow_sin"] = np.sin(2 * np.pi * data["dow"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["dow"] / 7)

    data = data.drop("pickup_datetime", axis=1)

    nyc = (-74.0063889, 40.7141667)
    jfk = (-73.7822222222, 40.6441666667)
    ewr = (-74.175, 40.69)
    lgr = (-73.87, 40.77)

    data["distance_to_center"] = dist(
        nyc[1], nyc[0], data["pickup_latitude"], data["pickup_longitude"]
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
    data["dist"] = dist(
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["dropoff_latitude"],
        data["dropoff_longitude"],
    )
    data["haversine_dist"] = haversine(
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["dropoff_latitude"],
        data["dropoff_longitude"],
    )
    data["log_dist"] = np.log1p(data["dist"])
    data["log_haversine"] = np.log1p(data["haversine_dist"])
    return data




## === cell 7
df = transform(df)

X = df.drop("fare_amount", axis=1)
y = df["fare_amount"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

dtrain = xgb.DMatrix(X_train, label=y_train_log)
dval = xgb.DMatrix(X_val, label=y_val_log)

df_full_X = df.drop("fare_amount", axis=1)
df_full_y_log = np.log1p(df["fare_amount"])
dfull = xgb.DMatrix(df_full_X, label=df_full_y_log)




## === cell 8
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 8,  # deeper trees for more complexity
    "eta": 0.05,  # smaller learning rate
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.0,
    "seed": 42,
}




## === cell 9
model_val = xgb.train(
    params,
    dtrain,
    num_boost_round=1500,  # increased rounds for the richer feature set
    evals=[(dval, "validation")],
    verbose_eval=False,
)

y_pred_log_val = model_val.predict(dval)
y_pred_val = np.expm1(y_pred_log_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"Validation RMSE (original scale): {rmse_val:.4f}")




## === cell 10
final_model = xgb.train(params, dfull, num_boost_round=1500, verbose_eval=False)




## === cell 11
test = transform(test)




## === cell 12
dtest = xgb.DMatrix(test)
y_pred_log_test = final_model.predict(dtest)
y_pred_test = np.expm1(y_pred_log_test)




## === cell 13
submission = pd.DataFrame({"key": test.index, "fare_amount": y_pred_test})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
