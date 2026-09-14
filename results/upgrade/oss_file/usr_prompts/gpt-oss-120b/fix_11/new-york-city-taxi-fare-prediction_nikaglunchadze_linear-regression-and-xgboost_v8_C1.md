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

3.12

# 3. Installed packages

geopandas==0.14.4
joblib==1.5.2
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

3.59144

# 6. Current score

7.3655

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.07409) has done: 'The changes reduce the amount of data processed and the size of the hyper‑parameter sample, which cuts the overall runtime dramatically while leaving every modeling step, feature engineering, and evaluation logic unchanged.  
- Cell 0 now reads only 1 000 000 rows instead of 5 000 000, keeping the same preprocessing pipeline.  
- Cell 18 samples only 2 % of the already‑filtered training set for the XGBoost grid search, preserving the same search logic but making each model fit much faster.'
- What this solution (achieved 10.72288) has done: 'We dramatically cut the hyper‑parameter search cost: sample only 2 % of the filtered data, shrink the grid to a single reasonable configuration, and set `n_jobs=1` inside each XGB fit (the outer `Parallel` already uses all cores). The rest of the pipeline – feature engineering, model fitting and prediction – stays unchanged, so the result accuracy is preserved while the total runtime drops below the 600 s limit.'
- What this solution (achieved 7.62582) has done: 'I increase the training sample size (load more rows) and make the final XGBoost model a bit stronger by using more trees and ensuring a minimum depth of 8. These modest adjustments should lower the RMSE toward the target without altering the core pipeline.'
- What this solution (achieved 8.18096) has done: 'I keep the overall pipeline unchanged but improve the XGBoost hyper‑parameter search and final model strength.  
- In the hyper‑parameter search cell I increase the sampled fraction from 2 % to 5 % so the grid evaluation uses more data and finds better settings.  
- In the final‑model cell I boost the number of trees by 3× instead of 2× and enforce a minimum depth of 10 instead of 8, which should lower the validation RMSE and move the score toward the target (without altering any other logic).'
- What this solution (achieved 7.3655) has done: 'The changes focus on eliminating the expensive exhaustive hyper‑parameter search that caused the timeout. Cell 17 now skips the large cross‑validation loop and directly sets a reasonable hyper‑parameter set, and Cell 18 uses those preset values to train the XGBoost model. All core feature engineering, data filtering, and model architecture remain unchanged, preserving the original prediction logic while drastically reducing runtime.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

np.random.seed(42)

dtypes = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=3_000_000,
    dtype=dtypes,
)
test_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    dtype={k: v for k, v in dtypes.items() if k != "fare_amount"},
)
df.dtypes




## === cell 1
df.describe()




## === cell 2
df.head()




## === cell 3
def filter_column(df, column, range_min, range_max):
    return df[(df[column] >= range_min) & (df[column] <= range_max)]


ny_latitude_min, ny_latitude_max = 40.4772, 45.0153
ny_longitude_min, ny_longitude_max = -79.7624, -71.7517

df = df.dropna()  # null value‑들을 제거

df = filter_column(df, "pickup_longitude", ny_longitude_min, ny_longitude_max)
df = filter_column(df, "pickup_latitude", ny_latitude_min, ny_latitude_max)

df = filter_column(df, "dropoff_longitude", ny_longitude_min, ny_longitude_max)
df = filter_column(df, "dropoff_latitude", ny_latitude_min, ny_latitude_max)

df = filter_column(df, "passenger_count", 1, 6)




## === cell 4
df[df["fare_amount"] > 200].describe()




## === cell 5
df = filter_column(df, "fare_amount", 1, 200)




## === cell 6
def refactor_datetime(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["hour"] = df["pickup_datetime"].dt.hour
    df.drop(columns=["pickup_datetime"], inplace=True)


refactor_datetime(df)
refactor_datetime(test_df)
df.head()




## === cell 7
def haversine(p1, p2):
    lon1, lat1, lon2, lat2 = map(np.radians, [p1[1], p1[0], p2[1], p2[0]])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    dist = 2 * np.arcsin(
        np.sqrt(
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
    )
    km = 6367 * dist
    return km


ny_center = ("ny_center", (40.7128, -74.0060))
jfk_airport = ("jfk_airport", (40.6446, -73.7797))
lga_airport = ("lga_airport", (40.7733, -73.8718))
ewr_airport = ("ewr_airport", (40.6895, -74.1745))

locs = [ny_center, jfk_airport, lga_airport, ewr_airport]




## === cell 8
def insert_haversine_dists(df, locations):
    for location in locations:
        df["pickup_dist_to_" + location[0]] = haversine(
            (df["pickup_latitude"], df["pickup_longitude"]), location[1]
        )
        df["dropoff_dist_to_" + location[0]] = haversine(
            (df["dropoff_latitude"], df["dropoff_longitude"]), location[1]
        )
    df["ride_distance"] = haversine(
        (df["pickup_latitude"], df["pickup_longitude"]),
        (df["dropoff_latitude"], df["dropoff_longitude"]),
    )


insert_haversine_dists(df, locs)
insert_haversine_dists(test_df, locs)




## === cell 9
df.describe()




## === cell 10
df = df[df["ride_distance"] > 0]
df.describe()




## === cell 11
from sklearn.model_selection import train_test_split

train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)




## === cell 12
features = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "ride_distance",
]
features += ["pickup_dist_to_" + x[0] for x in locs]
features += ["dropoff_dist_to_" + x[0] for x in locs]
fare_amount = "fare_amount"

train_features = train_df[features]
train_fare_amount = train_df[fare_amount]

validation_features = validation_df[features]
validation_fare_amount = validation_df[fare_amount]
train_features.info()




## === cell 13
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()




## === cell 14
from sklearn.model_selection import cross_val_score


def estimate_model(model, df):
    X = df[features]
    y = df[fare_amount]
    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_error", n_jobs=-1
    )
    rmse_scores = np.sqrt(-cv_scores)
    print("RMSE scores for each fold:", rmse_scores)
    print("Mean RMSE:", rmse_scores.mean())
    print("Standard Deviation of RMSE:", rmse_scores.std())




## === cell 15
estimate_model(linear_model, train_df)




## === cell 16
linear_model.fit(train_features, train_fare_amount)




## === cell 17
best_lr = 0.1  # learning rate
best_ne = 500  # number of trees (minimum enforced later)
best_md = 12  # max depth
best_mcw = 1  # min child weight




## === cell 18
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    learning_rate=best_lr,
    n_estimators=best_ne,
    max_depth=best_md,
    min_child_weight=best_mcw,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    tree_method="hist",  # fast histogram based algorithm
    random_state=42,
)

xgb_model.fit(train_features.values, train_fare_amount.values)

xgb_predictions = xgb_model.predict(validation_features.values)
xgb_predictions = np.clip(xgb_predictions, 1, 200)
print(
    "Validation RMSE:",
    mean_squared_error(validation_fare_amount, xgb_predictions, squared=False),
)




## === cell 19
xgb_predictions_train = xgb_model.predict(train_features.values)
xgb_predictions_train = np.clip(xgb_predictions_train, 1, 200)
print(
    "Train RMSE:",
    mean_squared_error(train_fare_amount, xgb_predictions_train, squared=False),
)




## === cell 20
xgb_test_predictions = xgb_model.predict(test_df[features].values)
xgb_test_predictions = np.clip(xgb_test_predictions, 1, 200)
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": xgb_test_predictions})
submission.to_csv("submission.csv", index=False)
