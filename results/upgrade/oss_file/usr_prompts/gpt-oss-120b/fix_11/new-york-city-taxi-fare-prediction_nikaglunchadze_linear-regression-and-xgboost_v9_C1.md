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

3.6129

# 6. Current score

11.41747

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.07333) has done: 'I keep the overall pipeline unchanged but select the best learning‑rate and number of estimators found in the cross‑validation step, and train the final XGBoost model with early stopping on the held‑out validation set. This modest tweak should lower the validation RMSE, moving the score nearer to the target 3.6129 without altering the core feature engineering or model type.'
- What this solution (achieved 6.64042) has done: 'We speed up the heavy cross‑validation steps while keeping the model definitions, feature engineering and evaluation unchanged.  
- In the linear‑regression CV we enable `n_jobs=-1` so all CPU cores are used.  
- The XGBoost hyper‑parameter search now runs the five CV folds in parallel (outer loop already parallel over parameter combos). Each XGBRegressor is limited to a single thread (`n_jobs=1`) to avoid oversubscription, preserving the exact training behaviour.  
These adjustments drastically cut runtime without altering any algorithmic logic or results.'
- What this solution (achieved 6.57538) has done: 'I combined the repeated geographic filtering into a single boolean mask to avoid multiple passes over the DataFrame and replaced the expensive hyper‑parameter grid search with a fixed, reasonable set of XGBoost parameters (learning_rate = 0.1, n_estimators = 150). This removes the 60 costly model fits while keeping the same model architecture, feature set, and training‑procedure logic, so the final predictions remain unchanged in semantics but the script now finishes well within the 600‑second limit.'
- What this solution (achieved 8.88376) has done: 'I add two simple geographic difference features (`delta_lat`, `delta_lon`) to give the model a bit more information, and increase the XGBoost n_estimators from 150 to 300 so the model can better fit the data while early‑stopping still prevents over‑training. These minor tweaks keep the overall pipeline unchanged but are expected to lower the validation RMSE, moving the score closer to the target.'
- What this solution (achieved 10.8862) has done: 'The update keeps the overall pipeline intact but adjusts the XGBoost hyper‑parameters to give the model more capacity and a smoother learning rate, which should reduce the validation RMSE and move the score closer to the target. The changes are limited to cell 17 where the `XGBRegressor` is instantiated.'
- What this solution (achieved 13.33209) has done: 'I keep the overall pipeline unchanged but remove the log‑transform of the target when training the XGBoost model, because exponentiating the predictions after fitting on log1p can introduce bias that inflates RMSE. By training directly on the raw fare amounts and still using early‑stopping, the model can learn a better scale and is expected to lower the validation error, moving the score closer to the target.'
- What this solution (achieved 10.8862) has done: 'I train the XGBoost model on a log‑transformed target (log1p) and exponentiate the predictions back to the original scale before computing RMSE and creating the submission. This simple change aligns the loss with the distribution of fare values and is expected to reduce the validation RMSE, moving the score closer to the target while preserving all other pipeline logic.'
- What this solution (achieved 11.41747) has done: 'I simplify the XGBoost training by removing the log‑transform of the target and using a more standard set of hyper‑parameters (lower max depth, fewer estimators, slightly higher learning rate). This keeps the overall pipeline intact while making the model easier to generalise, which should lower the validation RMSE and move the score toward the target. The prediction steps are also updated to work on the raw fare amounts.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=5_000_000
)
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
df.dtypes



## === cell 1
df.describe()




## === cell 2
def filter_column(df, column, range_min, range_max):
    return df[(df[column] >= range_min) & (df[column] <= range_max)]


ny_latitude_min, ny_latitude_max = 40.4772, 45.0153
ny_longitude_min, ny_longitude_max = -79.7624, -71.7517

df = df.dropna()

mask = (
    df["pickup_longitude"].between(ny_longitude_min, ny_longitude_max)
    & df["pickup_latitude"].between(ny_latitude_min, ny_latitude_max)
    & df["dropoff_longitude"].between(ny_longitude_min, ny_longitude_max)
    & df["dropoff_latitude"].between(ny_latitude_min, ny_latitude_max)
    & df["passenger_count"].between(1, 6)
)
df = df[mask]



## === cell 3
df[df["fare_amount"] > 200].describe()



## === cell 4
df = filter_column(df, "fare_amount", 1, 200)




## === cell 5
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




## === cell 6
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




## === cell 7
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



## === cell 8
df.describe()
df["delta_lat"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
df["delta_lon"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
test_df["delta_lat"] = (test_df["dropoff_latitude"] - test_df["pickup_latitude"]).abs()
test_df["delta_lon"] = (
    test_df["dropoff_longitude"] - test_df["pickup_longitude"]
).abs()



## === cell 9
df = df[df["ride_distance"] > 0]
df.describe()



## === cell 10
from sklearn.model_selection import train_test_split

train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)



## === cell 11
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
    "delta_lat",
    "delta_lon",
]
features += ["pickup_dist_to_" + x[0] for x in locs]
features += ["dropoff_dist_to_" + x[0] for x in locs]
fare_amount = "fare_amount"

train_features = train_df[features]
train_fare_amount = train_df[fare_amount]

validation_features = validation_df[features]
validation_fare_amount = validation_df[fare_amount]
train_features.info()



## === cell 12
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()



## === cell 13
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




## === cell 14
estimate_model(linear_model, train_df)



## === cell 15
linear_model.fit(train_features, train_fare_amount)



## === cell 16
from sklearn.metrics import mean_squared_error

linear_predictions = linear_model.predict(validation_features)
mean_squared_error(validation_fare_amount, linear_predictions, squared=False)



## === cell 17
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

best_lr = 0.1
best_ne = 300
best_max_depth = 6
print(
    f"Using learning_rate={best_lr}, n_estimators={best_ne}, max_depth={best_max_depth}"
)

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    learning_rate=best_lr,
    n_estimators=best_ne,
    max_depth=best_max_depth,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    n_jobs=-1,
    eval_metric="rmse",
)

xgb_model.fit(
    train_features,
    train_fare_amount,
    eval_set=[(validation_features, validation_fare_amount)],
    early_stopping_rounds=30,
    verbose=False,
)

val_pred = xgb_model.predict(validation_features)
val_rmse = mean_squared_error(validation_fare_amount, val_pred, squared=False)
print(f"Validation RMSE (original target scale): {val_rmse}")



## === cell 18
train_pred = xgb_model.predict(train_features)
train_rmse = mean_squared_error(train_fare_amount, train_pred, squared=False)
print(f"Training RMSE (original target scale): {train_rmse}")



## === cell 19
test_pred = xgb_model.predict(test_df[features])
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission.to_csv("submission.csv", index=False)
