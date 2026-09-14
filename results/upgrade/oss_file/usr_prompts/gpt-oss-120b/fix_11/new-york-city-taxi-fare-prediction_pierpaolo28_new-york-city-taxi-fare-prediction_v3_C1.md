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

4.10469

# 6. Current score

5.10743

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.24975) has done: 'I fixed the seaborn `regplot` and `lmplot` calls (they now use keyword arguments, matching the current seaborn API) and added a few cheap datetime‑based features (hour, weekday, month) that are known to help fare prediction. The XGBoost model is given modest hyper‑parameters to improve performance without changing the core logic. These changes resolve the runtime errors, produce a proper “sub_*.csv” file, and should move the RMSE from 5.34 closer to the target 4.10.'
- What this solution (achieved 5.27172) has done: 'I keep the overall pipeline unchanged but improve the model by (1) training on a log‑transformed target (log1p) and converting predictions back with exp‑1, (2) slightly strengthening the XGBoost hyper‑parameters (more trees, deeper depth, smaller learning rate) which usually lowers RMSE, and (3) applying a final clipping of negative predictions to zero. These modest tweaks keep the core logic intact while moving the validation RMSE closer to the target.'
- What this solution (achieved 5.19201) has done: 'I add signed latitude/longitude differences and the Euclidean distance to the feature set, then include these new columns in the model’s training and prediction pipeline. These inexpensive features usually capture direction and actual distance better than only absolute differences, which should lower the RMSE and move the score closer to the target.'
- What this solution (achieved 5.25959) has done: 'I fix the haversine distance calculation, add cyclical hour features (sin & cos) to better capture time‑of‑day patterns, include those features in the model, and slightly strengthen the XGBoost hyper‑parameters (more trees, a smaller learning rate). These modest, targeted changes keep the core pipeline intact while expected to lower the validation RMSE and move the score closer to the target.'
- What this solution (achieved 4.94947) has done: 'I add the raw latitude/longitude columns to the feature set so the model can directly learn location effects, and I slightly strengthen the XGBoost model (more trees and a lower learning rate) to improve its capacity without altering the overall pipeline. These minimal, targeted changes keep the core logic intact while aiming to close the RMSE gap toward the target score.'
- What this solution (achieved 4.94722) has done: 'I keep the overall pipeline unchanged but strengthen the XGBoost model by increasing the maximum number of trees and lowering the learning rate, then let early stopping pick the optimal number of boosting rounds on the validation set. This typically reduces over‑fitting and lowers the RMSE, moving the score closer to the target while preserving all existing features and processing steps.'
- What this solution (achieved 4.99204) has done: 'I add cyclical encodings for month and weekday (sin & cos) to give the model a smoother representation of temporal patterns, and I slightly increase model capacity (deeper trees, more estimators, a smaller learning rate, and slightly higher subsample/colsample rates). These minimal, targeted changes preserve the existing pipeline while expected to lower the validation RMSE closer to the target.'
- What this solution (achieved 4.99406) has done: 'I add log‑transformed distance features (`log_haversine` and `log_euclidean`) to give the model a smoother view of the distances, and include these new columns in the feature list for both training and test data. This small feature enhancement is expected to lower the validation RMSE, moving the score closer to the target while keeping the core pipeline unchanged.'
- What this solution (achieved 5.10743) has done: 'I filter out rows with invalid passenger counts and replace any infinite values in the feature matrix before training, then apply the same cleaning to the test set. This prevents XGBoost from receiving `inf` values, allowing the model to fit and generate a valid submission CSV. The core feature engineering and model logic remain unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=500_000,
    parse_dates=["pickup_datetime"],
).drop(columns="key")

df = df.dropna()
df.head()




## === cell 2
df.describe()




## === cell 3
plt.figure(figsize=(10, 8))
plt.hist(df["fare_amount"])
plt.title("Fare Distribution")
plt.show()




## === cell 4
print(f"Number of negative fares: {len(df[df['fare_amount'] < 0])}")
print(f"Number of fares equal to 0: {len(df[df['fare_amount'] == 0])}")




## === cell 5
df = df[df["fare_amount"].between(left=2.5, right=df["fare_amount"].max())]




## === cell 6
def ecdf(x):
    x = np.sort(x)
    n = len(x)
    y = np.arange(1, n + 1) / n
    return x, y




## === cell 7
x, y = ecdf(df["fare_amount"])
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.ylabel("Percentile")
plt.xlabel("Fare Amount")
plt.title("Fare Amount ECDF")
plt.show()




## === cell 8
df = df[df["fare_amount"].between(left=2.5, right=70)]




## === cell 9
x, y = ecdf(df["fare_amount"])
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.ylabel("Percentile")
plt.xlabel("Fare Amount")
plt.title("Fare Amount ECDF (capped)")
plt.show()




## === cell 10
df["passenger_count"].value_counts().plot.bar()
plt.title("Passenger Counts")
plt.xlabel("Passengers Numbers")
plt.ylabel("Frequency")
plt.show()




## === cell 11
df = df.loc[df["passenger_count"] < 6]

df = df[df["passenger_count"] > 0]




## === cell 12
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
axes = axes.flatten()

sns.regplot(
    x="pickup_longitude", y="pickup_latitude", data=df, ax=axes[0], fit_reg=False
)
sns.regplot(
    x="dropoff_longitude", y="dropoff_latitude", data=df, ax=axes[1], fit_reg=False
)

axes[0].set_title("Pickup Locations")
axes[1].set_title("Dropoff Locations")
plt.show()




## === cell 13
df["abs_lat_diff"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
df["abs_lon_diff"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
df["lat_diff"] = df["dropoff_latitude"] - df["pickup_latitude"]
df["lon_diff"] = df["dropoff_longitude"] - df["pickup_longitude"]




## === cell 14
sns.lmplot(
    x="abs_lat_diff", y="abs_lon_diff", data=df, fit_reg=False, height=6, aspect=1.2
)
plt.title("Absolute latitude diff vs absolute longitude diff")
plt.show()




## === cell 15
zero_diff = df[(df["abs_lat_diff"] == 0) & (df["abs_lon_diff"] == 0)]
zero_diff.shape




## === cell 16
def minkowski_distance(x1, x2, y1, y2, p):
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1) ** p)) ** (1 / p)




## === cell 17
df["euclidean"] = minkowski_distance(
    df["pickup_longitude"],
    df["dropoff_longitude"],
    df["pickup_latitude"],
    df["dropoff_latitude"],
    2,
)




## === cell 18
plt.figure(figsize=(10, 8))
plt.hist(df["euclidean"], bins=100)
plt.title("Euclidean Distance Distribution")
plt.xlim([0, 500])
plt.show()




## === cell 19
plt.figure(figsize=(10, 6))
for p, grp in df.groupby("passenger_count"):
    sns.kdeplot(grp["fare_amount"], label=f"{p} passengers")
plt.xlabel("Fare Amount")
plt.ylabel("Density")
plt.title("Fare Amount Distribution by Passenger Count")
plt.legend()
plt.show()




## === cell 20
df.groupby("passenger_count")["fare_amount"].agg(["mean", "count"])




## === cell 21
df.groupby("passenger_count")["fare_amount"].mean().plot.bar(color="b")
plt.title("Average Fare by Passenger Count")
plt.show()




## === cell 22
R = 6378  # Earth radius in km


def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = R * c
    return km




## === cell 23
df["haversine"] = haversine_np(
    df["pickup_longitude"],
    df["pickup_latitude"],
    df["dropoff_longitude"],
    df["dropoff_latitude"],
)

df["log_haversine"] = np.log1p(df["haversine"])
df["log_euclidean"] = np.log1p(df["euclidean"])

df["total_distance"] = df["haversine"] + df["euclidean"]
df["log_total_distance"] = np.log1p(df["total_distance"])
df["dist_per_passenger"] = df["haversine"] / df["passenger_count"]
df["log_dist_per_passenger"] = np.log1p(df["dist_per_passenger"])




## === cell 24
sns.kdeplot(df["haversine"])
plt.title("Haversine Distance KDE")
plt.show()




## === cell 25
df["pickup_hour"] = df["pickup_datetime"].dt.hour
df["pickup_weekday"] = df["pickup_datetime"].dt.weekday
df["pickup_month"] = df["pickup_datetime"].dt.month

df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)

df["weekday_sin"] = np.sin(2 * np.pi * df["pickup_weekday"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["pickup_weekday"] / 7)

df["month_sin"] = np.sin(2 * np.pi * df["pickup_month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["pickup_month"] / 12)




## === cell 26
from sklearn.model_selection import train_test_split

features = [
    "haversine",
    "log_haversine",
    "euclidean",
    "log_euclidean",
    "abs_lat_diff",
    "abs_lon_diff",
    "lat_diff",
    "lon_diff",
    "passenger_count",
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "total_distance",
    "log_total_distance",
    "dist_per_passenger",
    "log_dist_per_passenger",
]

X = df[features]
y = df["fare_amount"].values

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

y_log = np.log1p(y)

X_train, X_valid, y_train_log, y_valid_log = train_test_split(
    X, y_log, test_size=0.30, random_state=42
)




## === cell 27
import xgboost as xgb

xgbr = xgb.XGBRegressor(
    n_estimators=3000,
    max_depth=12,
    learning_rate=0.01,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective="reg:squarederror",
    n_jobs=4,
    random_state=42,
    verbosity=0,
)

xgbr.fit(
    X_train,
    y_train_log,
    eval_set=[(X_valid, y_valid_log)],
    early_stopping_rounds=150,
    verbose=False,
)




## === cell 28
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def metrics(train_pred_log, valid_pred_log, y_train_log, y_valid_log):
    """RMSE and MAPE on the original (non‑log) scale."""
    train_pred = np.expm1(train_pred_log)
    valid_pred = np.expm1(valid_pred_log)
    y_train = np.expm1(y_train_log)
    y_valid = np.expm1(y_valid_log)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))

    train_ape = np.abs((y_train - train_pred) / y_train)
    valid_ape = np.abs((y_valid - valid_pred) / y_valid)

    train_ape[~np.isfinite(train_ape)] = 0
    valid_ape[~np.isfinite(valid_ape)] = 0

    train_mape = 100 * np.mean(train_ape)
    valid_mape = 100 * np.mean(valid_ape)

    return train_rmse, valid_rmse, train_mape, valid_mape


def evaluate(model, X_train, X_valid, y_train_log, y_valid_log):
    train_pred_log = model.predict(X_train)
    valid_pred_log = model.predict(X_valid)

    train_rmse, valid_rmse, train_mape, valid_mape = metrics(
        train_pred_log, valid_pred_log, y_train_log, y_valid_log
    )

    print(f"Training:   rmse = {round(train_rmse, 2)} \t mape = {round(train_mape, 2)}")
    print(f"Validation: rmse = {round(valid_rmse, 2)} \t mape = {round(valid_mape, 2)}")




## === cell 29
evaluate(xgbr, X_train, X_valid, y_train_log, y_valid_log)




## === cell 30
test = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)

test_id = test["key"].tolist()
test = test.drop(columns="key")

test["abs_lat_diff"] = (test["dropoff_latitude"] - test["pickup_latitude"]).abs()
test["abs_lon_diff"] = (test["dropoff_longitude"] - test["pickup_longitude"]).abs()
test["lat_diff"] = test["dropoff_latitude"] - test["pickup_latitude"]
test["lon_diff"] = test["dropoff_longitude"] - test["pickup_longitude"]
test["euclidean"] = minkowski_distance(
    test["pickup_longitude"],
    test["dropoff_longitude"],
    test["pickup_latitude"],
    test["dropoff_latitude"],
    2,
)
test["haversine"] = haversine_np(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["dropoff_longitude"],
    test["dropoff_latitude"],
)

test["log_haversine"] = np.log1p(test["haversine"])
test["log_euclidean"] = np.log1p(test["euclidean"])

test["total_distance"] = test["haversine"] + test["euclidean"]
test["log_total_distance"] = np.log1p(test["total_distance"])
test["dist_per_passenger"] = test["haversine"] / test["passenger_count"].replace(
    0, np.nan
)
test["log_dist_per_passenger"] = np.log1p(test["dist_per_passenger"])

test["pickup_hour"] = test["pickup_datetime"].dt.hour
test["pickup_weekday"] = test["pickup_datetime"].dt.weekday
test["pickup_month"] = test["pickup_datetime"].dt.month

test["hour_sin"] = np.sin(2 * np.pi * test["pickup_hour"] / 24)
test["hour_cos"] = np.cos(2 * np.pi * test["pickup_hour"] / 24)

test["weekday_sin"] = np.sin(2 * np.pi * test["pickup_weekday"] / 7)
test["weekday_cos"] = np.cos(2 * np.pi * test["pickup_weekday"] / 7)

test["month_sin"] = np.sin(2 * np.pi * test["pickup_month"] / 12)
test["month_cos"] = np.cos(2 * np.pi * test["pickup_month"] / 12)

test = test.replace([np.inf, -np.inf], np.nan)
test = test.fillna(0)

preds_log = xgbr.predict(test[features])
preds = np.expm1(preds_log)
preds = np.clip(preds, a_min=0, a_max=None)

sub = pd.DataFrame({"key": test_id, "fare_amount": preds})
sub.to_csv("sub_xgb_simple.csv", index=False)

sns.histplot(sub["fare_amount"], bins=100, kde=True)
plt.title("Predicted Fare Amount Distribution")
plt.show()
