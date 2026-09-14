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

3.51454

# 6. Current score

5.22106

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.34116) has done: 'I fixed the haversine distance calculation (which was swapping latitude/longitude) and added a small log‑transform model with slightly tuned XGBoost parameters; this improves distance quality and lets the regressor work on a smoother target, which is expected to lower the RMSE toward the target score. The final prediction uses this updated model and writes a proper `submission.csv`.'
- What this solution (achieved 5.29195) has done: 'I keep the overall pipeline unchanged but improve the log‑transformed XGBoost model that generates the final submission. By increasing the number of trees and using a slightly lower learning rate we can obtain a modest gain in validation RMSE, moving the score closer to the target. The only change is in the definition of `log_model` (cell 66); the rest of the script stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 5.30754) has done: 'Implemented a fix for the feature‑mismatch error that prevented model prediction.  
The test data now includes the required **log_distance** column by selecting the full `features1` list (used during training). This change lets both the regular XGB model and the log‑transformed XGB model generate predictions and write a valid `submission.csv`. No other logic was altered, preserving the original pipeline and score‑related behavior.'
- What this solution (achieved 5.30298) has done: 'I add sinusoidal hour features (`sin_hour`, `cos_hour`) to better capture daily patterns, include them in the feature set used for training and prediction, and keep the rest of the pipeline unchanged. This small augmentation should lower the RMSE toward the target while preserving the original model logic.'
- What this solution (achieved 5.22106) has done: 'I keep the existing preprocessing and feature engineering unchanged and only modify the final modeling step. The log‑transformed XGBoost model that is already trained be combined with the earlier XGBRegressor (`xgb_model3`) by averaging their predictions both on the validation split and on the test set. This simple ensemble usually lowers RMSE modestly without altering the core pipeline, moving the score nearer to the target. The submission file is still written to `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import pandas as pd




## === cell 1
train_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=10_000_000
)
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")




## === cell 2
train_df.info()




## === cell 3
test_df.info()




## === cell 4
train_df.describe()




## === cell 5
train_df.isnull().sum()




## === cell 6
test_df.isnull().sum()




## === cell 7
train_df = train_df.dropna()




## === cell 8
len(train_df[train_df["fare_amount"] < 0])




## === cell 9
train_df = train_df[train_df["fare_amount"] >= 0]




## === cell 10
len(train_df[train_df["fare_amount"] == 0])




## === cell 11
train_df[
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
].describe()




## === cell 12
test_df[
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
].describe()




## === cell 13
mask1 = (train_df["pickup_latitude"] < -90) | (train_df["pickup_latitude"] > 90)
mask2 = (train_df["pickup_longitude"] < -180) | (train_df["pickup_longitude"] > 180)
mask3 = (train_df["dropoff_latitude"] < -90) | (train_df["dropoff_latitude"] > 90)
mask4 = (train_df["dropoff_longitude"] < -180) | (train_df["dropoff_longitude"] > 180)
print(
    "number of entries with alogical coordinates:",
    len(train_df[mask1 | mask2 | mask3 | mask4]),
)




## === cell 14
train_df = train_df[~mask1 & ~mask2 & ~mask3 & ~mask4]




## === cell 15
train_df[
    ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
].describe()




## === cell 16
test_df[
    ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
].describe()




## === cell 17
train_df["passenger_count"].describe()




## === cell 18
test_df["passenger_count"].describe()




## === cell 19
mask = (train_df["passenger_count"] <= 0) | (train_df["passenger_count"] > 6)
print(
    "ისეთი მონაცემების რიცხვი, οποίთაც ალოგიკური რაოდენობის მგზავრი ჰყავთ გაწერილი:",
    len(train_df[mask]),
)




## === cell 20
train_df = train_df[~mask]




## === cell 21
train_df.info()




## === cell 22
train_df["fare_amount"].describe()




## === cell 23
import numpy as np




## === cell 24
def haversine(df):
    lon1, lat1 = df["pickup_longitude"], df["pickup_latitude"]
    lon2, lat2 = df["dropoff_longitude"], df["dropoff_latitude"]
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    df["distance"] = 6371 * 2 * np.arcsin(np.sqrt(a))
    return




## === cell 25
haversine(train_df)
haversine(test_df)
train_df["log_distance"] = np.log1p(train_df["distance"])
test_df["log_distance"] = np.log1p(test_df["distance"])




## === cell 26
len(train_df[train_df["distance"] == 0])




## === cell 27
len(test_df[test_df["distance"] == 0])




## === cell 28
train_df.loc[train_df["distance"] == 0, "fare_amount"] = 0




## === cell 29
mask1 = train_df["distance"] > 0
mask2 = train_df["fare_amount"] == 0
len(train_df[mask1 & mask2])




## === cell 30
train_df = train_df[~(mask1 & mask2)]




## === cell 31
nyc_east_long = -73.699710
nyc_west_long = -74.256890
nyc_north_lat = 40.921678
nyc_south_lat = 40.496160




## === cell 32
mask1 = (test_df["pickup_longitude"] <= nyc_east_long) & (
    test_df["pickup_longitude"] >= nyc_west_long
)
mask2 = (test_df["pickup_latitude"] <= nyc_north_lat) & (
    test_df["pickup_latitude"] >= nyc_south_lat
)
mask3 = (test_df["dropoff_longitude"] <= nyc_east_long) & (
    test_df["dropoff_longitude"] >= nyc_west_long
)
mask4 = (test_df["dropoff_latitude"] <= nyc_north_lat) & (
    test_df["dropoff_latitude"] >= nyc_south_lat
)
mask5 = test_df["distance"] != 0
test_df_not_nyc = test_df[~(mask1 & mask2 & mask3 & mask4) & mask5]
test_df_not_nyc.describe()




## === cell 33
tst_east_long = max(
    test_df["pickup_longitude"].max(), test_df["dropoff_longitude"].max()
)
tst_west_long = min(
    test_df["pickup_longitude"].min(), test_df["dropoff_longitude"].min()
)
tst_north_lat = max(test_df["pickup_latitude"].max(), test_df["dropoff_latitude"].max())
tst_south_lat = min(test_df["pickup_latitude"].min(), test_df["dropoff_latitude"].min())
print(tst_east_long, tst_west_long, tst_north_lat, tst_south_lat)




## === cell 34
mask1 = (train_df["pickup_longitude"] <= tst_east_long) & (
    train_df["pickup_longitude"] >= tst_west_long
)
mask2 = (train_df["pickup_latitude"] <= tst_north_lat) & (
    train_df["pickup_latitude"] >= tst_south_lat
)
mask3 = (train_df["dropoff_longitude"] <= tst_east_long) & (
    train_df["dropoff_longitude"] >= tst_west_long
)
mask4 = (train_df["dropoff_latitude"] <= tst_north_lat) & (
    train_df["dropoff_latitude"] >= tst_south_lat
)
mask5 = train_df["distance"] != 0
len(train_df[~(mask1 & mask2 & mask3 & mask4) & mask5])




## === cell 35
train_df = train_df[(mask1 & mask2 & mask3 & mask4) | ~mask5]




## === cell 36
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])




## === cell 37
train_df["year"] = train_df["pickup_datetime"].dt.year
train_df["month"] = train_df["pickup_datetime"].dt.month
train_df["weekday"] = train_df["pickup_datetime"].dt.dayofweek
train_df["hour"] = train_df["pickup_datetime"].dt.hour
train_df["sin_hour"] = np.sin(2 * np.pi * train_df["hour"] / 24)
train_df["cos_hour"] = np.cos(2 * np.pi * train_df["hour"] / 24)

test_df["year"] = test_df["pickup_datetime"].dt.year
test_df["month"] = test_df["pickup_datetime"].dt.month
test_df["weekday"] = test_df["pickup_datetime"].dt.dayofweek
test_df["hour"] = test_df["pickup_datetime"].dt.hour
test_df["sin_hour"] = np.sin(2 * np.pi * test_df["hour"] / 24)
test_df["cos_hour"] = np.cos(2 * np.pi * test_df["hour"] / 24)




## === cell 38
train_df.describe()




## === cell 39
import matplotlib.pyplot as plt
import seaborn as sns




## === cell 40
train_df.describe()




## === cell 41
fig, ax = plt.subplots(2, 1, figsize=(14, 10))

sns.histplot(train_df["fare_amount"], bins=200, ax=ax[0])
ax[0].set_title("Fare Amount Distribution")
ax[0].set_xlabel("Fare Amount ($)")
ax[0].set_ylabel("Frequency")

sns.boxplot(x=train_df["fare_amount"], ax=ax[1])
ax[1].set_title("Box Plot of Fare Amount")
ax[1].set_xlabel("Fare Amount ($)")

plt.tight_layout()
plt.show()




## === cell 42
import math

n = 50
for i in range(math.floor(1000 / n)):
    mask1 = train_df["fare_amount"] >= i * n
    mask2 = train_df["fare_amount"] < (i + 1) * n
    x = train_df[mask1 & mask2]
    if len(x) != 0:
        print(
            "there are", len(x), "entries in the range [", i * n, ",", (i + 1) * n, ")"
        )




## === cell 43
plt.figure(figsize=(14, 10))
plt.scatter(train_df["distance"], train_df["fare_amount"], alpha=0.2)
plt.xlabel("Distance (km)")
plt.ylabel("Fare Amount ($)")
plt.title("Fare Amount vs. Distance")
plt.grid(True)
plt.show()




## === cell 44
expencives = train_df[train_df["fare_amount"] >= 100]
expencives.describe()
print(
    f"fares that costed more than 100 USD make {len(expencives)/len(train_df) * 100:.4f}% of currently present rows"
)




## === cell 45
train_df = train_df[train_df["fare_amount"] < 100]




## === cell 46
plt.figure(figsize=(14, 10))
plt.scatter(train_df["weekday"], train_df["fare_amount"], alpha=0.4)
plt.xlabel("day of the week")
plt.ylabel("Fare Amount ($)")
plt.title("Fare Amount vs. day of the week")
plt.grid(True)
plt.show()




## === cell 47
plt.figure(figsize=(14, 10))
plt.scatter(train_df["month"], train_df["fare_amount"], alpha=0.4)
plt.xlabel("day of the week")
plt.ylabel("Fare Amount ($)")
plt.title("Fare Amount vs. day of the week")
plt.grid(True)
plt.show()




## === cell 48
plt.figure(figsize=(14, 10))
plt.scatter(train_df["year"], train_df["fare_amount"], alpha=0.3)
plt.xlabel("day of the week")
plt.ylabel("Fare Amount ($)")
plt.title("Fare Amount vs. day of the week")
plt.grid(True)
plt.show()




## === cell 49
plt.figure(figsize=(14, 10))
plt.scatter(train_df["hour"], train_df["fare_amount"], alpha=0.2)
plt.xlabel("hour")
plt.ylabel("Fare Amount ($)")
plt.title("Fare Amount vs. hour")
plt.grid(True)
plt.show()




## === cell 50
average_fare_per_month = train_df.groupby("month")["fare_amount"].mean()

plt.figure(figsize=(10, 6))
average_fare_per_month.plot(kind="bar")
plt.title("Average Fare Amount per Month")
plt.xlabel("Month")
plt.ylabel("Average Fare Amount ($)")
plt.grid(axis="y")
plt.show()




## === cell 51
average_fare_per_hour = train_df.groupby("hour")["fare_amount"].mean()

plt.figure(figsize=(12, 6))
average_fare_per_month.plot(kind="bar")
plt.title("Average Fare Amount per Hour")
plt.xlabel("Hour")
plt.ylabel("Average Fare Amount ($)")
plt.grid(axis="y")
plt.show()




## === cell 52
average_fare_per_year = train_df.groupby("year")["fare_amount"].mean()

plt.figure(figsize=(12, 6))
average_fare_per_month.plot(kind="bar")
plt.title("Average Fare Amount per Year")
plt.xlabel("year")
plt.ylabel("Average Fare Amount ($)")
plt.grid(axis="y")
plt.show()




## === cell 53
average_fare_per_passenger_count = train_df.groupby("passenger_count")[
    "fare_amount"
].mean()

plt.figure(figsize=(12, 6))
average_fare_per_month.plot(kind="bar")
plt.title("Average Fare Amount per Hour")
plt.xlabel("Hour")
plt.ylabel("Average Fare Amount ($)")
plt.grid(axis="y")
plt.show()




## === cell 54
train_df.info()




## === cell 55
corrs = train_df[
    [
        "fare_amount",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
        "year",
        "month",
        "weekday",
        "hour",
        "sin_hour",
        "cos_hour",
    ]
].corr()

plt.figure(figsize=(10, 10))
sns.heatmap(corrs, annot=True, vmin=-1, vmax=1, fmt=".3f")




## === cell 56
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

features1 = [
    "distance",
    "log_distance",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "year",
    "month",
    "weekday",
    "hour",
    "sin_hour",
    "cos_hour",
]
target = "fare_amount"

X1 = train_df[features1]
y1 = train_df[target]

X_train1, X_val1, y_train1, y_val1 = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)




## === cell 57
def test_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    y_pred_train = model.predict(X_train)

    mse_val = mean_squared_error(y_val, y_pred_val)
    mse_train = mean_squared_error(y_train, y_pred_train)

    rmse_val = np.sqrt(mse_val)
    rmse_train = np.sqrt(mse_train)
    print(f"{type(model).__name__}: RMSE on validation set: {rmse_val}")
    print(f"{type(model).__name__}: RMSE on train set: {rmse_train}")




## === cell 58
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
test_model(lr_model, X_train1, y_train1, X_val1, y_val1)




## === cell 59
from xgboost import XGBRegressor

xgb_model1 = XGBRegressor(objective="reg:squarederror", n_jobs=-1, random_state=42)
test_model(xgb_model1, X_train1, y_train1, X_val1, y_val1)




## === cell 60
xgb_model2 = XGBRegressor(
    n_estimators=100,
    learning_rate=0.2,
    max_depth=4,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42,
)
test_model(xgb_model2, X_train1, y_train1, X_val1, y_val1)




## === cell 61
xgb_model3 = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    n_jobs=-1,
    objective="reg:squarederror",
    random_state=42,
)
test_model(xgb_model3, X_train1, y_train1, X_val1, y_val1)




## === cell 62
features2 = ["distance", "pickup_longitude", "pickup_latitude", "year", "month", "hour"]

X2 = train_df[features1]  # kept as originally written
y2 = train_df[target]

X_train2, X_val2, y_train2, y_val2 = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)




## === cell 63
xgb_model4 = XGBRegressor(
    n_estimators=120,
    learning_rate=0.05,
    max_depth=6,
    n_jobs=-1,
    min_child_weight=3,
    objective="reg:squarederror",
    random_state=42,
)
test_model(xgb_model4, X_train2, y_train2, X_val2, y_val2)




## === cell 64
test_pred = test_df[features1]  # corrected: matches training features
pred = xgb_model3.predict(test_pred)
submission = pd.DataFrame(
    {"key": test_df["key"], "fare_amount": pred}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)




## === cell 65
y_train_log = np.log1p(y_train1)
y_val_log = np.log1p(y_val1)

log_model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42,
)

log_model.fit(X_train1, y_train_log)

val_pred_log = np.expm1(log_model.predict(X_val1))
val_pred_xgb3 = xgb_model3.predict(X_val1)

val_pred_ens = (val_pred_log + val_pred_xgb3) / 2.0
val_rmse_ens = np.sqrt(mean_squared_error(y_val1, val_pred_ens))
print(f"Ensemble RMSE on validation set: {val_rmse_ens}")

test_pred_log = np.expm1(log_model.predict(test_pred))
test_pred_xgb3 = xgb_model3.predict(test_pred)

test_pred_ens = (test_pred_log + test_pred_xgb3) / 2.0

submission = pd.DataFrame(
    {"key": test_df["key"], "fare_amount": test_pred_ens},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)
