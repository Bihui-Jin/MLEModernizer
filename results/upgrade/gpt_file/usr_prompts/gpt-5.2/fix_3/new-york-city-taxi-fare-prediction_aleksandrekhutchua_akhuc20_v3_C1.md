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

4.6494

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.18792) has done: 'I make a few minimal, score-relevant fixes that should reduce RMSE toward your target by correcting a bug and tightening training stability without changing the overall approach. First, your `haversine` function accidentally swaps latitude/longitude, which creates wrong distances and hurts model quality; fixing that is a direct improvement. Second, your `features2` block mistakenly still uses `features1`, so your “second feature set” experiment isn’t real; I correct it but keep your chosen final model (xgb_model3) as the one used for submission to preserve core logic. Finally, I add deterministic `random_state` for the split and XGBoost, and clip negative predictions to 0 (fares can’t be negative), which usually improves RMSE slightly without changing evaluation semantics.'
- What this solution (achieved 4.6494) has done: 'I keep your overall pipeline and XGBoost approach intact, but make a few score-relevant fixes that typically reduce RMSE for this competition. First, I remove the line that forces all zero-distance trips to have zero fare (that injects label noise because many zero-distance rows in this dataset are data errors or have nonzero fares). Second, I add two standard, minimal geospatial features (absolute lat/lon deltas) computed from existing columns—this preserves the same model class/training loop but gives XGBoost a much easier signal than raw coordinates alone. Finally, I train the final model on all cleaned training data (still the same `xgb_model3` hyperparameters) before predicting test, which usually improves generalization vs using the fold-trained model.'

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
    "ისეთი მონაცემების რიცხვი, რომელთაც ალოგიკური რაოდენობის მგზავრი ჰყავთ გაწერილი:",
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
    lon1, lat1, lon2, lat2 = (
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = np.sin((lat2 - lat1) / 2.0) ** 2 + (
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    df["distance"] = 6371 * 2 * np.arcsin(np.sqrt(a))
    return




## === cell 25
haversine(train_df)
haversine(test_df)



## === cell 26
len(train_df[train_df["distance"] == 0])



## === cell 27
len(test_df[test_df["distance"] == 0])



## === cell 28
_ = None



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



## === cell 38
test_df["year"] = test_df["pickup_datetime"].dt.year
test_df["month"] = test_df["pickup_datetime"].dt.month
test_df["weekday"] = test_df["pickup_datetime"].dt.dayofweek
test_df["hour"] = test_df["pickup_datetime"].dt.hour



## === cell 39
train_df.describe()



## === cell 40
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 41
train_df.describe()



## === cell 42
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



## === cell 43
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



## === cell 44
plt.figure(figsize=(14, 10))
plt.scatter(train_df["distance"], train_df["fare_amount"], alpha=0.2)
plt.xlabel("Distance (km)")
plt.ylabel("Fare Amount ($)")
plt.title("Fare Amount vs. Distance")
plt.grid(True)
plt.show()



## === cell 45
expencives = train_df[train_df["fare_amount"] >= 100]
expencives.describe()
print(
    f"fares that costed more than 100 USD make {len(expencives)/len(train_df) * 100:.4f}% of currently present rows"
)



## === cell 46
train_df = train_df[train_df["fare_amount"] < 100]



## === cell 47
plt.figure(figsize=(14, 10))
plt.scatter(train_df["weekday"], train_df["fare_amount"], alpha=0.4)
plt.xlabel("day of the week")
plt.ylabel("Fare Amount ($)")
plt.title("Fare Amount vs. day of the week")
plt.grid(True)
plt.show()



## === cell 48
plt.figure(figsize=(14, 10))
plt.scatter(train_df["month"], train_df["fare_amount"], alpha=0.4)
plt.xlabel("day of the week")
plt.ylabel("Fare Amount ($)")
plt.title("Fare Amount vs. day of the week")
plt.grid(True)
plt.show()



## === cell 49
plt.figure(figsize=(14, 10))
plt.scatter(train_df["year"], train_df["fare_amount"], alpha=0.3)
plt.xlabel("day of the week")
plt.ylabel("Fare Amount ($)")
plt.title("Fare Amount vs. day of the week")
plt.grid(True)
plt.show()



## === cell 50
plt.figure(figsize=(14, 10))
plt.scatter(train_df["hour"], train_df["fare_amount"], alpha=0.2)
plt.xlabel("hour")
plt.ylabel("Fare Amount ($)")
plt.title("Fare Amount vs. hour")
plt.grid(True)
plt.show()



## === cell 51
average_fare_per_month = train_df.groupby("month")["fare_amount"].mean()

plt.figure(figsize=(10, 6))
average_fare_per_month.plot(kind="bar")
plt.title("Average Fare Amount per Month")
plt.xlabel("Month")
plt.ylabel("Average Fare Amount ($)")
plt.grid(axis="y")
plt.show()



## === cell 52
average_fare_per_hour = train_df.groupby("hour")["fare_amount"].mean()

plt.figure(figsize=(12, 6))
average_fare_per_month.plot(kind="bar")
plt.title("Average Fare Amount per Hour")
plt.xlabel("Hour")
plt.ylabel("Average Fare Amount ($)")
plt.grid(axis="y")
plt.show()



## === cell 53
average_fare_per_year = train_df.groupby("year")["fare_amount"].mean()

plt.figure(figsize=(12, 6))
average_fare_per_month.plot(kind="bar")
plt.title("Average Fare Amount per Year")
plt.xlabel("year")
plt.ylabel("Average Fare Amount ($)")
plt.grid(axis="y")
plt.show()



## === cell 54
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



## === cell 55
train_df.info()



## === cell 56
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
    ]
].corr()

plt.figure(figsize=(10, 10))
sns.heatmap(corrs, annot=True, vmin=-1, vmax=1, fmt=".3f")



## === cell 57
train_df.info()



## === cell 58
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

for df in (train_df, test_df):
    df["abs_lon_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()

features1 = [
    "distance",
    "abs_lon_diff",
    "abs_lat_diff",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "year",
    "month",
    "weekday",
    "hour",
]
target = "fare_amount"

X1 = train_df[features1]
y1 = train_df[target]

X_train1, X_val1, y_train1, y_val1 = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)




## === cell 59
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




## === cell 60
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
test_model(lr_model, X_train1, y_train1, X_val1, y_val1)



## === cell 61
from xgboost import XGBRegressor

xgb_model1 = XGBRegressor(random_state=42, n_jobs=-1)
test_model(xgb_model1, X_train1, y_train1, X_val1, y_val1)



## === cell 62
xgb_model2 = XGBRegressor(
    n_estimators=100, learning_rate=0.2, max_depth=4, random_state=42, n_jobs=-1
)
test_model(xgb_model2, X_train1, y_train1, X_val1, y_val1)



## === cell 63
xgb_model3 = XGBRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=5, n_jobs=-1, random_state=42
)
test_model(xgb_model3, X_train1, y_train1, X_val1, y_val1)



## === cell 64
features2 = ["distance", "pickup_longitude", "pickup_latitude", "year", "month", "hour"]

X2 = train_df[features2]
y2 = train_df[target]

X_train2, X_val2, y_train2, y_val2 = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)



## === cell 65
xgb_model4 = XGBRegressor(
    n_estimators=120,
    learning_rate=0.05,
    max_depth=6,
    n_jobs=-1,
    min_child_weight=3,
    random_state=42,
)
test_model(xgb_model4, X_train2, y_train2, X_val2, y_val2)



## === cell 66
final_model = XGBRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=5, n_jobs=-1, random_state=42
)
final_model.fit(X1, y1)

test_pred = test_df[features1]
pred = final_model.predict(test_pred)

pred = np.clip(pred, 0, None)

submission = pd.DataFrame(
    {"key": test_df["key"], "fare_amount": pred}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
