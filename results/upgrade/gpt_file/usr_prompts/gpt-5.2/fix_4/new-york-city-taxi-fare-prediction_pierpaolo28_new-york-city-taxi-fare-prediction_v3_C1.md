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

6.88736

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.37712) has done: 'I fix the runtime errors caused by seaborn’s updated API by switching positional arguments in `regplot`/`lmplot` to explicit `x=`/`y=` keywords. I also fix a major modeling bug in the train/validation split where `fare_amount` was mistakenly included as an input feature, causing leakage and harming generalization; the model now train on `df` with `fare_amount` removed, while keeping the same feature set and XGBoost regressor approach. Finally, I make the run deterministic (`random_state`) and ensure the submission is written with a `.csv` suffix and correct columns (`key,fare_amount`). These changes are minimal, preserve the core approach, and should improve RMSE toward your target.'
- What this solution (achieved 5.44725) has done: 'I make two minimal changes aimed at lowering RMSE toward your 4.10469 target without changing the overall XGBoost-regression approach: (1) add a few simple, standard time-based features from `pickup_datetime` (hour/day/month/weekday) that are already present but currently unused, and (2) apply a basic, competition-common geographic outlier filter on lat/lon and distance to remove clearly invalid trips that hurt generalization. I keep the same model type and training loop, only adjusting the feature list and adding the same feature engineering to both train and test to preserve semantics. This should improve score materially from ~5.38 while remaining stable and within Kaggle constraints, and it still writes a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.88736) has done: 'Your current RMSE (5.44725) is worse than the target (4.10469), so we should make small, legitimate improvements without changing the core XGBoost regression approach. The biggest low-risk gain here is to add a couple of standard, competition-common geographic features (Manhattan distance and bearing) derived from the same coordinates you already use; this keeps the same model type/training loop but gives the model more signal. I also make the train/test feature engineering fully symmetric by applying the same geographic filters to test (with safe clipping rather than dropping rows) and by ensuring no NaNs/Infs reach XGBoost, which can otherwise degrade predictions. These changes are minimal, keep evaluation semantics intact, and still write a valid `submission.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
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



## === cell 4
print(f"Number of negative fares: {len(df[df['fare_amount'] < 0])}")
print(f"Number of fares equal to 0: {len(df[df['fare_amount'] == 0])}")



## === cell 5
df = df[df["fare_amount"].between(left=2.5, right=df["fare_amount"].max())]




## === cell 6
def ecdf(x):
    x = np.sort(x)
    n = len(x)
    y = np.arange(1, n + 1, 1) / n
    return x, y




## === cell 7
x, y = ecdf(df["fare_amount"])
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.ylabel("Percentile")
plt.xlabel("Fare Amount")
plt.title("Fare Amount ECDF")



## === cell 8
df = df[df["fare_amount"].between(left=2.5, right=70)]



## === cell 9
x, y = ecdf(df["fare_amount"])
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.ylabel("Percentile")
plt.xlabel("Fare Amount")
plt.title("Fare Amount ECDF")



## === cell 10
df["passenger_count"].value_counts().plot.bar()
plt.title("Passenger Counts")
plt.xlabel("Passengers Numbers")
plt.ylabel("Frequency")



## === cell 11
df = df.loc[df["passenger_count"] < 6]



## === cell 12
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
axes = axes.flatten()

sns.regplot(
    x="pickup_longitude", y="pickup_latitude", fit_reg=False, data=df, ax=axes[0]
)
sns.regplot(
    x="dropoff_longitude", y="dropoff_latitude", fit_reg=False, data=df, ax=axes[1]
)
axes[0].set_title("Pickup Locations")
axes[1].set_title("Dropoff Locations")



## === cell 13
df["abs_lat_diff"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
df["abs_lon_diff"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()



## === cell 14
sns.lmplot(x="abs_lat_diff", y="abs_lon_diff", fit_reg=False, data=df)
plt.title("Absolute latitude difference vs Absolute longitude difference")



## === cell 15
zero_diff = df[(df["abs_lat_diff"] == 0) & (df["abs_lon_diff"] == 0)]
zero_diff.shape




## === cell 16
def minkowski_distance(x1, x2, y1, y2, p):
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)




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
plt.hist(df["euclidean"])
plt.title("Euclidean Distance Distribution")
ax = plt.subplot(111)
ax.set_xlim([0, 500])



## === cell 19
plt.figure(figsize=(10, 6))
for p, grouped in df.groupby("passenger_count"):
    sns.kdeplot(grouped["fare_amount"], label=f"{p} passengers")

plt.xlabel("Fare Amount")
plt.ylabel("Density")
plt.title("Distribution of Fare Amount by Number of Passengers")



## === cell 20
df.groupby("passenger_count")["fare_amount"].agg(["mean", "count"])



## === cell 21
df.groupby("passenger_count")["fare_amount"].mean().plot.bar(color="b")
plt.title("Average Fare by Passenger Count")



## === cell 22
R = 6378


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees)
    All args must be of equal length.
    source: https://stackoverflow.com/a/29546836
    """
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



## === cell 24
sns.kdeplot(df["haversine"])



## === cell 25
corrs = df.corr(numeric_only=True)
corrs["fare_amount"].plot.bar(color="b")
plt.title("Correlation with Fare Amount")




## === cell 26
def bearing_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.degrees(np.arctan2(y, x))
    return (brng + 360.0) % 360.0


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    dt = frame["pickup_datetime"]
    frame["pickup_hour"] = dt.dt.hour.astype(np.int16)
    frame["pickup_day"] = dt.dt.day.astype(np.int16)
    frame["pickup_month"] = dt.dt.month.astype(np.int16)
    frame["pickup_weekday"] = dt.dt.weekday.astype(np.int16)
    return frame


def apply_geo_filters(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[
        frame["pickup_longitude"].between(-75, -72)
        & frame["dropoff_longitude"].between(-75, -72)
        & frame["pickup_latitude"].between(40, 42)
        & frame["dropoff_latitude"].between(40, 42)
    ]
    frame = frame[frame["haversine"].between(0.05, 100)]
    return frame


def add_geo_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame["manhattan"] = frame["abs_lat_diff"] + frame["abs_lon_diff"]
    frame["bearing"] = bearing_np(
        frame["pickup_longitude"],
        frame["pickup_latitude"],
        frame["dropoff_longitude"],
        frame["dropoff_latitude"],
    ).astype(np.float32)
    return frame


df = add_time_features(df)
df = add_geo_features(df)
df = apply_geo_filters(df)

df = df.replace([np.inf, -np.inf], np.nan).dropna()



## === cell 27
test = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)

test["abs_lat_diff"] = (test["dropoff_latitude"] - test["pickup_latitude"]).abs()
test["abs_lon_diff"] = (test["dropoff_longitude"] - test["pickup_longitude"]).abs()

test_id = list(test.pop("key"))

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

test = add_time_features(test)
test = add_geo_features(test)

test["pickup_longitude"] = test["pickup_longitude"].clip(-75, -72)
test["dropoff_longitude"] = test["dropoff_longitude"].clip(-75, -72)
test["pickup_latitude"] = test["pickup_latitude"].clip(40, 42)
test["dropoff_latitude"] = test["dropoff_latitude"].clip(40, 42)
test["haversine"] = test["haversine"].clip(0.05, 100)

test = test.replace([np.inf, -np.inf], np.nan).fillna(0)

test.describe()



## === cell 28
from sklearn.model_selection import train_test_split

y = df["fare_amount"].to_numpy()
X = df.drop(columns=["fare_amount"])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.30, random_state=42
)



## === cell 29
import xgboost as xgb

features = [
    "haversine",
    "manhattan",
    "bearing",
    "abs_lat_diff",
    "abs_lon_diff",
    "passenger_count",
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
]

xgbr = xgb.XGBRegressor(random_state=42)
xgbr.fit(X_train[features], y_train)



## === cell 30
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def metrics(train_pred, valid_pred, y_train, y_valid):
    """Calculate metrics: Root mean squared error and mean absolute percentage error"""
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))

    train_ape = abs((y_train - train_pred) / y_train)
    valid_ape = abs((y_valid - valid_pred) / y_valid)

    train_ape[train_ape == np.inf] = 0
    train_ape[train_ape == -np.inf] = 0
    valid_ape[valid_ape == np.inf] = 0
    valid_ape[valid_ape == -np.inf] = 0

    train_mape = 100 * np.mean(train_ape)
    valid_mape = 100 * np.mean(valid_ape)

    return train_rmse, valid_rmse, train_mape, valid_mape


def evaluate(model, features, X_train, X_valid, y_train, y_valid):
    """Print RMSE and MAPE on train/validation splits"""
    train_pred = model.predict(X_train[features])
    valid_pred = model.predict(X_valid[features])

    train_rmse, valid_rmse, train_mape, valid_mape = metrics(
        train_pred, valid_pred, y_train, y_valid
    )

    print(f"Training:   rmse = {round(train_rmse, 2)} \t mape = {round(train_mape, 2)}")
    print(f"Validation: rmse = {round(valid_rmse, 2)} \t mape = {round(valid_mape, 2)}")




## === cell 31
evaluate(
    xgbr,
    features,
    X_train,
    X_valid,
    y_train,
    y_valid,
)



## === cell 32
preds = xgbr.predict(test[features])

preds = np.clip(preds, 0, None)

sub = pd.DataFrame({"key": test_id, "fare_amount": preds})
sub.to_csv("submission.csv", index=False)

sns.histplot(sub["fare_amount"], kde=True)
plt.title("Distribution of Predicted Fare Amount")
