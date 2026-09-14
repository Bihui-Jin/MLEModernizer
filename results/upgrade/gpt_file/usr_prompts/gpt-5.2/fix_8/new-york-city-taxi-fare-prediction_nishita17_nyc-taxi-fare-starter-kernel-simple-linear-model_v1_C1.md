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

3.9

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

5.34463

# 6. Current score

1266.36145

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 872.47979) has done: 'I fix the immediate runtime error caused by scikit-learn removing `normalize=` from `LinearRegression`, while keeping the same linear regression core logic by moving normalization into an explicit `StandardScaler` pipeline. I also ensure the train/test one-hot weekday columns are aligned (so prediction can’t fail or silently mismatch columns) and add a small fare cleanup step to drop non-positive/outlier fares that otherwise hurt RMSE. Finally, I make sure the submission is written with the required `key,fare_amount` columns to a `.csv` file in the working directory.'
- What this solution (achieved 793.00212) has done: 'Your RMSE is extremely high because the feature scaling/normalization is inconsistent between train and test: you manually normalize `abs_diff_longitude` using the *test set’s own mean/variance* instead of using the training statistics, which breaks the relationship the linear model learned. I make a minimal fix by computing the mean/variance of `abs_diff_longitude` on the training data once, then applying the same transform to both train and test (core model/training stays identical). I also guard against division-by-zero in case the variance is tiny, and keep the rest of your pipeline unchanged so behavior is stable while the score moves sharply toward the target. The submission writing remains the same and still produces `Submission.csv` with `key,fare_amount`.'
- What this solution (achieved 761.82938) has done: 'Your current RMSE is still far from the target, so we need a small but meaningful correction that improves generalization without changing the core “linear regression on engineered features” approach. The biggest remaining issue is your manual transform on `abs_diff_longitude`: it uses variance (not standard deviation) and applies an absolute deviation, which badly distorts a key feature and can make predictions explode. I replace that single-column custom normalization with a standard z-score computed from the training data (mean/std) and applied consistently to train and test; this keeps the model and training loop identical but fixes the feature scale. I also clip negative predictions to 0 (fares can’t be negative) to reduce RMSE tail harm while preserving the same regression model.'
- What this solution (achieved 761.83695) has done: 'Your current RMSE is still far above the target, and the biggest remaining cause is that several engineered numeric features are on wildly different scales (especially the airport-distance and haversine distance), while you only standardized `abs_diff_longitude` manually and then standardized everything again in a pipeline—this partial/manual scaling plus rounding is unnecessarily distorting the linear fit. I make a minimal, metric-aligned change by removing the single-column manual normalization (cell 31) and letting the existing `StandardScaler` in your pipeline handle consistent scaling for *all* features using training statistics. I also remove the rounding of the engineered distances (cell 29) and of predictions (cell 36) to avoid injecting quantization error that hurts RMSE. Core logic remains the same: same features, same LinearRegression+StandardScaler pipeline, same train/test split, and same submission format.'
- What this solution (achieved 1242.35165) has done: 'Your current RMSE is catastrophically high for this competition, which usually indicates broken/insufficient feature signal rather than a “tuning” issue. The smallest change that preserves your core “LinearRegression on engineered features” approach but should move sharply toward the target is to stop throwing away the raw lat/long coordinates (they’re essential for NYC taxi fares) and keep them alongside your existing distance/weekday/time features. I also add a minimal coordinate validity filter (NYC-ish bounds) before training, because out-of-range points are common noise in this dataset and heavily inflate RMSE for linear models. Everything else (feature engineering, scaling pipeline, model, train/test split, submission format) stays the same.'
- What this solution (achieved 1267.40082) has done: 'Your current RMSE is far worse than the target, and the most likely cause (given the feature set is reasonable) is that training is being dominated by remaining bad/outlier rows and the linear model is over-extrapolating. I make the smallest changes that keep your exact core approach (same engineered features + StandardScaler + LinearRegression) but tighten data validity filters to remove obvious coordinate/passenger/time anomalies and extreme distances that inflate RMSE. I also ensure the feature columns are strictly numeric before fitting/predicting, preventing silent object-dtype issues from the string-based time extraction. The submission format and file writing remain unchanged.'
- What this solution (achieved 1266.36145) has done: 'Your RMSE is still extremely high, which strongly suggests a train/test feature mismatch or a feature that is harming the linear fit rather than helping it. The most likely culprit here is the weekday one-hot encoding: train and test can end up with the same named columns but in a different order, and your later `reindex(columns=X.columns)` then silently rearranges/zeros them in a way that can degrade predictions; we force a fixed weekday category order before `get_dummies` so those columns are consistent and stable. Next, we make a minimal but very impactful metric-aligned fix by removing the raw `key`-derived string slicing loops and instead parse `pickup_datetime` once with pandas to produce numeric `hour` and `minute` features (replacing only the *implementation* of the same time-based signal, not changing the modeling approach). Finally, we ensure the exact same numeric cleaning is applied to both train and test feature matrices (inf/NaN handling), avoiding train-only row drops that can skew scaling/fit.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import os  # reading the input files we have access to
import matplotlib.pyplot as plt
import seaborn as sns
import time

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"

train_df = pd.read_csv(TRAIN_PATH, nrows=10_000_000)
train_df.head()



## === cell 2
train_df.shape



## === cell 3
train_df.info()



## === cell 4
test_data = pd.read_csv(TEST_PATH)
test_data.head()



## === cell 5
test_data.info()



## === cell 6
train_df.isna().sum()




## === cell 7
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_data)



## === cell 8
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 9
_ = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")



## === cell 10
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))



## === cell 11
train_df = train_df[
    (train_df["fare_amount"] > 0) & (train_df["fare_amount"] <= 300)
].copy()



## === cell 12
train_df = train_df.loc[train_df["passenger_count"].between(1, 6)].copy()
test_data["passenger_count"] = test_data["passenger_count"].clip(lower=1, upper=6)



## === cell 13
coord_mask = (
    train_df["pickup_longitude"].between(-74.5, -72.8)
    & train_df["dropoff_longitude"].between(-74.5, -72.8)
    & train_df["pickup_latitude"].between(40.4, 41.9)
    & train_df["dropoff_latitude"].between(40.4, 41.9)
)
print("Old size (pre-NYC-bounds): %d" % len(train_df))
train_df = train_df.loc[coord_mask].copy()
print("New size (post-NYC-bounds): %d" % len(train_df))



## === cell 14
train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"], errors="coerce", utc=True
)
test_data["pickup_datetime"] = pd.to_datetime(
    test_data["pickup_datetime"], errors="coerce", utc=True
)

train_df["hour"] = train_df["pickup_datetime"].dt.hour.astype("float32")
train_df["minute"] = train_df["pickup_datetime"].dt.minute.astype("float32")

test_data["hour"] = test_data["pickup_datetime"].dt.hour.astype("float32")
test_data["minute"] = test_data["pickup_datetime"].dt.minute.astype("float32")

train_df["pickuptime"] = (train_df["hour"] * 100.0 + train_df["minute"]).astype(
    "float32"
)
test_data["pickuptime"] = (test_data["hour"] * 100.0 + test_data["minute"]).astype(
    "float32"
)



## === cell 15
train_df.head()



## === cell 16
test_data.head()



## === cell 17
train_df["Weekday"] = train_df["pickup_datetime"].dt.weekday
test_data["Weekday"] = test_data["pickup_datetime"].dt.weekday



## === cell 18
train_df.head()



## === cell 19
test_data.head()



## === cell 20
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 21
train_df["Weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)
test_data["Weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)



## === cell 22
weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
train_df["Weekday"] = pd.Categorical(train_df["Weekday"], categories=weekday_order)
test_data["Weekday"] = pd.Categorical(test_data["Weekday"], categories=weekday_order)

train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])

train_df = pd.concat([train_df, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 23
train_df.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 24
dummy_cols = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
for c in dummy_cols:
    if c not in train_df.columns:
        train_df[c] = 0
    if c not in test_data.columns:
        test_data[c] = 0



## === cell 25
train_df = train_df.loc[train_df["pickuptime"].between(0, 2359)].copy()
test_data["pickuptime"] = test_data["pickuptime"].clip(lower=0, upper=2359)



## === cell 26
train_df.head()



## === cell 27
R = 6373.0
lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_df["Distance"] = np.asarray(distance) * 0.621

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data["Distance"] = np.asarray(distance) * 0.621



## === cell 28
test_data.head()



## === cell 29
R = 6373.0
lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

lat3 = np.zeros(len(train_df)) + np.radians(40.6413111)
lon3 = np.zeros(len(train_df)) + np.radians(-73.7781391)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff = lon3 - lon2
d_lat_dropoff = lat3 - lat2
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
train_df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

lat3 = np.zeros(len(test_data)) + np.radians(40.6413111)
lon3 = np.zeros(len(test_data)) + np.radians(-73.7781391)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff = lon3 - lon2
d_lat_dropoff = lat3 - lat2
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
test_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## === cell 30
test_data.head()



## === cell 31
print("Old size (pre-distance-filter): %d" % len(train_df))
train_df = train_df.loc[train_df["Distance"].between(0.0, 100.0)].copy()
print("New size (post-distance-filter): %d" % len(train_df))



## === cell 32
pass



## === cell 33
train_df.shape



## === cell 34
test_data.shape



## === cell 35
from sklearn.model_selection import train_test_split

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 36
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lr = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LinearRegression()),
    ]
)

lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 37
X_testdata = test_data.drop("key", axis=1)

X_testdata = X_testdata.reindex(columns=X.columns, fill_value=0)

X_testdata = X_testdata.apply(pd.to_numeric, errors="coerce")
X_testdata = X_testdata.replace([np.inf, -np.inf], np.nan).fillna(0.0)

pred = lr.predict(X_testdata)
pred = np.clip(pred, 0.0, None)



## === cell 38
pd.read_csv(SAMPLE_SUB_PATH).head()



## === cell 39
Submission = pd.DataFrame(data=pred, columns=["fare_amount"])
Submission["key"] = test_data["key"]
Submission = Submission[["key", "fare_amount"]]



## === cell 40
Submission.head()



## === cell 41
Submission.to_csv("Submission.csv", index=False)
print("Wrote Submission.csv with shape:", Submission.shape)
print(Submission.head())
