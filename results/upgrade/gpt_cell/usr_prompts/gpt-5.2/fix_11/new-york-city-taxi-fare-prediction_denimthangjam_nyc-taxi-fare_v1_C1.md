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

5.689

# 6. Current score

988.4304

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'Diagnosis: The crash happens because `LinearRegression(normalize=True)` is no longer a valid argument in scikit-learn 1.2.2; the `normalize` parameter was removed, so passing it raises `TypeError`. The rest of the cell (fit + score) is fine and should remain unchanged. To keep execution unblocked without changing the modeling approach beyond what’s necessary, remove the unsupported keyword argument.

Patch summary: Update the `LinearRegression` constructor call to omit `normalize=True`, keeping the same estimator type and the same subsequent `.fit()` and `.score()` calls.

Updated cells: Only cell 34 is modified.

Compatibility notes for cell k+1: Variable `lr` remains a fitted `LinearRegression` instance with the same interface, so `lr.predict(...)` in cell 35 continue to work unchanged.

Assumptions: No other preprocessing or scaling pipeline is required to run; the goal is to fix the runtime error with minimal change under scikit-learn 1.2.2.'
- What this solution (achieved 752.76803) has done: 'Your very high RMSE is most likely caused by a train/test feature mismatch: `get_dummies()` is applied separately, so if a weekday category is missing in the 10M training sample (or present in test only), the column sets differ and the linear model’s coefficients are applied to the wrong features at inference. I make a minimal change that forces identical one-hot columns by fitting dummies on the concatenated weekday values and then splitting back, preserving the same “one-hot weekday” core feature logic. I also apply your “Difference_*” normalization to test using the train mean/variance (instead of test’s own), which keeps train/test feature scaling consistent and typically reduces error without changing the model type or training loop. Finally, I write the submission as `submission.csv` (lowercase name is conventional) while keeping the required columns and row alignment the same.'
- What this solution (achieved 866.13609) has done: 'Your RMSE is still extremely high for this competition, which strongly suggests the model is producing invalid/implausible fare values (typically from remaining dirty training rows like negative/zero fares or wildly out-of-NYC coordinates). To move the score toward the 5.689 target without changing the model/feature core logic, I add minimal but standard data cleaning filters on the training set (positive fares, passenger_count bounds, NYC-ish coordinate bounds) before feature engineering. I also clip negative predictions to 0.0 (fares can’t be negative), which is a small post-processing step that prevents catastrophic errors on the leaderboard. Everything else (LinearRegression, same engineered features, same training approach) stays the same and it still writes `submission.csv`.'
- What this solution (achieved 866.13609) has done: 'Your RMSE is catastrophically high because the model is effectively being trained on a (nearly) constant target: after you filter `Difference_longitude/latitude < 5`, you never recompute those differences, so most remaining rows have `Difference_*` pegged near 5 and then you “normalize” by variance, creating near-constant features and junk predictions. I make the smallest fix that preserves your exact feature set and LinearRegression approach: after the coordinate/fare cleaning step, recompute `Difference_longitude` and `Difference_latitude` from the raw coordinates, then continue your existing pipeline. I also guard the variance division against zero to avoid blow-ups, without changing semantics when variance is normal. This should move RMSE drastically down toward the 5.689 target while keeping your core logic intact and still producing `submission.csv`.'
- What this solution (achieved 1160.34506) has done: 'Your current RMSE is far above the target, so we should make a small, legitimate change that drastically reduces catastrophic errors without changing your model type or feature set. The biggest remaining risk is that a plain `LinearRegression` is extremely sensitive to any remaining outliers in the 10M-row sample, so a few bad rows can explode coefficients and ruin test predictions. I keep the exact same engineered features and training flow, but switch to a robust linear model (`HuberRegressor`) which is still linear regression (same semantics/prediction interface) and is designed to dampen outlier influence; this typically brings NYC Taxi Fare RMSE down by orders of magnitude versus OLS when data is imperfect. I also keep your existing clipping/rounding and ensure the submission is written with the required `key,fare_amount` columns.'
- What this solution (achieved 1057.24047) has done: 'Your RMSE is far worse than the target, so we need a small change that fixes the most likely remaining source of catastrophic predictions without changing your feature set or training flow. Right now `HuberRegressor` is being fit on raw (unscaled) features where some columns (e.g., “Difference_*” divided by a tiny variance) can become extremely large, which can destabilize optimization and lead to terrible coefficients. The minimal, core-logic-preserving fix is to standardize all numeric features (train/test with the same scaler) immediately before fitting and predicting, while keeping the same model and engineered features. I also ensure the train/test matrices have identical column order before scaling/predicting to avoid any silent misalignment.'
- What this solution (achieved 1132.71743) has done: 'Your RMSE is still catastrophically high (worse than target), so the smallest safe improvement is to remove the prediction-time inconsistencies that can blow up errors: you currently “normalize” `Difference_*` by dividing by variance (not std), then you standardize again, which can create extreme feature magnitudes and instability. I keep your exact features and HuberRegressor approach, but change that step to standard-score the `Difference_*` columns using train mean and **std** (still the same “center + scale” intent, just numerically stable). I also cap `pickuptime` to valid HHMM range and ensure all feature columns are float before scaling to avoid any dtype surprises. Everything else (data filters, engineered features, model, training loop, and submission schema/path) remains unchanged.'
- What this solution (achieved 988.4304) has done: 'Your RMSE is still catastrophic because the model is being fit with a random split (so “future” rides can leak into training) and—more importantly—because `HuberRegressor` can behave poorly with the current `alpha/epsilon` on this feature set, often producing badly biased predictions at inference. To move sharply toward the 5.689 target without changing your feature engineering or overall linear-regression approach, I switch to a plain `Ridge` regressor (still linear, same `fit/predict` flow) and keep the exact same scaling and post-processing. I also change the split to be time-ordered (based on `pickup_datetime`) to stabilize coefficient estimation and avoid leakage, which typically reduces leaderboard RMSE for this competition. Submission format/path stays identical and still write `submission.csv`.'
- What this solution (achieved 988.4304) has done: 'Your RMSE is still catastrophic compared to the 5.689 target, which strongly suggests the model is learning the wrong mapping due to a key bug: you’re extracting the timestamp from `key` (which includes a trailing unique integer), so most parsed datetimes become `NaT`, the time-sort becomes meaningless, and the time-based split is effectively broken. I make the smallest fix that preserves your model/features: parse `pickup_datetime` into a real datetime *before you drop it*, keep it only for sorting/splitting, then drop it from features as you already do. I also clip the training label to the same bounds you already filter (0–300) just as a safety guard against any remaining bad rows sneaking through after `dropna`, without changing the training approach. Everything else (feature engineering, StandardScaler + Ridge fit/predict, submission schema) stays the same.'
- What this solution (achieved 988.4304) has done: 'Your current RMSE is catastrophically worse than the target, so the smallest meaningful fix is to address the most likely remaining root cause: `pickuptime` is being extracted with a brittle string slice that can yield invalid/shifted times and degrade the model heavily. I replace the manual string-slicing loops for `pickuptime` and `Weekday` with a robust `pd.to_datetime` parse from `pickup_datetime` (same information, same features, just correctly computed), while keeping all downstream feature columns (`pickuptime`, weekday one-hot) and the Ridge+scaling training flow unchanged. This should substantially reduce error without changing the model family, loss, or overall approach. The submission schema/path remains identical and still writes `submission.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_data = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=10_000_000
)
train_data.head()



## === cell 2
train_data.shape



## === cell 3
train_data.info()



## === cell 4
test_data = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
test_data.head()



## === cell 5
test_data.info()



## === cell 6
test_data.shape



## === cell 7
train_data.isna().sum()



## === cell 8
train_data.isnull().sum()



## === cell 9
train_data["Difference_longitude"] = np.abs(
    np.asarray(train_data["pickup_longitude"] - train_data["dropoff_longitude"])
)
train_data["Difference_latitude"] = np.abs(
    np.asarray(train_data["pickup_latitude"] - train_data["dropoff_latitude"])
)

test_data["Difference_longitude"] = np.abs(
    np.asarray(test_data["pickup_longitude"] - test_data["dropoff_longitude"])
)
test_data["Difference_latitude"] = np.abs(
    np.asarray(test_data["pickup_latitude"] - test_data["dropoff_latitude"])
)



## === cell 10
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")



## === cell 11
plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## === cell 12
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]



## === cell 13
train_data = train_data[
    (train_data["fare_amount"] > 0)  # fares should be positive
    & (train_data["fare_amount"] < 300)  # remove extreme outliers
    & (train_data["passenger_count"] >= 1)
    & (train_data["passenger_count"] <= 6)
]

train_data = train_data[
    (train_data["pickup_longitude"].between(-75, -72))
    & (train_data["dropoff_longitude"].between(-75, -72))
    & (train_data["pickup_latitude"].between(40, 42))
    & (train_data["dropoff_latitude"].between(40, 42))
].copy()

train_data["Difference_longitude"] = np.abs(
    train_data["pickup_longitude"] - train_data["dropoff_longitude"]
)
train_data["Difference_latitude"] = np.abs(
    train_data["pickup_latitude"] - train_data["dropoff_latitude"]
)

train_data.shape



## === cell 14
train_pickup_ts = pd.to_datetime(
    train_data["pickup_datetime"], errors="coerce", utc=False
)
test_pickup_ts = pd.to_datetime(
    test_data["pickup_datetime"], errors="coerce", utc=False
)

train_data = train_data.loc[train_pickup_ts.notna()].copy()
train_pickup_ts = train_pickup_ts.loc[train_pickup_ts.notna()]

train_data["pickuptime"] = (
    train_pickup_ts.dt.hour * 100 + train_pickup_ts.dt.minute
).astype(np.int32)
test_data["pickuptime"] = (
    test_pickup_ts.dt.hour * 100 + test_pickup_ts.dt.minute
).astype(np.int32)

train_data["Weekday"] = train_pickup_ts.dt.weekday.astype(np.int8)
test_data["Weekday"] = test_pickup_ts.dt.weekday.astype(np.int8)

train_data.head()



## === cell 15
train_data.head()



## === cell 16
train_data.head()



## === cell 17
test_data.head()



## === cell 18
train_data.head()



## === cell 19
train_data["_pickup_ts"] = pd.to_datetime(
    train_data["pickup_datetime"], errors="coerce"
)
train_data = train_data.dropna(subset=["_pickup_ts"]).copy()

train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 20
train_data["Weekday"].replace(
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



## === cell 21
weekday_all = pd.concat(
    [train_data["Weekday"].astype(str), test_data["Weekday"].astype(str)],
    axis=0,
    ignore_index=True,
)
weekday_dummies_all = pd.get_dummies(weekday_all)

train_one_hot = weekday_dummies_all.iloc[: len(train_data)].reset_index(drop=True)
test_one_hot = weekday_dummies_all.iloc[len(train_data) :].reset_index(drop=True)

train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 22
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 23
test_data["pickuptime"] = test_data["pickuptime"].clip(lower=0, upper=2359)
train_data["pickuptime"] = train_data["pickuptime"].clip(lower=0, upper=2359)



## === cell 24
train_data.head()



## === cell 25
R = 6373.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_data["Distance"] = np.asarray(distance) * 0.621

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



## === cell 26
R = 6373.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

lat3 = np.zeros(len(train_data)) + np.radians(40.6413111)
lon3 = np.zeros(len(train_data)) + np.radians(-73.7781391)
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
train_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

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



## === cell 27
train_data["Distance"] = np.round(train_data["Distance"], 2)
train_data["Pickup_Distance_airport"] = np.round(
    train_data["Pickup_Distance_airport"], 2
)
train_data["Dropoff_Distance_airport"] = np.round(
    train_data["Dropoff_Distance_airport"], 2
)
test_data["Distance"] = np.round(test_data["Distance"], 2)
test_data["Pickup_Distance_airport"] = np.round(test_data["Pickup_Distance_airport"], 2)
test_data["Dropoff_Distance_airport"] = np.round(
    test_data["Dropoff_Distance_airport"], 2
)



## === cell 28
train_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 29
diff_lon_mean = np.mean(train_data["Difference_longitude"])
diff_lon_std = np.std(train_data["Difference_longitude"])
diff_lon_std = diff_lon_std if diff_lon_std > 1e-6 else 1e-6

train_data["Difference_longitude"] = train_data["Difference_longitude"] - diff_lon_mean
train_data["Difference_longitude"] = train_data["Difference_longitude"] / diff_lon_std



## === cell 30
diff_lat_mean = np.mean(train_data["Difference_latitude"])
diff_lat_std = np.std(train_data["Difference_latitude"])
diff_lat_std = diff_lat_std if diff_lat_std > 1e-6 else 1e-6

train_data["Difference_latitude"] = train_data["Difference_latitude"] - diff_lat_mean
train_data["Difference_latitude"] = train_data["Difference_latitude"] / diff_lat_std



## === cell 31
test_data["Difference_longitude"] = test_data["Difference_longitude"] - diff_lon_mean
test_data["Difference_longitude"] = test_data["Difference_longitude"] / diff_lon_std

test_data["Difference_latitude"] = test_data["Difference_latitude"] - diff_lat_mean
test_data["Difference_latitude"] = test_data["Difference_latitude"] / diff_lat_std



## === cell 32
train_data.shape



## === cell 33
test_data.shape



## === cell 34
from sklearn.model_selection import train_test_split

train_data["fare_amount"] = train_data["fare_amount"].clip(lower=0.0, upper=300.0)

train_data = train_data.sort_values("_pickup_ts").reset_index(drop=True)

X = train_data.drop(["key", "fare_amount", "_pickup_ts"], axis=1)
y = train_data["fare_amount"]
split_idx = int(len(train_data) * 0.99)
X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()



## === cell 35
from sklearn.preprocessing import StandardScaler

feature_cols = list(X_train.columns)
X_test = X_test[feature_cols]
test_features = test_data.drop("key", axis=1)[feature_cols]

X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
test_features = test_features.astype(np.float64)

scaler = StandardScaler(with_mean=True, with_std=True)
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
test_s = scaler.transform(test_features)



## === cell 36
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0, random_state=80)
ridge.fit(X_train_s, y_train)
print(ridge.score(X_test_s, y_test))



## === cell 37
pred = ridge.predict(test_s)
pred = np.clip(pred, 0.0, None)
pred = np.round(pred, 2)



## === cell 38
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 39
Submission = pd.DataFrame(data=pred, columns=["fare_amount"])
Submission["key"] = test_data["key"]
Submission = Submission[["key", "fare_amount"]]



## === cell 40
Submission.set_index("key", inplace=True)



## === cell 41
Submission.head()



## === cell 42
Submission.to_csv("submission.csv")
