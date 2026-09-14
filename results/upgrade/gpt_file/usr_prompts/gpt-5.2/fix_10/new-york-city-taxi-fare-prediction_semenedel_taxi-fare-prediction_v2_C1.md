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

5.69073

# 6. Current score

20986800.19266

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.40056) has done: 'I fix the scikit-learn API error by removing the deprecated `normalize=True` argument so the LinearRegression model can be instantiated and trained. I also make the weekday one-hot encoding deterministic between train and test by aligning dummy columns (this prevents column-mismatch bugs and is score-stable). Finally, I ensure the submission is written end-to-end with the exact required columns (`key`, `fare_amount`) and a `.csv` suffix. These changes preserve your core feature engineering and linear regression approach while making the notebook run successfully.'
- What this solution (achieved 751.68646) has done: 'Your score is extremely poor because the feature scaling is inconsistent: you standardize train features using train stats, but you standardize test features using *test* stats (and you even forget to scale `test_df["abs_diff_latitude"]` until the next cell). That distribution mismatch makes the linear regression coefficients apply incorrectly at inference time, exploding RMSE. I make the smallest change that preserves your exact feature set and LinearRegression training: compute normalization parameters (means/vars) on `train_df` only and apply them to both train and test. I also set a fixed `random_state` in the split for stability (doesn’t change core logic) and keep the submission format/path unchanged.'
- What this solution (achieved 949.92244) has done: 'Your RMSE is still catastrophically high because the model is being trained on many invalid/outlier target rows (e.g., negative/huge fares, passenger_count outside expected range, and impossible coordinates), which strongly distorts linear regression coefficients and explodes test predictions. I add minimal, standard NYC Taxi Fare competition filters on the training data only (valid fare range, passenger_count range, and a coarse NYC bounding box) while preserving your exact feature engineering and LinearRegression training. I also add a tiny numerical safeguard to avoid division-by-zero if a variance becomes 0, without changing the intended normalization logic. These changes should move RMSE dramatically downward toward the target band while keeping the same overall approach and producing the same submission format.'
- What this solution (achieved 994.95061) has done: 'Your current RMSE is still far from the target because the “normalization” step is mathematically wrong (you divide by variance instead of standard deviation and also take an absolute value), which badly distorts feature scales and linear regression coefficients. I make the smallest metric-relevant fix by switching that step to standard z-score scaling using train-only mean and **std** (no abs), applied identically to both train and test, while keeping your exact features, LinearRegression model, and overall pipeline unchanged. I also add a tiny post-prediction clamp to keep fares non-negative (a harmless constraint for RMSE on this task that prevents pathological negatives from dominating error). The script still run end-to-end and write `Submission.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 6.96638) has done: 'Your RMSE is still catastrophically high relative to the target, so the most likely issue is that your LinearRegression is being trained with extreme outliers in the *feature space* (especially huge “Distance” / airport-distance values from bad coordinates that survive the coarse bounding box), which makes coefficients explode and ruins generalization. I add one minimal, standard sanity filter that removes rows with implausibly large computed trip/airport distances **after** those features are created, while keeping your exact feature set, LinearRegression training, and inference pipeline unchanged. I also apply the same “abs diff” sanity constraint to the test set (dropping only impossible rows is not allowed because submission must include all keys), so instead I just clip the engineered distance features in test to the maximum seen in filtered training to avoid pathological predictions. These changes are directly metric-relevant (reduce coefficient distortion), preserve your core approach, and still produce a valid `Submission.csv`.'
- What this solution (achieved 6.96638) has done: 'Your current RMSE (6.966) is above the target (5.691), so we should make a small, metric-aligned improvement without changing the model or feature set. The biggest low-risk gain here is fixing inconsistent scaling: you only standardize the two “abs_diff” features, but you also feed unscaled “Distance/airport distances” and “pickuptime”, which can dominate linear regression and hurt generalization. I extend the exact same train-only z-score standardization you already use to the remaining numeric features (pickuptime, passenger_count, Distance, Pickup_Distance_airport, Dropoff_Distance_airport) and apply it identically to test, keeping everything else (filters, rounding, LinearRegression, prediction clip) unchanged. This should move RMSE downward toward the target band with minimal code changes and stable submission formatting.'
- What this solution (achieved 20986800.19266) has done: 'We should move RMSE down (lower is better) from 6.966 toward 5.691 with minimal, metric-relevant changes while keeping your linear regression + engineered features pipeline intact. The biggest low-risk gain is to align the loss/metric with training: train a linear model on `log1p(fare_amount)` and invert with `expm1` at prediction time; this preserves the same model family and feature set but makes the model less dominated by large-fare outliers, which typically reduces RMSE on this competition. I also add a tiny, standard extra fare filter (remove very small “non-real” fares, e.g., < 2.5) to reduce label noise without changing your feature engineering or inference format. Submission writing and column order remain unchanged.'
- What this solution (achieved 20986800.19266) has done: 'Your current RMSE is astronomically worse than the target, which strongly suggests the submission is invalidly aligned (predictions not matching the correct `key` rows) rather than a modeling issue. The smallest, most score-relevant fix is to preserve the original test row order/keys throughout preprocessing and explicitly re-align predictions back onto the original `sample_submission.csv` key order before writing. I also add a deterministic safeguard for any duplicate keys by grouping to a single prediction per key (mean), which avoids Kaggle interpreting duplicate keys unpredictably. These changes keep your exact feature engineering and LinearRegression(log1p) training intact while making the submission semantics correct.'
- What this solution (achieved 20986800.19266) has done: 'Your current RMSE is astronomically worse than the target (lower is better), which is consistent with a submission-format/alignment issue rather than a modeling-quality issue. The smallest, most metric-relevant fix is to stop reordering/aggregating predictions by `key` (the `groupby` + `merge` step can silently scramble alignment and inject NaNs), and instead write predictions in the exact test row order with the exact `key` values from `test.csv`. I keep your exact feature engineering, filtering, scaling, and `LinearRegression(log1p)` training intact, and only change the submission construction to be one-to-one with the test set. I also add a minimal safety check to assert the submission row count matches the test set before writing.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=10_000_000
)
train_df.dtypes



## === cell 2
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
test_df.dtypes



## === cell 3
print("Train DF shape ", train_df.shape)
print("Test DF Shape: ", test_df.shape)




## === cell 4
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)



## === cell 5
test_df.head()



## === cell 6
print(train_df.isnull().sum())



## === cell 7
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 8
print(test_df.isnull().sum())



## === cell 9
print("Old size (pre-sanity-filters): %d" % len(train_df))

train_df = train_df[(train_df["fare_amount"] >= 2.5) & (train_df["fare_amount"] <= 250)]

train_df = train_df[
    (train_df["passenger_count"] >= 1) & (train_df["passenger_count"] <= 6)
]

train_df = train_df[
    (train_df["pickup_longitude"].between(-74.3, -72.9))
    & (train_df["dropoff_longitude"].between(-74.3, -72.9))
    & (train_df["pickup_latitude"].between(40.5, 41.8))
    & (train_df["dropoff_latitude"].between(40.5, 41.8))
]

print("New size (post-sanity-filters): %d" % len(train_df))



## === cell 10
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))




## === cell 11
def get_input_matrix(df):
    return np.column_stack(
        (df.abs_diff_longitude, df.abs_diff_latitude, np.ones(len(df)))
    )


train_X = get_input_matrix(train_df)
train_y = np.array(train_df["fare_amount"])

print(train_X.shape)
print(train_y.shape)



## === cell 12
ls1 = list(train_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_df["pickuptime"] = ls1

ls1 = list(test_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_df["pickuptime"] = ls1



## === cell 13
ls1 = list(train_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
train_df["Weekday"] = ls1

ls1 = list(test_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
test_df["Weekday"] = ls1



## === cell 14
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)



## === cell 15
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
test_df["Weekday"].replace(
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

train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_df["Weekday"])

test_one_hot = test_one_hot.reindex(columns=train_one_hot.columns, fill_value=0)

train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)

train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)

ls1 = list(train_df["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
train_df["pickuptime"] = ls1

ls1 = list(test_df["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
test_df["pickuptime"] = ls1



## === cell 16
train_df.head()



## === cell 17
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

lat1 = np.asarray(np.radians(test_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_df["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_df["Distance"] = np.asarray(distance) * 0.621



## === cell 18
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

lat1 = np.asarray(np.radians(test_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_df["dropoff_longitude"]))

lat3 = np.zeros(len(test_df)) + np.radians(40.6413111)
lon3 = np.zeros(len(test_df)) + np.radians(-73.7781391)
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
test_df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## === cell 19
train_df["Distance"] = np.round(train_df["Distance"], 2)
train_df["Pickup_Distance_airport"] = np.round(train_df["Pickup_Distance_airport"], 2)
train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)
test_df["Distance"] = np.round(test_df["Distance"], 2)
test_df["Pickup_Distance_airport"] = np.round(test_df["Pickup_Distance_airport"], 2)
test_df["Dropoff_Distance_airport"] = np.round(test_df["Dropoff_Distance_airport"], 2)



## === cell 20
print("Old size (pre-distance-sanity): %d" % len(train_df))
train_df = train_df[
    (train_df["Distance"].between(0, 100))
    & (train_df["Pickup_Distance_airport"].between(0, 100))
    & (train_df["Dropoff_Distance_airport"].between(0, 100))
].copy()
print("New size (post-distance-sanity): %d" % len(train_df))

dist_max = float(train_df["Distance"].max())
pda_max = float(train_df["Pickup_Distance_airport"].max())
dda_max = float(train_df["Dropoff_Distance_airport"].max())

test_df["Distance"] = test_df["Distance"].clip(lower=0, upper=dist_max)
test_df["Pickup_Distance_airport"] = test_df["Pickup_Distance_airport"].clip(
    lower=0, upper=pda_max
)
test_df["Dropoff_Distance_airport"] = test_df["Dropoff_Distance_airport"].clip(
    lower=0, upper=dda_max
)



## === cell 21
train_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 22
eps = 1e-12

numeric_cols_to_scale = [
    "abs_diff_longitude",
    "abs_diff_latitude",
    "pickuptime",
    "passenger_count",
    "Distance",
    "Pickup_Distance_airport",
    "Dropoff_Distance_airport",
]

for c in numeric_cols_to_scale:
    train_df[c] = train_df[c].astype("float64")
    test_df[c] = test_df[c].astype("float64")

mu = train_df[numeric_cols_to_scale].mean()
std = train_df[numeric_cols_to_scale].std(ddof=0)
std = std.where(std > 0, eps)

train_df[numeric_cols_to_scale] = (train_df[numeric_cols_to_scale] - mu) / std
test_df[numeric_cols_to_scale] = (test_df[numeric_cols_to_scale] - mu) / std



## === cell 23
from sklearn.model_selection import train_test_split

X = train_df.drop(["key", "fare_amount"], axis=1)
y = np.log1p(train_df["fare_amount"].astype("float64"))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=42
)



## === cell 24
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 25
pred_log = lr.predict(test_df.drop("key", axis=1))
pred = np.expm1(pred_log)
pred = np.clip(pred, 0, None)
pred = np.round(pred, 2)

Submission = pd.DataFrame({"key": test_df["key"].values, "fare_amount": pred})

assert Submission.shape[0] == test_df.shape[0], "Row count mismatch vs test set"
assert Submission["key"].isna().sum() == 0, "NaNs in key column"
assert Submission["fare_amount"].isna().sum() == 0, "NaNs in fare_amount predictions"

Submission.head()



## === cell 26
Submission.to_csv("Submission.csv", index=False)
print("Wrote submission to Submission.csv with shape:", Submission.shape)
print("Submission columns:", Submission.columns.tolist())
print("Any NaNs in fare_amount:", Submission["fare_amount"].isna().any())
