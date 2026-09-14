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

5.57556

# 6. Current score

12.63594

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 762.93433) has done: 'The crash happens because scikit-learn 1.2 removed the `normalize` parameter from `LinearRegression`, so passing `normalize=True` raises a `TypeError`. The smallest safe fix is to instantiate `LinearRegression()` without that argument while keeping the same fit/predict logic and variables (`lr`) used by the next cell. This resolves the error and preserves downstream compatibility (cell 34 still calls `lr.predict(...)`). No other cells need changes.'
- What this solution (achieved 1054.56859) has done: 'Your current score is extremely worse than the target (lower RMSE is better), so we should make a small, legitimate improvement without changing the core model choice (still `LinearRegression`). The biggest issue is that `train_test_split` is random and non-reproducible, and the model is being trained on raw, very noisy targets (including invalid/negative/extreme fares) which strongly degrades generalization and can explode RMSE on Kaggle. I add minimal, standard NYC Taxi cleaning filters (fare > 0 and reasonable upper bound, passenger_count bounds) and basic coordinate bounding to remove obvious corrupt rows, plus align one-hot weekday columns between train/test and fill any missing dummies with 0 to prevent column mismatch. These changes preserve your overall feature engineering and training approach, but should move RMSE dramatically toward your 5.57556 target.'
- What this solution (achieved 15.26361) has done: 'Your RMSE is blowing up mainly because the model is being trained on a cleaned `train_df` but the early features (`abs_diff_longitude/abs_diff_latitude`) were created before cleaning and then never recomputed, so many rows have inconsistent features vs coordinates. Also, your submission currently sets `key` as the index, which can produce an invalid Kaggle file (Kaggle expects `key` as a column), and negative predictions can heavily worsen RMSE. I make three minimal changes that preserve your model/feature logic: recompute travel-vector features after the cleaning filter, clip predictions to a reasonable positive range, and write a submission with `key` as a column (not the index) to `submission.csv`.'
- What this solution (achieved 15.2636) has done: 'Your RMSE (15.26) is far above the target (5.58), so we should make a small, legitimate improvement without changing the core model (still `LinearRegression`) or feature set. The biggest remaining score drag is that the model is trained on unscaled features with very different magnitudes (e.g., `Distance` vs weekday dummies vs `pickuptime`), which makes plain least-squares numerically unstable and can generalize poorly; adding a `StandardScaler` in a `Pipeline` keeps the same linear model while improving conditioning. I also enforce identical column order between train and test before predicting to avoid any subtle mismatch risk. Everything else (data loading, cleaning, feature engineering, split, loss/eval semantics, submission format) stays the same.'
- What this solution (achieved 15.31808) has done: 'Your current RMSE is far above the target (lower is better), so we should make a small, legitimate improvement without changing the model (still LinearRegression in a StandardScaler Pipeline) or the overall feature set. The biggest remaining issue is that plain LinearRegression is very sensitive to remaining outliers; switching to a robust linear estimator (HuberRegressor) keeps the same “linear model” approach and loss semantics for submission but reduces the impact of bad labels, typically improving RMSE a lot on this dataset. I also ensure all feature columns are numeric (your weekday dummies can be bool) and add a minimal “zero-distance must be low fare” cleaning rule that removes common corrupted rows without changing feature engineering. Submission format and paths remain identical and it still writes `submission.csv`.'
- What this solution (achieved 12.63595) has done: 'Your current RMSE (15.318) is far worse than the target (5.576), so we should make a small, legitimate improvement that keeps the same overall linear/robust approach and feature set. The biggest remaining score drag is that the robust model is still being trained on very large-but-noisy data with label/feature outliers; tightening the standard NYC Taxi filters slightly (without changing feature engineering) typically reduces RMSE a lot. I also remove the rounding of engineered distance features (rounding throws away signal) and ensure we train/predict on float32/float64 consistently without changing semantics. Everything still runs end-to-end and writes a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 30.56881) has done: 'Your current RMSE (12.63595) is far above the target (5.57556), so we need a modest, legitimate improvement without changing the overall “robust linear model + scaler” approach. The biggest remaining drag is that the model is trained to predict raw dollars while Kaggle evaluates RMSE; heavy-tailed fares make the regression harder, and Huber still gets pulled by the upper tail. With minimal semantic change, we train Huber on `log1p(fare_amount)` and then inverse-transform predictions with `expm1`, which typically improves RMSE substantially on this competition while keeping the same model family and training loop. We also add one small, standard cleaning rule (remove impossible 0 passenger trips) and keep the same submission format/paths.'
- What this solution (achieved 30.5611) has done: 'Your current RMSE (30.57) is far worse than the target (5.58), so we need a modest but still “same core logic” improvement: the main issue is that training on 10M rows with `HuberRegressor(max_iter=100)` is under-converging (and likely timing out in practice), yielding a weak model. I keep the exact same pipeline (StandardScaler + HuberRegressor on `log1p(fare_amount)` with `expm1` at inference), but train on a small, deterministic subsample of the cleaned data so the optimizer can actually converge within the 600s budget. I also remove prediction rounding (rounding harms RMSE) and increase `max_iter` (same model, just better convergence), which should move RMSE substantially toward your target while preserving the approach and semantics. The submission file format/path remains identical and still writes `submission.csv`.'
- What this solution (achieved 12.63594) has done: 'Your current RMSE is much worse than the target (lower is better), and the biggest likely cause without changing your overall approach is the training target transform: using `log1p(fare)` + `expm1` often *helps*, but here it’s plausibly hurting because your pipeline already has robust loss (Huber) and strong clipping/filters, so the log transform can miscalibrate dollar-level errors. I keep the exact same feature engineering and the same `StandardScaler + HuberRegressor` pipeline, but train Huber on the original `fare_amount` scale and predict directly in dollars (still clipped). I also make the train/test feature column alignment explicit (both sides reindexed to the same `feature_cols` order) to avoid any subtle ordering drift, while keeping the same data paths and submission format.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import os  # reading the input files we have access to

print(os.listdir("../input"))



## === cell 1
train_df = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=10_000_000
)
train_df.dtypes



## === cell 2
test_df = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
test_df.dtypes




## === cell 3
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)



## === cell 4
test_df.head()



## === cell 5
print(train_df.isnull().sum())



## === cell 6
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 7
plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")



## === cell 8
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))




## === cell 9
def get_input_matrix(df):
    return np.column_stack(
        (df.abs_diff_longitude, df.abs_diff_latitude, np.ones(len(df)))
    )


train_X = get_input_matrix(train_df)
train_y = np.array(train_df["fare_amount"])

print(train_X.shape)
print(train_y.shape)



## === cell 10
train_df.head()



## === cell 11
print("Cleaning obvious outliers/corrupt rows...")
old_len = len(train_df)

train_df = train_df[
    (train_df["fare_amount"] > 0)
    & (train_df["fare_amount"] <= 200)
    & (train_df["passenger_count"] >= 1)
    & (train_df["passenger_count"] <= 6)
    & (train_df["pickup_longitude"].between(-74.3, -72.9))
    & (train_df["dropoff_longitude"].between(-74.3, -72.9))
    & (train_df["pickup_latitude"].between(40.5, 41.2))
    & (train_df["dropoff_latitude"].between(40.5, 41.2))
].copy()

print(f"Old size: {old_len}")
print(f"New size: {len(train_df)}")



## === cell 12
add_travel_vector_features(train_df)



## === cell 13
train_df = train_df[
    ~(
        (train_df["abs_diff_longitude"] < 1e-6)
        & (train_df["abs_diff_latitude"] < 1e-6)
        & (train_df["fare_amount"] > 15.0)
    )
].copy()



## === cell 14
ls1 = list(train_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_df["pickuptime"] = ls1

ls1 = list(test_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_df["pickuptime"] = ls1



## === cell 15
train_df.head()



## === cell 16
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



## === cell 17
train_df.head()



## === cell 18
test_df.head()



## === cell 19
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)



## === cell 20
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



## === cell 21
train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_df["Weekday"])
train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)



## === cell 22
weekday_cols = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
for c in weekday_cols:
    if c not in train_df.columns:
        train_df[c] = 0
    if c not in test_df.columns:
        test_df[c] = 0



## === cell 23
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)



## === cell 24
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



## === cell 25
train_df.head()



## === cell 26
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



## === cell 27
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



## === cell 28
train_df["Distance"] = train_df["Distance"].astype(float)
train_df["Pickup_Distance_airport"] = train_df["Pickup_Distance_airport"].astype(float)
train_df["Dropoff_Distance_airport"] = train_df["Dropoff_Distance_airport"].astype(
    float
)
test_df["Distance"] = test_df["Distance"].astype(float)
test_df["Pickup_Distance_airport"] = test_df["Pickup_Distance_airport"].astype(float)
test_df["Dropoff_Distance_airport"] = test_df["Dropoff_Distance_airport"].astype(float)



## === cell 29
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



## === cell 30
train_df.head()



## === cell 31
test_df.head()



## === cell 32
train_df.head()



## === cell 33
from sklearn.model_selection import train_test_split

feature_cols = [c for c in train_df.columns if c not in ["key", "fare_amount"]]

for c in feature_cols:
    train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
    test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

train_df = train_df.dropna(subset=feature_cols + ["fare_amount"]).copy()
test_df[feature_cols] = test_df[feature_cols].fillna(0.0)

X = train_df[feature_cols].astype(float)
y = train_df["fare_amount"].astype(float)

max_train_rows = 2_000_000
if len(X) > max_train_rows:
    keep_idx = X.sample(n=max_train_rows, random_state=42).index
    X = X.loc[keep_idx]
    y = y.loc[keep_idx]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=42
)



## === cell 34
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lr = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=300)),
    ]
)

lr.fit(X_train, y_train)

print(lr.score(X_test, y_test))



## === cell 35
test_X = test_df[feature_cols].astype(float)
test_X = test_X.reindex(columns=X.columns, fill_value=0.0)

pred = lr.predict(test_X)

pred = np.clip(pred, 0.0, 200.0)
pred = pred.astype(float)



## === cell 36
Submission = pd.DataFrame({"key": test_df["key"].values, "fare_amount": pred})



## === cell 37
Submission.head()



## === cell 38
Submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", Submission.shape)
