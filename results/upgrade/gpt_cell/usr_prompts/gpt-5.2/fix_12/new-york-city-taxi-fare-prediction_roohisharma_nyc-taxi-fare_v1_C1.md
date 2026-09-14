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

5.68923

# 6. Current score

14.89689

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'Diagnosis: The crash happens because scikit-learn 1.2 removed the `normalize` argument from `sklearn.linear_model.LinearRegression`, so `LinearRegression(normalize=True)` raises a `TypeError`. The rest of the pipeline expects a fitted `lr` model object with the same `.fit/.predict/.score` interface. Since normalization is not available via that parameter anymore, the minimal fix is to instantiate `LinearRegression()` without `normalize` so the code runs and produces predictions as expected.

Patch summary: In cell 24, remove the unsupported `normalize=True` keyword when creating the `LinearRegression` instance. Keep training and scoring logic unchanged so downstream code (cell 25) continues to use the same `lr` variable.

Updated cells: Only cell 24 is changed.

Compatibility notes for cell k+1: `lr` remains a trained `LinearRegression` model with `.predict`, so `pred = np.round(lr.predict(...), 2)` in cell 25 work unchanged.

Assumptions: No other preprocessing/scaling steps are required elsewhere; the primary goal is to unblock execution under scikit-learn 1.2.2 without changing the model type or training flow.'
- What this solution (achieved 941.33069) has done: 'Your huge RMSE indicates the model is producing wildly wrong fares, which commonly happens here because the feature preprocessing is inconsistent between train and test (you normalize each dataset using its own mean/variance) and because NaNs/invalid coordinates in the training set can distort the fit. To move the score toward the target with minimal logic change, I (1) compute scaling parameters (mean/variance) on the training data once and apply them to both train and test, and (2) add a lightweight, standard NYC taxi cleanup filter (keep realistic lat/lon bounds and fare range) to remove extreme outliers that dominate linear regression error. I also ensure one-hot weekday columns are aligned between train/test so the feature matrix columns match exactly. The model (LinearRegression) and the overall feature set remain the same; we’re just making preprocessing consistent and removing obviously bad rows.'
- What this solution (achieved 941.33069) has done: 'Your RMSE is catastrophically high for this competition, which strongly suggests a submission alignment/format issue rather than “just” model quality. The smallest score-moving fix is to write the submission exactly in the required schema: keep `key` as a normal column (not the index), preserve the exact row order from `test.csv`, and use the standard lowercase filename `submission.csv` (Kaggle accepts any name, but this avoids accidental mishandling). I also add a minimal safety step to ensure train/test feature columns are identical and in the same order at prediction time (a common silent bug when one-hot columns or concatenations differ). Core model/training/feature logic remains unchanged.'
- What this solution (achieved 865.03537) has done: 'Your RMSE is so far from the target that it’s almost certainly dominated by a feature bug rather than model capacity. The biggest issue here is the “normalization” in cell 23: you divide by variance instead of standard deviation, shrinking those features by ~1/σ and badly distorting the linear model; switching to std is a minimal, semantics-preserving fix that typically improves RMSE dramatically. I also add a tiny safety clip to keep predicted fares non-negative (prevents rare negative predictions from blowing up RMSE) while preserving the same LinearRegression training and the same feature set. Everything else (filters, feature engineering, train/test split, model) stays the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved 990.63985) has done: 'Your current RMSE (865) is so far from the target (5.69) that the dominant issue is almost certainly a severe feature bug rather than model capacity. The biggest minimal-impact fix is to remove the `np.abs(...)` inside your normalization step: taking absolute value after mean-centering destroys sign information and warps the linear relationship, which can massively degrade linear regression performance. I keep your exact model (LinearRegression), features, and training flow, but change normalization to standard z-scoring (center then divide by std) using train-derived statistics applied to both train and test (still consistent with your approach). Everything else remains the same and it still write a valid `submission.csv`.'
- What this solution (achieved 990.63985) has done: 'Your RMSE is still orders of magnitude off, which most often means the feature matrix reaching `LinearRegression` contains non-numeric/object columns or mis-parsed time features that silently explode the fit. I make two minimal fixes that preserve your exact model and feature engineering intent: (1) replace the slow string-slicing loops for `pickup_datetime` with vectorized `pd.to_datetime` extraction of hour/minute and weekday (same semantics, far fewer parse mistakes), and (2) force all model features to numeric (dropping/repairing any unexpected object columns) and align train/test columns deterministically before fitting and predicting. These changes should move RMSE sharply down toward the target without changing the learning algorithm, loss, or feature set conceptually. The script still write a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 990.95508) has done: 'Your RMSE is still catastrophically high, which most strongly suggests either a train/test feature mismatch at predict-time or numerical blow-ups from a few remaining outliers that dominate linear regression. I keep your exact LinearRegression approach and feature set, but make the preprocessing consistent by z-scoring *all numeric feature columns* using train-derived mean/std and applying the same transform to test (your current code only scales the two “Difference_*” columns). I also add a very small, standard additional cleanup to remove rows with implausibly large computed trip distances (still within your existing “remove obvious outliers” philosophy) and then re-align columns deterministically before fitting/predicting. These are minimal changes that should move RMSE sharply down toward the target without changing the model type or training flow.'
- What this solution (achieved 15.25839) has done: 'Your RMSE is still wildly off-target, which strongly suggests the submission is numerically mis-scaled (features z-scored, but the learned linear model’s intercept/coefficients are being applied in a way that yields unrealistic fare magnitudes for many rows). To move the score sharply toward the target with minimal semantic change, I keep your exact features, filters, and LinearRegression, but fit with `fit_intercept=False` after explicitly adding a constant bias column that is scaled consistently (so the intercept is learned in the same standardized space as the other features). I also add a very small, competition-standard post-processing step to clip extreme predictions to a plausible range (0–250), which prevents a few extreme rows from dominating RMSE without changing the model type or training loop. Finally, I ensure the exact same column set/order (including the bias) is used for train, validation, and submission.'
- What this solution (achieved 17.71515) has done: 'Your current RMSE (15.26) is still far above the target (5.69), so we should make a small, legitimate improvement that reduces large errors without changing your core LinearRegression setup or feature set. The single biggest remaining issue is that you z-score the feature columns and then add an unscaled `bias` column; this makes the constant term live on a different scale than the other features and can destabilize coefficient magnitudes. I keep `fit_intercept=False` and the explicit bias column (same training approach), but include `bias` in the same train-derived standardization as all other features, so the model learns in a consistent space. I also ensure test feature scaling uses the exact same column list as train (including the bias) to avoid any subtle alignment gaps.'
- What this solution (achieved 14.89689) has done: 'Your RMSE is still far from the target, so we make the smallest changes that directly address likely remaining error drivers without changing your model type or feature set. First, we stop taking absolute values for the coordinate deltas (keeping directional information usually helps linear regression) while keeping the same delta features. Second, we remove the manual scaling/bias trick and simply let `LinearRegression` learn its intercept normally (`fit_intercept=True`), while keeping the exact same features and standardization approach (train-derived z-score applied to both train/test). Finally, we keep your submission alignment safeguards and still write a valid `submission.csv`.'
- What this solution (achieved 14.89689) has done: 'Your RMSE is still much higher than the target, so the most likely remaining issue is a subtle train/test preprocessing mismatch rather than model capacity. I make two minimal, score-relevant fixes while keeping the exact same features and the same `LinearRegression` training flow: (1) compute the one-hot weekday columns with a fixed Monday–Sunday category list so train/test always have identical dummy columns (no reliance on observed categories), and (2) ensure the exact same `feature_cols` list (order and membership) is used for both train scaling and test scaling (right now test uses `test_feature_cols`, which can silently differ). These are small consistency fixes that typically reduce large generalization errors without changing the model or adding new features.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import os  # reading the input files we have access to

print(os.listdir("../input"))



## === cell 1
train_data = pd.read_csv("../input/train.csv", nrows=10_000_000)
train_data.dtypes



## === cell 2
test_data = pd.read_csv("../input/test.csv")
test_data.head()



## === cell 3
train_data["Difference_longitude"] = np.asarray(
    train_data["pickup_longitude"] - train_data["dropoff_longitude"]
)
train_data["Difference_latitude"] = np.asarray(
    train_data["pickup_latitude"] - train_data["dropoff_latitude"]
)

test_data["Difference_longitude"] = np.asarray(
    test_data["pickup_longitude"] - test_data["dropoff_longitude"]
)
test_data["Difference_latitude"] = np.asarray(
    test_data["pickup_latitude"] - test_data["dropoff_latitude"]
)



## === cell 4
print(train_data.isnull().sum())



## === cell 5
print("Old size: %d" % len(train_data))
train_data = train_data.dropna(how="any", axis="rows")
print("New size: %d" % len(train_data))



## === cell 6
plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## === cell 7
print("Old size: %d" % len(train_data))
train_data = train_data[
    (train_data["Difference_longitude"].abs() < 5.0)
    & (train_data["Difference_latitude"].abs() < 5.0)
]
print("New size: %d" % len(train_data))



## === cell 8
print("Old size (before geo/fare filter): %d" % len(train_data))
train_data = train_data[
    (train_data["fare_amount"] > 0)
    & (train_data["fare_amount"] <= 250)
    & (train_data["passenger_count"] >= 1)
    & (train_data["passenger_count"] <= 6)
    & (train_data["pickup_longitude"].between(-75, -72))
    & (train_data["dropoff_longitude"].between(-75, -72))
    & (train_data["pickup_latitude"].between(40, 42))
    & (train_data["dropoff_latitude"].between(40, 42))
]
print("New size (after geo/fare filter): %d" % len(train_data))



## === cell 9
train_dt = pd.to_datetime(train_data["pickup_datetime"], errors="coerce", utc=True)
test_dt = pd.to_datetime(test_data["pickup_datetime"], errors="coerce", utc=True)

train_data = train_data.loc[train_dt.notna()].copy()
train_dt = train_dt.loc[train_dt.notna()]

train_data["pickuptime"] = (
    train_dt.dt.hour.astype(np.int16) * 100 + train_dt.dt.minute.astype(np.int16)
).astype(np.int16)
test_data["pickuptime"] = (
    test_dt.dt.hour.fillna(0).astype(np.int16) * 100
    + test_dt.dt.minute.fillna(0).astype(np.int16)
).astype(np.int16)

train_data["Weekday"] = train_dt.dt.weekday.astype(np.int8)
test_data["Weekday"] = test_dt.dt.weekday.fillna(0).astype(np.int8)



## === cell 10
train_data.head()



## === cell 11
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 12
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



## === cell 13
train_data.head()



## === cell 14
weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
train_data["Weekday"] = pd.Categorical(train_data["Weekday"], categories=weekday_order)
test_data["Weekday"] = pd.Categorical(test_data["Weekday"], categories=weekday_order)

train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])

train_one_hot, test_one_hot = train_one_hot.align(
    test_one_hot, join="outer", axis=1, fill_value=0
)

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 15
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 16
R = 6373.0  # radius of earth
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



## === cell 17
R = 6373.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

lat3 = np.zeros(len(train_data)) + np.radians(40.6413111)  # latitude of jfk airport
lon3 = np.zeros(len(train_data)) + np.radians(-73.7781391)  # longitude of jfk airport
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



## === cell 18
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



## === cell 19
print("Old size (before distance filter): %d" % len(train_data))
train_data = train_data[(train_data["Distance"] >= 0) & (train_data["Distance"] <= 100)]
print("New size (after distance filter): %d" % len(train_data))



## === cell 20
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



## === cell 21
non_feature_cols = {"key", "fare_amount"}
feature_cols = [c for c in train_data.columns if c not in non_feature_cols]

for c in feature_cols:
    train_data[c] = pd.to_numeric(train_data[c], errors="coerce")
for c in [c for c in test_data.columns if c != "key"]:
    test_data[c] = pd.to_numeric(test_data[c], errors="coerce")

train_data = (
    train_data.replace([np.inf, -np.inf], np.nan)
    .dropna(subset=feature_cols + ["fare_amount"])
    .copy()
)
test_data = test_data.replace([np.inf, -np.inf], np.nan).copy()

test_data = test_data.reindex(columns=["key"] + feature_cols, fill_value=0.0)

mu = train_data[feature_cols].mean()
sigma = train_data[feature_cols].std().replace(0.0, 1.0)

train_data.loc[:, feature_cols] = (train_data[feature_cols] - mu) / sigma
test_data.loc[:, feature_cols] = (test_data[feature_cols] - mu) / sigma
test_data.loc[:, feature_cols] = test_data.loc[:, feature_cols].fillna(0.0)



## === cell 22
from sklearn.model_selection import train_test_split

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]

X = X.reindex(sorted(X.columns), axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 23
from sklearn.linear_model import LinearRegression

lr = LinearRegression(fit_intercept=True)
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 24
X_test_submit = test_data.drop("key", axis=1)
X_test_submit = X_test_submit.reindex(columns=X.columns, fill_value=0.0)

pred = lr.predict(X_test_submit)

pred = np.clip(pred, 0.0, 250.0)
pred = np.round(pred, 2)



## === cell 25
Submission = pd.DataFrame({"key": test_data["key"], "fare_amount": pred})

Submission = Submission[["key", "fare_amount"]]
assert len(Submission) == len(test_data), "Submission row count must match test.csv"



## === cell 26
Submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", Submission.shape)
