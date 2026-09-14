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

5.68914

# 6. Current score

6.90379

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'Diagnosis: The crash happens because scikit-learn 1.2.2 removed the `normalize` parameter from `sklearn.linear_model.LinearRegression`, so passing `normalize=True` raises `TypeError`. The rest of the cell’s logic (fit + score) is fine, and downstream cell 31 expects a fitted `lr` object with `.predict()` available.  
Patch summary: Remove the unsupported `normalize=True` argument and instantiate `LinearRegression()` with default parameters to restore compatibility without changing the training/evaluation flow.  
Updated cells: Only cell 30 is modified.  
Compatibility notes for cell k+1: `lr` remains a trained `LinearRegression` model; `.predict()` usage in cell 31 is unchanged and work.  
Assumptions: No other preprocessing pipeline is required for this fix; keeping defaults preserves the notebook’s intended flow while unblocking execution.'
- What this solution (achieved 936.92806) has done: 'Your score is extremely far from the target (lower is better), which strongly suggests the submission predictions are misaligned with the `key` order (indexing/sorting mismatch) or contain invalid values rather than the model being fundamentally weak. I make two minimal, core-logic-preserving fixes: (1) keep `key` as a normal column (do not set it as index) and write the CSV in the exact expected schema/order, and (2) ensure train/test one-hot weekday columns are aligned and numeric, preventing silent column-order differences that can explode errors. These changes do not alter the model type or training loop; they mainly fix submission correctness and feature alignment to move RMSE drastically down toward the target band. The output be a valid `submission.csv` in the working directory.'
- What this solution (achieved 752.76803) has done: 'Your current RMSE (936) is far worse than the target, so we should make a small correctness fix that reduces error without changing the model/training core. The biggest issue is your train/test scaling: you normalize test features using the *test mean/variance* (cells 26) while training uses the *train mean/variance* (cells 24–25), which makes the model see differently-scaled distributions at inference and can explode RMSE. I minimally change cells 24–26 to compute normalization statistics on the training set once and apply those same stats to both train and test (keeping the same “abs(center)/var” logic). I also add a tiny numeric safety guard to prevent division-by-zero if variance is extremely small, preserving semantics while stabilizing predictions.'
- What this solution (achieved 798.40694) has done: 'Your RMSE is so far from the target that the most likely remaining issue is data quality rather than model capacity: the NYC Taxi Fare dataset contains many invalid/garbage rows (negative/huge fares, out-of-range coordinates, zero-distance rides) that severely distort a linear model. I add minimal, standard “sanity filters” on the training data (fare range, passenger_count range, coordinate bounds, and non-zero trip distance) while keeping your exact feature engineering and LinearRegression training unchanged. This should move the score dramatically downward (better) toward the target band without changing the core approach. I also add a tiny safety clip on predictions to avoid negative fares, which often improves RMSE slightly and never breaks submission validity.'
- What this solution (achieved 1209.86905) has done: 'Your RMSE is still orders of magnitude above the target, which strongly indicates a remaining correctness issue in feature preprocessing rather than model capacity. The smallest impactful fix is to apply the same normalization logic to the other continuous engineered features (`Distance`, `Pickup_Distance_airport`, `Dropoff_Distance_airport`, and `pickuptime`) using training-set statistics and then reuse those stats for test; currently these are left on very different scales and can destabilize a plain linear regression. I also add a minimal safety cleanup to drop any non-finite values produced during feature engineering before fitting, which otherwise can silently corrupt coefficients and explode predictions. Core model (LinearRegression), feature set, and training flow remain unchanged; the changes only make preprocessing consistent and numerically stable.'
- What this solution (achieved 7.43241) has done: 'Your RMSE is still extremely far from the target, which usually happens when the model is being trained on one feature set but predicted on a mismatched one (different columns/order) or when a few extreme engineered-feature values dominate a linear model. I make the feature matrix construction identical for train/validation/test by defining a single `feature_cols` list and using it everywhere (including the train/val split), which prevents subtle column drift from exploding predictions. I also add a minimal, standard winsorization (clipping by high quantiles) on the engineered continuous features using *training* quantiles and applying the same caps to test, which improves numerical stability without changing the model or feature definitions. Finally, I ensure the submission keys stay aligned with predictions and the CSV schema remains exactly `key,fare_amount`.'
- What this solution (achieved 24.19387) has done: 'Your current RMSE (7.432) is worse than the target (5.689), so we should make a small, low-risk improvement without changing the model type or feature set. The biggest remaining gap for a linear regression on this dataset is usually that the target `fare_amount` is heavy-tailed and heteroscedastic; applying a log1p transform during training and inverting it at prediction time often reduces RMSE substantially while keeping the same core approach (still LinearRegression, same features, same fitting/predict loop). To keep semantics safe, we only apply the transform after your existing fare filters (so `fare_amount > 0` holds), and we keep the same clipping/rounding and submission alignment. This should move the score down toward the target band with minimal code changes and no new dependencies.'
- What this solution (achieved 7.43241) has done: 'Your current RMSE (24.19) is still far above the target (5.69), and with your existing feature set + LinearRegression this typically happens when the model is trained on a distorted target rather than a feature mismatch (you already fixed alignment/normalization). The smallest core-logic-preserving improvement is to remove the log1p target transform (it optimizes relative error and can worsen RMSE on the original dollar scale), and instead keep training directly on `fare_amount` while adding a minimal post-processing clip to a realistic fare range to reduce the impact of occasional outlier predictions. I keep the same features, same LinearRegression, same split/training flow, and the same submission schema/key alignment. This should move RMSE down toward the target band without changing the overall approach.'
- What this solution (achieved 7.43137) has done: 'Your current RMSE (7.432) is worse than the target (5.689), so we should make a small, low-risk improvement without changing the model or feature set. The biggest remaining issue is that `LinearRegression` is very sensitive to a few remaining outlier feature rows and to collinearity; switching to `Ridge` (still linear regression, same training/predict flow) usually improves RMSE on this competition while preserving the core logic. I keep your exact feature engineering, scaling, clipping, split, and submission-writing semantics, and only replace the estimator and add a tiny alpha setting. This should move the score downward toward the target band with minimal code changes and no new dependencies.'
- What this solution (achieved 7.43137) has done: 'Your current RMSE (7.431) is above the target (5.689), so we want a small, low-risk improvement that keeps your linear model and engineered features intact. The largest remaining easy win is to include a standard NYC baseline feature: straight-line trip distance between pickup and dropoff (Haversine), which your current features only approximate via abs lat/long deltas and airport distances. I add this single feature and keep the same scaling/winsorization pipeline by including it in the existing continuous-columns lists, so train/test preprocessing stays consistent. This should nudge RMSE downward toward the target band without changing the training loop, loss/metric, or submission semantics.'
- What this solution (achieved 7.43133) has done: 'Your current RMSE (7.431) is worse than the target (5.689), so we want a small, safe improvement without changing your feature set or linear-model training flow. The most impactful minimal tweak for a Ridge model here is to tune `alpha` a bit (regularization strength) because your engineered features can still be somewhat collinear/outlier-prone; this often improves RMSE while keeping identical semantics. I add a tiny validation-based selection over a short fixed list of alphas (no early stopping, no new model type) and then refit on the full training data with the chosen alpha before predicting test. Everything else (filters, feature engineering, scaling/winsorization, clipping, and submission writing) stays the same.'
- What this solution (achieved 7.43039) has done: 'Your current RMSE (7.43133) is worse than the target (5.68914), so we want a small, low-risk improvement that keeps the same overall Ridge-on-engineered-features approach. The biggest remaining easy win without changing the model type is to add a couple of standard, lightweight time-derived features (hour, month, year) from `pickup_datetime`, because fare strongly depends on time patterns and your current `pickuptime` encoding is coarse. I implement this by parsing `pickup_datetime` once (vectorized, avoiding slow Python loops), creating the new numeric columns for both train/test, then keeping them in the same scaling/winsorization pipeline as other continuous features to preserve your preprocessing semantics. Everything else (filters, feature engineering, Ridge + alpha grid selection, clipping, submission writing) stays the same.'
- What this solution (achieved 6.90377) has done: 'Your RMSE (7.430) is still above the target (5.689), so we want a small improvement without changing the overall Ridge-on-engineered-features approach. The biggest remaining low-risk gain is adding a standard linear-friendly nonlinearity: simple interaction terms between trip distance and time/passenger_count (e.g., `Distance * pickup_hour`, `Distance * passenger_count`), which often helps a linear model capture fare structure without changing the model family. To keep preprocessing semantics consistent, the new interaction columns are added for both train/test, then put through the same train-stat scaling and winsorization you already apply to other continuous features. Everything else (filters, feature set construction, Ridge alpha selection, clipping, and submission writing) stays the same.'
- What this solution (achieved 7.37243) has done: 'Your current RMSE (6.90377) is worse than the target (5.68914), so we should make a small, low-risk improvement that keeps the same Ridge-on-engineered-features pipeline. The biggest remaining issue is that the model can produce negative fares and is sensitive to remaining training outliers; we (1) train Ridge with `positive=True` to enforce non-negative coefficients (tends to reduce bad negatives without changing the linear family) and (2) add a minimal extra training filter removing extremely low fares (often label noise) while keeping your existing filters intact. These changes are small, keep the same features/training flow, and should move RMSE down toward the target band. Submission writing and column alignment remain unchanged.'
- What this solution (achieved 6.90379) has done: 'Your current RMSE (7.372) is still above the target (5.689), so we need a small, low-risk improvement that keeps the same Ridge + engineered-features pipeline. The biggest remaining issue is that `positive=True` can significantly hurt fit quality here because NYC taxi fares require both positive and negative coefficients across correlated features; removing that constraint usually improves RMSE while keeping the same model family and training loop. I only change the Ridge instantiation (both during alpha selection and final refit) to drop `positive=True`, leaving features, scaling/winsorization, split, and submission writing untouched. This should move the score downward (better) toward the target without altering the core approach.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

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
train_data.isna().sum()



## === cell 7
train_data["Difference_longitude"] = np.abs(
    np.asarray(train_data["pickup_longitude"] - train_data["dropoff_longitude"])
)
train_data["Difference_latitude"] = np.abs(
    np.asarray(train_data["pickup_latitude"] - train_data["dropoff_latitude"])
)



## === cell 8
test_data["Difference_longitude"] = np.abs(
    np.asarray(test_data["pickup_longitude"] - test_data["dropoff_longitude"])
)
test_data["Difference_latitude"] = np.abs(
    np.asarray(test_data["pickup_latitude"] - test_data["dropoff_latitude"])
)



## === cell 9
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")



## === cell 10
plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## === cell 11
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]



## === cell 12
train_data = train_data[
    (train_data["fare_amount"] >= 2.5)
    & (train_data["fare_amount"] <= 250.0)
    & (train_data["passenger_count"] >= 1)
    & (train_data["passenger_count"] <= 6)
]

train_data = train_data[
    (train_data["pickup_longitude"] >= -75.0)
    & (train_data["pickup_longitude"] <= -72.0)
    & (train_data["dropoff_longitude"] >= -75.0)
    & (train_data["dropoff_longitude"] <= -72.0)
    & (train_data["pickup_latitude"] >= 40.0)
    & (train_data["pickup_latitude"] <= 42.0)
    & (train_data["dropoff_latitude"] >= 40.0)
    & (train_data["dropoff_latitude"] <= 42.0)
]

train_data.shape



## === cell 13
train_dt = pd.to_datetime(train_data["pickup_datetime"], errors="coerce", utc=True)
test_dt = pd.to_datetime(test_data["pickup_datetime"], errors="coerce", utc=True)

train_data["pickuptime"] = (train_dt.dt.hour * 100 + train_dt.dt.minute).astype(
    "float32"
)
test_data["pickuptime"] = (test_dt.dt.hour * 100 + test_dt.dt.minute).astype("float32")

train_data["pickup_hour"] = train_dt.dt.hour.astype("float32")
test_data["pickup_hour"] = test_dt.dt.hour.astype("float32")

train_data["pickup_month"] = train_dt.dt.month.astype("float32")
test_data["pickup_month"] = test_dt.dt.month.astype("float32")

train_data["pickup_year"] = train_dt.dt.year.astype("float32")
test_data["pickup_year"] = test_dt.dt.year.astype("float32")

train_data["Weekday"] = train_dt.dt.weekday
test_data["Weekday"] = test_dt.dt.weekday

train_data.head()



## === cell 14
train_data.head()



## === cell 15
test_data.head()



## === cell 16
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 17
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



## === cell 18
train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])

all_cols = sorted(set(train_one_hot.columns).union(set(test_one_hot.columns)))
train_one_hot = train_one_hot.reindex(columns=all_cols, fill_value=0)
test_one_hot = test_one_hot.reindex(columns=all_cols, fill_value=0)

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 19
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 20
train_data.head()



## === cell 21
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



## === cell 22
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



## === cell 23
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



## === cell 24
train_data = train_data[train_data["Distance"] > 0.0].copy()
train_data.shape



## === cell 25
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



## === cell 26
_eps = 1e-12

dlon_mean = float(np.mean(train_data["Difference_longitude"]))
dlon_var = float(np.var(train_data["Difference_longitude"]))
dlat_mean = float(np.mean(train_data["Difference_latitude"]))
dlat_var = float(np.var(train_data["Difference_latitude"]))

dlon_var = dlon_var if dlon_var > _eps else _eps
dlat_var = dlat_var if dlat_var > _eps else _eps

train_data["Difference_longitude"] = (
    np.abs(train_data["Difference_longitude"] - dlon_mean) / dlon_var
)
train_data["Difference_latitude"] = (
    np.abs(train_data["Difference_latitude"] - dlat_mean) / dlat_var
)

test_data["Difference_longitude"] = (
    np.abs(test_data["Difference_longitude"] - dlon_mean) / dlon_var
)
test_data["Difference_latitude"] = (
    np.abs(test_data["Difference_latitude"] - dlat_mean) / dlat_var
)




## === cell 27
def _abs_center_div_var(
    series_train: pd.Series, series_apply: pd.Series, eps: float = 1e-12
):
    m = float(np.mean(series_train))
    v = float(np.var(series_train))
    v = v if v > eps else eps
    return (np.abs(series_apply - m) / v), m, v


for _df in (train_data, test_data):
    _df["Distance_x_hour"] = _df["Distance"].astype("float32") * _df[
        "pickup_hour"
    ].astype("float32")
    _df["Distance_x_passengers"] = _df["Distance"].astype("float32") * _df[
        "passenger_count"
    ].astype("float32")
    _df["AirportDistSum"] = _df["Pickup_Distance_airport"].astype("float32") + _df[
        "Dropoff_Distance_airport"
    ].astype("float32")

for col in [
    "Distance",
    "Pickup_Distance_airport",
    "Dropoff_Distance_airport",
    "pickuptime",
    "pickup_hour",
    "pickup_month",
    "pickup_year",
    "Distance_x_hour",
    "Distance_x_passengers",
    "AirportDistSum",
]:
    train_data[col] = pd.to_numeric(train_data[col], errors="coerce")
    test_data[col] = pd.to_numeric(test_data[col], errors="coerce")

    train_scaled, m, v = _abs_center_div_var(train_data[col], train_data[col], eps=_eps)
    test_scaled, _, _ = _abs_center_div_var(train_data[col], test_data[col], eps=_eps)
    train_data[col] = train_scaled
    test_data[col] = test_scaled



## === cell 28
_cont_cols = [
    "Difference_longitude",
    "Difference_latitude",
    "Distance",
    "Pickup_Distance_airport",
    "Dropoff_Distance_airport",
    "pickuptime",
    "pickup_hour",
    "pickup_month",
    "pickup_year",
    "Distance_x_hour",
    "Distance_x_passengers",
    "AirportDistSum",
]
for col in _cont_cols:
    train_data[col] = pd.to_numeric(train_data[col], errors="coerce")
    test_data[col] = pd.to_numeric(test_data[col], errors="coerce")

    lo = float(train_data[col].quantile(0.001))
    hi = float(train_data[col].quantile(0.999))
    train_data[col] = train_data[col].clip(lower=lo, upper=hi)
    test_data[col] = test_data[col].clip(lower=lo, upper=hi)



## === cell 29
feature_cols_all = [c for c in train_data.columns if c not in ["fare_amount"]]
train_data = (
    train_data.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols_all).copy()
)
test_data = test_data.replace([np.inf, -np.inf], np.nan)

print(train_data.shape)
print(test_data.shape)



## === cell 30
from sklearn.model_selection import train_test_split

feature_cols = [c for c in train_data.columns if c not in ["key", "fare_amount"]]

X = train_data[feature_cols]
y = train_data["fare_amount"].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 31
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

alpha_grid = [0.1, 0.3, 1.0, 3.0, 10.0]

best_alpha = None
best_rmse = np.inf
for a in alpha_grid:
    m = Ridge(alpha=a, random_state=80)
    m.fit(X_train, y_train)
    pred_val = m.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred_val)))
    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = a

print("Holdout RMSE:", best_rmse, "best_alpha:", best_alpha)

lr = Ridge(alpha=float(best_alpha), random_state=80)
lr.fit(X, y)



## === cell 32
test_X = test_data.reindex(columns=["key"] + feature_cols).copy()
test_features = test_X[feature_cols].fillna(0)

pred = lr.predict(test_features)

pred = np.clip(pred, 0.0, 250.0)
pred = np.round(pred, 2)



## === cell 33
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 34
Submission = pd.DataFrame({"key": test_data["key"].values, "fare_amount": pred})



## === cell 35
Submission = Submission[["key", "fare_amount"]]

Submission.to_csv("submission.csv", index=False)
print(Submission.head())
print("Wrote submission.csv with shape:", Submission.shape)
