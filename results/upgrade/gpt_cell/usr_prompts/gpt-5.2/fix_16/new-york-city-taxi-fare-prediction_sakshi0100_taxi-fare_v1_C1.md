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

5.68932

# 6. Current score

36.68502

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'Diagnosis: The crash in cell 29 is due to using the deprecated/removed `normalize` parameter in `sklearn.linear_model.LinearRegression`. In scikit-learn 1.2.2, `normalize` is no longer accepted, so passing it raises `TypeError`. The rest of the cell (fit/score) is fine and must remain unchanged.

Patch summary: Remove the unsupported `normalize=True` argument when constructing `LinearRegression`, keeping the same model type and training/evaluation flow so downstream cells can still use `lr` for prediction.

Updated cells: Only cell 29 is modified.

Compatibility notes for cell k+1: Cell 30 expects a fitted `lr` object with `.predict(...)`; this remains unchanged because `lr` is still a `LinearRegression` instance trained via `.fit(...)`.

Assumptions: No other scikit-learn API incompatibilities exist earlier, and data types/shapes are already compatible with `LinearRegression.fit`/`predict`.'
- What this solution (achieved 937.3389) has done: 'Your RMSE is extremely high because the model is being trained on the first 10M rows without filtering out clearly invalid/outlier target values, so the linear regression gets dominated by huge/negative fares and bad records. To move the score toward the target with minimal disruption, I add a small, standard NYC Taxi cleanup step that only removes impossible coordinates, passenger counts, and extreme/invalid `fare_amount` values before training. I also ensure the one-hot weekday columns align between train and test (in case a weekday is missing in the sampled train slice), which prevents silent feature mismatch. These changes preserve your model (LinearRegression) and feature set logic while making the training data consistent with the competition’s expected domain.'
- What this solution (achieved 1114.26746) has done: 'Your current RMSE is far from the target largely because the `Difference_*` normalization is computed separately on train and test (and with the wrong mean/variance), which badly shifts feature scales at inference time; we instead normalize test using the train mean/variance (same core feature, just consistent scaling). Also, the submission is being written with `key` as an index, but the competition expects `key` as a column; we write the CSV with `key` as a normal column to avoid format-related scoring issues. These are minimal changes that preserve your model (LinearRegression) and all feature engineering, but remove two high-impact sources of error. No training loop, model type, loss, or feature set is changed—only consistent preprocessing and correct submission formatting.'
- What this solution (achieved 15.3192) has done: 'Your current RMSE is far above target because the LinearRegression predictions can go negative or explode on rare test patterns, and RMSE heavily penalizes these outliers; adding a simple non-negative clamp (and a mild upper cap consistent with your training filter) stabilizes predictions without changing the model or features. I also make sure train/test columns are strictly aligned before fitting/predicting to avoid any subtle column-order mismatch from earlier concatenations. These are minimal post-processing and safety checks that preserve your core logic while typically bringing RMSE down substantially on this competition. The submission writing stays the same format (key as a column, correct header) and still produces a valid CSV.'
- What this solution (achieved 15.31904) has done: 'Your current RMSE (15.3192) is far above the target (5.68932), so we should improve performance with minimal, safe changes that keep your LinearRegression and feature logic intact. The biggest likely remaining issue is that `LinearRegression` is being fit without any feature scaling, while your engineered features (especially distances vs one-hot weekdays vs pickuptime) have very different magnitudes; standardizing features (fit on train only, apply to test) typically reduces RMSE a lot without changing the model class or training loop. I add a `StandardScaler` step after the final column alignment (so train/val/test all share identical column order) and keep your existing clipping/rounding and submission format unchanged. This keeps the same core pipeline (same engineered features and same LinearRegression) but makes the linear model behave much closer to what it expects numerically.'
- What this solution (achieved 15.31902) has done: 'Your current RMSE (15.319) is much worse than the target (5.689), so we should improve predictive accuracy with minimal changes while keeping your LinearRegression + existing engineered features. The biggest remaining “minimal but high impact” issue is the nonstandard “variance” normalization in cells 24–26: you divide by `np.var(...)` instead of `np.std(...)`, which overly shrinks/warps those features and can hurt the linear fit; switching to standard deviation keeps the same feature and intent (standardization) but fixes scaling. I also add a tiny numerical guard to avoid any divide-by-zero edge case and keep the rest (scaler, clipping, submission format) unchanged. This should move RMSE substantially toward the target without changing model type, training loop, or feature set.'
- What this solution (achieved 15.3192) has done: 'We need to move your RMSE down from 15.319 toward the 5.689 target, so the smallest high-impact change is to make the linear regression less sensitive to remaining mislabeled/outlier training rows while keeping the same model family and features. I switch `LinearRegression()` to `Ridge()` (still a linear model with the same training flow and prediction semantics) and keep your existing StandardScaler and clipping, which typically stabilizes coefficients and reduces RMSE substantially on this dataset. I also ensure the exact same column alignment is applied to `X_test` as well (using `align`) to prevent any silent feature-order mismatches between the validation split and the fitted model. Everything else (feature engineering, filtering, scaling, train/test split, submission format/path) stays the same and the script still writes a valid `Submission.csv`.'
- What this solution (achieved 15.3192) has done: 'Your RMSE (15.319) is far above the target (5.689), so we should improve accuracy with the smallest change that keeps your same feature engineering and linear-model training flow. The biggest remaining issue is that `pickuptime` is encoded as an HHMM integer (e.g., 930 vs 1030), which creates an artificial discontinuity at hour boundaries and weakens the linear fit; converting it to “minutes since midnight” preserves the same information but makes it linear-friendly. I keep your Ridge + StandardScaler + clipping exactly as-is, and only change how `pickuptime` is derived (for both train and test) so the rest of the pipeline remains identical. This should reduce RMSE materially without changing model class, loss, or the overall approach, and it still writes a valid `Submission.csv`.'
- What this solution (achieved 15.30423) has done: 'Your RMSE is still far above the target, so we need a small but high-impact fix that keeps your exact feature set and linear-model training flow. The main issue is the `abs(x - mean) / std` transform on `Difference_longitude/latitude`, which destroys the sign/linearity and makes the relationship to fare harder for Ridge to learn; switching to standard z-scoring `(x - mean) / std` preserves the same “normalize this feature” intent but restores linear structure. I keep your Ridge + StandardScaler + clipping + submission format unchanged, and only adjust the two normalization cells (and the corresponding test transform) to be consistent and linear-friendly. This should materially lower RMSE while staying within your core logic constraints.'
- What this solution (achieved 15.30091) has done: 'Your score is much worse than the target (RMSE 15.30 vs 5.69, lower is better), so we need a small but high-impact fix while keeping your Ridge + StandardScaler pipeline and existing feature set. The biggest remaining issue is that Ridge is highly sensitive to remaining outliers even after your fare/coordinate filters; we can make it robust by winsorizing (clipping) the training targets `y_train` only before fitting, without changing the model, features, or metric semantics. This usually reduces RMSE substantially on this competition because a tiny fraction of noisy labels otherwise dominate a squared-error objective. I also keep your existing prediction clipping and submission format unchanged.'
- What this solution (achieved 15.30091) has done: 'Your RMSE (15.30091) is far above the target (5.68932, lower is better), so we should improve accuracy with the smallest changes that keep your Ridge + StandardScaler pipeline and all existing features. The biggest issue is that `Ridge(alpha=1.0)` is likely too strongly regularized for your standardized features, causing underfitting and systematically worse predictions; tuning `alpha` downward is a minimal, legitimate adjustment that preserves the same model/approach. I select `alpha` from a small fixed grid using your existing `X_test/y_test` split (no extra training loops beyond choosing among a few alphas) and then refit once on the full training split with the chosen `alpha`. Everything else (filters, feature engineering, scaling, clipping, submission format/path) remains the same and it still write `Submission.csv`.'
- What this solution (achieved 15.07803) has done: 'We need to move RMSE down from 15.30 toward the 5.69 target (lower is better) with minimal changes while preserving your Ridge + StandardScaler pipeline and existing features. The highest-impact, low-risk issue is that your train/test feature construction is asymmetric: you compute `Distance` and airport distances, but only normalize `Difference_*`, leaving other continuous distance features on very different scales (and partially redundant), which can destabilize Ridge even after global scaling due to heavy-tailed distributions. I add a small, consistent log1p transform to the three distance-like features (`Distance`, `Pickup_Distance_airport`, `Dropoff_Distance_airport`) for both train and test (after rounding) to reduce leverage from long trips/outliers while keeping the same features and model. I also ensure the `align` uses `join="inner"` (not left) to guarantee identical columns/order across train/holdout/submit, preventing silent mismatch that can hurt generalization.'
- What this solution (achieved 21.04071) has done: 'Your RMSE (15.078) is far above the target (5.689, lower is better), so we should make a minimal, high-impact improvement without changing the core Ridge + engineered-features pipeline. The biggest remaining systematic issue is underfitting from using only linear terms: fare depends nonlinearly on trip distance and deltas, and adding a tiny set of polynomial interaction terms for the already-computed continuous features lets the same linear Ridge model capture curvature while keeping the same training loop and loss. I add `PolynomialFeatures` only on the 5 continuous columns (`Difference_*`, `pickuptime`, `Distance`, airport distances) and then concatenate with the existing weekday one-hots, keeping your scaler, alpha selection, clipping, and submission format unchanged. This should materially reduce RMSE toward the target while staying within Kaggle constraints and still writing a valid `Submission.csv`.'
- What this solution (achieved 21.04635) has done: 'Your current RMSE (21.04) is far above the target (5.689), so we should improve with the smallest changes that keep your Ridge + PolynomialFeatures + StandardScaler pipeline intact. The biggest remaining gap is usually caused by keeping noisy long-distance/outlier trips in training (even with fare/coord filters), which hurts squared-error fits; adding a light “reasonable trip length” filter on your already-computed `Distance` (in miles) is a minimal, domain-standard cleanup that often produces a large RMSE drop. To keep feature construction consistent, we apply the filter right after `Distance` is computed (before log1p) and reapply the same `Difference_* < 5` constraint after dropping NA (to avoid any ordering edge cases). Everything else—features, model, alpha selection, scaling, clipping, and submission writing—stays the same.'
- What this solution (achieved 36.68502) has done: 'Your current RMSE (21.046) is much worse than the target (5.689, lower is better), and the main minimal lever left without changing your model/feature set is to make the linear Ridge fit less dominated by high-variance residuals. I keep your exact feature engineering (including PolynomialFeatures, scaling, and existing clipping), but change the Ridge solver to use `sample_weight` based on trip distance so very long trips (which are rarer and noisier) don’t disproportionately drive the squared-error fit. This is a small, legitimate adjustment that preserves the same training flow and prediction semantics while typically reducing RMSE a lot on this competition. I also keep your submission format identical and still write `Submission.csv`.'

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



## === cell 7
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")



## === cell 8
train_data = train_data[
    (train_data["fare_amount"] > 0)
    & (train_data["fare_amount"] <= 250)
    & (train_data["passenger_count"] >= 1)
    & (train_data["passenger_count"] <= 6)
    & (train_data["pickup_longitude"].between(-74.3, -72.7))
    & (train_data["dropoff_longitude"].between(-74.3, -72.7))
    & (train_data["pickup_latitude"].between(40.4, 41.3))
    & (train_data["dropoff_latitude"].between(40.4, 41.3))
].copy()



## === cell 9
plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## === cell 10
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
].copy()



## === cell 11
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_data["pickuptime"] = ls1

ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_data["pickuptime"] = ls1



## === cell 12
train_data.head()



## === cell 13
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
train_data["Weekday"] = ls1

ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
test_data["Weekday"] = ls1
test_data.head()



## === cell 14
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 15
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



## === cell 16
train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])

train_one_hot, test_one_hot = train_one_hot.align(
    test_one_hot, join="outer", axis=1, fill_value=0
)

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 17
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 18
ls1 = list(train_data["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 60 + int(z[1])
train_data["pickuptime"] = ls1

ls1 = list(test_data["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 60 + int(z[1])
test_data["pickuptime"] = ls1



## === cell 19
train_data.head()



## === cell 20
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



## === cell 21
train_data = train_data[train_data["Distance"].between(0.0, 60.0)].copy()



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
for col in ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]:
    train_data[col] = np.log1p(train_data[col].astype(float))
    test_data[col] = np.log1p(test_data[col].astype(float))



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
diff_lon_mean = np.mean(train_data["Difference_longitude"])
diff_lon_std = np.std(train_data["Difference_longitude"])
if diff_lon_std == 0:
    diff_lon_std = 1.0

train_data["Difference_longitude"] = (
    train_data["Difference_longitude"] - diff_lon_mean
) / diff_lon_std



## === cell 27
diff_lat_mean = np.mean(train_data["Difference_latitude"])
diff_lat_std = np.std(train_data["Difference_latitude"])
if diff_lat_std == 0:
    diff_lat_std = 1.0

train_data["Difference_latitude"] = (
    train_data["Difference_latitude"] - diff_lat_mean
) / diff_lat_std



## === cell 28
test_data["Difference_longitude"] = (
    test_data["Difference_longitude"] - diff_lon_mean
) / diff_lon_std

test_data["Difference_latitude"] = (
    test_data["Difference_latitude"] - diff_lat_mean
) / diff_lat_std



## === cell 29
train_data.shape



## === cell 30
test_data.shape



## === cell 31
from sklearn.model_selection import train_test_split

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 32
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

X_test_submit = test_data.drop("key", axis=1)

X_train, X_test_submit = X_train.align(X_test_submit, join="inner", axis=1)
X_train, X_test = X_train.align(X_test, join="inner", axis=1)

weekday_cols = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
weekday_cols = [c for c in weekday_cols if c in X_train.columns]

cont_cols = [
    "Difference_longitude",
    "Difference_latitude",
    "pickuptime",
    "Distance",
    "Pickup_Distance_airport",
    "Dropoff_Distance_airport",
    "passenger_count",
]
cont_cols = [c for c in cont_cols if c in X_train.columns]

Xtr_cont = X_train[cont_cols].astype(float)
Xte_cont = X_test[cont_cols].astype(float)
Xsub_cont = X_test_submit[cont_cols].astype(float)

Xtr_cat = X_train[weekday_cols].astype(float) if len(weekday_cols) else None
Xte_cat = X_test[weekday_cols].astype(float) if len(weekday_cols) else None
Xsub_cat = X_test_submit[weekday_cols].astype(float) if len(weekday_cols) else None

poly = PolynomialFeatures(degree=2, include_bias=False)
Xtr_cont_poly = poly.fit_transform(Xtr_cont)
Xte_cont_poly = poly.transform(Xte_cont)
Xsub_cont_poly = poly.transform(Xsub_cont)

if Xtr_cat is not None:
    X_train_final = np.hstack([Xtr_cont_poly, Xtr_cat.to_numpy()])
    X_test_final = np.hstack([Xte_cont_poly, Xte_cat.to_numpy()])
    X_sub_final = np.hstack([Xsub_cont_poly, Xsub_cat.to_numpy()])
else:
    X_train_final = Xtr_cont_poly
    X_test_final = Xte_cont_poly
    X_sub_final = Xsub_cont_poly

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)
X_test_submit_scaled = scaler.transform(X_sub_final)

y_train_fit = y_train.clip(lower=y_train.quantile(0.001), upper=y_train.quantile(0.999))

dist_col = "Distance" if "Distance" in X_train.columns else None
if dist_col is not None:
    d = X_train[dist_col].astype(float).to_numpy()
    sample_weight = 1.0 / (
        1.0 + np.expm1(np.clip(d, 0.0, 10.0))
    )  # ~ 1/(1+distance) in log-space
    sample_weight = np.clip(sample_weight, 0.05, 1.0)
else:
    sample_weight = None

alphas = [0.01, 0.03, 0.1, 0.3, 1.0]
best_alpha = None
best_rmse = np.inf
for a in alphas:
    m = Ridge(alpha=a, random_state=80)
    if sample_weight is None:
        m.fit(X_train_scaled, y_train_fit)
    else:
        m.fit(X_train_scaled, y_train_fit, sample_weight=sample_weight)
    val_pred = m.predict(X_test_scaled)
    val_pred = np.clip(val_pred, 0, 250)
    rmse = mean_squared_error(y_test, val_pred, squared=False)
    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = a

lr = Ridge(alpha=best_alpha, random_state=80)
if sample_weight is None:
    lr.fit(X_train_scaled, y_train_fit)
else:
    lr.fit(X_train_scaled, y_train_fit, sample_weight=sample_weight)

print("Chosen alpha:", best_alpha, "| Validation RMSE:", best_rmse)
print("R^2 on holdout:", lr.score(X_test_scaled, y_test))



## === cell 33
pred = lr.predict(X_test_submit_scaled)
pred = np.clip(pred, 0, 250)
pred = np.round(pred, 2)



## === cell 34
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 35
Submission = pd.DataFrame(data=pred, columns=["fare_amount"])
Submission["key"] = test_data["key"]
Submission = Submission[["key", "fare_amount"]]



## === cell 36
Submission.to_csv("Submission.csv", index=False)
print("Wrote Submission.csv with shape:", Submission.shape)
print(Submission.head())
