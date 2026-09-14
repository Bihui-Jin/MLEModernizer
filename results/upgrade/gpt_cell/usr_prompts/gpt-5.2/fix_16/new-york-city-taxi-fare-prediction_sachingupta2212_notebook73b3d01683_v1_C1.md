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

5.689

# 6. Current score

75.91126

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'Diagnosis: The crash happens in cell 30 because scikit-learn 1.2.2 removed the `normalize` parameter from `sklearn.linear_model.LinearRegression`, so passing `normalize=True` raises `TypeError`. The rest of the cell (train/test split, fitting, scoring) is valid and produces `lr`, which cell 31 depends on for `.predict(...)`.  
Patch summary: Remove the unsupported `normalize=True` argument while keeping the same model class, training procedure, and variables (`X_train`, `X_test`, `y_train`, `y_test`, `lr`) so downstream code continues to work unchanged.  
Updated cells: Only cell 30 is modified.  
Compatibility notes for cell k+1: `lr` is still a fitted `LinearRegression` instance, so `lr.predict(...)` in cell 31 run as before.  
Assumptions: No additional preprocessing (e.g., adding a scaler) is introduced to respect the “no core logic changes” constraint; the goal is strictly to unblock execution under the installed scikit-learn version.'
- What this solution (achieved 752.76803) has done: 'Your current RMSE (936) indicates the model is producing wildly mis-scaled predictions; the most likely cause is inconsistent feature scaling between train and test (you normalize each dataset using its own mean/variance in cell 26). To move the score toward the target, we keep the exact same model and features, but compute normalization statistics only from the training set and apply those to both train and test. We also ensure the one-hot weekday columns are perfectly aligned between train and test (missing dummy columns can silently shift feature meaning). These are minimal, metric-relevant fixes that should dramatically reduce RMSE while preserving the core approach and still writing a valid submission CSV.'
- What this solution (achieved 869.40855) has done: 'Your current RMSE is far above target, which is consistent with the model seeing out-of-range or inconsistent feature values at inference time. To move the score down toward the target while keeping the same features and LinearRegression training, I (1) filter training rows with obviously invalid coordinates/fare values (a minimal, standard cleanup for this dataset that stabilizes linear regression), (2) ensure `passenger_count` is in a valid range and non-missing, and (3) clip negative predictions to 0 since fares can’t be negative (this typically reduces RMSE). These are small, metric-relevant changes that preserve your overall approach and still produce a valid submission CSV with the required columns.'
- What this solution (achieved 802.91791) has done: 'Your RMSE is still extremely high, which is consistent with the regression being dominated by a few very large “error” cases caused by mislabeled/invalid training rows (even after your current bounds). To move the score down toward the target without changing the model or feature set, I add one more minimal, standard NYC Taxi cleanup: drop trips with (near-)zero distance but non-trivial fare (and also drop extreme-distance outliers) using your already-computed `Distance` feature. I also ensure the exact same feature columns (and order) are used for fitting and predicting to avoid any silent column mismatch. These are small, metric-relevant fixes that keep your LinearRegression training and all existing feature engineering intact while typically reducing RMSE substantially.'
- What this solution (achieved 1207.2087) has done: 'Your RMSE is still far from the 5.689 target, so we need a small, metric-aligned correction that doesn’t change your model or feature set: the linear model is currently being trained on raw `Distance`/airport-distance magnitudes that can be very different in scale from your (oddly) normalized difference features, making regression numerically unstable and causing extreme predictions. I keep the same LinearRegression and the same engineered columns, but apply the same train-derived normalization to the three distance features (`Distance`, `Pickup_Distance_airport`, `Dropoff_Distance_airport`) in addition to the two difference features, using identical semantics (abs(x-mu)/var) to preserve your approach. I also enforce a strict, identical feature column order for both fitting and prediction to avoid any silent mismatch. The submission writing stays the same and still produces `Submission.csv` with `key,fare_amount`.'
- What this solution (achieved 1207.2087) has done: 'Your current RMSE is catastrophically high for this competition, which strongly suggests the prediction rows are misaligned with the required submission `key` ordering (you set `key` as the index and write it as an index column, which Kaggle often treats differently than an explicit `key` column). To move the score down toward the 5.689 target without changing your feature engineering or LinearRegression core logic, I (1) enforce that predictions are generated in exactly the same row order as the original test file, and (2) write the submission with an explicit `key` column (not as an index) and with the exact row count/order of `sample_submission.csv`. This is a minimal, metric-relevant fix that often turns “huge RMSE” submissions into reasonable ones when the model itself is otherwise sane. I also ensure `test_data[feature_cols]` columns are reindexed to `X_train`’s column order to prevent any silent column-order mismatch.'
- What this solution (achieved 1207.2087) has done: 'Your RMSE is still extremely large, which is most consistent with a broken normalization formula that explodes feature magnitudes (you divide by variance instead of standard deviation) and destabilizes LinearRegression predictions. To move the score sharply downward toward the 5.689 target while preserving the same features, model, and training approach, I keep your “train-derived scaling applied to both train and test” idea but switch the denominator from variance to standard deviation (with a small epsilon guard). I also add a minimal safeguard to replace any inf/NaN created during scaling with 0.0 so training/prediction remain numerically stable. Submission generation (key alignment via sample_submission merge) remains the same.'
- What this solution (achieved 28.88079) has done: 'Your RMSE is far worse than the baseline, which is most consistent with the model producing extreme predictions due to a few remaining bad/inconsistent rows and a scaling transform that unintentionally destroys linear relationships (using `abs(x-mu)/std`). To move the score sharply down toward the 5.689 target while keeping the same LinearRegression and the same engineered features, I (1) make scaling linear and consistent by using standard z-score `(x-mu)/std` (no `abs`), (2) add a minimal extra cleanup that removes “too-cheap for very long distance” outliers that otherwise dominate least-squares, and (3) clip predictions to a realistic upper bound to prevent catastrophic submission errors without changing the model. These are small, metric-aligned changes that keep your overall pipeline intact and still write `Submission.csv` in the required `key,fare_amount` format.'
- What this solution (achieved 28.94905) has done: 'Your RMSE (28.88) is still far above the 5.689 target, so we need a small, metric-relevant fix without changing the model or features. The biggest remaining issue is that you’re training on 10M rows but doing only minimal cleanup; LinearRegression is very sensitive to remaining label noise/outliers, especially extreme fares at moderate distances. I add one additional standard NYC Taxi cleanup that removes “fare per mile” outliers using your already-computed `Distance` (no new features/model), and I also ensure `test_X` has any missing columns filled with 0 after reindexing to avoid NaNs affecting predictions. These are minimal changes expected to reduce catastrophic errors and move RMSE downward toward the target.'
- What this solution (achieved 75.32697) has done: 'Your RMSE (28.95) is still far above the 5.689 target, so we should reduce catastrophic prediction errors without changing your model or feature set. The smallest high-impact fix is to train the LinearRegression on a log-transformed target (`log1p(fare_amount)`) and then invert with `expm1` at prediction time; this keeps the same model/approach but makes least-squares far less sensitive to remaining outliers and heavy-tailed fares, typically lowering RMSE a lot for this dataset. I’m also adding one minimal cleanup to remove extremely high “fare-per-mile” cases at very short distances (where sensor noise creates huge ratios) using your existing `Distance` column. Submission generation and column alignment remain the same, and the script still writes a valid `Submission.csv`.'
- What this solution (achieved 28.95433) has done: 'Your RMSE got worse after introducing the log1p target transform; to move the score back down toward the 5.689 target while keeping the same LinearRegression model and the same engineered features, I’m reverting to training directly on `fare_amount` (no log/expm1). To prevent a few remaining outliers from blowing up least-squares, I’m also adding one minimal, standard cleanup: drop rows with `fare_amount` far above what the trip distance would plausibly allow (a distance-conditioned upper bound using your existing `Distance`). Finally, I keep your existing feature alignment/reindexing and submission merge to ensure keys and row order are correct and a valid `Submission.csv` is always produced.'
- What this solution (achieved 28.85242) has done: 'Your RMSE is still far above the 5.689 target, so we should remove avoidable sources of large error while keeping your exact LinearRegression + current features. The biggest minimal fix is to stop throwing away the raw latitude/longitude coordinates (cell 29), because with a linear model they carry crucial location information that your current distance-only features can’t fully represent. I also add a small, standard outlier cleanup that removes “too-high fare for near-zero distance” and “too-low fare for long distance” using your existing `Distance` feature to reduce least-squares sensitivity. Finally, I normalize the restored coordinate columns using the same train-derived z-score transform you already apply, ensuring consistent scaling and stable predictions, while keeping the submission alignment logic unchanged.'
- What this solution (achieved 28.85242) has done: 'Your pipeline is training the linear model before applying your normalization (cell 32), then training again after normalization (cell 34). To move RMSE down toward the 5.689 target with minimal risk and without changing the model/feature set, I remove the first (pre-normalization) train/test split + fit so you only train once on the correctly scaled features. I also make sure the scaling statistics (mu/std) are computed from `X_train` only (not the full cleaned training set) to avoid slight train-test leakage that can hurt generalization on the Kaggle test set. Finally, I keep the exact same submission alignment logic but ensure `pred_df` uses the keys from the original `sample_submission` order for maximum safety.'
- What this solution (achieved 74.46969) has done: 'Your RMSE is still far above the 5.689 target, so the smallest high-impact move is to make the linear model less sensitive to remaining label noise/outliers without changing the model class, features, or training loop. I keep your exact feature engineering and scaling, but switch the regression objective from ordinary least squares to a robust linear regression (`HuberRegressor`) which is still a linear model and trains the same way (fit once, predict once) while typically cutting large-error tails that dominate RMSE here. I also apply the same scaling to `test_X` (instead of scaling `test_data` in-place before reindexing) to ensure perfect train/test transform consistency and avoid any subtle column drift. Submission writing/key alignment stays identical and still output a valid `Submission.csv`.'
- What this solution (achieved 75.91126) has done: 'Your current RMSE (74.47) is still far above the 5.689 target, so we should address the most likely source of catastrophic errors while keeping your same feature set and “fit once, predict once” workflow. The biggest minimal fix is that `HuberRegressor` is still being trained on raw-dollar `fare_amount` while you clip predictions to [0, 500]; switching to a log1p-target with expm1 inversion typically reduces large-error tails without changing the model family or features. To keep semantics stable, we also compute the fallback fill value from the same transformed target and invert it consistently. Everything else (data cleanup, feature engineering, scaling, key alignment, submission writing) stays the same.'

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

test_data["Difference_longitude"] = np.abs(
    np.asarray(test_data["pickup_longitude"] - test_data["dropoff_longitude"])
)
test_data["Difference_latitude"] = np.abs(
    np.asarray(test_data["pickup_latitude"] - test_data["dropoff_latitude"])
)



## === cell 8
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")



## === cell 9
train_data = train_data[
    (train_data["fare_amount"] > 0) & (train_data["fare_amount"] <= 500)
].copy()

train_data = train_data[
    train_data["pickup_longitude"].between(-75, -72)
    & train_data["dropoff_longitude"].between(-75, -72)
    & train_data["pickup_latitude"].between(40, 42)
    & train_data["dropoff_latitude"].between(40, 42)
].copy()

train_data = train_data[train_data["passenger_count"].between(1, 6)].copy()

test_data["passenger_count"] = test_data["passenger_count"].fillna(1).clip(1, 6)



## === cell 10
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]



## === cell 11
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_data["pickuptime"] = ls1

ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_data["pickuptime"] = ls1
train_data.head()



## === cell 12
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



## === cell 13
train_data.head()



## === cell 14
test_data.head()



## === cell 15
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 16
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



## === cell 17
train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])

train_one_hot, test_one_hot = train_one_hot.align(
    test_one_hot, join="outer", axis=1, fill_value=0
)

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 18
ls1 = list(train_data["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
train_data["pickuptime"] = ls1

ls1 = list(test_data["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
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



## === cell 21
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



## === cell 23
a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2

train_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## === cell 24
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



## === cell 25
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



## === cell 26
train_data = train_data[
    (train_data["Distance"] >= 0.05)  # drop near-zero distance artifacts
    & (train_data["Distance"] <= 100.0)  # drop extreme distance outliers
    & ~(
        (train_data["Distance"] < 0.2) & (train_data["fare_amount"] > 50)
    )  # inconsistent rows
].copy()

train_data = train_data[
    ~((train_data["Distance"] > 20.0) & (train_data["fare_amount"] < 5.0))
].copy()



## === cell 27
fare_per_mile = train_data["fare_amount"] / (train_data["Distance"] + 1e-3)
train_data = train_data[fare_per_mile.between(0.5, 60.0)].copy()

train_data = train_data[
    ~((train_data["Distance"] < 1.0) & (fare_per_mile > 25.0))
].copy()



## === cell 28
max_fare_by_dist = 3.5 + 25.0 * train_data["Distance"]  # generous upper envelope
train_data = train_data[train_data["fare_amount"] <= max_fare_by_dist].copy()



## === cell 29
min_fare_by_dist = (
    2.5 + 0.8 * train_data["Distance"]
)  # gentle floor (covers base fare + small per-mile)
train_data = train_data[train_data["fare_amount"] >= min_fare_by_dist].copy()
train_data = train_data[
    ~((train_data["Distance"] < 0.5) & (train_data["fare_amount"] > 100.0))
].copy()



## === cell 30
train_data.shape



## === cell 31
test_data.shape



## === cell 32
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor

feature_cols = [c for c in train_data.columns if c not in ["key", "fare_amount"]]

X = train_data[feature_cols]
y = train_data["fare_amount"].astype(float)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 33
norm_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "Difference_longitude",
    "Difference_latitude",
    "Distance",
    "Pickup_Distance_airport",
    "Dropoff_Distance_airport",
]
eps = 1e-12

X_train = X_train.copy()
X_valid = X_valid.copy()

test_X = test_data.reindex(columns=feature_cols, fill_value=0.0).copy()

for col in norm_cols:
    if col in X_train.columns:
        mu = float(np.mean(X_train[col]))
        std = float(np.std(X_train[col]))
        if (not np.isfinite(std)) or std < eps:
            std = 1.0

        X_train[col] = (X_train[col] - mu) / std
        X_valid[col] = (X_valid[col] - mu) / std
        if col in test_X.columns:
            test_X[col] = (test_X[col] - mu) / std

X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_valid.replace([np.inf, -np.inf], np.nan, inplace=True)
test_X.replace([np.inf, -np.inf], np.nan, inplace=True)

X_train.fillna(0.0, inplace=True)
X_valid.fillna(0.0, inplace=True)
test_X.fillna(0.0, inplace=True)

y_train_t = np.log1p(y_train.values)
y_valid_t = np.log1p(y_valid.values)

lr = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=200, fit_intercept=True)
lr.fit(X_train, y_train_t)

print("Valid R^2 on log1p target:", lr.score(X_valid, y_valid_t))

pred_t = lr.predict(test_X)
pred = np.expm1(pred_t)

pred = np.clip(pred, 0, 500)
pred = np.round(pred, 2)

pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 34
sample_sub = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)

pred_df = pd.DataFrame({"key": test_data["key"].values, "fare_amount": pred})

Submission = sample_sub[["key"]].merge(pred_df, on="key", how="left")

if Submission["fare_amount"].isna().any():
    fill_val = float(
        np.expm1(np.mean(np.log1p(train_data["fare_amount"].astype(float))))
    )
    Submission["fare_amount"] = Submission["fare_amount"].fillna(fill_val)

Submission.to_csv("Submission.csv", index=False)
print("Wrote Submission.csv with shape:", Submission.shape)
print(Submission.head())
