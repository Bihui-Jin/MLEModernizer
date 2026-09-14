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

3.40466

# 6. Current score

4.4145

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.32387) has done: 'The changes reduce the training size (1 M rows instead of 2 M) and eliminate the second full‑data fit, keeping the same RandomForest architecture while cutting the work roughly in half; this ensures the script finishes well under the 600 s limit without affecting prediction logic. Vectorized haversine already remains, and the model is still trained on a proper train/validation split, preserving the internal RMSE check.'
- What this solution (achieved 4.65062) has done: 'The changes keep the same preprocessing, feature engineering, and model type while dramatically cutting the training workload.  
- Load a smaller random subset of the training data (300 k rows) instead of 1 M rows, reducing memory and computation.  
- Train the RandomForest with only 100 trees (instead of 300); this keeps the same depth and overall algorithm, preserving the model’s structure but cutting training time by ~⅔.  
- Minor comment additions explain why each modification speeds up execution without altering the core logic.'
- What this solution (achieved 4.34764) has done: 'The update keeps the same RandomForest model and preprocessing but adds a modest amount of extra training data, a useful logarithmic distance feature, and a stronger forest (more trees and deeper). These changes are lightweight enough to stay within the 600 s limit while clearly moving the internal RMSE down toward the target of 3.40466.'
- What this solution (achieved 4.33708) has done: 'I add a cheap weekend indicator feature and adjust the RandomForest to use the classic “sqrt” feature subset with a small leaf‑size regularizer. These tweaks keep the same model type and overall pipeline but usually improve generalisation, moving the internal RMSE closer to the target while staying within the time limit.'
- What this solution (achieved 4.41079) has done: 'We replace infinite values in the test features with NaNs before filling them, preventing the “contains infinity” error that stopped prediction. After this fix the script can compute `test_pred` and write a proper `submission_*.csv` file with the required columns.'
- What this solution (achieved 4.45418) has done: 'We slightly boost the RandomForest’s capacity (more trees and a smaller leaf size) which usually lowers validation RMSE without altering the overall pipeline or feature set, moving the score closer to the target.'
- What this solution (achieved 4.47528) has done: 'I add a modest clipping of unrealistic distance values and a reasonable upper bound on fares, then slightly regularize the RandomForest (more trees, shallower depth, and a larger leaf size). These changes keep the same model type and core pipeline but should improve generalisation and bring the RMSE closer to the target while staying within the runtime limit.'
- What this solution (achieved 4.4145) has done: 'I add the missing imports, define the feature matrix X, target y, and the list of feature columns, and include a tiny feature‑engineering step (haversine distance, hour and weekday) so the model can be trained and used. After training the RandomForest, I generate predictions for the test set and write a proper `submission.csv` containing the required `key` and `fare_amount` columns.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



## === cell 1
train_path = "../input/train.csv"
test_path = "../input/test.csv"

train_df = pd.read_csv(
    train_path,
    nrows=800_000,
    parse_dates=["pickup_datetime"],
)
test_df = pd.read_csv(
    test_path,
    parse_dates=["pickup_datetime"],
)

num_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
for col in num_cols:
    if col in train_df.columns:
        train_df[col] = pd.to_numeric(train_df[col], downcast="float")
    if col in test_df.columns:
        test_df[col] = pd.to_numeric(test_df[col], downcast="float")


def haversine_distance(lat1, lon1, lat2, lon2):
    """Return distance in kilometers between two lat/lon points."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


train_df["distance"] = haversine_distance(
    train_df["pickup_latitude"],
    train_df["pickup_longitude"],
    train_df["dropoff_latitude"],
    train_df["dropoff_longitude"],
)
test_df["distance"] = haversine_distance(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
)

for df in (train_df, test_df):
    df["hour"] = df["pickup_datetime"].dt.hour.astype(np.float32)
    df["weekday"] = df["pickup_datetime"].dt.weekday.astype(np.float32)

feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance",
    "hour",
    "weekday",
]

train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)

medians = train_df[feature_cols].median()
train_df[feature_cols] = train_df[feature_cols].fillna(medians)
test_df[feature_cols] = test_df[feature_cols].fillna(medians)

train_df[feature_cols] = train_df[feature_cols].astype(np.float32)
test_df[feature_cols] = test_df[feature_cols].astype(np.float32)

sampled_train = train_df.sample(300_000, random_state=42)
X = sampled_train[feature_cols]
y = sampled_train["fare_amount"]


## === cell 2
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(
    n_estimators=200,  # balance speed / performance
    max_depth=25,
    max_features="sqrt",
    min_samples_leaf=1,
    n_jobs=5,
    random_state=42,
)
rf.fit(X_train, y_train)

val_pred = rf.predict(X_val)
rmse = mean_squared_error(y_val, val_pred, squared=False)
print(f"Internal validation RMSE: {rmse:.5f}")



## === cell 3
test_features = test_df[feature_cols]
test_pred = rf.predict(test_features)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})

submission_path = "./submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
