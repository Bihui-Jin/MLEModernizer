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

5.69761

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1003.87868) has done: 'Your RMSE is exploding because the linear model is being trained on many invalid/outlier trips (bad coordinates, zero-distance rides, extreme fares) and you’re also rounding predictions before scoring/submitting, which adds avoidable error for an RMSE metric. I keep the same core approach (OLS on abs coordinate deltas + intercept) but add minimal, standard NYC Taxi cleaning filters and ensure feature engineering happens after filtering. I also remove rounding for validation and submission (you can still round for display), and make the train/validation split deterministic to stabilize results. These changes should drastically reduce the RMSE and move it much closer to your target.'
- What this solution (achieved 1003.87868) has done: 'Your current RMSE is massively worse than the target because the model is being fit with a `key` string column present in `X`, so the train/validation split and downstream matrix construction become misaligned (and predictions don’t correspond to the right rows), and also because your `add_travel_vector_features` function doesn’t return the modified frame (so it’s easy to accidentally use a non-featured frame later). I keep the exact same core model (OLS on abs coordinate deltas + intercept) but make minimal, score-relevant fixes: explicitly drop non-feature columns (`key`, `pickup_datetime`) before splitting, reset indices to guarantee alignment, and ensure features are created before building matrices for both train/val/test. This should dramatically reduce the RMSE toward your target while preserving the same modeling approach and submission semantics. The script still write a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 15.15766) has done: 'You’re already using the right minimal linear OLS core, but your score is still far from the target because the model is being trained on raw coordinate deltas without a distance-like scale, and the least-squares solution can produce negative/implausible fares that heavily hurt RMSE. I keep the exact same OLS setup (same training loop/solver) and only add one standard, minimal feature (`manhattan` = abs_dx + abs_dy) to stabilize the linear fit while preserving the travel-vector logic. I also clip predictions to a reasonable non-negative range for RMSE (this is post-processing, not a new model) and ensure train/val/test use identical feature construction. These are small, score-relevant changes that should move RMSE sharply down toward your target without changing the overall approach.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

print(os.listdir("../input"))



## === cell 1
data = pd.read_csv("../input/train.csv", nrows=20_000_000)




## === cell 2
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["abs_diff_latitude"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
    df["manhattan"] = df["abs_diff_longitude"] + df["abs_diff_latitude"]

    mean_lat = ((df["pickup_latitude"] + df["dropoff_latitude"]) / 2.0).astype(float)
    cos_lat = np.cos(np.deg2rad(mean_lat.to_numpy()))
    dlon = (df["dropoff_longitude"] - df["pickup_longitude"]).astype(float).to_numpy()
    dlat = (df["dropoff_latitude"] - df["pickup_latitude"]).astype(float).to_numpy()

    dx_km = (dlon * 111.32) * cos_lat
    dy_km = dlat * 110.57

    df["euclidean_km"] = np.sqrt(dx_km * dx_km + dy_km * dy_km)
    df["manhattan_km"] = np.abs(dx_km) + np.abs(dy_km)

    return df




## === cell 3
print(data.isnull().sum())



## === cell 4
print("Old size: %d" % len(data))
data = data.dropna(how="any", axis="rows")
print("New size: %d" % len(data))



## === cell 5
print("Old size: %d" % len(data))

data = data[
    (data.fare_amount > 0.0)
    & (data.fare_amount <= 250.0)
    & (data.passenger_count >= 1)
    & (data.passenger_count <= 6)
    & (data.pickup_longitude.between(-74.3, -73.7))
    & (data.dropoff_longitude.between(-74.3, -73.7))
    & (data.pickup_latitude.between(40.5, 41.0))
    & (data.dropoff_latitude.between(40.5, 41.0))
].copy()

data = add_travel_vector_features(data)

data = data[
    (data.abs_diff_longitude < 3.0)
    & (data.abs_diff_latitude < 3.0)
    & (data.manhattan > 0.0)
    & np.isfinite(data["euclidean_km"])
    & np.isfinite(data["manhattan_km"])
    & (data["euclidean_km"] > 0.0)
].copy()

data = data.reset_index(drop=True)

print("New size: %d" % len(data))



## === cell 6
plot = data.iloc[:10000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")



## === cell 7
from sklearn.model_selection import train_test_split

y = data["fare_amount"].astype(float)

feature_df = data.drop(
    columns=["fare_amount", "key", "pickup_datetime"], errors="ignore"
).copy()

train_df, val_df, train_y, val_y = train_test_split(
    feature_df, y, test_size=0.2, random_state=42
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
train_y = train_y.reset_index(drop=True)
val_y = val_y.reset_index(drop=True)

print(train_df.dtypes)




## === cell 8
def get_input_matrix(df):
    return np.column_stack(
        (
            df["abs_diff_longitude"].to_numpy(),
            df["abs_diff_latitude"].to_numpy(),
            df["manhattan"].to_numpy(),
            df["euclidean_km"].to_numpy(),
            df["manhattan_km"].to_numpy(),
            np.ones(len(df)),
        )
    )


train_X = get_input_matrix(train_df)

print(train_X.shape)
print(train_y.shape)



## === cell 9
(w, _, _, _) = np.linalg.lstsq(train_X, train_y.to_numpy(), rcond=None)
print(w)



## === cell 10
w_OLS = np.matmul(
    np.matmul(np.linalg.inv(np.matmul(train_X.T, train_X)), train_X.T),
    train_y.to_numpy(),
)
print(w_OLS)



## === cell 11
test_df = pd.read_csv("../input/test.csv")
print(test_df.dtypes)

test_df = test_df.dropna(how="any", axis="rows").copy()
test_df = test_df[
    (test_df.pickup_longitude.between(-74.3, -73.7))
    & (test_df.dropoff_longitude.between(-74.3, -73.7))
    & (test_df.pickup_latitude.between(40.5, 41.0))
    & (test_df.dropoff_latitude.between(40.5, 41.0))
].copy()
test_df = test_df.reset_index(drop=True)

test_df = add_travel_vector_features(test_df)

test_feature_df = test_df.drop(
    columns=["key", "pickup_datetime"], errors="ignore"
).copy()

test_X = get_input_matrix(test_feature_df)
val_X = get_input_matrix(val_df)

test_y_predictions = np.matmul(test_X, w)
val_y_predictions = np.matmul(val_X, w)

test_y_predictions = np.clip(test_y_predictions, 0.0, 250.0)
val_y_predictions = np.clip(val_y_predictions, 0.0, 250.0)

from sklearn.metrics import mean_squared_error

print(np.sqrt(mean_squared_error(val_y.to_numpy(), val_y_predictions)))

submission = pd.DataFrame(
    {"key": test_df["key"], "fare_amount": test_y_predictions},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)

print(os.listdir("."))
print(submission.head())
print("Submission rows:", len(submission))
