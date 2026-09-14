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

3.12

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

5.53411

# 6. Current score

29.18355

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1069.49091) has done: 'I fix the data‑filtering logic (remove extreme fares and correctly apply both passenger‑count and downtown‑distance conditions) and clamp the predictions to non‑negative values. These minimal changes clean noisy rows that were inflating the RMSE and prevent absurd negative fare predictions, moving the score much closer to the target while keeping the original linear‑regression workflow. I also ensure the output directory exists before saving the submission.'
- What this solution (achieved 962.90876) has done: 'I remove the overly‑restrictive `idx` filter so the linear model is trained on all cleaned rows (instead of only those with non‑zero passengers and a downtown distance < 15 km). This gives the model more representative data and should dramatically lower the RMSE, moving the result toward the target value while keeping the original linear‑regression pipeline unchanged.'
- What this solution (achieved 4.9509772773362635e+69) has done: 'I add a simple log‑distance feature to give the linear model a tiny extra signal, which should nudge the validation RMSE closer to the target without altering the core workflow. I create the column in both train and test sets and include it in the feature list, then keep the existing pipeline and submission steps unchanged.'
- What this solution (achieved 29.18852) has done: 'I safeguard the predictions by replacing any non‑finite values (NaN or ±inf) with a reasonable fallback (the median fare from the training data) and clip the fares to a realistic upper bound. This prevents extreme or undefined numbers from inflating the RMSE, moving the score toward the target while leaving the core modeling pipeline unchanged.'
- What this solution (achieved 29.18355) has done: 'I remove the log‑transform of the target and train the Ridge model directly on the fare amount, add the raw latitude/longitude coordinates to the feature set, and set the regularisation strength to 0 (equivalent to an ordinary linear regression). These modest changes keep the overall pipeline intact while giving the model more useful signals, which should lower the RMSE toward the target. The prediction step is also updated to work without exponentiation.'
- What this solution (achieved 29.19991) has done: 'I train the model on the log‑transformed fare (log1p) and add a modest Ridge regularisation (α=1.0).  This keeps the same linear‑model pipeline while giving the model a smoother target and preventing over‑fitting, which should lower the RMSE toward the target.  Predictions are transformed back with `expm1` before clipping and saving, and the median fare is still computed from the original (non‑log) values for fallback handling.'
- What this solution (achieved 29.18355) has done: 'I switch the model to predict the raw fare amount directly (removing the log‑transform of the target) and use a weaker Ridge regularisation (α = 0.1). This keeps the same feature set and pipeline while giving the linear model a clearer signal, which should lower the RMSE toward the target. I also adjust the prediction step to omit the inverse log‑transform and keep the existing safety‑checks.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

np.random.seed(42)

INPUT_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction"

for dirname, _, filenames in os.walk(INPUT_PATH):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_data_set = pd.read_csv(
    os.path.join(INPUT_PATH, "train.csv"),
    nrows=5_000_000,  # reduced for quicker iteration
    parse_dates=["pickup_datetime"],
)
train_data_set.head(5)



## === cell 2
old_len = len(train_data_set)
train_data_set = train_data_set[
    (train_data_set.fare_amount >= 0.1) & (train_data_set.fare_amount <= 200)
]
new_len = len(train_data_set)
print(f"Removed {(old_len - new_len)} rows with extreme fares")
old_len = new_len



## === cell 3
train_data_set = train_data_set.dropna(how="any", axis="rows")
new_len = len(train_data_set)
print(f"Removed {(old_len - new_len)} rows with NA values")
old_len = new_len




## === cell 4
def select_within_boundingbox(df, box):
    return (
        (df.pickup_longitude >= box[0])
        & (df.pickup_longitude <= box[1])
        & (df.pickup_latitude >= box[2])
        & (df.pickup_latitude <= box[3])
        & (df.dropoff_longitude >= box[0])
        & (df.dropoff_longitude <= box[1])
        & (df.dropoff_latitude >= box[2])
        & (df.dropoff_latitude <= box[3])
    )


new_york_box = (-74.763379, -72.856164, 40.502009, 41.915509)
train_data_set = train_data_set[select_within_boundingbox(train_data_set, new_york_box)]
new_len = len(train_data_set)
print(f"Removed {(old_len - new_len)} rows outside NYC bounding box")
old_len = new_len




## === cell 5
def distance_on_the_sphere(lat1, lon1, lat2, lon2):
    earth_radius = 6371  # km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return earth_radius * c


train_data_set["distance"] = distance_on_the_sphere(
    train_data_set["pickup_latitude"],
    train_data_set["pickup_longitude"],
    train_data_set["dropoff_latitude"],
    train_data_set["dropoff_longitude"],
)

train_data_set["log_distance"] = np.log1p(train_data_set["distance"])



## === cell 6
old_len = len(train_data_set)
train_data_set = train_data_set[
    (train_data_set.distance > 0)
    & (train_data_set.distance < 100)
    & (train_data_set.passenger_count >= 0)  # allow zero passengers
    & (train_data_set.passenger_count <= 6)
]
new_len = len(train_data_set)
print(
    f"Removed {(old_len - new_len)} rows with unrealistic distance or passenger count"
)
old_len = new_len



## === cell 7
train_data_set["pickup_datetime"] = pd.to_datetime(train_data_set["pickup_datetime"])
train_data_set["hour"] = train_data_set["pickup_datetime"].dt.hour
train_data_set["year"] = train_data_set["pickup_datetime"].dt.year
train_data_set["day_of_week"] = train_data_set["pickup_datetime"].dt.dayofweek
train_data_set["is_rush_hour"] = train_data_set["hour"].apply(
    lambda x: 1 if (7 <= x <= 10) or (16 <= x <= 19) else 0
)



## === cell 8
nyc_down_town = (-74.0063889, 40.7141667)  # (lon, lat)
train_data_set["distance_to_downtown"] = distance_on_the_sphere(
    nyc_down_town[1],
    nyc_down_town[0],
    train_data_set["pickup_latitude"],
    train_data_set["pickup_longitude"],
)

train_data_set["distance_per_passenger"] = train_data_set["distance"] / (
    train_data_set["passenger_count"].replace(0, np.nan)
)
train_data_set["distance_per_passenger"] = train_data_set[
    "distance_per_passenger"
].fillna(train_data_set["distance"])



## === cell 9
features = [
    "hour",
    "year",
    "distance",
    "log_distance",
    "passenger_count",
    "is_rush_hour",
    "day_of_week",
    "distance_to_downtown",
    "distance_per_passenger",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
target = "fare_amount"

X = train_data_set[features].values
y_raw = train_data_set[target].values  # original fares
y = y_raw

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

linear_model = make_pipeline(StandardScaler(), Ridge(alpha=0.1))
linear_model.fit(X_train, y_train)

y_val_pred = linear_model.predict(X_val)
rmse = np.sqrt(((y_val_pred - y_val) ** 2).mean())
print(f"Validation RMSE (fare scale, raw target): {rmse:.4f}")



## === cell 10
test_data_set = pd.read_csv(os.path.join(INPUT_PATH, "test.csv"))
test_data_set["distance"] = distance_on_the_sphere(
    test_data_set["pickup_latitude"],
    test_data_set["pickup_longitude"],
    test_data_set["dropoff_latitude"],
    test_data_set["dropoff_longitude"],
)

test_data_set["log_distance"] = np.log1p(test_data_set["distance"])

test_data_set["distance_to_downtown"] = distance_on_the_sphere(
    nyc_down_town[1],
    nyc_down_town[0],
    test_data_set["pickup_latitude"],
    test_data_set["pickup_longitude"],
)
test_data_set["pickup_datetime"] = pd.to_datetime(test_data_set["pickup_datetime"])
test_data_set["hour"] = test_data_set["pickup_datetime"].dt.hour
test_data_set["year"] = test_data_set["pickup_datetime"].dt.year
test_data_set["day_of_week"] = test_data_set["pickup_datetime"].dt.dayofweek
test_data_set["is_rush_hour"] = test_data_set["hour"].apply(
    lambda x: 1 if (7 <= x <= 10) or (16 <= x <= 19) else 0
)

test_data_set["distance_per_passenger"] = test_data_set["distance"] / (
    test_data_set["passenger_count"].replace(0, np.nan)
)
test_data_set["distance_per_passenger"] = test_data_set[
    "distance_per_passenger"
].fillna(test_data_set["distance"])



## === cell 11
XTEST = test_data_set[features].values
y_test_pred = linear_model.predict(XTEST)

median_fare = np.median(y_raw)  # fallback from original fares
y_test_pred = np.where(np.isfinite(y_test_pred), y_test_pred, median_fare)

y_test_pred = np.clip(y_test_pred, a_min=0, a_max=200)

submission = pd.DataFrame(
    {"key": test_data_set["key"], "fare_amount": y_test_pred},
    columns=["key", "fare_amount"],
)
submission_path = os.path.join("/kaggle/working", "submission.csv")

os.makedirs(os.path.dirname(submission_path), exist_ok=True)

submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
