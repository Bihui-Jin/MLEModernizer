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

5.51467

# 6. Current score

12.56343

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1038.76494) has done: 'I fix the boolean filter used for selecting training rows – the original expression lacked parentheses, causing an incorrect mask and a near‑empty training set, which led to a huge RMS error. I also clip negative predictions to zero, which is sensible for fare amounts and helps lower the RMSE. The rest of the pipeline remains unchanged.'
- What this solution (achieved 969.4979) has done: 'I broaden the training data by removing the overly‑strict distance filter and include additional useful time‑ and location‑based features (day‑of‑week, rush‑hour flag, distance‑to‑downtown) in both training and test sets. This keeps the linear‑regression core unchanged while giving the model more relevant information, which should substantially lower the RMSE toward the target.'
- What this solution (achieved 251.37551) has done: 'I add modest log‑transform features for the highly skewed distance‑related columns (and passenger count) and use them instead of the raw values, then refit the same Ridge‑regression pipeline with a slightly stronger regularisation (alpha = 1.0). These changes keep the overall model structure identical while improving linear fit calibration, which should lower the RMSE toward the target without over‑hauling the core logic.'
- What this solution (achieved 251.22971) has done: 'Implemented a modest outlier filter on the fare amount (capping extreme values) and reduced the Ridge regularization strength. These tweaks clean the training distribution and allow the model to fit more closely, which should lower the validation RMSE and move the score toward the target without altering the core pipeline.'
- What this solution (achieved 1.6215297917026728e+97) has done: 'Implemented a modest feature expansion and slight regularisation tweak while retaining the original pipeline.  
- Added the raw `distance` and `passenger_count` columns to the feature set, giving the linear model direct access to these strong predictors.  
- Switched ridge regularisation from `alpha=0.1` to `alpha=1.0` to improve stability after the extra features.  
- Updated the feature list in both training and test preprocessing cells so the model uses the same columns throughout.'
- What this solution (achieved 12.56312) has done: 'I add a safe upper‑bound on the log‑predictions before converting them back to the original scale, preventing overflow that caused the astronomically large RMSE. I also raise the Ridge regularisation (α) modestly to keep coefficient magnitudes in check. These tweaks keep the overall pipeline unchanged while ensuring the model’s predictions stay within a realistic range, moving the score toward the target.'
- What this solution (achieved 12.56343) has done: 'I increase the training sample size (from 2 M to 5 M rows) and lessen the ridge regularisation (α = 0.5). More data gives the model a better view of the fare distribution, while a weaker penalty lets it fit the log‑fare relationship more closely, which should lower the RMSE toward the target without altering the overall pipeline.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

train_data_set = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=5_000_000,
    parse_dates=["pickup_datetime"],
)
print("Training rows loaded:", len(train_data_set))




## === cell 1
old_len = len(train_data_set)
train_data_set = train_data_set[train_data_set.fare_amount >= 0.1]
new_len = len(train_data_set)
print(f"Removed {old_len - new_len} rows with fare < 0.1")




## === cell 2
old_len = len(train_data_set)
train_data_set = train_data_set[train_data_set.fare_amount <= 200]
new_len = len(train_data_set)
print(f"Removed {old_len - new_len} rows with fare > 200")




## === cell 3
old_len = len(train_data_set)
train_data_set = train_data_set.dropna(how="any", axis="rows")
new_len = len(train_data_set)
print(f"Removed {old_len - new_len} rows with NaNs")




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

old_len = len(train_data_set)
train_data_set = train_data_set[select_within_boundingbox(train_data_set, new_york_box)]
new_len = len(train_data_set)
print(f"Removed {old_len - new_len} rows outside NYC bounding box")




## === cell 5
def distance_on_the_sphere(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometers."""
    earth_radius = 6371.0
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

old_len = len(train_data_set)
train_data_set = train_data_set[train_data_set["distance"] <= 100]
new_len = len(train_data_set)
print(f"Removed {old_len - new_len} rows with distance > 100 km")




## === cell 6
train_data_set["hour"] = train_data_set["pickup_datetime"].dt.hour
train_data_set["year"] = train_data_set["pickup_datetime"].dt.year
train_data_set["day_of_week"] = train_data_set["pickup_datetime"].dt.dayofweek
train_data_set["is_rush_hour"] = train_data_set["hour"].apply(
    lambda x: 1 if (7 <= x <= 10) or (16 <= x <= 19) else 0
)




## === cell 7
nyc_down_town = (-74.0063889, 40.7141667)  # (lon, lat)

train_data_set["distance_to_downtown"] = distance_on_the_sphere(
    train_data_set["pickup_latitude"],
    train_data_set["pickup_longitude"],
    nyc_down_town[1],
    nyc_down_town[0],
)

train_data_set["log_distance"] = np.log1p(train_data_set["distance"])
train_data_set["log_distance_to_downtown"] = np.log1p(
    train_data_set["distance_to_downtown"]
)
train_data_set["log_passenger_count"] = np.log1p(train_data_set["passenger_count"])




## === cell 8
features = [
    "hour",
    "year",
    "log_distance",
    "log_passenger_count",
    "log_distance_to_downtown",
    "is_rush_hour",
    "day_of_week",
    "distance",  # raw distance
    "passenger_count",  # raw passenger count
]
target = "fare_amount"

X = train_data_set[features].values
y = train_data_set[target].values

y_log = np.log1p(y)

X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

ridge_model = Ridge(alpha=0.5, random_state=42)
ridge_model.fit(X_train_scaled, y_train_log)

y_val_pred_log = ridge_model.predict(X_val_scaled)

MAX_LOG = np.log1p(200)
y_val_pred_log = np.clip(y_val_pred_log, a_min=None, a_max=MAX_LOG)

y_val_pred = np.expm1(y_val_pred_log)
y_val_true = np.expm1(y_val_log)
val_rmse = np.sqrt(((y_val_true - y_val_pred) ** 2).mean())
print(f"Validation RMSE (log‑model, original scale): {val_rmse:.5f}")

X_scaled_full = scaler.fit_transform(X)
ridge_model.fit(X_scaled_full, y_log)

linear_model = ridge_model  # Alias for downstream cells




## === cell 9
test_data_set = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)

test_data_set["distance"] = distance_on_the_sphere(
    test_data_set["pickup_latitude"],
    test_data_set["pickup_longitude"],
    test_data_set["dropoff_latitude"],
    test_data_set["dropoff_longitude"],
)

test_data_set["distance_to_downtown"] = distance_on_the_sphere(
    test_data_set["pickup_latitude"],
    test_data_set["pickup_longitude"],
    nyc_down_town[1],
    nyc_down_town[0],
)

test_data_set["hour"] = test_data_set["pickup_datetime"].dt.hour
test_data_set["year"] = test_data_set["pickup_datetime"].dt.year
test_data_set["day_of_week"] = test_data_set["pickup_datetime"].dt.dayofweek
test_data_set["is_rush_hour"] = test_data_set["hour"].apply(
    lambda x: 1 if (7 <= x <= 10) or (16 <= x <= 19) else 0
)

test_data_set["log_distance"] = np.log1p(test_data_set["distance"])
test_data_set["log_distance_to_downtown"] = np.log1p(
    test_data_set["distance_to_downtown"]
)
test_data_set["log_passenger_count"] = np.log1p(test_data_set["passenger_count"])

X_test = test_data_set[features].values
X_test_scaled = scaler.transform(X_test)

y_pred_log = linear_model.predict(X_test_scaled)

y_pred_log = np.clip(y_pred_log, a_min=None, a_max=MAX_LOG)

y_pred_final = np.expm1(y_pred_log)

y_pred_final = np.maximum(y_pred_final, 0)

submission = pd.DataFrame(
    {"key": test_data_set["key"], "fare_amount": y_pred_final},
    columns=["key", "fare_amount"],
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")
