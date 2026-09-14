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

3.10

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
seaborn==0.12.2
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

10.01913

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'I fix the LinearRegression initialization (remove the deprecated `normalize` argument) and add a small step to align one‑hot encoded weekday columns between train and test so the model sees identical feature sets. These changes resolve the runtime errors, ensure a valid prediction array, and allow the script to write a proper `Submission.csv` file without altering the core modeling logic.'
- What this solution (achieved 15.47835) has done: 'I adjust the feature scaling to use standard deviation (more sensible than variance) and switch the linear model to a regularized Ridge regression, then compute the proper RMSE metric and clip extreme predictions to a realistic fare range. These targeted tweaks keep the overall pipeline intact while expectedly moving the RMSE far closer to the target score.'
- What this solution (achieved 19.16724) has done: 'I fix the NaN issue caused by taking the log of negative fares, ensure the model is properly fitted, and make the submission creation use the resulting predictions. This resolves the runtime errors and lets the script produce a valid `Submission.csv` while keeping the core modeling approach unchanged.'
- What this solution (achieved 19.16459) has done: 'I keep the overall pipeline unchanged but stop scaling the one‑hot weekday columns, which were being centered and divided by their standard deviation and hurting the Ridge model. Instead, I use the raw feature values (no StandardScaler) so numeric distances stay on their original scale while dummy variables remain 0/1. This small change should lower the validation RMSE and bring the score closer to the target without altering the core modeling logic.'
- What this solution (achieved 18.41943) has done: 'I keep the overall pipeline unchanged but add the most predictive geographic coordinates back into the model and standardize the distance‑based features. I also lower the Ridge regularisation (α = 0.1) so the linear model can utilise the richer feature set. These small, targeted tweaks should lower the validation RMSE and move the score toward the target without altering the core modelling approach.'
- What this solution (achieved 17.63121) has done: 'I added the missing imports, fixed the file‑loading paths, ensured all intermediate variables are defined, aligned the one‑hot weekday columns between train and test, incorporated `passenger_count` into the numeric scaling, and correctly built and saved the submission CSV. These changes remove the runtime errors, let the model train and predict, and produce a valid `Submission.csv` that can be submitted to Kaggle.'
- What this solution (achieved 10.01913) has done: 'The script now samples a manageable subset of the training rows (2 million by default) before any scaling or model fitting, which cuts the data‑size‑dependent work dramatically while keeping the same feature engineering, Ridge model, and evaluation logic. All other steps stay unchanged, and the random seed guarantees reproducible results.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from pathlib import Path




## === cell 1
def read_csv_fallback(filename: str, **kwargs):
    """Read a CSV from /kaggle/input or a sub‑folder if needed."""
    base = Path("/kaggle/input")
    possible = [
        base / filename,
        base / "new-york-city-taxi-fare-prediction" / filename,
    ]
    for p in possible:
        if p.exists():
            return pd.read_csv(p, **kwargs)
    raise FileNotFoundError(f"{filename} not found in /kaggle/input")


train_dtype = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
test_dtype = {
    "key": "object",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

train_data = read_csv_fallback("train.csv", dtype=train_dtype, low_memory=False)
test_data = read_csv_fallback("test.csv", dtype=test_dtype, low_memory=False)



## === cell 2
train_data.head()



## === cell 3
test_data.head()



## === cell 4
train_data["Difference_longitude"] = np.abs(
    train_data["pickup_longitude"] - train_data["dropoff_longitude"]
)
train_data["Difference_latitude"] = np.abs(
    train_data["pickup_latitude"] - train_data["dropoff_latitude"]
)

test_data["Difference_longitude"] = np.abs(
    test_data["pickup_longitude"] - test_data["dropoff_longitude"]
)
test_data["Difference_latitude"] = np.abs(
    test_data["pickup_latitude"] - test_data["dropoff_latitude"]
)



## === cell 5
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")
train_data = train_data[train_data["fare_amount"] > 0].reset_index(drop=True)
print(f"After removing non‑positive fares: {len(train_data)}")



## === cell 6
dt_train = pd.to_datetime(train_data["pickup_datetime"], errors="coerce")
dt_test = pd.to_datetime(test_data["pickup_datetime"], errors="coerce")

train_data = train_data[dt_train.notna()].reset_index(drop=True)
dt_train = dt_train[dt_train.notna()]

train_data["pickuptime"] = dt_train.dt.hour * 100 + dt_train.dt.minute
test_data["pickuptime"] = dt_test.dt.hour * 100 + dt_test.dt.minute

train_data["Weekday"] = dt_train.dt.weekday
test_data["Weekday"] = dt_test.dt.weekday

train_data.drop(columns=["pickup_datetime"], inplace=True)
test_data.drop(columns=["pickup_datetime"], inplace=True)



## === cell 7
pass



## === cell 8
pass



## === cell 9
pass



## === cell 10
pass



## === cell 11
pass



## === cell 12
pass



## === cell 13
R = 6373.0  # Earth radius in km


def haversine(df):
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c * 0.621  # convert km → miles


train_data["Distance"] = haversine(train_data)
test_data["Distance"] = haversine(test_data)



## === cell 14
airport_lat = np.radians(40.6413111)
airport_lon = np.radians(-73.7781391)


def airport_distance(lat_col, lon_col):
    lat = np.radians(lat_col)
    lon = np.radians(lon_col)
    dlon = airport_lon - lon
    dlat = airport_lat - lat
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat) * np.cos(airport_lat) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c * 0.621


train_data["Pickup_Distance_airport"] = airport_distance(
    train_data["pickup_latitude"], train_data["pickup_longitude"]
)
train_data["Dropoff_Distance_airport"] = airport_distance(
    train_data["dropoff_latitude"], train_data["dropoff_longitude"]
)

test_data["Pickup_Distance_airport"] = airport_distance(
    test_data["pickup_latitude"], test_data["pickup_longitude"]
)
test_data["Dropoff_Distance_airport"] = airport_distance(
    test_data["dropoff_latitude"], test_data["dropoff_longitude"]
)



## === cell 15
dist_cols = ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]
for col in dist_cols:
    train_data[col] = np.round(train_data[col], 2)
    test_data[col] = np.round(test_data[col], 2)



## === cell 16
sample_size = 2_000_000  # adjust as needed for the 600 s budget
if len(train_data) > sample_size:
    train_data = train_data.sample(n=sample_size, random_state=42).reset_index(
        drop=True
    )

num_cols = [
    "Difference_longitude",
    "Difference_latitude",
    "Distance",
    "Pickup_Distance_airport",
    "Dropoff_Distance_airport",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "pickuptime",
    "passenger_count",
]

stats = train_data[num_cols].agg(["mean", "std"]).T
train_data[num_cols] = (train_data[num_cols] - stats["mean"]) / stats["std"]
test_data[num_cols] = (test_data[num_cols] - stats["mean"]) / stats["std"]

for base in ["Distance", "passenger_count", "pickuptime"]:
    train_data[f"{base}_sq"] = train_data[base] ** 2
    test_data[f"{base}_sq"] = test_data[base] ** 2



## === cell 17
train_features = set(train_data.columns) - {"key", "fare_amount"}
test_features = set(test_data.columns) - {"key"}

for col in train_features - test_features:
    test_data[col] = 0
for col in test_features - train_features:
    train_data[col] = 0



## === cell 18
X = train_data.drop(columns=["key", "fare_amount"])
y = train_data["fare_amount"]
X_test = test_data.drop(columns=["key"])

X_array = X.values.astype(np.float32)
X_test_array = X_test.values.astype(np.float32)



## === cell 19
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_array, y, test_size=0.01, random_state=80
)



## === cell 20
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

alpha_val = 0.001
ridge_raw = Ridge(
    alpha=alpha_val,
    random_state=80,
    solver="sag",
    max_iter=100,
    tol=1e-3,
    fit_intercept=True,
)
ridge_raw.fit(X_train, y_train)
pred_raw_val = ridge_raw.predict(X_val)
rmse_raw = mean_squared_error(y_val, pred_raw_val, squared=False)

ridge_log = Ridge(
    alpha=alpha_val,
    random_state=80,
    solver="sag",
    max_iter=100,
    tol=1e-3,
    fit_intercept=True,
)
y_train_log = np.log1p(y_train.clip(lower=0))
ridge_log.fit(X_train, y_train_log)
pred_log_val = np.expm1(ridge_log.predict(X_val))
rmse_log = mean_squared_error(y_val, pred_log_val, squared=False)

if rmse_log < rmse_raw:
    chosen_model = ridge_log
    use_log = True
    best_rmse = rmse_log
else:
    chosen_model = ridge_raw
    use_log = False
    best_rmse = rmse_raw

print(f"Validation RMSE (raw): {rmse_raw:.4f}")
print(f"Validation RMSE (log): {rmse_log:.4f}")
print(
    f"Chosen model uses {'log‑target' if use_log else 'raw'} with RMSE: {best_rmse:.4f}"
)



## === cell 21
if use_log:
    y_full = np.log1p(y.clip(lower=0))
else:
    y_full = y

chosen_model.fit(X_array, y_full)



## === cell 22
test_pred = chosen_model.predict(X_test_array)
if use_log:
    test_pred = np.expm1(test_pred)

test_pred = np.clip(test_pred, 0, 300)
test_pred = np.round(test_pred, 2)



## === cell 23
Submission = pd.DataFrame({"key": test_data["key"], "fare_amount": test_pred})
Submission = Submission[["key", "fare_amount"]]



## === cell 24
output_path = Path("/kaggle/working/Submission.csv")
Submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
