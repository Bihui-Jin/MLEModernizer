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

No external packages required in the script and installed.

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

4.13939939564504

# 6. Current score

5.51407

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.02934) has done: 'I replace the failing TensorFlow‑based pipeline with a minimal, reliable solution: load the training data, compute the overall mean fare, and use this constant prediction for every test row. This removes the TF 1.x incompatibilities, fixes the date‑part feature errors, and guarantees a valid `submission.csv` file with the required columns. The change is small, keeps the overall workflow (load data → generate predictions → save submission) and moves the script from “no output” to a working baseline submission.'
- What this solution (achieved 10.02257) has done: 'Implemented robust data cleaning and a regularized linear regression to avoid SVD convergence errors. The script now drops rows with missing or non‑positive fare values, adds a small ridge regularization term when solving for coefficients, and proceeds to generate predictions and a proper `submission.csv`. These changes fix the runtime errors while preserving the original feature engineering and model approach, moving the solution toward the targeted RMSE.'
- What this solution (achieved 10.33833) has done: 'I add a few inexpensive feature enhancements (distance‑squared) and train the linear model on the log‑transformed fare to better capture multiplicative patterns. I also standardise the non‑intercept columns before solving the ridge regression, then back‑transform predictions with `expm1`. These changes keep the overall linear‑regression pipeline intact while aiming to lower the RMSE toward the target.'
- What this solution (achieved 10.34269) has done: 'I extend the feature set with cyclical time encodings (sin/cos of hour, day‑of‑week, month), an interaction between distance and passenger count, and a slightly weaker ridge regularisation. These inexpensive additions keep the linear‑regression pipeline unchanged while giving the model more expressive power, which should lower the RMSE toward the target.'
- What this solution (achieved 6.31136) has done: 'I increase the training sample size, tighten data cleaning, add a simple √distance feature, and reduce ridge regularisation. These minor tweaks keep the linear‑regression pipeline intact while giving the model more data and expressive power, which should lower the RMSE toward the target.'
- What this solution (achieved 5.51407) has done: 'I increase the training sample size, add two inexpensive nonlinear distance features (log‑distance and distance per passenger), and strengthen the ridge regularisation slightly. These changes keep the linear‑regression pipeline intact while giving the model a bit more data and expressive power, which should lower the RMSE toward the target.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd

print("Input directories:", os.listdir("../input"))



## === cell 1
test_path = "../input/new-york-city-taxi-fare-prediction/test.csv"
df_test = pd.read_csv(
    test_path,
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    parse_dates=["pickup_datetime"],
)
print("Test rows:", df_test.shape[0])



## === cell 2
train_path = "../input/new-york-city-taxi-fare-prediction/train.csv"

sample_n = 5_000_000  # 5 M rows ≈ 300 MB

df_train = pd.read_csv(
    train_path,
    usecols=[
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    parse_dates=["pickup_datetime"],
    nrows=sample_n,
)

df_train = df_train.dropna()
df_train = df_train[df_train["fare_amount"] > 0]


def haversine(lon1, lat1, lon2, lat2):
    """km distance between two lon/lat points (vectorised)."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km


dist_train = haversine(
    df_train["pickup_longitude"].values,
    df_train["pickup_latitude"].values,
    df_train["dropoff_longitude"].values,
    df_train["dropoff_latitude"].values,
)
df_train = df_train[dist_train <= 100.0]  # keep trips ≤ 100 km


def build_features(df):
    """Return a 2‑D array with intercept and engineered features."""
    dist = haversine(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    )
    dist_sq = dist**2
    dist_sqrt = np.sqrt(dist)
    log_dist = np.log1p(dist)
    pax = df["passenger_count"].values.astype(float)
    dist_per_pax = dist / (pax + 1e-3)  # avoid division by zero

    hour = df["pickup_datetime"].dt.hour.values
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    dow = df["pickup_datetime"].dt.dayofweek.values
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    month = df["pickup_datetime"].dt.month.values
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    dist_pax = dist * pax

    X = np.column_stack(
        (
            np.ones_like(dist),  # intercept
            dist,
            dist_sq,
            dist_sqrt,
            log_dist,
            dist_per_pax,
            pax,
            hour,
            dow,
            month,
            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos,
            month_sin,
            month_cos,
            dist_pax,
        )
    )
    return X


X_train_raw = build_features(df_train)
y_train = np.log1p(df_train["fare_amount"].values)  # log‑transform target

feature_means = X_train_raw[:, 1:].mean(axis=0, keepdims=True)
feature_stds = X_train_raw[:, 1:].std(axis=0, keepdims=True) + 1e-8
X_train = X_train_raw.copy()
X_train[:, 1:] = (X_train[:, 1:] - feature_means) / feature_stds

lambda_reg = 1e-4
XtX = X_train.T @ X_train
XtX_reg = XtX + lambda_reg * np.eye(XtX.shape[0])
beta = np.linalg.solve(XtX_reg, X_train.T @ y_train)

print("Linear model coefficients (log‑target):", beta)



## === cell 3
X_test_raw = build_features(df_test)
X_test = X_test_raw.copy()
X_test[:, 1:] = (X_test[:, 1:] - feature_means) / feature_stds

preds_log = X_test @ beta
preds = np.expm1(preds_log)  # inverse of log1p

preds = np.clip(preds, a_min=0.0, a_max=500.0)

submission = pd.DataFrame({"key": df_test["key"], "fare_amount": preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
