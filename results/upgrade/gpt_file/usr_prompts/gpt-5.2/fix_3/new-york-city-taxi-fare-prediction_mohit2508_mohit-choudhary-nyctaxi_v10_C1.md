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

5.83413681510617

# 6. Current score

7.67241

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1094.89132) has done: 'I remove the unavailable `feather` dependency and stop relying on an intermediate `.feather` file so `df` is always defined by reading the training CSV directly. I also fix the broken datetime parsing (`Series.str.split` API change) by using `pd.to_datetime` and extracting `year` and `hour` robustly for both train and test, keeping the same feature intent. To ensure the notebook runs end-to-end within the time limit and produces a valid submission, I keep the same LinearRegression model/features and write `submission.csv` with the required `key,fare_amount` columns. Finally, I guard optional plotting/correlation cells so missing optional libraries won’t crash execution.'
- What this solution (achieved 7.67241) has done: 'Your score is extremely far from the target (RMSE 1094 vs 5.83), which strongly suggests the submission has a systemic issue (usually wildly wrong predictions from out-of-distribution rows or invalid/negative fares). I keep your same LinearRegression model and same feature set, but make two minimal fixes that materially reduce RMSE: (1) clip predictions to a realistic fare range consistent with your training cleaning (prevents huge negative/positive outputs that dominate RMSE), and (2) apply the same basic sanity filters to the test features (replace impossible coordinates/passenger_count with NaN then fill with safe defaults) to prevent the model from extrapolating catastrophically. These changes preserve your core logic/semantics (same model, same features, same training approach) while addressing the most likely cause of the enormous error. The script still run end-to-end and write a valid `submission.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd

try:
    from scipy import stats as st
except Exception:
    st = None

try:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-whitegrid")
except Exception:
    plt = None

try:
    import seaborn as sns  # noqa: F401
except Exception:
    sns = None

from sklearn.linear_model import LinearRegression

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "../input/new-york-city-taxi-fare-prediction"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

print("Train exists:", os.path.exists(TRAIN_PATH))
print("Test exists:", os.path.exists(TEST_PATH))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB_PATH))



## === cell 1
NROWS = 5_000_000  # original intent

usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
df = pd.read_csv(TRAIN_PATH, nrows=NROWS, usecols=usecols, low_memory=True)
print(df.shape)
print(df.head())



## === cell 2
df = df[df.passenger_count > 0]

df = df[df.dropoff_latitude != 0]
df = df[df.pickup_longitude != 0]
df = df[df.pickup_latitude != 0]
df = df[df.dropoff_longitude != 0]

df = df[df.fare_amount > 2.5]
df = df[df.fare_amount < 100]

df = df.dropna()
print("After basic cleaning:", df.shape)



## === cell 3
dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
df["year"] = dt.dt.year
df["hour"] = dt.dt.hour * 100 + dt.dt.minute  # matches original "HHMM" integer intent
df = df.dropna(subset=["year", "hour"])

df[["year", "hour"]] = df[["year", "hour"]].astype(np.int32)
print(df[["pickup_datetime", "year", "hour"]].head())




## === cell 4
def select_within_newYork(_df, BB):
    return (
        (_df.pickup_longitude >= BB[0])
        & (_df.pickup_longitude <= BB[1])
        & (_df.pickup_latitude >= BB[2])
        & (_df.pickup_latitude <= BB[3])
        & (_df.dropoff_longitude >= BB[0])
        & (_df.dropoff_longitude <= BB[1])
        & (_df.dropoff_latitude >= BB[2])
        & (_df.dropoff_latitude <= BB[3])
    )


NYC = (-74.5, -72.8, 40.5, 41.8)
df = df[select_within_newYork(df, NYC)]
print("After NYC bounding box:", df.shape)




## === cell 5
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    miles = 6367 * c * 0.62137
    return miles


df["distance"] = haversine_np(
    df.pickup_longitude, df.pickup_latitude, df.dropoff_longitude, df.dropoff_latitude
)
print(df[["distance", "fare_amount"]].head())



## === cell 6
if st is not None:
    print("Co-relation b/w Fare and Distance")
    try:
        print(st.pearsonr(df.distance, df.fare_amount))
    except Exception as e:
        print("pearsonr failed:", repr(e))
    print(df["distance"].corr(df["fare_amount"], method="pearson"))

df = df[df.distance <= 30]
print("After distance<=30:", df.shape)



## === cell 7
if plt is not None:
    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    con = (
        (df.distance < 30)
        & (df.distance > 0.5)
        & (df.fare_amount > 0)
        & (df.fare_amount < 200)
    )
    axs.scatter(df.loc[con, "distance"], df.loc[con, "fare_amount"], alpha=0.2)
    axs.set_xlabel("Distance")
    axs.set_ylabel("Fare")
    axs.set_title("Distance vs Fare")
    plt.show()



## === cell 8
df["diff_lon"] = (df.dropoff_longitude - df.pickup_longitude).abs()
df["diff_lat"] = (df.dropoff_latitude - df.pickup_latitude).abs()
print(df[["diff_lon", "diff_lat"]].head())



## === cell 9
FEATURES = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "distance",
    "passenger_count",
]
TARGET = "fare_amount"

lr = LinearRegression()
lr.fit(df[FEATURES], df[TARGET])

print("Fitted LinearRegression.")
print("Intercept", round(float(lr.intercept_), 4))
print("Coefficients:", dict(zip(FEATURES, [round(float(c), 6) for c in lr.coef_])))



## === cell 10
test = pd.read_csv(TEST_PATH, low_memory=True)
print(test.head())

test["diff_lat"] = (test.dropoff_latitude - test.pickup_latitude).abs()
test["diff_long"] = (test.dropoff_longitude - test.pickup_longitude).abs()
test["distance"] = haversine_np(
    test.pickup_longitude,
    test.pickup_latitude,
    test.dropoff_longitude,
    test.dropoff_latitude,
)

dt_test = pd.to_datetime(test["pickup_datetime"], errors="coerce", utc=True)
test["year"] = dt_test.dt.year
test["hour"] = dt_test.dt.hour * 100 + dt_test.dt.minute
test[["year", "hour"]] = test[["year", "hour"]].fillna(0).astype(np.int32)

test_id = test["key"].astype(str).tolist()

test.loc[
    test["passenger_count"].isna() | (test["passenger_count"] <= 0), "passenger_count"
] = 1

mask_nyc = select_within_newYork(test, NYC)
for col in [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]:
    test.loc[~mask_nyc, col] = np.nan

fill_values = {
    "pickup_latitude": float(df["pickup_latitude"].median()),
    "pickup_longitude": float(df["pickup_longitude"].median()),
    "dropoff_latitude": float(df["dropoff_latitude"].median()),
    "dropoff_longitude": float(df["dropoff_longitude"].median()),
    "distance": float(df["distance"].median()),
    "passenger_count": float(df["passenger_count"].median()),
}
test["distance"] = test["distance"].replace([np.inf, -np.inf], np.nan)
test[FEATURES] = test[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(fill_values)



## === cell 11
preds = lr.predict(test[FEATURES])

preds = np.clip(preds, 2.5, 100.0)

sub = pd.DataFrame({"key": test_id, "fare_amount": preds})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("File exists:", os.path.exists("submission.csv"))
print("Submission columns:", list(sub.columns))
print(
    "fare_amount min/max:",
    float(sub["fare_amount"].min()),
    float(sub["fare_amount"].max()),
)
