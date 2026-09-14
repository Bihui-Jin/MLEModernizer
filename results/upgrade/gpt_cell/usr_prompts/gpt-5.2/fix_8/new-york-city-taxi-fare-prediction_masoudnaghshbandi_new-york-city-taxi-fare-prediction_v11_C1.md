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
xgboost==2.0.3

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

3.7787

# 6. Current score

7.92696

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.32437) has done: 'You don’t currently have a reliable submission because (a) scaling is computed but never applied to the data you train/predict on, and (b) you drop rows from `test` then predict on the filtered rows, which can silently change row counts and make alignment fragile. I keep your exact model and feature set, but make scaling actually take effect by transforming both train/valid/test consistently (same scaler fit on train only). I also preserve every test row by filtering only the training data (so the submission always has exactly the same keys as `test.csv`), and ensure the submission is built directly from the original test `key` order. These minimal fixes typically improve RMSE (moving toward your 3.7787 target) while keeping the core logic intact and guaranteeing a valid `submission1.csv`.'
- What this solution (achieved 5.31854) has done: 'You’re currently underperforming the target (RMSE 5.324 > 3.7787), so we make the smallest changes that legitimately reduce RMSE without changing your model type or feature set. The biggest avoidable error is training on “fare_amount” outliers (including negative/huge fares), which XGBoost try to fit and hurt RMSE on the normal test distribution; we add a standard NYC Taxi fare cleaning filter. We also ensure the model is trained with the up-to-date XGBoost objective name (`reg:squarederror`) for correct RMSE-oriented optimization (same semantics, fewer quirks than deprecated `reg:linear`). Everything else (same rows limit, same features, same scaling usage, same submission schema and path) stays the same.'
- What this solution (achieved 5.42354) has done: 'Your RMSE (5.31854) is worse than the target (3.7787), so we should make a small, legitimate change that typically improves generalization without changing your model type or feature set. The biggest issue left is that `pickup_datetime` is dropped entirely, yet it contains strong signal (hour/day-of-week/month/year) for fare; we add simple time-derived numeric features for both train and test (same column set) while keeping the same XGBoost regressor and training loop. We also ensure train/test preprocessing is consistent (datetime parsing, NA handling, column order), and keep your existing cleaning rules intact. This should move RMSE downward toward the target band while remaining a minimal extension of your current pipeline.'
- What this solution (achieved 5.77097) has done: 'Your current RMSE (5.42354) is worse than the target (3.7787), so we should make a small, legitimate improvement that typically reduces error without changing the model type or overall training flow. The biggest remaining gap is feature signal: you already add time features, but you still lack the core distance feature that dominates fare; we add a simple haversine distance (plus a couple of minimal, standard geo deltas) using the same input columns and keep XGBoost exactly as-is. We also apply the same basic coordinate/passenger sanity filtering to the training set in a single boolean mask (no semantic change, just safer) and keep the test rows untouched to guarantee submission alignment. These changes usually move RMSE down materially while staying within your existing pipeline and producing the same `submission1.csv`.'
- What this solution (achieved 8.15747) has done: 'Your current RMSE (5.77097) is worse than the target (3.7787), so we should make a small, legitimate improvement that usually reduces error without changing your model type or overall training flow. The biggest avoidable miss is that the model never sees the strong relationship between passenger count and pricing structure beyond being a raw integer; adding the common “passenger_count==1” and capped/log variants tends to improve generalization while keeping the same features + XGBoost approach. We also clip negative/zero/implausible distance-derived rows in training (but never drop test rows) to reduce label noise that disproportionately hurts RMSE. Finally, we keep submission alignment exactly as before (keys from `test_raw` in original order), so the output remains valid.'
- What this solution (achieved 7.92696) has done: 'Your current RMSE (8.15747) is far worse than the target (3.7787), and the main issue is not the model but a data bug: you trained on `train.csv` while `labels.csv` is available and (in this environment) is typically the cleaned/curated training set used for this competition; switching to `labels.csv` keeps the exact same pipeline but reduces label noise/outliers. I keep your exact feature engineering functions and XGBoost regressor settings, but read from `labels.csv` (same columns) and keep all cleaning rules identical. I also make sure the train/test column alignment stays deterministic by capturing the final feature column order from the processed training data and reindexing test to it (same semantics, fewer accidental mismatches). This is a minimal change that usually moves RMSE materially downward toward the target without changing architecture or training approach.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE



## === cell 2
train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/labels.csv", nrows=1000000
)
test_raw = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 3
train.shape, test_raw.shape



## === cell 4
train.head()



## === cell 5
train.isnull().sum()



## === cell 6
train = train.dropna(how="any", axis="rows")



## === cell 7
test_raw.isnull().sum()



## === cell 8
train.head()



## === cell 9
train["fare_amount"].describe()



## === cell 10
mask_basic = (
    (train["pickup_longitude"] != 0)
    & (train["pickup_latitude"] != 0)
    & (train["dropoff_longitude"] != 0)
    & (train["dropoff_latitude"] != 0)
    & (train["passenger_count"] != 208)
    & (train["passenger_count"] > 0)
    & (train["passenger_count"] <= 5)
)
train = train.loc[mask_basic].copy()



## === cell 11
train.head()



## === cell 12
train = train[(train["fare_amount"] > 0) & (train["fare_amount"] <= 250)].copy()



## === cell 13
train.drop(["key"], axis=1, inplace=True)




## === cell 14
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["pickup_hour"] = dt.dt.hour.astype("float32")
    df["pickup_dayofweek"] = dt.dt.dayofweek.astype("float32")
    df["pickup_month"] = dt.dt.month.astype("float32")
    df["pickup_year"] = dt.dt.year.astype("float32")
    return df


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    dlon = (df["dropoff_longitude"] - df["pickup_longitude"]).astype("float32")
    dlat = (df["dropoff_latitude"] - df["pickup_latitude"]).astype("float32")
    df["delta_lon"] = dlon
    df["delta_lat"] = dlat
    df["manhattan_dist"] = (np.abs(dlon) + np.abs(dlat)).astype("float32")

    R = 6371.0
    lat1 = np.radians(df["pickup_latitude"].astype("float64"))
    lat2 = np.radians(df["dropoff_latitude"].astype("float64"))
    dlat_r = lat2 - lat1
    dlon_r = np.radians(
        df["dropoff_longitude"].astype("float64")
        - df["pickup_longitude"].astype("float64")
    )
    a = np.sin(dlat_r / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon_r / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    df["haversine_km"] = (R * c).astype("float32")
    return df


def add_passenger_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pc = (
        pd.to_numeric(df["passenger_count"], errors="coerce")
        .fillna(0)
        .astype("float32")
    )
    df["passenger_count"] = pc
    df["is_single_passenger"] = (pc == 1).astype("float32")
    df["passenger_count_capped"] = np.minimum(pc, 4).astype("float32")
    df["log1p_passenger_count"] = np.log1p(pc).astype("float32")
    return df


train = add_time_features(train)
train = add_geo_features(train)
train = add_passenger_features(train)



## === cell 15
train.drop(["pickup_datetime"], axis=1, inplace=True)

train.dropna(inplace=True)

train.drop(
    train.index[
        (train.pickup_longitude < -75)
        | (train.pickup_longitude > -72)
        | (train.pickup_latitude < 40)
        | (train.pickup_latitude > 42)
    ],
    inplace=True,
)
train.drop(
    train.index[
        (train.dropoff_longitude < -75)
        | (train.dropoff_longitude > -72)
        | (train.dropoff_latitude < 40)
        | (train.dropoff_latitude > 42)
    ],
    inplace=True,
)

train = train[(train["haversine_km"] > 0) & (train["haversine_km"] <= 100)].copy()
train = train[(train["manhattan_dist"] > 0) & (train["manhattan_dist"] <= 2.0)].copy()



## === cell 16
train.head()



## === cell 17
X, y = train.drop("fare_amount", axis=1), train["fare_amount"]
feature_cols = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12
)



## === cell 18
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)



## === cell 19
xgb_r = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=400, seed=123)



## === cell 20
xgb_r.fit(X_train_s, y_train)



## === cell 21
y_pred = xgb_r.predict(X_test_s)



## === cell 22
rmse = np.sqrt(MSE(y_test, y_pred))
print("RMSE : % f" % (rmse))



## === cell 23
test = test_raw.copy()



## === cell 24
test.head()



## === cell 25
test = test.fillna(0)



## === cell 26
test.head()



## === cell 27
test = add_time_features(test)
test = add_geo_features(test)
test = add_passenger_features(test)

test_features = test.drop(["key", "pickup_datetime"], axis=1)
test_features = test_features.reindex(columns=feature_cols, fill_value=0)



## === cell 28
test_s = scaler.transform(test_features)



## === cell 29
new_pred = xgb_r.predict(test_s)



## === cell 30
submission = pd.DataFrame({"key": test_raw["key"], "fare_amount": new_pred})



## === cell 31
submission.head()



## === cell 32
submission.to_csv("submission1.csv", index=False)
print("Wrote submission1.csv with shape:", submission.shape)
