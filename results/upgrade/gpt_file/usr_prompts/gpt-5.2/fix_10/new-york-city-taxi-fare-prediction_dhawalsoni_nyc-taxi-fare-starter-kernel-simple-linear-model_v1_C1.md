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

5.69253

# 6. Current score

37.29238

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 762.61089) has done: 'I fix the scikit-learn runtime error by removing the deprecated `normalize` argument from `LinearRegression` so the model can train. I also correct a bug in the Haversine distance feature where `lon2` mistakenly uses `dropoff_latitude` instead of `dropoff_longitude`, which would otherwise degrade predictions and score. To keep feature scaling consistent and avoid train/test drift, I compute normalization statistics on the training set and apply them to the test set. Finally, I ensure the submission file is written as a valid Kaggle CSV with the required `key,fare_amount` columns (no index), using a `.csv` suffix.'
- What this solution (achieved 762.61089) has done: 'I fix the pandas `Int64` dtype issue by converting `weekday` to a string/category-friendly type before mapping day names, which unblocks the one-hot encoding step. Then I fix the scikit-learn `feature_names` error by forcing all feature column names to strings consistently for train/test before fitting and predicting. Finally, I ensure the model is successfully fitted before inference and that the submission is written as a valid `submission.csv` with the required `key,fare_amount` columns and no index. These changes are execution/correctness fixes and should also substantially improve RMSE versus the currently broken pipeline (no valid score).'
- What this solution (achieved 995.38027) has done: 'Your RMSE is exploding because the linear model is being trained on unfiltered labels with many impossible/garbage fares and coordinates, so a small fraction of extreme outliers dominate the fit and produce wildly wrong predictions. To move RMSE toward the target with minimal logic change, I add standard NYC Taxi Fare “sanity filters” on the *training* data only (valid lat/lon ranges, passenger_count bounds, and plausible fare range) before feature engineering and fitting the same `LinearRegression`. I also clip negative predictions to 0 (fares can’t be negative), which typically reduces RMSE without changing the modeling approach. Everything else (features, model, training loop, submission format/path) stays the same.'
- What this solution (achieved 995.42416) has done: 'Your current RMSE (995) is far above target, so we should make small, legitimate fixes that improve generalization without changing the core “engineer a few geo/time features + LinearRegression” approach. The biggest low-risk issue here is that you never remove rows with invalid `pickup_datetime` (you coerce to NaT but keep them), and you also round engineered distance features to 2 decimals before fitting, which discards signal and can materially hurt RMSE. I (1) drop training rows where `pickup_datetime` can’t be parsed, and (2) stop rounding the distance features (keep full float precision) while keeping everything else (filters, features, LinearRegression, clipping, submission format) the same. These are minimal changes that should move RMSE sharply down toward the target band while preserving your pipeline.'
- What this solution (achieved 1140.21714) has done: 'Your current RMSE is catastrophically high (995 vs target 5.69), which strongly suggests the model is being dominated by remaining bad/outlier training rows rather than lacking sophistication. With minimal changes and preserving your “engineer geo/time features + LinearRegression” approach, I add one more standard NYC sanity filter to remove unrealistically long trips (based on your already-computed Haversine `Distance`). I also normalize `Distance`, `pickup_distance_a`, and `dropoff_distance_a` using train-set statistics just like you already do for the abs diffs, which keeps feature scales consistent and typically stabilizes LinearRegression. Everything else (features, model, training loop, prediction clipping, submission format) remains the same.'
- What this solution (achieved 37.29238) has done: 'Your current issue is that you don’t have a valid Kaggle score yet, so the first goal is to reliably generate a submission that runs end-to-end within the time limit and is aligned with the RMSE metric. With minimal changes and preserving your exact core pipeline (same engineered features + `LinearRegression` on `log1p(fare)`), I fix a high-impact bug in your weekday mapping (`"Thrusday"` typo) that silently creates a wrong one-hot column and harms generalization. I also add a tiny, standard guard to drop non-finite engineered distances created by rare numerical edge cases, and I clip extreme high predictions (still legitimate) to a reasonable max fare to reduce RMSE blow-ups from linear extrapolation while keeping the same model and loss. Everything else (filters, feature set, training approach, log-transform, and submission format) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import os  # reading the input files we have access to

INPUT_DIR = "../input"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "/kaggle/input"

print(os.listdir(INPUT_DIR))



## === cell 1
DATA_DIR = os.path.join(INPUT_DIR, "new-york-city-taxi-fare-prediction")
if not os.path.exists(DATA_DIR):
    DATA_DIR = INPUT_DIR

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

NROWS_TRAIN = int(os.environ.get("NROWS_TRAIN", "2000000"))

train_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
    "key": "string",
    "pickup_datetime": "string",
}

train_data = pd.read_csv(TRAIN_PATH, nrows=NROWS_TRAIN, dtype=train_dtypes)
train_data.dtypes




## === cell 2
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_data)



## === cell 3
print(train_data.isnull().sum())



## === cell 4
train_data = train_data.dropna(how="any", axis="rows")



## === cell 5
train_data = train_data[
    (train_data["fare_amount"] > 0.0)
    & (train_data["fare_amount"] <= 250.0)
    & (train_data["passenger_count"] >= 1)
    & (train_data["passenger_count"] <= 6)
    & (train_data["pickup_longitude"].between(-74.5, -72.8))
    & (train_data["dropoff_longitude"].between(-74.5, -72.8))
    & (train_data["pickup_latitude"].between(40.0, 41.8))
    & (train_data["dropoff_latitude"].between(40.0, 41.8))
].copy()



## === cell 6
try:
    plot = train_data.iloc[:2000].plot.scatter(
        "abs_diff_longitude", "abs_diff_latitude"
    )
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 7
train_data = train_data[
    (train_data.abs_diff_longitude < 5.0) & (train_data.abs_diff_latitude < 5.0)
]



## === cell 8
test_dtypes = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
    "key": "string",
    "pickup_datetime": "string",
}
test_data = pd.read_csv(TEST_PATH, dtype=test_dtypes)
test_data.dtypes



## === cell 9
add_travel_vector_features(test_data)



## === cell 10
test_data.dtypes



## === cell 11
train_data.dtypes



## === cell 12
train_data.head()



## === cell 13
train_dt = pd.to_datetime(train_data["pickup_datetime"], errors="coerce")
valid_dt_mask = train_dt.notna()
train_data = train_data.loc[valid_dt_mask].copy()
train_dt = train_dt.loc[valid_dt_mask]

test_dt = pd.to_datetime(test_data["pickup_datetime"], errors="coerce")

test_valid_dt_mask = test_dt.notna()
test_data_valid = test_data.loc[test_valid_dt_mask].copy()
test_dt_valid = test_dt.loc[test_valid_dt_mask]

train_data["pickup_time"] = (train_dt.dt.hour * 100 + train_dt.dt.minute).astype(
    "Int64"
)
test_data_valid["pickup_time"] = (
    test_dt_valid.dt.hour * 100 + test_dt_valid.dt.minute
).astype("Int64")

train_data["weekday"] = train_dt.dt.weekday.astype("Int64")
test_data_valid["weekday"] = test_dt_valid.dt.weekday.astype("Int64")



## === cell 14
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data_valid.drop("pickup_datetime", inplace=True, axis=1)



## === cell 15
train_data.head(10)



## === cell 16
weekday_map = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

train_data["weekday"] = train_data["weekday"].astype("int16").map(weekday_map)
test_data_valid["weekday"] = test_data_valid["weekday"].astype("int16").map(weekday_map)



## === cell 17
train_one_shot = pd.get_dummies(train_data["weekday"])
test_one_shot = pd.get_dummies(test_data_valid["weekday"])

all_cols = sorted(set(train_one_shot.columns).union(set(test_one_shot.columns)))
train_one_shot = train_one_shot.reindex(columns=all_cols, fill_value=0)
test_one_shot = test_one_shot.reindex(columns=all_cols, fill_value=0)

test_data_valid = pd.concat([test_data_valid, test_one_shot], axis=1)
train_data = pd.concat([train_data, train_one_shot], axis=1)



## === cell 18
train_data.drop("weekday", inplace=True, axis=1)
test_data_valid.drop("weekday", inplace=True, axis=1)



## === cell 19
train_data["pickup_time"] = train_data["pickup_time"].astype(int)
test_data_valid["pickup_time"] = test_data_valid["pickup_time"].astype(int)



## === cell 20
R = 6372.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))  # FIXED

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_data["Distance"] = np.asarray(distance) * 0.621

lat1 = np.asarray(np.radians(test_data_valid["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data_valid["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data_valid["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data_valid["dropoff_longitude"]))  # FIXED

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data_valid["Distance"] = np.asarray(distance) * 0.621



## === cell 21
R = 6372.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

lat3 = np.zeros(len(train_data)) + np.radians(40.641)
lon3 = np.zeros(len(train_data)) + np.radians(-73.778)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
dlon_dropoff = lon3 - lon2
dlat_dropoff = lat3 - lat2
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
train_data["pickup_distance_a"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_data["dropoff_distance_a"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(test_data_valid["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data_valid["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data_valid["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data_valid["dropoff_longitude"]))

lat3 = np.zeros(len(test_data_valid)) + np.radians(40.641)
lon3 = np.zeros(len(test_data_valid)) + np.radians(-73.778)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
dlon_dropoff = lon3 - lon2
dlat_dropoff = lat3 - lat2
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
test_data_valid["pickup_distance_a"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_data_valid["dropoff_distance_a"] = np.asarray(distance2) * 0.621



## === cell 22
train_data = train_data[train_data["Distance"].between(0.0, 50.0)].copy()



## === cell 23
finite_mask = (
    np.isfinite(train_data["Distance"].values)
    & np.isfinite(train_data["pickup_distance_a"].values)
    & np.isfinite(train_data["dropoff_distance_a"].values)
)
train_data = train_data.loc[finite_mask].copy()



## === cell 24
train_data.drop(
    ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"],
    inplace=True,
    axis=1,
)
test_data_valid.drop(
    ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"],
    inplace=True,
    axis=1,
)



## === cell 25
for col in [
    "abs_diff_longitude",
    "abs_diff_latitude",
    "Distance",
    "pickup_distance_a",
    "dropoff_distance_a",
]:
    mu = float(np.mean(train_data[col]))
    var = float(np.var(train_data[col]))
    std = float(np.sqrt(var))
    if std == 0.0 or not np.isfinite(std):
        std = 1.0
    train_data[col] = (train_data[col] - mu) / std
    test_data_valid[col] = (test_data_valid[col] - mu) / std



## === cell 26
from sklearn.model_selection import train_test_split

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]

X.columns = X.columns.astype(str)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.01, random_state=80)



## === cell 27
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(Xtrain, np.log1p(ytrain))

val_pred = np.expm1(lr.predict(Xtest))
val_pred = np.clip(val_pred, 0.0, 250.0)
rmse = float(np.sqrt(np.mean((val_pred - ytest.values) ** 2)))
print("Holdout RMSE (approx):", rmse)

X_sub_valid = test_data_valid.drop("key", axis=1)
X_sub_valid.columns = X_sub_valid.columns.astype(str)
X_sub_valid = X_sub_valid.reindex(columns=X.columns, fill_value=0)

pred_valid = np.expm1(lr.predict(X_sub_valid))
pred_valid = np.clip(pred_valid, 0.0, 250.0)

fallback_pred = float(np.median(pred_valid)) if len(pred_valid) else 11.35

pred_all = np.empty(len(test_data), dtype=np.float64)
pred_all[test_valid_dt_mask.values] = pred_valid.astype(np.float64)
pred_all[~test_valid_dt_mask.values] = fallback_pred

submission = pd.DataFrame({"key": test_data["key"].values, "fare_amount": pred_all})
submission = submission[["key", "fare_amount"]]
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print(
    "Test datetime valid rows:",
    int(test_valid_dt_mask.sum()),
    "invalid rows:",
    int((~test_valid_dt_mask).sum()),
)
