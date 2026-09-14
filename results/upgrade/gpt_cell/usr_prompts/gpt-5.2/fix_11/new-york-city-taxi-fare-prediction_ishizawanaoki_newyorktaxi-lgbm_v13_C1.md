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
geopy==2.4.1
lightgbm==4.6.0
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

3.27672

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.08803) has done: 'Your code likely didn’t yield a Kaggle score because it reads from `../input/...`, while your environment shows the data under `/kaggle/input/...`; that path mismatch can prevent the notebook from running through to writing `submission.csv`. I make the smallest possible change by adding a robust path resolver (tries `/kaggle/input/...` first, then falls back), keeping the same modeling and feature logic. To move RMSE downward toward the target (lower is better) without changing the approach, I also add a minimal, standard data cleaning step for NYC taxi fares (restrict coordinates to a NYC-ish bounding box and cap extreme fares/distances) which usually improves generalization with LightGBM while preserving the same model/training loop. The script still write a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.30544) has done: 'I make the script reliably yield a valid `submission.csv` by fixing the input path resolver to also try the non-nested `/kaggle/input/train.csv` layout that your environment actually has. To move RMSE down toward your target without changing the model/training loop, I add a minimal, standard NYC Taxi cleanup: remove extreme outliers (very long distances and implausibly low fare-per-mile), which usually improves generalization for LightGBM on this competition. I also ensure the `key` used in the submission stays aligned with the engineered `X_test` rows by carrying it through feature engineering instead of referencing the original `test` frame implicitly. All changes are small and keep the same core features, LightGBM training approach, and prediction averaging.'
- What this solution (achieved 5.85291) has done: 'I keep your LightGBM model/training loop and feature set intact, and focus on small data-quality and split-stability tweaks that typically reduce RMSE in this competition. The main score drag here is that the current “fare_per_mile >= 1.0” filter removes many legitimate short-trip records and biases the model; I replace it with a conservative, more standard cleaning rule that keeps short trips while still removing obvious outliers. I also add a minimal “distance > 0” handling (zero-distance artifacts) and a light cap on extreme distances/fare-per-mile to reduce label noise without changing the model. Submission formatting and key alignment remain unchanged, and the script still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
def _resolve_input_path(rel_path: str) -> str:
    candidates = [
        os.path.join("/kaggle/input", rel_path),
        os.path.join("../input", rel_path),
        os.path.join("/kaggle/data", rel_path),
    ]
    if "/" in rel_path:
        candidates.insert(1, os.path.join("/kaggle/input", os.path.basename(rel_path)))
        candidates.insert(2, os.path.join("/kaggle/data", os.path.basename(rel_path)))

    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


train_path = _resolve_input_path("new-york-city-taxi-fare-prediction/train.csv")
test_path = _resolve_input_path("new-york-city-taxi-fare-prediction/test.csv")
sample_path = _resolve_input_path(
    "new-york-city-taxi-fare-prediction/sample_submission.csv"
)

train = pd.read_csv(train_path, nrows=1_000_000)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)



## === cell 2
train.isnull().sum()



## === cell 3
train.dropna(inplace=True)



## === cell 4
train.describe()



## === cell 5
train.query("passenger_count > 6")



## === cell 6
train.query("passenger_count < 1")



## === cell 7
train.query("fare_amount < 0")



## === cell 8
train.query("pickup_longitude < -180 or pickup_longitude > 180")



## === cell 9
train.query("dropoff_longitude < -180 or dropoff_longitude > 180")



## === cell 10
train.query("pickup_latitude < -90 or pickup_latitude > 90")



## === cell 11
train.query("dropoff_latitude < -90 or dropoff_latitude > 90")



## === cell 12
train = train.query(
    "1 <= passenger_count <= 6 and "
    "0 <= fare_amount and "
    "-180 <= pickup_longitude <= 180 and "
    "-180 <= dropoff_longitude <= 180 and "
    "-90 <= pickup_latitude <= 90 and "
    "-90 <= dropoff_latitude <= 90"
)
train.describe()



## === cell 13
train = train.query(
    "(-74.5 <= pickup_longitude <= -72.8) and (40.4 <= pickup_latitude <= 41.3) and "
    "(-74.5 <= dropoff_longitude <= -72.8) and (40.4 <= dropoff_latitude <= 41.3) and "
    "(0 < fare_amount <= 250)"
).copy()

train.reset_index(drop=True, inplace=True)
train



## === cell 14
data = pd.concat([train, test], sort=False, ignore_index=True)



## === cell 15
data.head()



## === cell 16
pickup_dt = pd.to_datetime(data["pickup_datetime"], errors="coerce", utc=False)
data["weekday"] = pickup_dt.dt.weekday.astype("float32")
data["hour"] = pickup_dt.dt.hour.astype("float32")

data.head()



## === cell 17
data = data.drop("pickup_datetime", axis=1)

data.head()




## === cell 18
def haversine_miles(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(np.float64))
    lon1 = np.radians(lon1.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    earth_radius_miles = 3958.756
    return earth_radius_miles * c


data["distance"] = haversine_miles(
    data["pickup_latitude"].values,
    data["pickup_longitude"].values,
    data["dropoff_latitude"].values,
    data["dropoff_longitude"].values,
).astype("float32")

data["abs_lon_diff"] = np.abs(
    data["pickup_longitude"].values - data["dropoff_longitude"].values
).astype("float32")
data["abs_lat_diff"] = np.abs(
    data["pickup_latitude"].values - data["dropoff_latitude"].values
).astype("float32")

data.head()



## === cell 19
data["distance"] = np.clip(data["distance"].values, 0, 100).astype("float32")

train_fe = data.iloc[: len(train)].copy()
test_fe = data.iloc[len(train) :].copy()

dist = train_fe["distance"].astype(np.float32).values
fare = train_fe["fare_amount"].astype(np.float32).values

mask_nonzero_or_smallfare = (dist > 0.0) | (fare <= 15.0)

mask_dist = dist <= 60.0

fare_per_mile = fare / np.maximum(dist, 1e-3)
mask_ppm = (fare_per_mile >= 0.5) & (fare_per_mile <= 60.0)

mask = mask_nonzero_or_smallfare & mask_dist & mask_ppm

train_fe = train_fe.loc[mask].reset_index(drop=True)

y_train = train_fe["fare_amount"].astype(np.float32)
X_train = train_fe.drop("fare_amount", axis=1)
X_test = test_fe.drop("fare_amount", axis=1, errors="ignore")

X_train.head()



## === cell 20
from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train),), dtype=np.float32)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

categorical_features = []



## === cell 21
import lightgbm as lgb

test_keys = None
if "key" in X_test.columns:
    test_keys = X_test["key"].astype(str).values
else:
    test_keys = test["key"].astype(str).values

if "key" in X_train.columns:
    X_train = X_train.drop(columns=["key"])
if "key" in X_test.columns:
    X_test = X_test.drop(columns=["key"])

params = {
    "objective": "regression",
    "max_bin": 300,
    "learning_rate": 0.05,
    "num_leaves": 40,
    "feature_fraction_seed": 0,
    "bagging_seed": 0,
    "seed": 0,
    "verbosity": -1,
}

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
    X_tr = X_train.iloc[train_index, :]
    X_val = X_train.iloc[valid_index, :]
    y_tr = y_train.iloc[train_index]
    y_val = y_train.iloc[valid_index]

    lgb_train = lgb.Dataset(
        X_tr, y_tr, categorical_feature=categorical_features, free_raw_data=False
    )
    lgb_eval = lgb.Dataset(
        X_val,
        y_val,
        reference=lgb_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        num_boost_round=1000,
        callbacks=[
            lgb.log_evaluation(period=50),
        ],
    )

    best_iter = (
        model.best_iteration if (getattr(model, "best_iteration", 0) or 0) > 0 else 1000
    )

    oof_train[valid_index] = model.predict(X_val, num_iteration=best_iter).astype(
        np.float32
    )
    y_pred = model.predict(X_test, num_iteration=best_iter).astype(np.float32)

    y_preds.append(y_pred)
    models.append(model)



## === cell 22
pd.DataFrame(oof_train).to_csv("oof_train_kfold.csv", index=False)

try:
    scores = [m.best_score["valid_1"]["l2"] for m in models]
    score = sum(scores) / len(scores)
    print("===CV scores (l2 as reported by LightGBM)===")
    print(scores)
    print(score)
except Exception as e:
    print("LightGBM best_score not available (expected when not using early stopping).")



## === cell 23
from sklearn.metrics import mean_squared_error

y_pred_oof = oof_train
print("OOF RMSE:", float(np.sqrt(mean_squared_error(y_train, y_pred_oof))))



## === cell 24
len(y_preds)



## === cell 25
y_preds[0][:10]



## === cell 26
y_sub = sum(y_preds) / len(y_preds)

y_sub = np.clip(y_sub, 0, 250).astype(np.float32)

y_sub[:10]



## === cell 27
sub_lgb = pd.DataFrame({"key": test_keys, "fare_amount": y_sub})
sub_lgb.to_csv("submission.csv", index=False)

sub_lgb.head()
