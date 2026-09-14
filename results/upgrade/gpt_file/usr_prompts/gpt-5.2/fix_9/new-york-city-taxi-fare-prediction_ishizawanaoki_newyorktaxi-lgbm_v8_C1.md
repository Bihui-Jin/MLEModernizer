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

3.40781

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.2907) has done: 'Your notebook doesn’t yield a Kaggle score because it likely never produces a submission in the required `/kaggle/working/` location due to path assumptions (`../input/...`) and a nonstandard starting cell index (`cell 0`). I keep your exact model/training logic but switch to robust input paths that match your environment (`/kaggle/input/...`), add a safety check that the submission has the right row count/columns and no missing keys, and write the CSV explicitly to `/kaggle/working/submission_lightgbm.csv`. I also remove early stopping (since your requirements forbid it) while keeping the same boosting setup otherwise; this change is necessary for compliance and still produce a valid, scorable submission. Finally, I keep your existing feature engineering and post-processing intact so evaluation semantics remain the same.'
- What this solution (achieved 5.78981) has done: 'Your current RMSE (4.2907) is worse than the target (3.40781), so we should make the smallest legitimate improvements that reduce error without changing the overall LightGBM+KFold approach. The main issue is that the model is only using raw coordinates plus simple datetime parts; adding the standard NYC taxi distance features (haversine distance and simple Manhattan distance) and a few common geospatial helpers (bearing and absolute deltas) typically provides a large RMSE drop while keeping the same core training loop and model type. I also add a minimal, well-known coordinate/fare cleaning consistent with the competition (filter obvious outliers in lat/lon and fare range) to reduce noise in the 1M-row sample. Everything else (KFold setup, LightGBM training, prediction averaging, submission writing) stays the same.'
- What this solution (achieved 6.29238) has done: 'Your run currently doesn’t yield a Kaggle score because the notebook starts at `cell 0`, which commonly breaks the execution order in Kaggle Notebook exports; I renumber cells to start at 1 while keeping the same logic. To move RMSE down toward the 3.40781 target without changing the core LightGBM+KFold approach, I make two minimal, standard, competition-safe adjustments: (1) parse `pickup_datetime` in train/test separately (instead of concatenating), avoiding any subtle dtype/coercion interactions, and (2) add one well-known NYC-specific helper feature (trip distance to LaGuardia) symmetric to the existing JFK distance features. Finally, I keep the same training loop and parameters, but add a deterministic check to ensure the submission is written correctly to `/kaggle/working/submission_lightgbm.csv` with the required schema.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

INPUT_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "/kaggle/input"

print("Using INPUT_DIR:", INPUT_DIR)



## === cell 1
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")
sample_path = os.path.join(INPUT_DIR, "sample_submission.csv")

train = pd.read_csv(train_path, nrows=1_000_000)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)

print("train:", train.shape, "test:", test.shape, "sample:", sample_submission.shape)



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
train = train.query("1 <= passenger_count <= 6 and 0 <= fare_amount").copy()

train = train[
    (train["pickup_longitude"].between(-75, -72))
    & (train["dropoff_longitude"].between(-75, -72))
    & (train["pickup_latitude"].between(40, 42))
    & (train["dropoff_latitude"].between(40, 42))
].copy()

train = train[train["fare_amount"].between(2.5, 200)].copy()

train.describe()



## === cell 9
train.reset_index(drop=True, inplace=True)
train



## === cell 10
test_key = test["key"].copy()




## === cell 11
def _haversine_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(np.float64))
    lat1 = np.radians(lat1.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0088 * c
    return km


def _bearing_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(np.float64))
    lat1 = np.radians(lat1.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.arctan2(y, x)
    return brng




## === cell 12
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["abs_lon_diff"] = (
        (df["pickup_longitude"] - df["dropoff_longitude"]).abs().astype("float32")
    )
    df["abs_lat_diff"] = (
        (df["pickup_latitude"] - df["dropoff_latitude"]).abs().astype("float32")
    )

    df["center_latitude"] = (
        (df["pickup_latitude"] + df["dropoff_latitude"]) * 0.5
    ).astype("float32")

    dist_km = _haversine_np(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    )
    df["haversine_km"] = dist_km.astype("float32")

    dist_km_lon = _haversine_np(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["pickup_latitude"].values,
    )
    dist_km_lat = _haversine_np(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        df["dropoff_latitude"].values,
    )
    df["manhattan_km"] = (dist_km_lon + dist_km_lat).astype("float32")

    df["bearing"] = _bearing_np(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    ).astype("float32")

    df["log_haversine_km"] = np.log1p(
        np.maximum(df["haversine_km"].astype(np.float64), 0.0)
    ).astype("float32")
    df["log_manhattan_km"] = np.log1p(
        np.maximum(df["manhattan_km"].astype(np.float64), 0.0)
    ).astype("float32")

    JFK_LON, JFK_LAT = -73.7781, 40.6413
    df["pickup_dist_jfk_km"] = _haversine_np(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        np.full(len(df), JFK_LON, dtype=np.float64),
        np.full(len(df), JFK_LAT, dtype=np.float64),
    ).astype("float32")
    df["dropoff_dist_jfk_km"] = _haversine_np(
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
        np.full(len(df), JFK_LON, dtype=np.float64),
        np.full(len(df), JFK_LAT, dtype=np.float64),
    ).astype("float32")

    LGA_LON, LGA_LAT = -73.8740, 40.7769
    df["pickup_dist_lga_km"] = _haversine_np(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        np.full(len(df), LGA_LON, dtype=np.float64),
        np.full(len(df), LGA_LAT, dtype=np.float64),
    ).astype("float32")
    df["dropoff_dist_lga_km"] = _haversine_np(
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
        np.full(len(df), LGA_LON, dtype=np.float64),
        np.full(len(df), LGA_LAT, dtype=np.float64),
    ).astype("float32")

    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=True
    )
    df["pickup_year"] = df["pickup_datetime"].dt.year.astype("float32")
    df["pickup_month"] = df["pickup_datetime"].dt.month.astype("float32")
    df["pickup_day"] = df["pickup_datetime"].dt.day.astype("float32")
    df["pickup_hour"] = df["pickup_datetime"].dt.hour.astype("float32")
    df["pickup_weekday"] = df["pickup_datetime"].dt.weekday.astype("float32")
    df = df.drop(["pickup_datetime"], axis=1)

    NYC_LON, NYC_LAT = -73.9855, 40.7580  # Times Square / Midtown
    df["pickup_dist_nyc_km"] = _haversine_np(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        np.full(len(df), NYC_LON, dtype=np.float64),
        np.full(len(df), NYC_LAT, dtype=np.float64),
    ).astype("float32")
    df["dropoff_dist_nyc_km"] = _haversine_np(
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
        np.full(len(df), NYC_LON, dtype=np.float64),
        np.full(len(df), NYC_LAT, dtype=np.float64),
    ).astype("float32")

    return df


train_fe = add_features(train)
test_fe = add_features(test)

print("train_fe:", train_fe.shape, "test_fe:", test_fe.shape)



## === cell 13
y_train = train_fe["fare_amount"].astype("float32")
X_train = train_fe.drop(["fare_amount", "key"], axis=1)
X_test = test_fe.drop(["fare_amount", "key"], axis=1, errors="ignore")

X_train.head()



## === cell 14
from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train),), dtype=np.float32)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

categorical_features = []



## === cell 15
import lightgbm as lgb

params = {
    "objective": "regression",
    "metric": "rmse",
    "max_bin": 300,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
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
        valid_names=["train", "valid"],
        num_boost_round=1000,
        callbacks=[
            lgb.log_evaluation(period=10),
        ],
    )

    oof_train[valid_index] = model.predict(
        X_val, num_iteration=model.current_iteration()
    )
    y_pred = model.predict(X_test, num_iteration=model.current_iteration())

    y_preds.append(y_pred)
    models.append(model)



## === cell 16
pd.DataFrame(oof_train, columns=["oof_pred"]).to_csv(
    "/kaggle/working/oof_train_kfold.csv", index=False
)

scores = [m.best_score["valid"]["rmse"] for m in models]
score = sum(scores) / len(scores) if len(scores) > 0 else np.nan
print("===CV scores (RMSE)===")
print(scores)
print(score)



## === cell 17
from sklearn.metrics import mean_squared_error

y_pred_oof = oof_train
rmse = np.sqrt(mean_squared_error(y_train, y_pred_oof))
print("OOF RMSE:", rmse)



## === cell 18
len(y_preds)



## === cell 19
y_preds[0][:10]



## === cell 20
y_sub = sum(y_preds) / len(y_preds)
y_sub[:10]



## === cell 21
y_sub = np.asarray(y_sub, dtype=np.float64)
y_sub = np.where(np.isfinite(y_sub), y_sub, np.nan)
fill_value = float(np.nanmean(y_sub)) if np.isnan(y_sub).any() else None
if fill_value is not None and np.isnan(fill_value):
    fill_value = float(y_train.mean())
y_sub = np.nan_to_num(
    y_sub, nan=fill_value if fill_value is not None else float(y_train.mean())
)
y_sub = np.clip(y_sub, 0, None)

sub_lgb = pd.DataFrame({"key": test_key.values, "fare_amount": y_sub})

assert list(sub_lgb.columns) == ["key", "fare_amount"]
assert len(sub_lgb) == len(test_key) == len(test), (
    len(sub_lgb),
    len(test_key),
    len(test),
)
assert sub_lgb["key"].isna().sum() == 0

out_path = "/kaggle/working/submission_lightgbm.csv"
sub_lgb.to_csv(out_path, index=False)
print("Wrote submission to:", out_path)
sub_lgb.head()
