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

6.20286

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.60198) has done: 'Your notebook likely fails to yield a Kaggle score mainly because it is very slow/unstable on 1M+ rows due to the row-wise `geopy.distance` `.apply`, and it also uses early stopping which violates your “no early stopping” requirement. I keep your exact modeling approach (LightGBM regressor with KFold, same features) but replace the distance feature computation with a fast, vectorized Haversine implementation (same semantic feature: great-circle distance) so the notebook reliably finishes and writes a valid submission CSV. I also remove early stopping (fixed `num_boost_round`) to comply with the constraints, and ensure the submission uses the `key` values from `test.csv` to avoid any potential alignment issues. These minimal changes should produce a valid submission and typically improve RMSE vs. a partially-run/failed pipeline.'
- What this solution (achieved 5.48543) has done: 'I make the pipeline reliably yield a valid submission by preventing LightGBM from outputting NaN/inf predictions (which can happen with occasional extreme/invalid coordinate combos even after basic filtering) and by ensuring feature dtypes are plain numeric (no nullable Int64) for consistent training/prediction. These are minimal changes that keep your model, CV loop, and features the same, but stabilize training and prediction so a CSV is always written and the public RMSE typically improves versus a submission containing invalid values. I also add a light, standard NYC bounding-box filter (still the same core “clean bad rows” logic you already use) to reduce outlier noise without changing the modeling approach. Finally, I clip negative fares in predictions (fares can’t be negative), which tends to reduce RMSE on this competition with minimal semantic impact.'
- What this solution (achieved 5.71148) has done: 'To move your RMSE down toward the 3.27672 target (lower is better), I keep the exact same LightGBM/KFold training loop and the same existing features, but add two very small, competition-standard features that don’t change the modeling approach: absolute lat/lon deltas and Manhattan distance in miles. These are simple deterministic transforms of the same coordinates and typically reduce NYC taxi RMSE substantially compared with using only Haversine distance + time parts. I also ensure the datetime-derived columns are plain float (not nullable Int64) for LightGBM stability and consistent handling of missing values. Everything else (data reading, filtering, fixed num_boost_round=300, submission format/path) remains the same.'
- What this solution (achieved 4.69052) has done: 'I fix the runtime error in feature engineering by using NumPy’s `clip(min, max)` signature (your traceback comes from calling `.clip(lower=..., upper=...)` on a NumPy array). I also ensure the submission file matches the competition’s required filename extension and column names by writing `submission.csv` with exactly `key` and `fare_amount` columns. Finally, I add a small guard to guarantee `fare_amount` exists and is numeric in the final DataFrame before saving, preventing the “missing fare_amount column” submission error even if an earlier step produces an unexpected dtype/object.'
- What this solution (achieved 6.00557) has done: 'Your current notebook doesn’t yield a Kaggle score likely because it’s unstable/slow on 1M rows due to unnecessary memory copies and because `pd.concat(train,test)` can create subtle train/test feature mismatches if any filtering/NaNs differ. I keep your exact model, CV loop, params, features, and fixed 300 boosting rounds, but (1) compute features separately for train and test (same formulas) to avoid any cross-frame dtype/NaN surprises, (2) add a tiny, competition-standard coordinate sanity filter for the *train* only (tight NYC bbox + positive distance) without changing feature definitions, and (3) ensure `test` features get the same median-imputation columns in the same order to guarantee submission validity. These are minimal, execution-stability changes that should reliably produce `submission.csv` and typically reduce RMSE versus a broken/partially-run pipeline, moving you toward the target.'
- What this solution (achieved 5.29831) has done: 'Your current notebook likely fails to “yield” a Kaggle score because it may not finish reliably within the time/memory constraints when reading and featurizing 1,000,000 rows (many engineered features plus 5-fold LightGBM), so the submission CSV may never get written. To keep the exact same model/loop/feature definitions and evaluation semantics, I only reduce `nrows` to a smaller but still representative subset so the pipeline completes end-to-end and produces `submission.csv`. I also add a tiny safety check that the submission has the exact same row count/order as `test.csv` and enforce numeric dtypes to avoid any accidental invalid submission. This minimal change is directly aimed at producing a valid submission and improving RMSE versus no score at all, moving you toward the 3.27672 target.'
- What this solution (achieved 5.60773) has done: 'You’re currently far above the target RMSE (5.298 vs 3.277; lower is better), so we should make small, legitimate quality improvements that don’t change your core LightGBM/KFold setup. The biggest “minimal” gain for this competition is to stop learning on the log1p-transformed target (which mismatches the RMSE on raw dollars) and instead train directly on `fare_amount`, while keeping the same features and training loop. To avoid large behavior changes, we keep the same LightGBM params and boosting rounds, but we remove the expm1 postprocessing and keep the same submission writing/validity checks. This should move your score materially downward toward the target band without altering your overall approach.'
- What this solution (achieved 6.20286) has done: 'I make the notebook reliably “yield” a score by ensuring it completes within the time limit while keeping your exact LightGBM/KFold training loop, parameters, boosting rounds, and feature formulas unchanged. The only substantive adjustment is increasing the training sample size moderately (still using `nrows`) so RMSE moves downward toward your 3.27672 target without changing the modeling approach. I also add a deterministic random-row sampling step (after your existing filters) to cap the training set to a stable size if the larger `nrows` still leaves too many rows, preventing memory/time spikes that can stop the submission CSV from being written. Submission writing, column names, alignment to `test["key"]`, and NaN/inf guards stay the same.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
TRAIN_NROWS = 1_000_000
RANDOM_STATE = 0

train = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=TRAIN_NROWS
)
test = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
sample_submission = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)



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
    "-90 <= dropoff_latitude <= 90 and "
    "-75 <= pickup_longitude <= -72 and "
    "-75 <= dropoff_longitude <= -72 and "
    "40 <= pickup_latitude <= 42 and "
    "40 <= dropoff_latitude <= 42"
)
train.describe()



## === cell 13
train.reset_index(drop=True, inplace=True)
train.head()




## === cell 14
def haversine_miles(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    earth_radius_miles = 3958.7613
    return earth_radius_miles * c


def bearing_rad(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)


landmarks = {
    "jfk": (40.6413, -73.7781),
    "lga": (40.7769, -73.8740),
    "ewr": (40.6895, -74.1745),
    "manhattan": (40.7580, -73.9855),
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["pickup_datetime"] = pd.to_datetime(
        out["pickup_datetime"], errors="coerce", utc=False
    )
    out["hour"] = out["pickup_datetime"].dt.hour.astype("float64")
    out["month"] = out["pickup_datetime"].dt.month.astype("float64")
    out["year"] = out["pickup_datetime"].dt.year.astype("float64")
    out["weekday"] = out["pickup_datetime"].dt.weekday.astype("float64")

    for c in ["hour", "month", "year", "weekday"]:
        if out[c].isna().any():
            mode_val = out[c].mode(dropna=True)
            fill_val = float(mode_val.iloc[0]) if len(mode_val) else 0.0
            out[c] = out[c].fillna(fill_val).astype("float64")

    out = out.drop("pickup_datetime", axis=1)

    out["distance"] = haversine_miles(
        out["pickup_latitude"].to_numpy(),
        out["pickup_longitude"].to_numpy(),
        out["dropoff_latitude"].to_numpy(),
        out["dropoff_longitude"].to_numpy(),
    )

    out["abs_lon_diff"] = (
        out["pickup_longitude"].astype(float) - out["dropoff_longitude"].astype(float)
    ).abs()
    out["abs_lat_diff"] = (
        out["pickup_latitude"].astype(float) - out["dropoff_latitude"].astype(float)
    ).abs()

    mean_lat_rad = np.radians(
        (
            (
                out["pickup_latitude"].astype(float)
                + out["dropoff_latitude"].astype(float)
            )
            / 2.0
        ).to_numpy()
    )
    miles_per_deg_lat = 69.172
    miles_per_deg_lon = miles_per_deg_lat * np.cos(mean_lat_rad)

    out["manhattan_miles"] = (
        out["abs_lat_diff"].to_numpy() * miles_per_deg_lat
        + out["abs_lon_diff"].to_numpy() * miles_per_deg_lon
    )

    out["distance"] = np.clip(out["distance"].to_numpy(), 0.0, 100.0)
    out["manhattan_miles"] = np.clip(out["manhattan_miles"].to_numpy(), 0.0, 150.0)

    out["bearing"] = bearing_rad(
        out["pickup_latitude"].to_numpy(),
        out["pickup_longitude"].to_numpy(),
        out["dropoff_latitude"].to_numpy(),
        out["dropoff_longitude"].to_numpy(),
    )

    n_data = len(out)
    for name, (lat0, lon0) in landmarks.items():
        pickup_d = haversine_miles(
            out["pickup_latitude"].to_numpy(),
            out["pickup_longitude"].to_numpy(),
            np.full(n_data, lat0, dtype=float),
            np.full(n_data, lon0, dtype=float),
        )
        dropoff_d = haversine_miles(
            out["dropoff_latitude"].to_numpy(),
            out["dropoff_longitude"].to_numpy(),
            np.full(n_data, lat0, dtype=float),
            np.full(n_data, lon0, dtype=float),
        )
        out[f"pickup_dist_{name}"] = np.clip(pickup_d, 0.0, 100.0)
        out[f"dropoff_dist_{name}"] = np.clip(dropoff_d, 0.0, 100.0)

    return out


train_fe = add_features(train)
test_fe = add_features(test)

train_fe.head()



## === cell 15
train_fe["fare_amount"] = pd.to_numeric(
    train_fe["fare_amount"], errors="coerce"
).astype(float)
train_fe = train_fe.dropna(subset=["fare_amount"]).reset_index(drop=True)

train_fe = train_fe.query("not (distance < 0.01 and fare_amount > 3.0)").reset_index(
    drop=True
)
train_fe = train_fe.query("not (distance < 0.05 and fare_amount > 50.0)").reset_index(
    drop=True
)

train_fe = train_fe.query("fare_amount <= 250.0").reset_index(drop=True)

train_fe = train_fe.query("not (distance > 30.0 and fare_amount < 20.0)").reset_index(
    drop=True
)

MAX_TRAIN_ROWS_AFTER_FILTER = 800_000
if len(train_fe) > MAX_TRAIN_ROWS_AFTER_FILTER:
    train_fe = train_fe.sample(
        n=MAX_TRAIN_ROWS_AFTER_FILTER, random_state=RANDOM_STATE
    ).reset_index(drop=True)

y_train_raw = train_fe["fare_amount"].astype(float)
y_train = y_train_raw.astype(float)

X_train = train_fe.drop(["fare_amount", "key"], axis=1)
X_test = test_fe.drop(["fare_amount", "key"], axis=1, errors="ignore")

for c in X_train.columns:
    X_train[c] = pd.to_numeric(X_train[c], errors="coerce").astype(float)
for c in X_test.columns:
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce").astype(float)

X_test = X_test.reindex(columns=X_train.columns)

meds = X_train.median(numeric_only=True)
X_train = X_train.fillna(meds)
X_test = X_test.fillna(meds)

X_train.head()



## === cell 16
from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train),), dtype=float)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

categorical_features = []



## === cell 17
import lightgbm as lgb

params = {
    "objective": "regression",
    "metric": "rmse",
    "max_bin": 300,
    "learning_rate": 0.05,
    "num_leaves": 40,
    "verbosity": -1,
}

callbacks = [
    lgb.log_evaluation(period=50),
]

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
        num_boost_round=300,  # fixed training length (no early stopping)
        callbacks=callbacks,
    )

    oof_train[valid_index] = model.predict(
        X_val, num_iteration=model.current_iteration()
    )
    y_pred = model.predict(X_test, num_iteration=model.current_iteration())

    y_preds.append(y_pred)
    models.append(model)



## === cell 18
pd.DataFrame(oof_train).to_csv("oof_train_kfold.csv", index=False)

scores = []
for m in models:
    if "rmse" in m.best_score["valid"]:
        scores.append(m.best_score["valid"]["rmse"])
    else:
        scores.append(np.sqrt(m.best_score["valid"]["l2"]))

score = sum(scores) / len(scores)
print("===CV scores (RMSE on raw target)===")
print(scores)
print("Mean RMSE:", score)



## === cell 19
from sklearn.metrics import mean_squared_error

y_pred_oof_raw = np.asarray(oof_train, dtype=float)
bad_oof = ~np.isfinite(y_pred_oof_raw)
if bad_oof.any():
    y_pred_oof_raw[bad_oof] = float(np.nanmean(y_train_raw))

y_pred_oof_raw = np.maximum(y_pred_oof_raw, 0.0)

print(
    "OOF RMSE (original scale):",
    np.sqrt(mean_squared_error(y_train_raw, y_pred_oof_raw)),
)



## === cell 20
len(y_preds)



## === cell 21
y_preds[0][:10]



## === cell 22
y_sub = sum(y_preds) / len(y_preds)
y_sub = np.asarray(y_sub, dtype=float)

bad = ~np.isfinite(y_sub)
if bad.any():
    y_sub[bad] = float(np.nanmean(y_train_raw))

y_sub = np.maximum(y_sub, 0.0)

y_sub[:10]



## === cell 23
sub_lgb = pd.DataFrame(
    {
        "key": test["key"].astype(str).values,
        "fare_amount": pd.to_numeric(y_sub, errors="coerce").astype(float),
    }
)

if len(sub_lgb) != len(test):
    raise RuntimeError(
        f"Submission row mismatch: got {len(sub_lgb)} expected {len(test)}"
    )

sub_lgb["fare_amount"] = sub_lgb["fare_amount"].replace([np.inf, -np.inf], np.nan)
sub_lgb["fare_amount"] = (
    sub_lgb["fare_amount"].fillna(float(y_train_raw.mean())).astype(float)
)

sub_path = "submission.csv"
sub_lgb.to_csv(sub_path, index=False)

print("Wrote:", os.path.abspath(sub_path))
print(sub_lgb.head())
print("Columns:", list(sub_lgb.columns))
print("Submission rows:", len(sub_lgb), "Expected:", len(test))
