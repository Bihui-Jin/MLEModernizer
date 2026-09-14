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

geopandas==0.14.4
lightgbm==4.6.0
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

3.62101

# 6. Current score

5.63196

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.73731) has done: 'Your code currently can’t reliably produce a valid submission because the test-set filtering (dropping zero coordinates) removes rows, but you still need predictions for every `key` in `sample_submission.csv`; we keep all test rows and instead only filter train rows. To move RMSE toward your target with minimal core-logic changes, we (1) fix the Haversine distance bug (you’re converting radians twice), (2) use the intended strong models consistently for the test-time stacking (you trained the meta-model on `[GBoost, model_xgb, model_lgb1]` but predicted test features using different models), and (3) ensure correct key alignment by merging predictions onto `sample_submission.csv`. These changes keep your overall approach (feature engineering + tree models + simple stacking) intact while improving correctness and expected leaderboard RMSE. The script also avoids reading the full 55M rows by keeping your `nrows=20000`, ensuring it runs under the time limit.'
- What this solution (achieved 5.37291) has done: 'Your current RMSE (5.737) is much worse than the target (3.621), so we should make small, correctness-focused changes that reliably improve generalization without changing the overall approach (feature engineering + tree models + stacking). The biggest safe win here is to remove train/test feature mismatch caused by fitting the base models on `X_train` only while predicting the Kaggle test set from `test_data` (which corresponds to all rows, not the holdout split); we instead refit base models on the full filtered training data `X` before creating test meta-features. Second, we fix a subtle stacking bug: the meta-model is trained on OOF predictions generated from `X_train`, but it’s fit against `y_train` (same), so that part is fine; however, for best alignment we should also build OOF on the same `X_train`/`y_train` but then refit the meta-model on all available OOF rows (already) and then use base models refit on full `X` to produce test meta-features (main fix). Finally, we clip extreme test predictions to a reasonable upper bound based on the filtered training target range (still legitimate post-processing for RMSE and avoids rare exploding predictions).'
- What this solution (achieved 6.07135) has done: 'Your current gap to the target (5.37291 → 3.62101, lower is better) is large, so we apply a few correctness-focused, minimal changes that keep your overall feature set + tree-model stacking approach intact. The biggest safe improvement is to remove distribution shift introduced by the current `train_test_split`-based OOF stacking: we generate out-of-fold predictions on the full filtered training set `X, y` (not just `X_train, y_train`) and train the meta-model on those OOF features. Next, we make the meta-model consistent with your stronger base models by fitting it with a slightly stronger but still lightweight LGBM configuration (same algorithm, just avoids an underfit default). Finally, we keep the submission key alignment merge, and we clip predictions using training percentiles as you already do for stability.'
- What this solution (achieved 6.10381) has done: 'You’re still far from the target (6.07 vs 3.62; lower is better), so the safest way to move RMSE downward without changing your overall approach is to fix train/test feature mismatch and reduce obvious outlier noise in a metric-consistent way. I (1) add two standard, minimal NYC-taxi features (`abs_delta_longitude`, `abs_delta_latitude`) and an `is_airport` flag (JFK/LGA proximity) computed from your existing coordinates—this keeps the same “feature engineering + tree models + stacking” core logic but gives the models much more signal. (2) I apply the common “NYC bounding box” filter to the *training* set only (test remains untouched) to reduce label noise/outliers that harm RMSE. (3) I keep your stacking exactly as-is, only ensuring the new features are created for both train/test before modeling and that submission alignment stays correct.'
- What this solution (achieved 5.92285) has done: 'The timeout is dominated by repeated model fitting: you train several heavy models for diagnostics, then run 5-fold CV on an averaging ensemble (which refits all base models), then generate 5-fold out-of-fold predictions for stacking (refitting all base models again). To keep the core algorithm intact while cutting redundant work, the optimized script (1) removes all plotting/EDA and intermediate `.describe()` calls that don’t affect the final submission, and (2) eliminates the extra CV-on-average-model step that is not used downstream (stacking is still performed exactly as written). It also replaces slow `.iloc` slicing inside the OOF loop with pre-extracted NumPy arrays for identical data/targets but much less overhead, and avoids unnecessary `DataFrame.info()` printing.'
- What this solution (achieved 5.63196) has done: 'Your current RMSE (5.92285) is well above the target (3.62101), so we should make small, high-impact changes that improve signal without changing the overall “feature engineering + 3 base models + OOF stacking + meta-model” logic. The biggest safe improvement is to stop training on an extremely non-representative sample: with NYC Taxi Fare, taking the first 20k rows can bias by time/region and hurts RMSE; we keep the same row count but sample randomly from the file in a deterministic way. Next, we add one standard, minimal geodesic feature (`distance_km_sq`) and a simple interaction (`distance_km * passenger_count`) which preserves your feature-engineering approach but gives the tree models easier nonlinear structure to learn. Finally, we ensure all engineered numeric columns are float32 to reduce any unintended dtype quirks and keep prediction clipping unchanged for stability.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)



## === cell 1
from pathlib import Path


def resolve_path(fname):
    candidates = [
        Path("../input") / fname,
        Path("/kaggle/input") / fname,
        Path("/kaggle/input/new-york-city-taxi-fare-prediction") / fname,
        Path("../input/new-york-city-taxi-fare-prediction") / fname,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(Path("../input") / fname)


TRAIN_PATH = resolve_path("train.csv")
TEST_PATH = resolve_path("test.csv")
SAMPLE_PATH = resolve_path("sample_submission.csv")

train_full = pd.read_csv(
    TRAIN_PATH, nrows=250000
)  # still lightweight; enables better random sample
train_data = train_full.sample(n=20000, random_state=RANDOM_STATE).reset_index(
    drop=True
)
del train_full

test_data = pd.read_csv(TEST_PATH)




## === cell 2
def changeDataType(dataset):
    dataset["passenger_count"] = dataset.passenger_count.astype("uint8")
    dataset["pickup_longitude"] = dataset.pickup_longitude.astype("float32")
    dataset["pickup_latitude"] = dataset.pickup_latitude.astype("float32")
    dataset["dropoff_longitude"] = dataset.dropoff_longitude.astype("float32")
    dataset["dropoff_latitude"] = dataset.dropoff_latitude.astype("float32")
    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], errors="coerce", utc=True
    )


changeDataType(train_data)
changeDataType(test_data)
train_data["fare_amount"] = train_data.fare_amount.astype("float32")



## === cell 3
train_data = train_data.dropna(axis=0)

train_data = train_data.loc[train_data["fare_amount"] > 0]
train_data = train_data[
    (train_data.fare_amount < 400) & (train_data.fare_amount >= 2.5)
].copy()
train_data = train_data[train_data.passenger_count <= 6]



## === cell 4
train_data = train_data.drop(
    train_data[
        (train_data["pickup_latitude"] < -90) | (train_data["pickup_latitude"] > 90)
    ].index,
    axis=0,
)
train_data = train_data.drop(
    train_data[
        (train_data["pickup_longitude"] < -180) | (train_data["pickup_longitude"] > 180)
    ].index,
    axis=0,
)
train_data = train_data.drop(
    train_data[
        (train_data["dropoff_longitude"] < -180)
        | (train_data["dropoff_longitude"] > 180)
    ].index,
    axis=0,
)
train_data = train_data.drop(
    train_data[
        (train_data["dropoff_latitude"] < -90) | (train_data["dropoff_latitude"] > 90)
    ].index,
    axis=0,
)



## === cell 5
train_data = train_data[
    train_data.pickup_latitude.between(
        test_data.pickup_latitude.min(), test_data.pickup_latitude.max()
    )
]
train_data = train_data[
    train_data.pickup_longitude.between(
        test_data.pickup_longitude.min(), test_data.pickup_longitude.max()
    )
]
train_data = train_data[
    train_data.dropoff_latitude.between(
        test_data.dropoff_latitude.min(), test_data.dropoff_latitude.max()
    )
]
train_data = train_data[
    train_data.dropoff_longitude.between(
        test_data.dropoff_longitude.min(), test_data.dropoff_longitude.max()
    )
]




## === cell 6
def degree_to_radion(degree):
    return degree * (np.pi / 180.0)


def calculate_distance(
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
):
    from_lat = degree_to_radion(pickup_latitude)
    from_long = degree_to_radion(pickup_longitude)
    to_lat = degree_to_radion(dropoff_latitude)
    to_long = degree_to_radion(dropoff_longitude)

    radius = 6371.01  # km

    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long

    a = (
        np.sin(lat_diff / 2.0) ** 2
        + np.cos(from_lat) * np.cos(to_lat) * np.sin(long_diff / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return radius * c




## === cell 7
train_data["distance"] = calculate_distance(
    train_data.pickup_latitude,
    train_data.pickup_longitude,
    train_data.dropoff_latitude,
    train_data.dropoff_longitude,
).astype("float32")
test_data["distance"] = calculate_distance(
    test_data.pickup_latitude,
    test_data.pickup_longitude,
    test_data.dropoff_latitude,
    test_data.dropoff_longitude,
).astype("float32")



## === cell 8
train_data = train_data.loc[train_data.distance < 200]  # keep your cutoff



## === cell 9
train_data = train_data.drop(columns="key")
test_data_key = test_data["key"].copy()
test_data = test_data.drop(columns="key")



## === cell 10
for df in (train_data, test_data):
    dt = df["pickup_datetime"].dt
    df["Year"] = dt.year.astype("int16")
    df["Month"] = dt.month.astype("uint8")
    df["Date"] = dt.day.astype("uint8")
    df["Day of Week"] = dt.dayofweek.astype("uint8")
    df["Hour"] = dt.hour.astype("uint8")
    df["Minute"] = dt.minute.astype("uint8")
    df["IsWeekend"] = (dt.dayofweek >= 5).astype("uint8")



## === cell 11
train_data = train_data.loc[train_data.pickup_latitude != 0]
train_data = train_data.loc[train_data.pickup_longitude != 0]
train_data = train_data.loc[train_data.dropoff_latitude != 0]
train_data = train_data.loc[train_data.dropoff_longitude != 0]

test_data = test_data.copy()

train_data = train_data[
    (train_data["pickup_longitude"].between(-74.3, -73.7))
    & (train_data["dropoff_longitude"].between(-74.3, -73.7))
    & (train_data["pickup_latitude"].between(40.5, 41.0))
    & (train_data["dropoff_latitude"].between(40.5, 41.0))
].copy()




## === cell 12
def add_geo_features(df):
    df["abs_delta_longitude"] = (
        (df["pickup_longitude"] - df["dropoff_longitude"]).abs().astype("float32")
    )
    df["abs_delta_latitude"] = (
        (df["pickup_latitude"] - df["dropoff_latitude"]).abs().astype("float32")
    )

    jfk_lat, jfk_lon = 40.6413, -73.7781
    lga_lat, lga_lon = 40.7769, -73.8740

    d_pick_jfk = calculate_distance(
        df["pickup_latitude"], df["pickup_longitude"], jfk_lat, jfk_lon
    ).astype("float32")
    d_drop_jfk = calculate_distance(
        df["dropoff_latitude"], df["dropoff_longitude"], jfk_lat, jfk_lon
    ).astype("float32")
    d_pick_lga = calculate_distance(
        df["pickup_latitude"], df["pickup_longitude"], lga_lat, lga_lon
    ).astype("float32")
    d_drop_lga = calculate_distance(
        df["dropoff_latitude"], df["dropoff_longitude"], lga_lat, lga_lon
    ).astype("float32")

    df["is_airport"] = (
        (d_pick_jfk < 2.0)
        | (d_drop_jfk < 2.0)
        | (d_pick_lga < 2.0)
        | (d_drop_lga < 2.0)
    ).astype("uint8")

    df["distance_km_sq"] = (df["distance"] * df["distance"]).astype("float32")
    df["distance_x_passenger"] = (df["distance"] * df["passenger_count"]).astype(
        "float32"
    )
    return df


train_data = add_geo_features(train_data)
test_data = add_geo_features(test_data)



## === cell 13
train_data = train_data.drop(columns="pickup_datetime", axis=1)
test_data = test_data.drop(columns="pickup_datetime", axis=1)



## === cell 14
X = train_data.loc[:, train_data.columns != "fare_amount"]
y = train_data["fare_amount"]



## === cell 15
pass



## === cell 16
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator

import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor



## === cell 17
GBoost = GradientBoostingRegressor(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=4,
    max_features="sqrt",
    min_samples_leaf=15,
    min_samples_split=10,
    loss="huber",
    random_state=5,
)

model_xgb = XGBRegressor(
    colsample_bytree=0.4603,
    gamma=0.0468,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=1.7817,
    n_estimators=2200,
    reg_alpha=0.4640,
    reg_lambda=0.8571,
    subsample=0.5213,
    random_state=7,
    n_jobs=-1,
    verbosity=0,
    objective="reg:squarederror",
)

model_lgb1 = lgb.LGBMRegressor(
    objective="regression",
    num_leaves=5,
    learning_rate=0.05,
    n_estimators=720,
    max_bin=55,
    bagging_fraction=0.8,
    bagging_freq=5,
    feature_fraction=0.2319,
    feature_fraction_seed=9,
    bagging_seed=9,
    min_data_in_leaf=6,
    min_sum_hessian_in_leaf=11,
    random_state=RANDOM_STATE,
)




## === cell 18
class AverageModel(BaseEstimator):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=1)




## === cell 19
pass



## === cell 20
base_models = [GBoost, model_xgb, model_lgb1]



## === cell 21
from sklearn.base import clone


def make_oof_predictions(X_df, y_ser, models, n_splits=5, seed=156):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((X_df.shape[0], len(models)), dtype=np.float32)

    X_values = X_df.to_numpy(copy=False)
    y_values = y_ser.to_numpy(copy=False)

    for i, model in enumerate(models):
        for tr_idx, ho_idx in kfold.split(X_values, y_values):
            model_clone = clone(model)
            model_clone.fit(X_values[tr_idx], y_values[tr_idx])
            oof[ho_idx, i] = model_clone.predict(X_values[ho_idx]).astype(np.float32)
    return oof




## === cell 22
oof_full = make_oof_predictions(X, y, base_models, n_splits=5, seed=156)



## === cell 23
meta_model = lgb.LGBMRegressor(
    random_state=RANDOM_STATE,
    n_estimators=400,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
)
meta_model.fit(oof_full, y)



## === cell 24
X_values_full = X.to_numpy(copy=False)
test_values = test_data.to_numpy(copy=False)
y_values_full = y.to_numpy(copy=False)

base_preds_test = []
for m in base_models:
    m_full = clone(m)
    m_full.fit(X_values_full, y_values_full)
    base_preds_test.append(m_full.predict(test_values).astype(np.float32))

meta_features_test = np.column_stack(base_preds_test)
meta_y = meta_model.predict(meta_features_test).astype(np.float32)

lower_clip = float(np.percentile(y.values, 0.1))
upper_clip = float(np.percentile(y.values, 99.9))
meta_y = np.clip(meta_y, lower_clip, upper_clip)
meta_y = np.maximum(meta_y, 0.0).astype(np.float32)



## === cell 25
sample_sub = pd.read_csv(SAMPLE_PATH)
pred_df = pd.DataFrame({"key": test_data_key.values, "fare_amount": meta_y})

submission = sample_sub[["key"]].merge(pred_df, on="key", how="left")
submission["fare_amount"] = submission["fare_amount"].fillna(float(np.mean(meta_y)))

submission.to_csv("submission.csv", index=False)
submission.head()
