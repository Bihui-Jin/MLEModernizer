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
joblib==1.5.2
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

3.56321

# 6. Current score

4.5346

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.13672) has done: 'I fix the crashing haversine feature engineering by making `haversine_vec` accept scalar reference coordinates (your airport/center lat/lon floats) without calling `.astype` on them. That unblock creation of `ride_distance` and the distance-to-landmark features, which in turn fixes the downstream KeyErrors and NameErrors. I also ensure the test-set filtering doesn’t drop rows (dropping rows would break submission row alignment); instead we only clean NaNs and then clip predictions to a reasonable range. Finally, the script always write a valid `submission.csv` with exactly `key,fare_amount`.'
- What this solution (achieved 4.38809) has done: 'Your current gap is +1.57 RMSE (5.13672 vs target 3.56321; lower is better), so we should improve score with minimal risk. The biggest likely issue is train/test distribution mismatch from filtering only the training set but not the test set, plus potentially harmful log1p/expm1 post-processing for a metric that evaluates RMSE on the original scale; both can inflate RMSE. I (1) apply the same coordinate/passenger_count sanity filtering to the test set without dropping rows (mark invalid rows and later impute), and (2) keep the exact same XGBRegressor but train/predict on the original target scale (no log transform) to align the objective with the evaluation metric. Finally, I ensure submission row alignment matches the original test order by predicting for all test keys and filling any invalid rows with a robust fallback (median train fare), while still writing `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 4.30388) has done: 'I make two score-oriented, minimal changes that keep your model and training approach intact: (1) apply the same NaN cleanup and range filtering to `test_df` *without dropping rows* (just as you already do for train), so engineered distance features aren’t polluted by impossible coordinates; and (2) add a very small, standard NYC taxi baseline feature set (absolute lat/lon deltas and Manhattan distance) which is consistent with your existing haversine feature engineering and typically reduces RMSE without changing the core workflow. I also ensure the train/test feature columns are built identically and that any remaining invalid test rows still get the same median fallback, preserving submission alignment. These changes should move RMSE down from 4.388 toward your 3.563 target without altering the XGBoost model family or the overall pipeline.'
- What this solution (achieved 4.53375) has done: 'To move RMSE down toward your 3.56321 target (from 4.30388; lower is better) with minimal logic change, I keep the same feature engineering and XGBRegressor approach but (1) add standard, cheap geospatial features that are strongly predictive for this competition (bearing and a simple “NYC bounding-box” flag), and (2) handle extreme outliers more consistently by filtering *training* using a widely-used NYC fare cap (0 < fare ≤ 100) while keeping test rows intact via the existing validity mask + median fallback. I also add a tiny amount of regularization (`subsample`/`colsample_bytree`) to reduce overfitting on the 4M sample without changing the training loop/semantics. Finally, I ensure train/test feature columns are identical and submission formatting/alignment stays unchanged.'
- What this solution (achieved 4.53375) has done: 'To move RMSE down toward your 3.56321 target (from 4.53375; lower is better) with minimal disruption, I keep your exact feature set and XGBRegressor workflow but (1) train on the same 4M rows *without dropping most of them* by switching from sequential filtering to a single boolean mask (your current `filter_column` chaining unintentionally shrinks the dataset far more than intended). Then (2) I apply the same validity mask logic to training (instead of hard-dropping), keeping more realistic trips while still excluding invalid rows from training. Finally, I align imputation between train/validation/test by filling NaNs via train medians before any model fitting/scoring so XGBoost doesn’t see inconsistent missingness patterns; submission writing stays identical and produces `submission.csv`.'
- What this solution (achieved 4.58585) has done: 'Your current RMSE (4.53375) is worse than the target (3.56321), so we should improve it with the smallest safe changes. The biggest likely issue is that you drop all “invalid” coordinate rows from training, but you *keep* those rows in test (as fallback median), which creates a train/test mismatch and hurts generalization; we keep your same model/feature logic but add an explicit “valid_trip” feature so the model can learn a different baseline for borderline/invalid-ish patterns instead of relying on a hard median override. Second, your XGBRegressor is missing `reg_lambda`/`reg_alpha`/`gamma`, which are small, standard regularizers that usually reduce RMSE on this competition without changing the approach; we add light regularization while keeping the same objective and training flow. Finally, we keep your submission alignment logic, but compute the validity masks *before* dropping rows so the “valid_trip” feature is consistent and available everywhere.'
- What this solution (achieved 4.5346) has done: 'Your current RMSE (4.58585) is worse than the target (3.56321), so we should reduce it with minimal disruption. The biggest score drag here is that you train on only “valid” trips (after dropping) but still ask the model to predict all test rows, then override “invalid” ones with a global median; this mismatch is avoidable without changing the model family or training loop. I keep your exact feature engineering and XGBRegressor approach, but (1) stop dropping valid-mask rows in training and instead keep them with the existing `valid_trip_basic`/`ride_distance_valid` flags so the model can learn a baseline for those patterns, and (2) handle `ride_distance`<=0 by setting distance-derived features to 0 (instead of dropping rows), while (3) applying identical numeric coercion + median imputation to train/val/test to eliminate inconsistent NaNs. This should move RMSE down toward your target without introducing new algorithms or changing the evaluation semantics.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=4_000_000
)
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
df.dtypes



## === cell 1
df.describe()



## === cell 2
df.head()




## === cell 3
def filter_column(df, column, range_min, range_max):
    return df[(df[column] >= range_min) & (df[column] <= range_max)]


ny_latitude_min, ny_latitude_max = 40.4772, 45.0153
ny_longitude_min, ny_longitude_max = -79.7624, -71.7517


def compute_basic_valid_mask(df_):
    return (
        df_["pickup_longitude"].between(ny_longitude_min, ny_longitude_max)
        & df_["pickup_latitude"].between(ny_latitude_min, ny_latitude_max)
        & df_["dropoff_longitude"].between(ny_longitude_min, ny_longitude_max)
        & df_["dropoff_latitude"].between(ny_latitude_min, ny_latitude_max)
        & df_["passenger_count"].between(1, 6)
    )


df = df.dropna()
df["valid_trip_basic"] = compute_basic_valid_mask(df).astype("int8")



## === cell 4
df[df["fare_amount"] > 200].describe()



## === cell 5
df = filter_column(df, "fare_amount", 0.0, 100.0)




## === cell 6
def refactor_datetime(df_):
    df_["pickup_datetime"] = pd.to_datetime(df_["pickup_datetime"], errors="coerce")
    df_["year"] = df_["pickup_datetime"].dt.year
    df_["month"] = df_["pickup_datetime"].dt.month
    df_["day"] = df_["pickup_datetime"].dt.day
    df_["weekday"] = df_["pickup_datetime"].dt.weekday
    df_["hour"] = df_["pickup_datetime"].dt.hour
    df_.drop(columns=["pickup_datetime"], inplace=True)


refactor_datetime(df)
refactor_datetime(test_df)
df.head()




## === cell 7
def haversine_vec(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype="float64"))
    lon1 = np.radians(np.asarray(lon1, dtype="float64"))
    lat2 = np.radians(np.asarray(lat2, dtype="float64"))
    lon2 = np.radians(np.asarray(lon2, dtype="float64"))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367.0 * c
    return km


ny_center = ("ny_center", (40.7128, -74.0060))
jfk_airport = ("jfk_airport", (40.6446, -73.7797))
lga_airport = ("lga_airport", (40.7733, -73.8718))
ewr_airport = ("ewr_airport", (40.6895, -74.1745))

locs = [ny_center, jfk_airport, lga_airport, ewr_airport]




## === cell 8
def insert_haversine_dists(df_, locations):
    for name, (lat, lon) in locations:
        df_["pickup_dist_to_" + name] = haversine_vec(
            df_["pickup_latitude"], df_["pickup_longitude"], lat, lon
        )
        df_["dropoff_dist_to_" + name] = haversine_vec(
            df_["dropoff_latitude"], df_["dropoff_longitude"], lat, lon
        )
    df_["ride_distance"] = haversine_vec(
        df_["pickup_latitude"],
        df_["pickup_longitude"],
        df_["dropoff_latitude"],
        df_["dropoff_longitude"],
    )


insert_haversine_dists(df, locs)
insert_haversine_dists(test_df, locs)




## === cell 9
def add_coord_delta_features(df_):
    df_["abs_lon_diff"] = (df_["pickup_longitude"] - df_["dropoff_longitude"]).abs()
    df_["abs_lat_diff"] = (df_["pickup_latitude"] - df_["dropoff_latitude"]).abs()
    df_["manhattan_dist"] = df_["abs_lon_diff"] + df_["abs_lat_diff"]


add_coord_delta_features(df)
add_coord_delta_features(test_df)




## === cell 10
def add_bearing_and_box_features(df_):
    dlon = np.radians(df_["dropoff_longitude"].values - df_["pickup_longitude"].values)
    lat1 = np.radians(df_["pickup_latitude"].values)
    lat2 = np.radians(df_["dropoff_latitude"].values)
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(y, x)  # [-pi, pi]
    df_["bearing"] = bearing.astype("float64")
    df_["bearing_sin"] = np.sin(bearing).astype("float64")
    df_["bearing_cos"] = np.cos(bearing).astype("float64")

    man_lat_min, man_lat_max = 40.70, 40.83
    man_lon_min, man_lon_max = -74.03, -73.93
    pickup_in = df_["pickup_latitude"].between(man_lat_min, man_lat_max) & df_[
        "pickup_longitude"
    ].between(man_lon_min, man_lon_max)
    dropoff_in = df_["dropoff_latitude"].between(man_lat_min, man_lat_max) & df_[
        "dropoff_longitude"
    ].between(man_lon_min, man_lon_max)
    df_["pickup_in_manhattan_box"] = pickup_in.astype("int8")
    df_["dropoff_in_manhattan_box"] = dropoff_in.astype("int8")
    df_["both_in_manhattan_box"] = (pickup_in & dropoff_in).astype("int8")


add_bearing_and_box_features(df)
add_bearing_and_box_features(test_df)



## === cell 11
df.describe()



## === cell 12
df["ride_distance_valid"] = (df["ride_distance"] > 0).astype("int8")
test_df["ride_distance_valid"] = (test_df["ride_distance"] > 0).astype("int8")


def zero_distance_features_when_invalid(df_):
    invalid = ~(df_["ride_distance"] > 0)
    cols_to_zero = (
        [
            "ride_distance",
            "abs_lon_diff",
            "abs_lat_diff",
            "manhattan_dist",
            "bearing",
            "bearing_sin",
            "bearing_cos",
        ]
        + ["pickup_dist_to_" + x[0] for x in locs]
        + ["dropoff_dist_to_" + x[0] for x in locs]
    )
    for c in cols_to_zero:
        if c in df_.columns:
            df_.loc[invalid, c] = 0.0


zero_distance_features_when_invalid(df)
zero_distance_features_when_invalid(test_df)

df.describe()



## === cell 13
test_df["valid_trip_basic"] = compute_basic_valid_mask(test_df).astype("int8")

test_valid_mask = test_df["valid_trip_basic"].astype(bool) & test_df[
    "ride_distance_valid"
].astype(bool)

test_feature_cols_for_nan = (
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "year",
        "month",
        "day",
        "hour",
        "weekday",
        "ride_distance",
        "abs_lon_diff",
        "abs_lat_diff",
        "manhattan_dist",
        "bearing",
        "bearing_sin",
        "bearing_cos",
        "pickup_in_manhattan_box",
        "dropoff_in_manhattan_box",
        "both_in_manhattan_box",
        "valid_trip_basic",
        "ride_distance_valid",
    ]
    + ["pickup_dist_to_" + x[0] for x in locs]
    + ["dropoff_dist_to_" + x[0] for x in locs]
)

train_feature_cols_for_nan = [c for c in test_feature_cols_for_nan if c in df.columns]

for c in train_feature_cols_for_nan:
    if c in df.columns and c != "passenger_count":
        df[c] = pd.to_numeric(df[c], errors="coerce")

for c in test_feature_cols_for_nan:
    if c in test_df.columns and c != "passenger_count":
        test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

df[["year", "month", "day", "hour", "weekday"]] = df[
    ["year", "month", "day", "hour", "weekday"]
].fillna(0)
test_df[["year", "month", "day", "hour", "weekday"]] = test_df[
    ["year", "month", "day", "hour", "weekday"]
].fillna(0)



## === cell 14
from sklearn.model_selection import train_test_split

train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)



## === cell 15
features = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "ride_distance",
    "abs_lon_diff",
    "abs_lat_diff",
    "manhattan_dist",
    "bearing",
    "bearing_sin",
    "bearing_cos",
    "pickup_in_manhattan_box",
    "dropoff_in_manhattan_box",
    "both_in_manhattan_box",
    "valid_trip_basic",
    "ride_distance_valid",
]
features += ["pickup_dist_to_" + x[0] for x in locs]
features += ["dropoff_dist_to_" + x[0] for x in locs]
fare_amount = "fare_amount"

train_features = train_df[features].copy()
train_fare_amount = train_df[fare_amount].copy()

validation_features = validation_df[features].copy()
validation_fare_amount = validation_df[fare_amount].copy()

train_feature_medians = train_features.median(numeric_only=True)
train_features = train_features.fillna(train_feature_medians)
validation_features = validation_features.fillna(train_feature_medians)

train_features.info()



## === cell 16
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()



## === cell 17
from sklearn.model_selection import cross_val_score


def estimate_model(model, df_):
    X = df_[features]
    y = df_[fare_amount]
    X = X.copy()
    X = X.fillna(X.median(numeric_only=True))
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-cv_scores)
    print("RMSE scores for each fold:", rmse_scores)
    print("Mean RMSE:", rmse_scores.mean())
    print("Standard Deviation of RMSE:", rmse_scores.std())




## === cell 18
estimate_model(linear_model, train_df)



## === cell 19
linear_model.fit(train_features, train_fare_amount)



## === cell 20
from sklearn.metrics import mean_squared_error

linear_predictions = linear_model.predict(validation_features)
mean_squared_error(validation_fare_amount, linear_predictions, squared=False)



## === cell 21
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import seaborn as sns

learning_rates = [0.1, 0.15, 0.2]
n_estimators = [80, 100, 150]

sample_fraction = 0.1
train_sample = df.sample(frac=sample_fraction, random_state=42)
X = train_sample[features].copy()
y = train_sample[fare_amount].copy()

sample_medians = X.median(numeric_only=True)
X = X.fillna(sample_medians)


def cross_val_rmse(lr, ne, X_, y_):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_rmse = []
    for train_index, val_index in kf.split(X_):
        X_train, X_val = X_.iloc[train_index], X_.iloc[val_index]
        y_train, y_val = y_.iloc[train_index], y_.iloc[val_index]
        model = XGBRegressor(
            objective="reg:squarederror",
            learning_rate=lr,
            n_estimators=ne,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        rmse = mean_squared_error(y_val, predictions, squared=False)
        fold_rmse.append(rmse)
    avg_rmse = np.mean(fold_rmse)
    return lr, ne, avg_rmse


results = Parallel(n_jobs=-1)(
    delayed(cross_val_rmse)(lr, ne, X, y)
    for lr in learning_rates
    for ne in n_estimators
)

results_df = pd.DataFrame(results, columns=["learning_rate", "n_estimators", "rmse"])
results_df.replace([np.inf, -np.inf], np.nan, inplace=True)

plt.figure(figsize=(12, 8))
sns.lineplot(
    data=results_df, x="n_estimators", y="rmse", hue="learning_rate", marker="o"
)
plt.title("RMSE for Different Learning Rates and n_estimators")
plt.xlabel("Number of Estimators")
plt.ylabel("RMSE")
plt.legend(title="Learning Rate")
plt.show()



## === cell 22
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.15,
    n_estimators=150,
    max_depth=6,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.5,
    reg_alpha=0.0,
    gamma=0.0,
    n_jobs=-1,
    random_state=42,
)

xgb_model.fit(train_features, train_fare_amount)

val_pred = xgb_model.predict(validation_features)
val_pred = np.clip(val_pred, 0.0, None)

mean_squared_error(validation_fare_amount, val_pred, squared=False)



## === cell 23
train_pred = xgb_model.predict(train_features)
train_pred = np.clip(train_pred, 0.0, None)
mean_squared_error(train_fare_amount, train_pred, squared=False)



## === cell 24
test_X = test_df[features].copy()
test_X = test_X.fillna(train_feature_medians)

test_pred = xgb_model.predict(test_X)
test_pred = np.clip(test_pred, 0.0, 500.0)

fallback_fare = float(train_fare_amount.median())
test_pred = np.where(test_valid_mask.values, test_pred, fallback_fare)

holdout = pd.DataFrame({"key": test_df["key"].values, "fare_amount": test_pred})
holdout.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", holdout.shape)
print(holdout.head())
print("submission.csv exists:", os.path.exists("submission.csv"))
