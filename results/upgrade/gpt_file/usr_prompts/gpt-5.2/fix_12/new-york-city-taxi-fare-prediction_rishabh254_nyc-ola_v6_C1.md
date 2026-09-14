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

5.91416

# 6. Current score

6.60309

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 28.8042) has done: 'Your very high RMSE is mainly coming from (1) predicting negative/near-zero fares for some rides (no non-negativity constraint) and (2) rounding predictions to cents before scoring, which adds avoidable error. I keep your exact OLS-with-two-features approach, but make three minimal fixes: set a fixed `random_state` for stable validation, stop rounding predictions before evaluation/submission, and clip predictions to a reasonable non-negative range to avoid catastrophic outliers. These changes preserve your core logic (same features, same linear least squares fit) while moving the score substantially toward the target. The script still writes a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 28.8139) has done: 'Your current RMSE is far above the target, so we should make small, score-relevant fixes without changing the core “OLS on abs diffs + bias” logic. The biggest gain (still within the same modeling approach) is to remove obviously bad training rows (invalid coordinates, unrealistic passenger counts, and extreme/invalid fares) that distort the least-squares fit and produce large errors. We also ensure the features are computed after cleaning and keep the existing non-negativity clipping for predictions (it reduces catastrophic outliers under RMSE). Finally, we keep the same split/random_state and still write a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 15.14804) has done: 'We keep your exact “OLS on abs coordinate diffs + bias” approach, but make small, score-relevant fixes that typically reduce RMSE a lot: (1) compute the same feature matrix, but fit with `lstsq` only (dropping the unstable explicit inverse that can amplify noise), (2) add one more minimal, competition-standard cleaning step to remove extreme long trips (using haversine distance only for filtering, not as a model feature), and (3) clip predictions to a tighter, more realistic max fare (reducing RMSE blow-ups from a few large over-predictions). These changes don’t alter the model class or training loop; they just improve conditioning and remove distortive outliers so the same linear model fits better. The script still runs end-to-end within time limits and writes a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 15.14776) has done: 'Your current RMSE (15.148) is far above the target (5.914), so the smallest safe way to move toward the target is to keep the exact same linear OLS-on-abs-diffs model but fix the single biggest remaining source of error: you’re not using `passenger_count` at all, even though it’s already in the data and is a standard helpful linear feature. I add `passenger_count` as an additional regressor (still the same least-squares fit, no new model/training loop), and I also remove the unused explicit-inverse OLS computation to avoid any accidental divergence from the `lstsq` weights you actually use. Everything else (cleaning, feature extraction, clipping, submission format/path) stays the same, and the script still write a valid `submission.csv`.'
- What this solution (achieved 15.31356) has done: 'I fix the length-mismatch bug by recomputing `trip_km` after you filter the dataframe on distance (the boolean mask changes the row count, so the old `trip_km` no longer aligns). Then I assign `trip_distance_km` from the filtered array and keep it carried through the train/validation split so `get_input_matrix()` can always find the column. Finally, I keep the same linear least-squares training and prediction flow and ensure the script writes a valid `submission.csv` with `key,fare_amount` without changing any core modeling choices.'
- What this solution (achieved 14.96948) has done: 'Your current RMSE (15.31) is still far above the target (5.91), so we should improve accuracy while keeping the same core “linear least-squares on simple features” approach. The biggest remaining issue is that the model only uses distance, passenger_count, and bias, even though you already compute the abs coordinate diffs; adding those two existing linear features typically drops RMSE substantially without changing the training loop or model class. I also apply the exact same basic coordinate sanity filter to the test set (not for score hacking—just to prevent invalid coordinates from producing nonsense distances/predictions), then keep your existing prediction clipping and submission format unchanged. This is a minimal, score-relevant change that should move RMSE toward the target band.'
- What this solution (achieved 14.96481) has done: 'Your current RMSE (14.97) is much worse than the target (5.91), so we should improve it while keeping the exact same linear least-squares core model and feature set. The biggest score-relevant issue remaining is that the model ignores time effects; extracting a few linear datetime parts (hour, day-of-week, month, year) and adding them as additional columns keeps the same OLS training approach but usually drops RMSE substantially for this competition. To keep semantics stable and avoid distortive rows, I also add one minimal, standard cleaning step: remove cases with zero-distance trips (often bad data) after your existing distance computation. Everything else (cleaning style, lstsq fit, clipping, submission format/path) stays the same and it still writes `submission.csv`.'
- What this solution (achieved 14.96481) has done: 'Your RMSE (14.96) is still far above the target (5.91), so we need a small, legitimate accuracy gain without changing the core “linear least-squares on engineered features” approach. The main issue is feature scaling/conditioning: `pickup_year` (~2010+) and the bias term can dominate numerically and distort the least-squares fit; we keep the same features but center the datetime parts using medians computed on the training split (a pure linear reparameterization). We also evaluate on the same clipped predictions you submit (so validation reflects leaderboard behavior) and keep everything else (cleaning, features, lstsq, clipping, file paths, submission schema) unchanged. This should typically reduce RMSE noticeably while preserving identical model class and training semantics.'
- What this solution (achieved 14.92551) has done: 'Your RMSE (14.96) is far above the target (5.91), so we need a legitimate accuracy gain while keeping the same core “linear least-squares on engineered numeric features” approach. The smallest high-impact change for this competition is to add the standard airport/Midtown distance features (JFK/LGA/EWR and a Manhattan center), which are still simple linear regressors and don’t change your training loop or loss. I also add a minimal, standard cleanup: remove very low-fare “non-trips” relative to distance (and extreme per-km values) so OLS isn’t distorted by bad labels; this keeps the same modeling semantics but improves fit stability. Finally, prediction clipping and submission format remain unchanged, and the script still writes `submission.csv`.'
- What this solution (achieved 6.60309) has done: 'Your current RMSE (14.93) is far worse than the target (5.91), so we should improve accuracy while keeping the same core “linear least-squares on engineered features” approach. The biggest score-relevant bug is that you clean/train on `fare_per_km` outliers but do not apply any consistent cap to `trip_distance_km` and landmark-distance features at inference time, which can create extreme linear extrapolations and large RMSE blow-ups. I add a minimal, training-consistent clipping for `trip_distance_km` and all distance-based features in the test set (and also clip the validation features the same way for a fair local RMSE), without changing the model class, loss, or training loop. This keeps semantics the same (still OLS on the same columns) but reduces catastrophic errors from out-of-distribution distances, moving RMSE down toward the target.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))



## === cell 1
data = pd.read_csv("../input/train.csv", nrows=20_000_000)




## === cell 2
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()




## === cell 3
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * (2.0 * np.arcsin(np.sqrt(a)))




## === cell 4
def add_datetime_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["pickup_hour"] = dt.dt.hour.astype("float32")
    df["pickup_dow"] = dt.dt.dayofweek.astype("float32")
    df["pickup_month"] = dt.dt.month.astype("float32")
    df["pickup_year"] = dt.dt.year.astype("float32")

    for c in ["pickup_hour", "pickup_dow", "pickup_month", "pickup_year"]:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())




## === cell 5
LANDMARKS = {
    "jfk": (-73.7781, 40.6413),
    "lga": (-73.8740, 40.7769),
    "ewr": (-74.1745, 40.6895),
    "manhattan": (-73.985428, 40.748817),  # near Midtown
}


def add_landmark_distance_features(df):
    for name, (lon, lat) in LANDMARKS.items():
        df[f"pickup_to_{name}_km"] = haversine_km(
            df["pickup_longitude"].values,
            df["pickup_latitude"].values,
            np.full(len(df), lon, dtype=float),
            np.full(len(df), lat, dtype=float),
        ).astype("float32")
        df[f"dropoff_to_{name}_km"] = haversine_km(
            df["dropoff_longitude"].values,
            df["dropoff_latitude"].values,
            np.full(len(df), lon, dtype=float),
            np.full(len(df), lat, dtype=float),
        ).astype("float32")




## === cell 6
print(data.isnull().sum())



## === cell 7
print("Old size: %d" % len(data))
data = data.dropna(how="any", axis="rows")
print("New size: %d" % len(data))



## === cell 8
data = data[(data["passenger_count"] >= 1) & (data["passenger_count"] <= 6)]
data = data[(data["fare_amount"] > 0.0) & (data["fare_amount"] <= 250.0)]

data = data[
    (data["pickup_longitude"].between(-75.0, -72.0))
    & (data["dropoff_longitude"].between(-75.0, -72.0))
    & (data["pickup_latitude"].between(40.0, 42.0))
    & (data["dropoff_latitude"].between(40.0, 42.0))
]

trip_km = haversine_km(
    data["pickup_longitude"].values,
    data["pickup_latitude"].values,
    data["dropoff_longitude"].values,
    data["dropoff_latitude"].values,
)

mask = (
    trip_km < 60.0
)  # conservative NYC-centric cutoff; reduces catastrophic RMSE contributors
data = data.loc[mask].copy()
trip_km = trip_km[mask]

mask_nonzero = trip_km > 0.05  # 50 meters
data = data.loc[mask_nonzero].copy()
trip_km = trip_km[mask_nonzero]

print("After passenger/fare/coord/distance cleaning size: %d" % len(data))



## === cell 9
data["trip_distance_km"] = trip_km



## === cell 10
add_travel_vector_features(data)



## === cell 11
add_datetime_features(data)



## === cell 12
fare_per_km = data["fare_amount"].values / np.maximum(
    data["trip_distance_km"].values, 0.1
)
mask_fpk = (fare_per_km >= 0.5) & (fare_per_km <= 50.0)
data = data.loc[mask_fpk].copy()
print("After fare_per_km cleaning size: %d" % len(data))



## === cell 13
add_landmark_distance_features(data)



## === cell 14
try:
    plot = data.iloc[:10000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 15
print("Old size: %d" % len(data))
data = data[(data.abs_diff_longitude < 0.5) & (data.abs_diff_latitude < 0.5)].copy()
print("New size: %d" % len(data))



## === cell 16
from sklearn.model_selection import train_test_split

y = data.fare_amount
X = data.drop("fare_amount", axis=1)

train_df, val_df, train_y, val_y = train_test_split(
    X, y, test_size=0.2, random_state=42
)
train_df.dtypes



## === cell 17
DT_COLS = ["pickup_hour", "pickup_dow", "pickup_month", "pickup_year"]
dt_medians = {c: float(train_df[c].median()) for c in DT_COLS}

LANDMARK_FEATURES = []
for name in LANDMARKS.keys():
    LANDMARK_FEATURES.extend([f"pickup_to_{name}_km", f"dropoff_to_{name}_km"])


def get_input_matrix(df, dt_medians=None):
    if dt_medians is None:
        dt_medians = {c: 0.0 for c in DT_COLS}
    return np.column_stack(
        (
            df["trip_distance_km"].values.astype(float),
            df["abs_diff_longitude"].values.astype(float),
            df["abs_diff_latitude"].values.astype(float),
            df["passenger_count"].values.astype(float),
            (df["pickup_hour"].values.astype(float) - dt_medians["pickup_hour"]),
            (df["pickup_dow"].values.astype(float) - dt_medians["pickup_dow"]),
            (df["pickup_month"].values.astype(float) - dt_medians["pickup_month"]),
            (df["pickup_year"].values.astype(float) - dt_medians["pickup_year"]),
            *[df[c].values.astype(float) for c in LANDMARK_FEATURES],
            np.ones(len(df)),
        )
    )


train_X = get_input_matrix(train_df, dt_medians=dt_medians)

print(train_X.shape)
print(train_y.shape)



## === cell 18
(w, _, _, _) = np.linalg.lstsq(train_X, train_y, rcond=None)
print(w)



## === cell 19
print("Skipped explicit inverse OLS; using np.linalg.lstsq weights.")



## === cell 20
test_df = pd.read_csv("../input/test.csv")
test_df.dtypes



## === cell 21
test_df = test_df.copy()

test_df["pickup_longitude"] = test_df["pickup_longitude"].clip(-75.0, -72.0)
test_df["dropoff_longitude"] = test_df["dropoff_longitude"].clip(-75.0, -72.0)
test_df["pickup_latitude"] = test_df["pickup_latitude"].clip(40.0, 42.0)
test_df["dropoff_latitude"] = test_df["dropoff_latitude"].clip(40.0, 42.0)

test_df["passenger_count"] = test_df["passenger_count"].clip(lower=1, upper=6)

test_df["trip_distance_km"] = haversine_km(
    test_df["pickup_longitude"].values,
    test_df["pickup_latitude"].values,
    test_df["dropoff_longitude"].values,
    test_df["dropoff_latitude"].values,
)

add_travel_vector_features(test_df)
add_datetime_features(test_df)
add_landmark_distance_features(test_df)

DIST_CLIP_KM = 60.0
test_df["trip_distance_km"] = test_df["trip_distance_km"].clip(0.0, DIST_CLIP_KM)
for c in LANDMARK_FEATURES:
    test_df[c] = test_df[c].clip(0.0, DIST_CLIP_KM)

val_df = val_df.copy()
val_df["trip_distance_km"] = val_df["trip_distance_km"].clip(0.0, DIST_CLIP_KM)
for c in LANDMARK_FEATURES:
    val_df[c] = val_df[c].clip(0.0, DIST_CLIP_KM)

test_X = get_input_matrix(test_df, dt_medians=dt_medians)
val_X = get_input_matrix(val_df, dt_medians=dt_medians)

test_y_predictions = np.matmul(test_X, w)
val_y_predictions = np.matmul(val_X, w)

test_y_predictions = np.clip(test_y_predictions, 0.0, 250.0)
val_y_predictions = np.clip(val_y_predictions, 0.0, 250.0)

from sklearn.metrics import mean_squared_error

print(np.sqrt(mean_squared_error(val_y, val_y_predictions)))

submission = pd.DataFrame(
    {"key": test_df["key"].values, "fare_amount": test_y_predictions},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)

print(os.listdir("."))
