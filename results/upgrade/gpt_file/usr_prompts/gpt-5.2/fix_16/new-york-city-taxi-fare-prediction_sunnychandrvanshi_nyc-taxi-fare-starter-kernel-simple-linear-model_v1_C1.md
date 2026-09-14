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

5.6891

# 6. Current score

1183.96747

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'I fix the runtime error caused by the deprecated `normalize` argument in `sklearn.linear_model.LinearRegression` (removed in scikit-learn 1.2), by removing it so the model can fit. I also make the one-hot encoding robust so train/test always end up with identical weekday dummy columns (otherwise `lr.predict` can fail or silently misalign features). Finally, I keep the rest of your feature logic intact and ensure a valid `submission.csv` (lowercase is fine) is written with exactly the required columns `key,fare_amount`.'
- What this solution (achieved 941.33069) has done: 'Your score is extremely poor for this competition because the model is being trained on many invalid/outlier rows (e.g., negative/huge fares, passenger_count outside 1–6, and coordinates far outside NYC), which makes linear regression fit nonsense and yields massive RMSE. I keep your exact feature set and linear regression approach, but add the standard NYC Taxi Fare competition data-cleaning filters before training so the same core logic learns from realistic trips. I also fix the scaling bug where you normalize train and test using different means/variances (this miscalibrates predictions); instead, compute mean/variance on train and apply to both. These minimal changes should move RMSE dramatically down toward your target without changing the model architecture or feature engineering semantics.'
- What this solution (achieved 1077.60025) has done: 'Your current RMSE is far worse than the target (lower is better), so the smallest safe move is to keep your exact LinearRegression + existing features, but correct two issues that heavily hurt generalization: (1) your “normalization” divides by variance instead of standard deviation (making scaling wrong by orders of magnitude), and (2) you never filter out zero-distance / near-zero trips (which behave like outliers for a linear model with your engineered features). These are minimal changes that preserve the same core pipeline and semantics while typically dropping RMSE dramatically toward your target band. I also keep submission formatting identical but avoid setting `key` as the index (some Kaggle submissions are safer with `key` as a column).'
- What this solution (achieved 952.26147) has done: 'Your RMSE is catastrophically worse than the target, so we should make the smallest changes that fix the most damaging issue while keeping your LinearRegression + same engineered features. The main problem is that the `key`↔row alignment can be broken by `dropna`/filters because `test_df["key"].values` is used instead of the key from the *feature frame actually predicted on*; a single misalignment makes the submission essentially random and explodes RMSE. I preserve your feature pipeline and model, but keep `key` as a separate Series from the start, ensure `test_features` retains the same row order, and build the submission from `test_features.index`-aligned keys. I also clip negative predictions to 0 (valid fare domain) to avoid needless RMSE inflation without changing the learning approach.'
- What this solution (achieved 952.26147) has done: 'Your current RMSE indicates predictions are still effectively “nonsense” relative to the test rows, so the smallest high-impact fix is to ensure train/test feature scaling is consistent for all continuous engineered features (right now only `abs_diff_*` is scaled, while `Distance` and airport distances remain unscaled and dominate a linear model). I keep your exact LinearRegression + same features, but compute mean/std on the training split and apply to both train and test for `abs_diff_longitude`, `abs_diff_latitude`, `Distance`, `Pickup_Distance_airport`, and `Dropoff_Distance_airport`. I also make the train/test one-hot weekday columns deterministic (fixed Monday–Sunday order) to prevent subtle column-order/absence issues. These are minimal, metric-aligned changes that should dramatically reduce RMSE toward your target while preserving the pipeline and producing the same `submission.csv` format.'
- What this solution (achieved 952.26147) has done: 'Your RMSE is still orders of magnitude above the target, so we need a minimal but high-impact correction that preserves your LinearRegression + feature set: make the time features consistent and numeric. Right now `pickuptime` is derived via string slicing (and later converted to HHMM), which breaks for some timestamp formats and is not comparable across days; using proper `datetime` parsing stabilize both `pickuptime` and `Weekday` without changing the overall feature intent. I also add a small, standard scaling safety step to fill any remaining NaNs in the scaled columns (rare, but can explode predictions) and keep submission key alignment exactly as you already fixed. These changes should materially reduce RMSE toward the target without altering the model or engineered feature set.'
- What this solution (achieved 84.51145) has done: 'Your RMSE is still massively worse than the target, which strongly suggests the model is being trained on a target distribution that doesn’t match the competition test set (it “learn” absurd coefficients). The smallest high-impact fix that preserves your exact feature set and LinearRegression training is to (1) remove obviously corrupt fare rows using the standard NYC Taxi rule `fare_amount <= 2.5 + 1.56 * Distance_miles` (keeps the same core logic but prevents extreme-label contamination), and (2) enforce the standard coordinate “zero point” filter (some rows have 0/0 or invalid GPS that slip through your current bounds). These two filters are applied only to training (never to test), so submission row alignment remains intact and the pipeline still writes a valid `submission.csv` with `key,fare_amount`. Everything else (feature engineering, scaling, model, prediction post-processing) stays the same.'
- What this solution (achieved 951.72525) has done: 'Your current RMSE (84.51) is still far above the target (5.6891), so we should improve (lower) the score with the smallest changes that keep your LinearRegression + existing features intact. The main issue now is that the new “distance-based fare cap” filter is far too strict (it caps fares at about $2.5 + $1.56/mile), which removes a huge fraction of valid training data and biases the model badly; switching to the standard permissive NYC Taxi cap ($2.5 + $10/mile) keeps the same core logic (still a distance-based outlier filter) but restores a realistic label distribution. I also add a minimal “minimum distance” filter to avoid dividing by essentially-zero distance effects and a simple passenger_count sanity filter on test (fill invalids to 1) without dropping rows, preserving submission alignment. Everything else (feature engineering, scaling, model, training loop, submission writing) stays the same.'
- What this solution (achieved 949.97505) has done: 'Your RMSE is still catastrophically worse than the target, so we need a minimal, high-impact correction without changing the model or feature set: remove the distance-based fare cap you added, because it is still deleting a large fraction of valid training rides and strongly biases LinearRegression. Instead, keep the standard, lightweight NYC Taxi cleaning (fare bounds, passenger bounds, NYC coordinate bounds, non-zero & minimum distance) which preserves core logic but restores a realistic target distribution. I also keep the existing consistent scaling and robust weekday one-hot alignment, and ensure the submission keys remain aligned with the predicted rows. This should move RMSE sharply downward toward the target band while remaining a small change.'
- What this solution (achieved 949.97505) has done: 'Your RMSE is catastrophically worse than the target, so the smallest high-impact move is to fix feature–row misalignment caused by dropping rows from `test_df` (you currently drop NaT pickup_datetimes in test, which changes row count and order vs the required submission keys). I keep all core feature engineering and the LinearRegression training exactly as-is, but ensure **test rows are never dropped**: instead, fill invalid `pickup_datetime` with a deterministic default so every test `key` gets a prediction. I also make the final submission explicitly use the (unchanged) original `test_df["key"]` order and assert the submission length matches the test file length to prevent silent corruption. These changes should drastically reduce RMSE toward your target because they restore correct key↔prediction alignment without changing model logic.'
- What this solution (achieved 949.97505) has done: 'Your RMSE is still catastrophically worse than the target, so the smallest likely “high-impact” fix is to ensure we are training on the **correct training file**: your environment includes `labels.csv`, and if that file accidentally contains corrupted/shifted labels or different ordering, the model learns nonsense and produces huge RMSE. I keep your exact feature engineering + LinearRegression pipeline intact, but explicitly load `train.csv` from the competition subfolder and add a strict sanity check that `fare_amount` is numeric and non-null before fitting. I also keep test rows intact (no dropping) and preserve your existing key alignment and submission writing, so the only substantive change is training on the correct source data. These changes should move the score sharply downward (better) toward your target without changing the model/feature logic.'
- What this solution (achieved 960.02844) has done: 'Your RMSE is still catastrophic relative to the target, which strongly suggests the model is being trained on a badly contaminated target distribution and/or unstable regression due to heavy-tailed outliers, even after the basic filters. To move the score sharply downward while preserving your exact LinearRegression + feature set and training loop, I add a single, standard training-only cleanup: remove extreme fares relative to trip distance using a permissive upper bound (keeps almost all valid trips but drops corrupt labels that dominate least-squares). I also ensure we don’t accidentally drop additional rows late in the pipeline by applying the distance-based filter before scaling and the final `dropna`, keeping test untouched and key alignment intact. Everything else (feature engineering, scaling, model, prediction post-processing, and submission writing) remains the same.'
- What this solution (achieved 1184.54729) has done: 'Your current RMSE is far worse than the target (lower is better), so we should make a small, high-impact fix that preserves your LinearRegression + existing engineered features: the biggest remaining issue is likely silent contamination from “bad” labels/rows (especially `fare_amount==0`, extreme fares, and invalid GPS) and weak robustness to remaining heavy-tailed outliers. I keep the same feature set and training approach, but tighten the training-only cleaning with standard NYC Taxi constraints (reasonable fare lower bound, tighter coordinate box, and an upper bound on distance) while keeping test untouched to preserve row/key alignment. I also ensure the training filters are applied before `dropna` and scaling so we don’t accidentally create NaNs/infs that later get dropped unpredictably. These minimal, metric-aligned changes should move RMSE sharply downward toward your target band without changing the core modeling logic.'
- What this solution (achieved 1183.96747) has done: 'Your current RMSE is far worse than the target (lower is better), so the smallest likely high-impact change is to stop training on a 10M-row slice that is dominated by old distribution/outliers and instead train on a smaller but cleaner, more representative sample spread across the full file. I keep your exact feature engineering, scaling, and LinearRegression intact, but switch the training load to a deterministic sampled subset (using `skiprows`) from the full `train.csv`, then apply the same cleaning you already have. This typically improves generalization a lot for this competition without changing the model or features. I also add a tiny extra training-only filter `fare_amount <= 2.5 + 10 * Distance` (permissive) to reduce heavy-tailed label corruption that least-squares is very sensitive to, while keeping test untouched and key alignment intact.'
- What this solution (achieved 1183.96747) has done: 'Your current RMSE is far worse than the target (lower is better), so we should make the smallest changes that address the most likely cause: the 2M-row “skiprows random sampling” is not actually uniformly sampling rows (and can be very slow/biased), leading to a badly skewed training distribution and unstable coefficients. I keep your exact feature engineering and LinearRegression training, but switch to a deterministic two-pass approach: read only the `key` column to sample row indices uniformly, then load just those rows—this preserves the same sample size while making it representative and faster. I also apply the same `basic_nyc_cleaning` constraints to **test_df** safely (without dropping any test rows) by clipping invalid passenger counts and filling any invalid coordinates/datetimes, so feature ranges match train and the linear model doesn’t extrapolate wildly. Everything else (features, scaling, model, submission format and alignment checks) stays intact.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import time
import os  # reading the input files we have access to

print(os.listdir("../input"))



## === cell 1
TRAIN_PATH = "../input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "../input/new-york-city-taxi-fare-prediction/test.csv"

N_TRAIN = 2_000_000  # keep within time/memory; still enough for LinearRegression
rng = np.random.RandomState(80)

train_keys = pd.read_csv(TRAIN_PATH, usecols=["key"])
max_rows = train_keys.shape[0]
picked_pos = rng.choice(
    max_rows, size=min(N_TRAIN, max_rows), replace=False
)  # 0-based positions in data (excluding header)
picked_pos.sort()

keep_rows = set((picked_pos + 1).tolist())
skip = lambda i: (i != 0) and (i not in keep_rows)

train_df = pd.read_csv(TRAIN_PATH, skiprows=skip)
train_df.dtypes



## === cell 2
train_df.head()



## === cell 3
train_df.shape



## === cell 4
train_df.info()



## === cell 5
test_df = pd.read_csv(TEST_PATH)
test_df.dtypes



## === cell 6
test_df.head()



## === cell 7
test_df.info()



## === cell 8
test_df.shape



## === cell 9
train_df.isna().sum()




## === cell 10
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)



## === cell 11
print(f"Before Dropping null values: {len(train_df)}")
train_df.dropna(inplace=True)
print(f"After Dropping null values: {len(train_df)}")



## === cell 12
print(train_df.isnull().sum())



## === cell 13
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 14
plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")



## === cell 15
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))




## === cell 16
def basic_nyc_cleaning(df):
    df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 6)]
    df = df[
        (df["pickup_longitude"] >= -74.5)
        & (df["pickup_longitude"] <= -72.8)
        & (df["dropoff_longitude"] >= -74.5)
        & (df["dropoff_longitude"] <= -72.8)
        & (df["pickup_latitude"] >= 40.5)
        & (df["pickup_latitude"] <= 41.8)
        & (df["dropoff_latitude"] >= 40.5)
        & (df["dropoff_latitude"] <= 41.8)
    ]
    return df


train_df = basic_nyc_cleaning(train_df)

train_df["fare_amount"] = pd.to_numeric(train_df["fare_amount"], errors="coerce")

train_df = train_df[(train_df["fare_amount"] >= 2.5) & (train_df["fare_amount"] <= 200)]

train_df = train_df[
    (train_df["abs_diff_longitude"] + train_df["abs_diff_latitude"]) > 1e-6
]

print("After basic_nyc_cleaning + fare filter (+ nonzero vector):", train_df.shape)



## === cell 17
train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"], errors="coerce", utc=True
)
test_df["pickup_datetime"] = pd.to_datetime(
    test_df["pickup_datetime"], errors="coerce", utc=True
)

train_df = train_df.dropna(subset=["pickup_datetime", "fare_amount"]).copy()

test_df["pickup_datetime"] = test_df["pickup_datetime"].fillna(
    pd.Timestamp("2010-01-01", tz="UTC")
)




## === cell 18
def creating_time(df):
    dt = df["pickup_datetime"].dt.tz_convert(None)
    df["pickuptime"] = (dt.dt.hour * 100 + dt.dt.minute).astype(np.int32)


creating_time(train_df)
creating_time(test_df)



## === cell 19
train_df.head()



## === cell 20
test_df.head()




## === cell 21
def creating_weekdays(df):
    dt = df["pickup_datetime"].dt.tz_convert(None)
    df["Weekday"] = dt.dt.weekday.astype(np.int8)


creating_weekdays(train_df)
creating_weekdays(test_df)



## === cell 22
train_df.head()



## === cell 23
test_df.head()



## === cell 24
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)




## === cell 25
def replace_weekday(df):
    df["Weekday"].replace(
        to_replace=[i for i in range(0, 7)],
        value=[
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
        inplace=True,
    )


replace_weekday(train_df)
replace_weekday(test_df)



## === cell 26
train_df.head()



## === cell 27
test_df.head()



## === cell 28
weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
train_df["Weekday"] = pd.Categorical(
    train_df["Weekday"], categories=weekday_order, ordered=True
)
test_df["Weekday"] = pd.Categorical(
    test_df["Weekday"], categories=weekday_order, ordered=True
)

train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_df["Weekday"])

train_one_hot, test_one_hot = train_one_hot.align(
    test_one_hot, join="outer", axis=1, fill_value=0
)

train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)



## === cell 29
train_df.head()



## === cell 30
test_df.head()



## === cell 31
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)



## === cell 32
train_df["pickuptime"] = train_df["pickuptime"].astype(np.int32)
test_df["pickuptime"] = test_df["pickuptime"].astype(np.int32)



## === cell 33
train_df.head()



## === cell 34
test_df.head()




## === cell 35
def finding_distance(df):
    R = 6373.0
    lat1 = np.asarray(np.radians(df["pickup_latitude"]))
    lon1 = np.asarray(np.radians(df["pickup_longitude"]))
    lat2 = np.asarray(np.radians(df["dropoff_latitude"]))
    lon2 = np.asarray(np.radians(df["dropoff_longitude"]))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    df["Distance"] = np.asarray(distance) * 0.621


finding_distance(train_df)
finding_distance(test_df)




## === cell 36
def creating_pickup_dropoff_distance(df):
    R = 6373.0
    lat1 = np.asarray(np.radians(df["pickup_latitude"]))
    lon1 = np.asarray(np.radians(df["pickup_longitude"]))
    lat2 = np.asarray(np.radians(df["dropoff_latitude"]))
    lon2 = np.asarray(np.radians(df["dropoff_longitude"]))

    lat3 = np.zeros(len(df)) + np.radians(40.6413111)
    lon3 = np.zeros(len(df)) + np.radians(-73.7781391)
    dlon_pickup = lon3 - lon1
    dlat_pickup = lat3 - lat1
    d_lon_dropoff = lon3 - lon2
    d_lat_dropoff = lat3 - lat2

    a1 = (
        np.sin(dlat_pickup / 2) ** 2
        + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
    )
    c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
    distance1 = R * c1
    df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

    a2 = (
        np.sin(d_lat_dropoff / 2) ** 2
        + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
    )
    c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
    distance2 = R * c2

    df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621


creating_pickup_dropoff_distance(train_df)
creating_pickup_dropoff_distance(test_df)



## === cell 37
train_df["Distance"] = np.round(train_df["Distance"], 2)
train_df["Pickup_Distance_airport"] = np.round(train_df["Pickup_Distance_airport"], 2)
train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)
test_df["Distance"] = np.round(test_df["Distance"], 2)
test_df["Pickup_Distance_airport"] = np.round(test_df["Pickup_Distance_airport"], 2)
test_df["Dropoff_Distance_airport"] = np.round(test_df["Dropoff_Distance_airport"], 2)



## === cell 38
train_df = train_df[
    (train_df["pickup_longitude"].abs() > 1e-6)
    & (train_df["pickup_latitude"].abs() > 1e-6)
    & (train_df["dropoff_longitude"].abs() > 1e-6)
    & (train_df["dropoff_latitude"].abs() > 1e-6)
].copy()

train_df = train_df[
    (train_df["Distance"] >= 0.05) & (train_df["Distance"] <= 50.0)
].copy()

print("After GPS zero filter + min/max distance:", train_df.shape)

test_df["passenger_count"] = (
    test_df["passenger_count"].clip(lower=1, upper=6).fillna(1).astype(np.int16)
)
test_df["pickup_longitude"] = test_df["pickup_longitude"].clip(-74.5, -72.8)
test_df["dropoff_longitude"] = test_df["dropoff_longitude"].clip(-74.5, -72.8)
test_df["pickup_latitude"] = test_df["pickup_latitude"].clip(40.5, 41.8)
test_df["dropoff_latitude"] = test_df["dropoff_latitude"].clip(40.5, 41.8)



## === cell 39
train_df = train_df[
    train_df["fare_amount"] <= (2.5 + 10.0 * train_df["Distance"])
].copy()
print("After distance-based fare outlier filter (permissive):", train_df.shape)



## === cell 40
train_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 41
scale_cols = [
    "abs_diff_longitude",
    "abs_diff_latitude",
    "Distance",
    "Pickup_Distance_airport",
    "Dropoff_Distance_airport",
]

means = train_df[scale_cols].mean()
stds = train_df[scale_cols].std(ddof=0).replace(0, 1.0)

train_df[scale_cols] = (train_df[scale_cols] - means) / stds
test_df[scale_cols] = (test_df[scale_cols] - means) / stds

train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)

train_df.dropna(inplace=True)
test_df.fillna(0.0, inplace=True)



## === cell 42
print(train_df.shape)
print(test_df.shape)



## === cell 43
from sklearn.model_selection import train_test_split

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 44
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 45
test_keys = test_df["key"].copy()

test_features = test_df.drop("key", axis=1)
test_features = test_features.reindex(columns=X.columns, fill_value=0)

pred = lr.predict(test_features)

pred = np.clip(pred, 0, None)
pred = np.round(pred, 2)

print(pred)



## === cell 46
Submission = pd.DataFrame({"key": test_keys.values, "fare_amount": pred})



## === cell 47
Submission.head()



## === cell 48
orig_test_n = pd.read_csv(TEST_PATH, usecols=["key"]).shape[0]
assert Submission.shape[0] == orig_test_n, (
    f"Submission rows ({Submission.shape[0]}) != original test rows ({orig_test_n}). "
    "This would cause key↔prediction mismatch and catastrophic RMSE."
)

Submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", Submission.shape)
print("Submission columns:", Submission.columns.tolist())
print("First/last keys:", Submission["key"].iloc[0], Submission["key"].iloc[-1])
