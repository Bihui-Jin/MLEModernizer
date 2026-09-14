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

993.46117

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'I fix the runtime error by removing the deprecated `normalize` argument from `LinearRegression` (scikit-learn 1.2+) so the model can train and produce predictions. I also fix a likely hidden bug where one-hot weekday columns can differ between train and test by aligning the columns so `lr.predict()` won’t fail due to mismatched feature sets. Finally, I make the submission writing robust and ensure it outputs a valid `Submission.csv` with exactly `key,fare_amount` columns. These changes preserve the existing feature engineering and linear regression core logic while enabling an end-to-end run and a valid Kaggle submission file.'
- What this solution (achieved 993.4602) has done: 'I make two minimal, score-relevant fixes while preserving your exact feature engineering and LinearRegression core logic. First, I remove the `abs()` from your “normalization” step (it currently destroys directionality and harms RMSE), keeping the same mean/variance stats computed on train and applied to test. Second, I filter out obviously invalid training targets/coordinates (negative/huge fares and out-of-range lat/lon) before training; this is a standard NYC Taxi Fare cleanup that usually reduces RMSE without changing the model class or training loop. Everything else (feature creation, model, prediction clamp, submission format/path) stays the same and the script still writes `Submission.csv`.'
- What this solution (achieved 993.4602) has done: 'Your RMSE is catastrophically high relative to the target, which strongly suggests a correctness bug rather than a “needs a better model” issue. The most score-relevant minimal fix is to correct your “normalization” step: you’re dividing by the variance instead of the standard deviation, which badly mis-scales features and can explode errors. I change that to divide by `sqrt(var)` (std) while keeping the exact same features, LinearRegression model, and training flow. I also keep your train/test column alignment and ensure the submission remains `key,fare_amount` written to `Submission.csv`.'
- What this solution (achieved 993.4602) has done: 'Your RMSE is so far from the target that it’s almost certainly a feature/column mismatch bug at inference rather than “model quality”. The biggest minimal fix is to ensure the one-hot weekday columns are identical between train and test by doing `get_dummies` on the concatenated Weekday series (so you never silently drop a weekday column in test). Second, your current `X.align(..., join="outer")` can introduce train-only columns that don’t exist in test (or vice versa) and encourages mismatch; switching to `join="inner"` keeps only shared features and stabilizes inference without changing the model class or training flow. These are small, score-relevant correctness fixes that preserve your feature engineering intent and keep the same LinearRegression training approach while producing a valid `Submission.csv`.'
- What this solution (achieved 993.4602) has done: 'Your current RMSE is so far from the target that this still looks like a correctness/generalization issue, not a need for a different model. The most minimal, score-relevant change is to normalize the same kinds of scale-sensitive distance features you already use (Distance and airport distances) with the same train-stat z-scoring you’re already applying to abs diffs; LinearRegression is very sensitive to mixed feature scales, and this usually improves RMSE substantially without changing core logic. I also add a tiny amount of robustness around datetime parsing by using vectorized pandas operations (same semantics, fewer malformed-string edge cases) while keeping the same derived fields (pickuptime and weekday). Everything else—feature set intent, LinearRegression, train/valid split approach, prediction clamp, and submission format/path—remains the same and still produces `Submission.csv`.'
- What this solution (achieved 993.4602) has done: 'Your RMSE is still wildly above target, so this is almost certainly a correctness/alignment issue rather than “needs a better model.” I make two minimal, score-relevant fixes while keeping your exact feature set and LinearRegression training: (1) replace the slow list/loop time parsing with a vectorized hour/minute extraction that preserves the same “HHMM” semantics but avoids subtle string/NaT issues, and (2) ensure the training target aligns with the feature rows after all filtering/transforms by reindexing `y` to `X` right before splitting (protects against any silent index drift from earlier drops/resets). These changes don’t alter the model class, loss, or core feature engineering intent, but they prevent “garbage-in” training that can produce catastrophic RMSE. The script still runs end-to-end and writes a valid `Submission.csv` with `key,fare_amount`.'
- What this solution (achieved 993.46117) has done: 'Your RMSE is still massively above the target, which strongly indicates a correctness bug in feature construction rather than “model quality.” The biggest issue is that your haversine distance uses a radius in kilometers and then multiplies by 0.621 (km→miles), which is incorrect; it should convert km→miles by multiplying by 0.621371, and it’s even safer to compute directly with Earth radius in miles. I fix the distance/airport-distance computation to use a consistent Earth radius in miles (minimal change, same features), and keep everything else (cleaning, one-hot weekdays, normalization, LinearRegression, clamp, submission format) identical. This should move the score dramatically downward (better) toward the 5.6891 target without changing the model/loop structure.'
- What this solution (achieved 993.46117) has done: 'Your RMSE is still orders of magnitude off, which is very likely caused by a systematic prediction scaling problem rather than “model quality.” The smallest core-logic-preserving fix is to ensure we aren’t accidentally training/predicting with object-typed columns (common when datetime parsing produces mixed types) by forcing all model features to numeric and imputing any NaNs produced by `to_datetime(..., errors="coerce")` consistently in train/test. I also make the train/test feature alignment stricter by using the sample submission keys as the authoritative ordering for the output, preventing any silent key/prediction misalignment that can explode RMSE. These changes keep your exact feature set, LinearRegression model, training flow, and submission semantics, but remove common correctness pitfalls that can cause catastrophic leaderboard scores.'
- What this solution (achieved 993.46117) has done: 'Your RMSE is catastrophically worse than the target (lower is better), so this is almost certainly a data/label misalignment bug rather than a “needs a better model” issue. The smallest high-impact fix is to ensure `y` stays aligned with `X` after the multiple `reset_index/dropna/filter` operations: currently `y = y.loc[X.index]` can silently mismatch because `y` still has the pre-reset index while `X` is 0..n-1. I fix that by resetting `y`’s index right when it’s created (preserving the same rows) and then reindexing by position after `X.align`. I also make `train_test_split` deterministic without changing semantics by disabling shuffling=False? (No, that would change semantics), so we keep shuffle but just ensure indices are consistent; everything else (features, LinearRegression, cleaning, normalization, submission format/path) stays the same.'
- What this solution (achieved 993.46117) has done: 'Your RMSE being ~993 (lower is better) indicates a major correctness issue, not just weak modeling. The smallest high-impact fix is to stop mis-aligning the target `y` with features `X`: you currently reset indices and then do `y = y.iloc[X.index]`, which silently scrambles labels after filtering/concats and produces near-random training. I keep your exact feature engineering and `LinearRegression` approach, but rebuild `y` directly from the same cleaned dataframe row order as `X` (after alignment) and add a quick NaN-row drop on `X/y` together to ensure consistent training rows. This should dramatically reduce RMSE toward the target while still writing a valid `Submission.csv` in the required format.'
- What this solution (achieved 993.46117) has done: 'Your RMSE (~993) is so far from the target (5.6891, lower is better) that this still points to a correctness issue rather than incremental model quality. The most likely remaining bug is that you’re training on a 10M-row sample but leaving `y` as the unfiltered full-length target (because `mask_finite` is computed on the already-reset `X` and the full `y`, which can silently scramble alignment). I make the smallest fix: rebuild a single “model dataframe” right before training that contains both features and `fare_amount`, drop NaNs/infs once, and then split `X/y` from that same row order (preserving your exact feature engineering, LinearRegression, and normalization). I also ensure the test features get the same column order as training features (still using your `align(join="inner")`) and keep the submission-writing logic identical.'
- What this solution (achieved 993.46117) has done: 'Your RMSE (~993) is so far from the target (5.6891, lower is better) that this still looks like a correctness issue, not “weak linear regression.” The smallest high-impact fix is to stop silently training/predicting on different feature sets: right now `X.align(..., join="inner")` can drop important train-only columns (often the weekday one-hots), and it can also change column order; instead we reindex test to train columns and fill missing columns with 0 to preserve the intended one-hot features. Second, we remove `fare_amount` from the “all finite” feature filter (it can wrongly drop rows for large but valid fares and is not needed to ensure feature numeric validity), and we drop rows where `fare_amount` itself is non-finite separately to keep X/y perfectly aligned. These are minimal changes that keep your exact feature engineering, normalization, LinearRegression, and submission semantics intact, but should move the score dramatically downward toward the target band.'
- What this solution (achieved 993.46117) has done: 'Your RMSE being ~993 (lower is better) is consistent with a systematic feature mismatch between train and test (model expects one set of columns but test effectively provides a different set), which produces wildly wrong predictions. The smallest high-impact fix is to enforce identical one-hot weekday columns by explicitly setting them as a fixed categorical with all 7 days before `get_dummies`, instead of relying on whatever categories appear after datetime coercion. Second, we should drop rows where datetime parsing failed (NaT) in training (since those rows become “00:00 / Monday-like” artifacts) while keeping test NaTs imputed consistently, which stabilizes learned coefficients without changing the model class or training approach. Everything else (your feature engineering, z-scoring, LinearRegression fit/predict, clamping, and submission writing) stays the same and still writes a valid `Submission.csv`.'
- What this solution (achieved 993.46117) has done: 'Your RMSE is so far from the target that this still points to a fundamental correctness issue at inference: the test features are not being transformed the same way as train. The biggest minimal fix is to apply the same `pickuptime.notna()` / numeric coercion / finite-row filtering pipeline to `test_df` (but without dropping any rows) so we don’t feed NaNs/objects into the linear model and produce wildly wrong predictions. Second, we should stop using `sample_sub.merge(...)` (it can reorder/drop keys if there are subtle dtype/whitespace differences); instead, write predictions in the exact order of `test_df["key"]` which is the required submission order. These changes preserve your exact feature engineering, normalization, and `LinearRegression` training, but make train/test preprocessing consistent and prevent key/pred alignment errors that can explode RMSE.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import time
import os  # reading the input files we have access to

INPUT_DIR_CANDIDATES = [
    "../input",
    "/kaggle/input",
    "/kaggle/data/input",
    "/kaggle/data",
]
INPUT_DIR = None
for p in INPUT_DIR_CANDIDATES:
    if os.path.isdir(p):
        INPUT_DIR = p
        break
if INPUT_DIR is None:
    raise FileNotFoundError(
        "Could not find Kaggle input directory among: " + str(INPUT_DIR_CANDIDATES)
    )

print("Using INPUT_DIR:", INPUT_DIR)
print(os.listdir(INPUT_DIR)[:20])




## === cell 1
def resolve_path(filename: str) -> str:
    direct = os.path.join(INPUT_DIR, filename)
    nested = os.path.join(INPUT_DIR, "new-york-city-taxi-fare-prediction", filename)
    if os.path.exists(direct):
        return direct
    if os.path.exists(nested):
        return nested
    if os.path.exists(f"../input/{filename}"):
        return f"../input/{filename}"
    raise FileNotFoundError(f"Could not find {filename} in {direct} or {nested}")


TRAIN_PATH = resolve_path("train.csv")
TEST_PATH = resolve_path("test.csv")
SAMPLE_SUB_PATH = resolve_path("sample_submission.csv")

TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH



## === cell 2
train_df = pd.read_csv(TRAIN_PATH, nrows=10_000_000)
train_df.dtypes



## === cell 3
train_df.head()



## === cell 4
train_df.shape



## === cell 5
train_df.info()



## === cell 6
test_df = pd.read_csv(TEST_PATH)
test_df.dtypes



## === cell 7
test_df.head()



## === cell 8
test_df.info()



## === cell 9
test_df.shape



## === cell 10
train_df.isna().sum()




## === cell 11
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)



## === cell 12
print(f"Before Dropping null values: {len(train_df)}")
train_df.dropna(inplace=True)
print(f"After Dropping null values: {len(train_df)}")



## === cell 13
print(train_df.isnull().sum())



## === cell 14
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 15
try:
    plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 16
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))



## === cell 17
print("Old size (pre-clean): %d" % len(train_df))
train_df = train_df[
    (train_df["fare_amount"] > 0.0)
    & (train_df["fare_amount"] < 500.0)
    & (train_df["passenger_count"] >= 1)
    & (train_df["passenger_count"] <= 6)
    & (train_df["pickup_longitude"].between(-75.0, -72.0))
    & (train_df["dropoff_longitude"].between(-75.0, -72.0))
    & (train_df["pickup_latitude"].between(40.0, 42.0))
    & (train_df["dropoff_latitude"].between(40.0, 42.0))
]
print("New size (post-clean): %d" % len(train_df))




## === cell 18
def creating_time(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=False)
    df["pickuptime"] = dt.dt.strftime("%H:%M:%S")


creating_time(train_df)
creating_time(test_df)



## === cell 19
train_df.head()



## === cell 20
test_df.head()




## === cell 21
def creating_weekdays(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=False)
    df["Weekday"] = dt.dt.weekday


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
    df["Weekday"] = df["Weekday"].replace(
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
train_df["Weekday"] = pd.Categorical(train_df["Weekday"], categories=weekday_order)
test_df["Weekday"] = pd.Categorical(test_df["Weekday"], categories=weekday_order)

train_one_hot = pd.get_dummies(train_df["Weekday"], dtype=np.uint8)
test_one_hot = pd.get_dummies(test_df["Weekday"], dtype=np.uint8)

train_one_hot = train_one_hot.reindex(columns=weekday_order, fill_value=0)
test_one_hot = test_one_hot.reindex(columns=weekday_order, fill_value=0)

train_df = pd.concat(
    [train_df.reset_index(drop=True), train_one_hot.reset_index(drop=True)], axis=1
)
test_df = pd.concat(
    [test_df.reset_index(drop=True), test_one_hot.reset_index(drop=True)], axis=1
)



## === cell 29
train_df.head()



## === cell 30
test_df.head()



## === cell 31
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)




## === cell 32
def creating_pickupdate(df):
    dt = pd.to_datetime(df["pickuptime"], format="%H:%M:%S", errors="coerce")
    hhmm = (
        dt.dt.hour.fillna(0).astype(int) * 100 + dt.dt.minute.fillna(0).astype(int)
    ).astype(int)
    df["pickuptime"] = hhmm


creating_pickupdate(train_df)
creating_pickupdate(test_df)



## === cell 33
train_df.head()



## === cell 34
test_df.head()




## === cell 35
def finding_distance(df):
    R_miles = 3958.7613  # mean Earth radius in miles
    lat1 = np.asarray(np.radians(df["pickup_latitude"]))
    lon1 = np.asarray(np.radians(df["pickup_longitude"]))
    lat2 = np.asarray(np.radians(df["dropoff_latitude"]))
    lon2 = np.asarray(np.radians(df["dropoff_longitude"]))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R_miles * c

    df["Distance"] = np.asarray(distance)


finding_distance(train_df)
finding_distance(test_df)




## === cell 36
def creating_pickup_dropoff_distance(df):
    R_miles = 3958.7613  # mean Earth radius in miles
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
    distance1 = R_miles * c1
    df["Pickup_Distance_airport"] = np.asarray(distance1)

    a2 = (
        np.sin(d_lat_dropoff / 2) ** 2
        + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
    )
    c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
    distance2 = R_miles * c2
    df["Dropoff_Distance_airport"] = np.asarray(distance2)


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




## === cell 39
def normalize_with_train_stats(train_df, test_df, col):
    mu = float(np.mean(train_df[col]))
    var = float(np.var(train_df[col]))
    std = float(np.sqrt(var)) if var > 0.0 else 1.0
    if std == 0.0:
        std = 1.0
    train_df[col] = (train_df[col] - mu) / std
    test_df[col] = (test_df[col] - mu) / std


normalize_with_train_stats(train_df, test_df, "abs_diff_longitude")
normalize_with_train_stats(train_df, test_df, "abs_diff_latitude")

normalize_with_train_stats(train_df, test_df, "Distance")
normalize_with_train_stats(train_df, test_df, "Pickup_Distance_airport")
normalize_with_train_stats(train_df, test_df, "Dropoff_Distance_airport")



## === cell 40
print(train_df.shape)
print(test_df.shape)



## === cell 41
from sklearn.model_selection import train_test_split

train_df = train_df[train_df["pickuptime"].notna()].reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

model_df = train_df.drop(columns=["key"]).copy()
for c in model_df.columns:
    model_df[c] = pd.to_numeric(model_df[c], errors="coerce")

y = model_df["fare_amount"].reset_index(drop=True)
X = model_df.drop(columns=["fare_amount"]).reset_index(drop=True)

finite_X = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
X = X.loc[finite_X].reset_index(drop=True)
y = y.loc[finite_X].reset_index(drop=True)

finite_y = np.isfinite(y.to_numpy(dtype=float))
X = X.loc[finite_y].reset_index(drop=True)
y = y.loc[finite_y].reset_index(drop=True)

X_test_features = test_df.drop(["key"], axis=1).copy()
for c in X_test_features.columns:
    X_test_features[c] = pd.to_numeric(X_test_features[c], errors="coerce")

med = X.median(numeric_only=True)
X = X.fillna(med)
X_test_features = X_test_features.fillna(med)

missing_in_test = [c for c in X.columns if c not in X_test_features.columns]
for c in missing_in_test:
    X_test_features[c] = 0.0
X_test_features = X_test_features.reindex(columns=X.columns)

mask_finite_xy = np.isfinite(X.to_numpy(dtype=float)).all(axis=1) & np.isfinite(
    y.to_numpy(dtype=float)
)
X = X.loc[mask_finite_xy].reset_index(drop=True)
y = y.loc[mask_finite_xy].reset_index(drop=True)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 42
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_valid, y_valid))



## === cell 43
pred = lr.predict(X_test_features)
pred = np.maximum(pred, 0.0)
print(pred[:10])



## === cell 44
Submission = pd.DataFrame(
    {"key": test_df["key"].astype(str).to_numpy(), "fare_amount": pred.astype(float)}
)

Submission["fare_amount"] = Submission["fare_amount"].fillna(float(np.median(pred)))

out_path = "Submission.csv"
Submission.to_csv(out_path, index=False)
print("Wrote submission to:", out_path, "shape:", Submission.shape)
print(Submission.head())
