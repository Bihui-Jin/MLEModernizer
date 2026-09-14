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

21.02506

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'The crash happens because scikit-learn 1.2 removed the `normalize` parameter from `LinearRegression`, so passing `normalize=True` raises a `TypeError`. The minimal fix is to instantiate `LinearRegression()` without that deprecated argument while keeping the same fit/predict logic and outputs used by the next cell. This change unblocks training and preserves the `lr` variable and its interface for cell 43. No other cells need modification.'
- What this solution (achieved 936.92806) has done: 'Your huge RMSE is most consistent with a feature mismatch between train and test after one-hot encoding weekdays (train and test can end up with different dummy columns), which makes the linear model see columns in a different order/meaning at prediction time. I fix this with a minimal alignment step that reindexes test features to the exact train feature columns (filling missing weekday columns with 0), preserving your model and training loop. I also ensure the submission file matches Kaggle’s expected schema (keep `key` as a column, not the index) and use the standard lowercase `submission.csv` filename. These changes are small but directly address a common cause of extreme score blow-ups, and should move RMSE sharply toward your target.'
- What this solution (achieved 752.76803) has done: 'Your RMSE is extremely far from the target, so we need a small but high-impact correctness fix rather than tuning: the biggest issue is that you normalize `abs_diff_*` on the test set using *test* mean/variance instead of the *train* mean/variance, which can severely distort test features and blow up predictions. I change this to compute train mean/variance once and apply the same scaling to both train and test (keeping your exact feature set and LinearRegression training). I also add a minimal guard to prevent division-by-zero if a variance is 0 (rare, but avoids NaNs that would ruin RMSE). Everything else (feature engineering, model, submission schema/path) stays the same, and the script still writes `submission.csv`.'
- What this solution (achieved 1048.63221) has done: 'Your RMSE is far above the target, so we need a correctness fix that meaningfully improves predictions without changing the model or feature set. The largest remaining issue is that `abs_diff_*` is “normalized” by dividing by the *variance* (and after taking an extra absolute value), which badly distorts scale; I switch to standard z-score scaling using the train mean and train standard deviation and apply it consistently to test. I also add a minimal, competition-safe cleanup to remove clearly invalid fares/coordinates/passenger counts (common in this dataset) because those outliers can dominate a linear regression fit and inflate RMSE; this keeps the same LinearRegression approach but trains on cleaner data. Finally, I keep your one-hot alignment and ensure the submission CSV is still written in the required format.'
- What this solution (achieved 990.63985) has done: 'Your RMSE is still catastrophically high, which strongly suggests a remaining correctness issue rather than model capacity: the main one left is inconsistent preprocessing between train and test for “Distance” and the airport distance features (they’re computed, but never scaled; and train has aggressive coordinate filtering while test can contain out-of-range coords that produce extreme feature values). I make the smallest changes that keep your LinearRegression and all existing features: apply the same basic coordinate/passenger filtering logic to the test set (without dropping rows; instead clip to the train’s allowed ranges) and standardize the three distance-based features using train statistics, applied identically to test. I also clip negative predictions to 0 (fares can’t be negative), which is a minimal, metric-safe postprocess that prevents RMSE blow-ups from a few large negative outputs. These changes are directly aimed at reducing the huge error toward your target without changing the core approach.'
- What this solution (achieved 13.06232) has done: 'Your RMSE is still wildly off-target, which is most consistent with the model learning a bad global linear fit due to remaining extreme/out-of-distribution feature values that survive the current filters, plus a few high-leverage training points. To move the score sharply down toward the target without changing your model or features, I (1) add minimal, standard NYC-taxi sanity filters on training (distance bounds + abs-diff bounds + time bounds) and (2) apply matching clipping on test (no row drops) to prevent extreme standardized values. I also ensure all numeric feature columns are finite after scaling (replace inf/NaN with 0) so the linear regression doesn’t produce huge predictions. Core logic (feature engineering + LinearRegression + train/test split + submission schema) stays the same.'
- What this solution (achieved 13.06232) has done: 'The crash happens because `Weekday` was created as pandas nullable integer dtype (`Int64`), and `Series.replace(..., value=["Monday", ...])` tries to assign strings into that integer-typed column, which pandas now disallows. The minimal fix is to explicitly cast `Weekday` to a string/object dtype before doing the replacement, preserving the same mapping semantics. I update only cell 23 to coerce `Weekday` to `object` and then run the existing `replace` call. This keeps `train_df`/`test_df` structure the same except that `Weekday` becomes the intended weekday-name strings.'
- What this solution (achieved 13.06233) has done: 'Your current score (13.06 RMSE) is still far worse than the target (5.689), so we should make a small, correctness-focused change that reduces error without changing the model type or feature set. The biggest remaining issue is that `pickuptime` is treated as a raw “HHMM” integer, which is not a linear/continuous representation of time (e.g., 10:59→11:00 jumps by 41), and linear regression fits poorly to that. I keep the same idea (“time of day feature”) but convert it to minutes-since-midnight (0–1439) computed from the same datetime, then standardize it using train statistics like your other numeric features. Everything else (feature engineering set, LinearRegression training, submission format/path) stays the same and the script still writes `submission.csv`.'
- What this solution (achieved 13.06313) has done: 'Your RMSE (13.06) is still far above the target (5.689), so the smallest likely-to-help change is to remove the information-destroying rounding of the three distance features to 2 decimals, which makes a linear model much less accurate on this problem. I keep the exact same feature set, same LinearRegression training, same filters/clipping, and the same scaling approach—only stop rounding those distances (and also stop rounding predictions, since Kaggle evaluates on raw floats). This should move the score downward toward the target while preserving your core logic and runtime constraints. The submission schema/path remain identical and a valid `submission.csv` is still written.'
- What this solution (achieved 14.97389) has done: 'Your RMSE (13.06) is still far above the target (5.689), so we should make a small but high-impact correctness change rather than tuning the model. The biggest remaining issue is that plain `LinearRegression` is very sensitive to the many residual outliers in this dataset; switching to a robust linear fit (same linear modeling family, same features, same fit/predict flow) typically reduces RMSE substantially on NYC taxi fares. I replace `LinearRegression` with `HuberRegressor` (robust to outliers) while keeping your feature engineering, scaling, clipping, and submission generation unchanged. I also keep deterministic settings so results are stable and the script still runs end-to-end within the time limit.'
- What this solution (achieved 13.06309) has done: 'I make one minimal, high-impact change that keeps your exact feature engineering and training flow but improves the robustness/fit quality of the same linear modeling family: switch from `HuberRegressor` (which can underfit with these features) to `Ridge` (L2-regularized least squares), which is a common strong baseline for this competition and should reduce RMSE toward your target. I keep the same train/test split, scaling, clipping, and column alignment so evaluation semantics remain consistent. I also ensure we never accidentally include `fare_amount` inside the numeric “finite cleanup” step by restricting that cleanup to feature columns only (prevents any unintended mutation side-effects). The script still run end-to-end and write a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 13.06313) has done: 'Your RMSE (13.06) is still much worse than the target (5.689), so we should make a small change that improves the linear model fit without changing your feature set or overall pipeline. The biggest remaining issue is that this dataset has heavy-tailed outliers even after filtering, and plain Ridge with default L2 strength can still be pulled by high-leverage points; a tiny, metric-aligned improvement is to tune `alpha` via a small fixed grid using your existing train/holdout split and RMSE (not changing training approach, just picking the best regularization strength). This keeps the same Ridge model family, same features, same preprocessing, and same prediction postprocessing, but typically moves RMSE down meaningfully. The submission schema and file writing remain unchanged and a valid `submission.csv` is produced.'
- What this solution (achieved 21.02506) has done: 'Your RMSE (13.06) is still far above the target (5.689), so the most likely “small but high-impact” fix is correcting a remaining preprocessing mismatch: you clip longitude/latitude on the test set, but you compute the engineered distance and abs-diff features *before* that clipping, so the model can still see extreme engineered values in test that never occur in train. I keep your exact model (Ridge + small alpha grid), feature set, and training flow, but move the test clipping earlier (right after reading test.csv) so that all engineered features are computed from the clipped coordinates. I also apply the same coordinate clipping to train *before* distance/abs-diff feature creation (without changing your existing row filters) to ensure engineering is consistent and numerically stable. These changes are minimal, preserve the core logic, and should reduce the train/test distribution shift, moving RMSE downward toward your target.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import time
import os  # reading the input files we have access to

print(os.listdir("../input"))



## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=10_000_000)
train_df.dtypes



## === cell 2
train_df.head()



## === cell 3
train_df.shape



## === cell 4
train_df.info()



## === cell 5
test_df = pd.read_csv("../input/test.csv")
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
def clip_coordinates_inplace(df):
    df["pickup_longitude"] = df["pickup_longitude"].clip(-75, -72)
    df["dropoff_longitude"] = df["dropoff_longitude"].clip(-75, -72)
    df["pickup_latitude"] = df["pickup_latitude"].clip(40, 42)
    df["dropoff_latitude"] = df["dropoff_latitude"].clip(40, 42)


clip_coordinates_inplace(train_df)
clip_coordinates_inplace(test_df)




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
plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")



## === cell 16
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))




## === cell 17
def creating_time(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=False)
    df["pickuptime"] = (dt.dt.hour * 60 + dt.dt.minute).astype("Int64")


creating_time(train_df)
creating_time(test_df)



## === cell 18
train_df.head()



## === cell 19
test_df.head()




## === cell 20
def creating_weekdays(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=False)
    df["Weekday"] = dt.dt.weekday.astype("Int64")


creating_weekdays(train_df)
creating_weekdays(test_df)



## === cell 21
train_df.head()



## === cell 22
test_df.head()



## === cell 23
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)




## === cell 24
def replace_weekday(df):
    df["Weekday"] = df["Weekday"].astype("object")
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



## === cell 25
train_df.head()



## === cell 26
test_df.head()



## === cell 27
train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_df["Weekday"])

train_one_hot, test_one_hot = train_one_hot.align(
    test_one_hot, join="outer", axis=1, fill_value=0
)

train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)



## === cell 28
train_df.head()



## === cell 29
test_df.head()



## === cell 30
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)



## === cell 31
train_df["pickuptime"] = train_df["pickuptime"].fillna(0).astype(int)
test_df["pickuptime"] = test_df["pickuptime"].fillna(0).astype(int)



## === cell 32
train_df.head()



## === cell 33
test_df.head()




## === cell 34
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




## === cell 35
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



## === cell 36
print("Old size before basic filtering: %d" % len(train_df))
train_df = train_df[
    (train_df["fare_amount"] > 0) & (train_df["fare_amount"] <= 250)
].copy()

train_df = train_df[
    train_df["pickup_longitude"].between(-75, -72)
    & train_df["dropoff_longitude"].between(-75, -72)
    & train_df["pickup_latitude"].between(40, 42)
    & train_df["dropoff_latitude"].between(40, 42)
].copy()

train_df = train_df[train_df["passenger_count"].between(1, 6)].copy()

train_df = train_df[train_df["Distance"].between(0.0, 100.0)].copy()
train_df = train_df[train_df["Pickup_Distance_airport"].between(0.0, 200.0)].copy()
train_df = train_df[train_df["Dropoff_Distance_airport"].between(0.0, 200.0)].copy()
train_df = train_df[train_df["abs_diff_longitude"].between(0.0, 2.0)].copy()
train_df = train_df[train_df["abs_diff_latitude"].between(0.0, 2.0)].copy()

train_df = train_df[train_df["pickuptime"].between(0, 1439)].copy()

test_df["pickup_longitude"] = test_df["pickup_longitude"].clip(-75, -72)
test_df["dropoff_longitude"] = test_df["dropoff_longitude"].clip(-75, -72)
test_df["pickup_latitude"] = test_df["pickup_latitude"].clip(40, 42)
test_df["dropoff_latitude"] = test_df["dropoff_latitude"].clip(40, 42)
test_df["passenger_count"] = test_df["passenger_count"].clip(1, 6)
test_df["Distance"] = test_df["Distance"].clip(0.0, 100.0)
test_df["Pickup_Distance_airport"] = test_df["Pickup_Distance_airport"].clip(0.0, 200.0)
test_df["Dropoff_Distance_airport"] = test_df["Dropoff_Distance_airport"].clip(
    0.0, 200.0
)
test_df["abs_diff_longitude"] = test_df["abs_diff_longitude"].clip(0.0, 2.0)
test_df["abs_diff_latitude"] = test_df["abs_diff_latitude"].clip(0.0, 2.0)
test_df["pickuptime"] = test_df["pickuptime"].clip(0, 1439)

print("New size after basic filtering: %d" % len(train_df))



## === cell 37
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



## === cell 38
_eps = 1e-12

_abs_long_mean = float(np.mean(train_df["abs_diff_longitude"]))
_abs_long_std = float(np.std(train_df["abs_diff_longitude"]))
_abs_long_std = _abs_long_std if _abs_long_std > 0 else _eps

train_df["abs_diff_longitude"] = (
    train_df["abs_diff_longitude"] - _abs_long_mean
) / _abs_long_std
test_df["abs_diff_longitude"] = (
    test_df["abs_diff_longitude"] - _abs_long_mean
) / _abs_long_std



## === cell 39
_abs_lat_mean = float(np.mean(train_df["abs_diff_latitude"]))
_abs_lat_std = float(np.std(train_df["abs_diff_latitude"]))
_abs_lat_std = _abs_lat_std if _abs_lat_std > 0 else _eps

train_df["abs_diff_latitude"] = (
    train_df["abs_diff_latitude"] - _abs_lat_mean
) / _abs_lat_std
test_df["abs_diff_latitude"] = (
    test_df["abs_diff_latitude"] - _abs_lat_mean
) / _abs_lat_std



## === cell 40
for _col in [
    "Distance",
    "Pickup_Distance_airport",
    "Dropoff_Distance_airport",
    "pickuptime",
]:
    _m = float(np.mean(train_df[_col]))
    _s = float(np.std(train_df[_col]))
    _s = _s if _s > 0 else _eps
    train_df[_col] = (train_df[_col] - _m) / _s
    test_df[_col] = (test_df[_col] - _m) / _s

_feature_cols_train = train_df.drop(
    columns=["key", "fare_amount"], errors="ignore"
).columns
_feature_num_cols_train = (
    train_df[_feature_cols_train].select_dtypes(include=[np.number]).columns
)
train_df[_feature_num_cols_train] = (
    train_df[_feature_num_cols_train].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)

_feature_cols_test = test_df.drop(columns=["key"], errors="ignore").columns
_feature_num_cols_test = (
    test_df[_feature_cols_test].select_dtypes(include=[np.number]).columns
)
test_df[_feature_num_cols_test] = (
    test_df[_feature_num_cols_test].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)



## === cell 41
print(train_df.shape)
print(test_df.shape)



## === cell 42
from sklearn.model_selection import train_test_split

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 43
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

alphas = [0.1, 0.3, 1.0, 3.0, 10.0]
best_alpha = None
best_rmse = None
best_model = None

for a in alphas:
    m = Ridge(alpha=a, fit_intercept=True, random_state=80)
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    if (best_rmse is None) or (rmse < best_rmse):
        best_rmse = rmse
        best_alpha = a
        best_model = m

lr = best_model
print("Best alpha:", best_alpha, "Holdout RMSE:", best_rmse)



## === cell 44
test_X = test_df.drop("key", axis=1)
test_X = test_X.reindex(columns=X.columns, fill_value=0)

pred = lr.predict(test_X)
pred = np.clip(pred, 0, None)
print(pred)



## === cell 45
Submission = pd.DataFrame({"key": test_df["key"].values, "fare_amount": pred})



## === cell 46
Submission.head()



## === cell 47
Submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", Submission.shape)
