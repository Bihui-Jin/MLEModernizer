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

5.57556

# 6. Current score

8.59609

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 761.74243) has done: 'I fix the runtime error caused by the removed `normalize` argument in scikit-learn’s `LinearRegression` by replacing it with an equivalent `Pipeline(StandardScaler, LinearRegression)` so the same modeling approach remains intact. I also fix a silent feature-mismatch bug where weekday one-hot columns can differ between train and test by aligning columns before training and inference. Finally, I ensure the submission is written as a valid `.csv` with exactly the required columns (`key`, `fare_amount`) and without setting `key` as the index.'
- What this solution (achieved 965.39453) has done: 'Your RMSE is extremely high because the model is being trained on many obviously invalid/outlier training rows (e.g., wrong lat/long ranges, zero-distance rides with non-trivial fares, negative/huge fares), which causes the linear regression to fit nonsense and generalize poorly. To move the score toward the target while keeping the same core approach (feature engineering + linear regression), I add a minimal, standard NYC Taxi cleaning block: filter fares to a plausible range, passenger_count to 1–6, lat/long to NYC bounding boxes, and remove near-zero distance rides with high fares. I also clip negative predictions to 0 (a legitimate constraint for fares) to avoid catastrophic errors on a subset of test rows, which typically reduces RMSE. Everything else (features, model, pipeline, submission format/paths) stays the same.'
- What this solution (achieved 574.03455) has done: 'Your current RMSE is far from the target, so we need a small but meaningful correction that keeps the same core model (scaled LinearRegression) and features. The biggest issue is that you’re training on raw `fare_amount`, which makes the linear model overly sensitive to large-fare outliers; switching to a log1p target transform (and then inverting with expm1) is a standard, minimal change that preserves the same regression approach while drastically stabilizing RMSE. I also apply the same basic “valid row” filtering to the test set (coordinates/passenger_count) and fill any invalid predictions with a safe fallback (median train fare), preventing catastrophic errors from nonsensical test rows. Everything else (feature engineering, pipeline structure, and submission schema/path) stays the same and the script still writes a valid `Submission.csv`.'
- What this solution (achieved 16.37679) has done: 'Your current RMSE is still extremely far from the target, so we need a small but meaningful correction that keeps the same core approach (feature engineering + scaled LinearRegression + log1p target). The biggest remaining issue is that the model is trained with raw longitude/latitude *differences* that don’t reflect true ride geometry (and can explode under some bad rows); we add a single, standard geometric feature (Manhattan distance in miles) while keeping the same model and training loop. We also make the existing outlier filter on abs diffs consistent with NYC scale (use a tight threshold) to remove clearly broken coordinates that otherwise dominate a linear model. Finally, we keep your test “invalid row” handling but compute the mask before dropping columns and ensure the feature columns are aligned deterministically.'
- What this solution (achieved 8.82583) has done: 'Your current score (16.38 RMSE) is still far above the target (5.58), so we should make a small, low-risk improvement that keeps the same overall approach (feature engineering + scaled LinearRegression + log1p target). The largest remaining issue is that the model is still being trained on many “broken” rows (implausible coordinates, absurd distances, or extreme speed/fare relationships) that a linear model cannot handle; tightening the training cleaning using your already-computed `Distance` and `manhattan_dist` is a minimal change that typically yields a big RMSE reduction. I also ensure the same plausibility rules are applied consistently to test rows by using the already-computed `Distance`/`manhattan_dist` in `valid_test_mask`, so the fallback only triggers for truly invalid cases. Finally, I keep submission writing identical but add a small sanity clamp on extreme predictions (still legitimate) to prevent a few catastrophic outliers from dominating RMSE.'
- What this solution (achieved 8.82457) has done: 'We need to reduce RMSE from 8.83 toward 5.58 (lower is better), so we make the smallest changes that improve generalization without changing the core approach (same engineered features + log1p target + scaled LinearRegression). The biggest low-risk win is removing extreme-but-“valid” training rows that still break a linear model: very high speeds (distance too large for trip time) and unrealistic fare-per-mile relative to distance; we already have `Distance` and `pickuptime`, so we can add these filters cheaply. We also make `pickuptime` a true “minutes since midnight” (instead of HHMM as an integer), which preserves the same feature but makes it more linear-friendly. Finally, we keep the submission format identical and ensure masks/index alignment stays correct after cleaning.'
- What this solution (achieved 8.67267) has done: 'To move RMSE down from 8.82 toward 5.58 without changing your core approach (same engineered features + log1p target + scaled LinearRegression), the smallest high-impact fix is to make the “pickup_datetime” feature more informative in a linear model by adding standard linear-friendly time components (hour, day-of-week as integer, month, and year) while keeping your existing weekday one-hot and pickuptime intact. I also add one minimal, well-known feature interaction for taxi fares—`Distance * passenger_count`—which stays within your current linear-regression semantics but helps capture per-passenger scaling. Finally, I keep your current cleaning and submission logic unchanged, just ensuring the new columns are created consistently for train and test and included in the aligned feature matrix.'
- What this solution (achieved 8.59609) has done: 'I fix the runtime error by ensuring there are no NaNs/Infs in the feature matrix after datetime parsing and trip-minutes creation (those can introduce missing values even if the original rows weren’t NA). I do this minimally by coercing all feature columns to numeric, replacing inf with NaN, and imputing remaining NaNs with the training-column medians (and applying the same medians to test), so the existing LinearRegression pipeline can fit. I also keep your existing train/test column alignment and prediction post-processing, so evaluation semantics stay the same, while guaranteeing the model fits and a valid `Submission.csv` is written. Finally, I add a couple of small sanity casts to keep boolean masks aligned after cleaning.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import os  # reading the input files we have access to

for p in ["../input", "/kaggle/input"]:
    if os.path.exists(p):
        print(p, "->", os.listdir(p)[:20])



## === cell 1
train_path = "../input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "../input/new-york-city-taxi-fare-prediction/test.csv"
sample_path = "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"

if not os.path.exists(train_path):
    train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
    test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
    sample_path = (
        "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
    )

train_df = pd.read_csv(
    train_path,
    nrows=10_000_000,
    dtype={
        "fare_amount": "float32",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int16",
    },
)
train_df.dtypes



## === cell 2
test_df = pd.read_csv(
    test_path,
    dtype={
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int16",
    },
)
test_df.dtypes




## === cell 3
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


def add_manhattan_distance_miles(df):
    miles_per_degree_lat = 69.0
    miles_per_degree_lon = 69.0 * np.cos(
        np.radians(df["pickup_latitude"].astype(float))
    )
    df["manhattan_dist"] = (
        df["abs_diff_latitude"].astype(float) * miles_per_degree_lat
        + df["abs_diff_longitude"].astype(float) * miles_per_degree_lon
    )


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)
add_manhattan_distance_miles(train_df)
add_manhattan_distance_miles(test_df)



## === cell 4
test_df.head()



## === cell 5
print(train_df.isnull().sum())



## === cell 6
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 7
try:
    plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 8
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 1.0) & (train_df.abs_diff_latitude < 1.0)
]
print("New size: %d" % len(train_df))




## === cell 9
def get_input_matrix(df):
    return np.column_stack(
        (df.abs_diff_longitude, df.abs_diff_latitude, np.ones(len(df)))
    )


train_X = get_input_matrix(train_df)
train_y = np.array(train_df["fare_amount"])

print(train_X.shape)
print(train_y.shape)



## === cell 10
train_df.head()



## === cell 11
train_dt = pd.to_datetime(train_df["pickup_datetime"].str.slice(0, 19), errors="coerce")
test_dt = pd.to_datetime(test_df["pickup_datetime"].str.slice(0, 19), errors="coerce")

train_df["pickup_hour"] = train_dt.dt.hour.astype("int16")
test_df["pickup_hour"] = test_dt.dt.hour.astype("int16")

train_df["pickup_month"] = train_dt.dt.month.astype("int16")
test_df["pickup_month"] = test_dt.dt.month.astype("int16")

train_df["pickup_year"] = train_dt.dt.year.astype("int16")
test_df["pickup_year"] = test_dt.dt.year.astype("int16")

train_df["pickup_dayofyear"] = train_dt.dt.dayofyear.astype("int16")
test_df["pickup_dayofyear"] = test_dt.dt.dayofyear.astype("int16")

train_df["pickuptime"] = (
    train_dt.dt.hour.astype("int16") * 60 + train_dt.dt.minute.astype("int16")
).astype("int16")
test_df["pickuptime"] = (
    test_dt.dt.hour.astype("int16") * 60 + test_dt.dt.minute.astype("int16")
).astype("int16")



## === cell 12
train_df.head()



## === cell 13
train_df["Weekday"] = train_dt.dt.weekday.astype("int8")
test_df["Weekday"] = test_dt.dt.weekday.astype("int8")



## === cell 14
train_df.head()



## === cell 15
test_df.head()



## === cell 16
pass



## === cell 17
weekday_map = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}
train_df["Weekday"] = train_df["Weekday"].map(weekday_map)
test_df["Weekday"] = test_df["Weekday"].map(weekday_map)



## === cell 18
combined_weekday = pd.concat(
    [
        train_df[["Weekday"]].assign(_is_train=1),
        test_df[["Weekday"]].assign(_is_train=0),
    ],
    axis=0,
    ignore_index=True,
)
weekday_dummies = pd.get_dummies(combined_weekday["Weekday"])
combined_weekday = pd.concat([combined_weekday, weekday_dummies], axis=1)

train_week = combined_weekday.loc[
    combined_weekday["_is_train"] == 1, weekday_dummies.columns
].reset_index(drop=True)
test_week = combined_weekday.loc[
    combined_weekday["_is_train"] == 0, weekday_dummies.columns
].reset_index(drop=True)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df = pd.concat([train_df.drop(columns=["Weekday"]), train_week], axis=1)
test_df = pd.concat([test_df.drop(columns=["Weekday"]), test_week], axis=1)



## === cell 19
train_df.head()



## === cell 20
R = 6373.0
lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_df["Distance"] = np.asarray(distance) * 0.621

lat1 = np.asarray(np.radians(test_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_df["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_df["Distance"] = np.asarray(distance) * 0.621



## === cell 21
R = 6373.0
lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

lat3 = np.zeros(len(train_df)) + np.radians(40.6413111)
lon3 = np.zeros(len(train_df)) + np.radians(-73.7781391)
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
train_df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(test_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_df["dropoff_longitude"]))

lat3 = np.zeros(len(test_df)) + np.radians(40.6413111)
lon3 = np.zeros(len(test_df)) + np.radians(-73.7781391)
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
test_df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## === cell 22
train_df["Distance"] = np.round(train_df["Distance"], 2)
train_df["Pickup_Distance_airport"] = np.round(train_df["Pickup_Distance_airport"], 2)
train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)
test_df["Distance"] = np.round(test_df["Distance"], 2)
test_df["Pickup_Distance_airport"] = np.round(test_df["Pickup_Distance_airport"], 2)
test_df["Dropoff_Distance_airport"] = np.round(test_df["Dropoff_Distance_airport"], 2)

train_df["manhattan_dist"] = np.round(train_df["manhattan_dist"], 2)
test_df["manhattan_dist"] = np.round(test_df["manhattan_dist"], 2)

train_df["dist_x_passengers"] = (
    train_df["Distance"].astype("float32")
    * train_df["passenger_count"].astype("float32")
).astype("float32")
test_df["dist_x_passengers"] = (
    test_df["Distance"].astype("float32") * test_df["passenger_count"].astype("float32")
).astype("float32")



## === cell 23
train_key_dt = pd.to_datetime(train_df["key"].str.slice(0, 19), errors="coerce")
test_key_dt = pd.to_datetime(test_df["key"].str.slice(0, 19), errors="coerce")

train_trip_minutes = (train_key_dt - train_dt).dt.total_seconds() / 60.0
test_trip_minutes = (test_key_dt - test_dt).dt.total_seconds() / 60.0

train_df["trip_minutes"] = train_trip_minutes.astype("float32")
test_df["trip_minutes"] = test_trip_minutes.astype("float32")

train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)



## === cell 24
print("Before cleaning:", len(train_df))

train_df = train_df[(train_df["fare_amount"] > 0) & (train_df["fare_amount"] <= 250)]
train_df = train_df[
    (train_df["passenger_count"] >= 1) & (train_df["passenger_count"] <= 6)
]

train_df = train_df[
    (train_df["pickup_longitude"].between(-74.5, -72.8))
    & (train_df["dropoff_longitude"].between(-74.5, -72.8))
    & (train_df["pickup_latitude"].between(40.0, 41.8))
    & (train_df["dropoff_latitude"].between(40.0, 41.8))
]

train_df = train_df[(train_df["Distance"] >= 0.0) & (train_df["Distance"] <= 50.0)]
train_df = train_df[
    (train_df["manhattan_dist"] >= 0.0) & (train_df["manhattan_dist"] <= 60.0)
]

train_df = train_df[train_df["Distance"] >= 0.01]

train_df = train_df[~((train_df["Distance"] < 0.05) & (train_df["fare_amount"] > 3.0))]

ppm = train_df["fare_amount"] / np.maximum(train_df["Distance"].astype(float), 0.2)
train_df = train_df[(ppm >= 0.5) & (ppm <= 60.0)]

ppm2 = train_df["fare_amount"] / np.maximum(train_df["Distance"].astype(float), 1.0)
train_df = train_df[(train_df["Distance"] < 20.0) | (ppm2 <= 25.0)]

known_dur = train_df["trip_minutes"].notna() & (train_df["trip_minutes"] > 0)
mph = train_df["Distance"].astype("float32") / (
    train_df["trip_minutes"].astype("float32") / 60.0
)
train_df = train_df[~known_dur | (mph <= 75.0)]

print("After cleaning:", len(train_df))



## === cell 25
valid_test_mask = (
    (test_df["passenger_count"].between(1, 6))
    & (test_df["pickup_longitude"].between(-74.5, -72.8))
    & (test_df["dropoff_longitude"].between(-74.5, -72.8))
    & (test_df["pickup_latitude"].between(40.0, 41.8))
    & (test_df["dropoff_latitude"].between(40.0, 41.8))
    & (test_df["Distance"].between(0.01, 50.0))
    & (test_df["manhattan_dist"].between(0.0, 60.0))
)

known_dur_t = test_df["trip_minutes"].notna() & (test_df["trip_minutes"] > 0)
mph_t = test_df["Distance"].astype("float32") / (
    test_df["trip_minutes"].astype("float32") / 60.0
)
valid_test_mask = valid_test_mask & (~known_dur_t | (mph_t <= 75.0))
valid_test_mask = valid_test_mask.astype(bool)

print("Valid test rows:", int(valid_test_mask.sum()), " / ", len(test_df))



## === cell 26
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



## === cell 27
train_df.head()



## === cell 28
test_df.head()



## === cell 29
train_df.head()



## === cell 30
from sklearn.model_selection import train_test_split

X = train_df.drop(["key", "fare_amount"], axis=1)
y = np.log1p(train_df["fare_amount"].astype(np.float64))

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

if len(X) == 0:
    raise RuntimeError("No training rows left after cleaning; cannot train model.")

X_medians = X.median(numeric_only=True)
X = X.fillna(X_medians)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=42
)



## === cell 31
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lr = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LinearRegression()),
    ]
)
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 32
X_test_final = test_df.drop("key", axis=1)
X_test_final = X_test_final.reindex(columns=X.columns, fill_value=0)

X_test_final = X_test_final.apply(pd.to_numeric, errors="coerce")
X_test_final = X_test_final.replace([np.inf, -np.inf], np.nan)
X_test_final = X_test_final.fillna(X_medians)

pred_log = lr.predict(X_test_final)
pred = np.expm1(pred_log)

pred = np.clip(pred, 0, None)
pred = np.clip(pred, 0, 250.0)

fallback_fare = float(np.median(train_df["fare_amount"].values))
pred = np.where(valid_test_mask.values, pred, fallback_fare)

pred = np.round(pred, 2)



## === cell 33
Submission = pd.DataFrame({"key": test_df["key"].values, "fare_amount": pred})
Submission = Submission[["key", "fare_amount"]]
Submission.head()



## === cell 34
out_path = "Submission.csv"
Submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", Submission.shape)
print(Submission.head())
