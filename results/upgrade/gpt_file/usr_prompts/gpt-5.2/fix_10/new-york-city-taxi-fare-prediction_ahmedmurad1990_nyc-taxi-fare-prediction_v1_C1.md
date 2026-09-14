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

3.8

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

3.09737

# 6. Current score

5.64905

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.03218) has done: 'I fix the LightGBM API break by removing the deprecated `silent` argument and replacing `early_stopping_rounds` with the supported callback API so training runs on lightgbm==4.6.0. I also fix a logic bug where `direction` is added to `train_df` (raw) instead of `df_train` (the cleaned/feature-engineered training frame), which would otherwise cause a missing feature mismatch at training/inference. Finally, I ensure the test preprocessing matches training by dropping `pickup_datetime` only after features are created (and keeping `key` separately), and then write a valid `taxi_fare_submission.csv` with the required columns.'
- What this solution (achieved 13.90867) has done: 'Your current score (4.03218 RMSE) is worse than the target (3.09737), so we should make small, legitimate improvements that reduce RMSE without changing the core model/training approach. The biggest avoidable issue is that you’re training on only 2,000,000 rows while already having the full 55M-row `train.csv` available; increasing training rows (within the 600s budget) usually yields a clear RMSE drop for this competition. I also add one standard, minimal cleaning step to remove extreme/invalid fares and coordinates (without altering feature logic) so the model isn’t distorted by obvious outliers. Finally, I ensure train/test feature columns align exactly before prediction and keep the submission format unchanged.'
- What this solution (achieved 13.86529) has done: 'You’re currently far worse than the target (13.91 vs 3.10 RMSE; lower is better), so the most direct minimal fix is to remove two sources of major error without changing your model/feature logic: (1) you accidentally train without the `key` column but keep it in `df_train` features (and then force `key=0` at test time), which badly hurts performance; and (2) your `direction` function has a sign bug in `dlon` that makes the bearing inconsistent. I drop `key` from training features (but keep it for submission) and compute `direction` with the correct longitude delta, preserving the rest of your pipeline and LightGBM training approach. I also ensure train/test are aligned identically (drop `key` from test features before reindex) so inference matches training exactly and the submission remains valid.'
- What this solution (achieved 12.91888) has done: 'You’re still far from the target (13.865 vs 3.097 RMSE; lower is better), so we should make one small change that legitimately improves generalization without changing the model/training loop: add a standard “haversine distance” feature while keeping your existing Manhattan-style distance and direction features. This preserves the same LightGBM approach and semantics, but gives the model a much stronger geometric signal that typically drops RMSE substantially for this competition. I also make the train/test datetime parsing consistent by dropping rows with invalid pickup_datetime (NaT) after coercion (currently those NaTs can silently create NaN date features and hurt fit). Everything else (data size, cleaning ranges, LightGBM params, early stopping, submission format/path) stays the same.'
- What this solution (achieved 12.70448) has done: 'Your score is much worse than the target (12.91888 vs 3.09737 RMSE; lower is better), so we should make small, legitimate fixes that typically yield a large RMSE drop without changing the LightGBM training approach. The biggest issue is that `df_test` still contains NaT-derived NaNs in the datetime features (and possibly other NaNs), while the model is trained on a cleaned `df_train`; we drop/repair those consistently and ensure feature columns are strictly aligned. Second, your train cleaning currently allows `fare_amount == 0` and doesn’t remove impossible coordinate (0,0) points; adding these standard sanity filters reduces noise substantially while preserving the same feature set and model. Finally, we clip negative predictions to 0 (fares can’t be negative), which usually improves RMSE a bit without changing core logic.'
- What this solution (achieved 6.04431) has done: 'You don’t yet have a current score for this exact script, so the priority is to (1) make sure the pipeline always produces a valid 9,914-row submission (even if test rows get filtered out), and (2) make small, legitimate RMSE-improving fixes without changing the core LightGBM approach. The main score-impacting issue is that you currently filter the test set by coordinate/passenger ranges, which drops rows and breaks alignment with the required submission keys; instead we keep all test rows and only *clip/repair* invalid values to the training-clean ranges. We also make train/test feature handling consistent by ensuring datetime-derived feature NaNs are repaired in test (not dropped) and by safely filling any remaining NaNs. These changes preserve your model/feature set and training loop, but should materially reduce error and produce a valid submission every run.'
- What this solution (achieved 5.64916) has done: 'Your current RMSE (6.04431) is worse than the target (3.09737), so we should make a small, legitimate improvement that typically reduces error without changing the model/training loop. The biggest score issue left is that the split is random across all years, which leaks temporal structure and encourages overfitting to mixed-era patterns; switching to a time-based split (train on earlier rides, validate on later rides) usually yields a model that generalizes better to the test distribution for this competition while preserving the exact LightGBM training approach. I also ensure the datetime-derived categorical features are explicitly integer-typed in both train and test so LightGBM treats them consistently as categoricals (this avoids subtle dtype-driven behavior differences). Everything else (feature set, cleaning rules, LightGBM params, early stopping callback, submission writing) stays the same.'
- What this solution (achieved 5.64905) has done: 'I make two minimal, score-relevant fixes that typically reduce RMSE for this competition without changing your model/loop: (1) add the standard `abs_diff_longitude/latitude` and `distance` (Manhattan-like) features to the datetime-derived feature set by ensuring `calculate_abs_different()` actually returns the modified dataframe instead of silently doing nothing, and (2) ensure train/test handle remaining NaNs in engineered features consistently (fill test with train medians; fill train with medians after feature creation so LightGBM doesn’t see unintended missingness). These changes preserve your LightGBM training approach, parameters, and split logic, but remove a major feature bug and improve stability. The submission writing and format remain identical, producing `taxi_fare_submission.csv` with 9,914 rows.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("dark_background")
sns.set_style("darkgrid")



## === cell 2
train_path = "../input/train.csv"

traintypes = {
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
    "key": "str",
}

cols = list(traintypes.keys())

NROWS = 5_000_000
train_df = pd.read_csv(train_path, usecols=cols, dtype=traintypes, nrows=NROWS)



## === cell 3
train_df.to_feather("nyc_taxi_data_raw.feather")



## === cell 4
df_train = pd.read_feather("nyc_taxi_data_raw.feather")



## === cell 5
df_train.dtypes



## === cell 6
df_train.describe()



## === cell 7
len(df_train[df_train.fare_amount > 0])



## === cell 8
df_train = df_train[df_train.fare_amount > 0]



## === cell 9
pass



## === cell 10
df_train.isnull().sum()



## === cell 11
df_train = df_train.dropna(how="any", axis="rows")



## === cell 12
df_test = pd.read_csv("../input/test.csv")
df_test.head(5)



## === cell 13
df_test.describe()



## === cell 14
df_train["pickup_datetime"] = pd.to_datetime(
    df_train["pickup_datetime"], utc=True, errors="coerce"
)



## === cell 15
df_train["pickup_datetime"]



## === cell 16
df_test["pickup_datetime"] = pd.to_datetime(
    df_test["pickup_datetime"], utc=True, errors="coerce"
)




## === cell 17
def add_new_date_time_features(dataset):
    dataset["hour"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["year"] = dataset.pickup_datetime.dt.year
    dataset["day_of_week"] = dataset.pickup_datetime.dt.dayofweek
    return dataset




## === cell 18
df_train = add_new_date_time_features(df_train)
df_test = add_new_date_time_features(df_test)



## === cell 19
df_train.describe()



## === cell 20
df_train = df_train[(df_train["fare_amount"] > 0) & (df_train["fare_amount"] <= 250)]
df_train = df_train[
    (df_train["passenger_count"] >= 1) & (df_train["passenger_count"] <= 6)
]

df_train = df_train[df_train.pickup_longitude.between(-75, -72)]
df_train = df_train[df_train.dropoff_longitude.between(-75, -72)]
df_train = df_train[df_train.pickup_latitude.between(40, 42)]
df_train = df_train[df_train.dropoff_latitude.between(40, 42)]

df_train = df_train[
    (df_train["pickup_longitude"] != 0)
    & (df_train["pickup_latitude"] != 0)
    & (df_train["dropoff_longitude"] != 0)
    & (df_train["dropoff_latitude"] != 0)
]

df_test["passenger_count"] = df_test["passenger_count"].clip(1, 6).astype("int16")

for col, lo, hi in [
    ("pickup_longitude", -75.0, -72.0),
    ("dropoff_longitude", -75.0, -72.0),
    ("pickup_latitude", 40.0, 42.0),
    ("dropoff_latitude", 40.0, 42.0),
]:
    df_test[col] = pd.to_numeric(df_test[col], errors="coerce")
    df_test[col] = df_test[col].clip(lo, hi)

for col in [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]:
    df_test.loc[df_test[col] == 0, col] = np.nan



## === cell 21
df_train.shape




## === cell 22
def calculate_abs_different(df):
    df = df.copy()
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()
    return df


df_train = calculate_abs_different(df_train)
df_test = calculate_abs_different(df_test)




## === cell 23
def convert_different_miles(df):
    df = df.copy()
    df["abs_diff_longitude"] = df.abs_diff_longitude * 50
    df["abs_diff_latitude"] = df.abs_diff_latitude * 69
    return df


df_train = convert_different_miles(df_train)
df_test = convert_different_miles(df_test)



## === cell 24
meas_ang = 0.506  # 29 degrees = 0.506 radians (https://en.wikipedia.org/wiki/Commissioners%27_Plan_of_1811)
import math


def add_distance(df):
    df = df.copy()
    eps = 1e-12
    df["Euclidean"] = (df.abs_diff_latitude**2 + df.abs_diff_longitude**2) ** 0.5
    angle = np.arctan(df.abs_diff_longitude / (df.abs_diff_latitude + eps)) - meas_ang
    df["delta_manh_long"] = (df.Euclidean * np.sin(angle)).abs()
    df["delta_manh_lat"] = (df.Euclidean * np.cos(angle)).abs()
    df["distance"] = df.delta_manh_long + df.delta_manh_lat
    df.drop(
        [
            "abs_diff_longitude",
            "abs_diff_latitude",
            "Euclidean",
            "delta_manh_long",
            "delta_manh_lat",
        ],
        axis=1,
        inplace=True,
    )
    return df


df_train = add_distance(df_train)
df_test = add_distance(df_test)



## === cell 25
df_train.head()




## === cell 26
def calculate_direction(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Return direction/bearing-like angle between pickup and dropoff coordinates.
    """
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(
        np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    )
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon

    a = np.arctan2(
        np.sin(dlon) * np.cos(dropoff_lat),
        np.cos(pickup_lat) * np.sin(dropoff_lat)
        - np.sin(pickup_lat) * np.cos(dropoff_lat) * np.cos(dlon),
    )
    return a




## === cell 27
df_train["direction"] = calculate_direction(
    df_train["pickup_latitude"].values,
    df_train["pickup_longitude"].values,
    df_train["dropoff_latitude"].values,
    df_train["dropoff_longitude"].values,
)
df_test["direction"] = calculate_direction(
    df_test["pickup_latitude"].values,
    df_test["pickup_longitude"].values,
    df_test["dropoff_latitude"].values,
    df_test["dropoff_longitude"].values,
)



## === cell 28
df_train["pickup_latitude"].apply(lambda x: np.radians(x))
df_train["pickup_longitude"].apply(lambda x: np.radians(x))
df_train["dropoff_latitude"].apply(lambda x: np.radians(x))
df_train["dropoff_longitude"].apply(lambda x: np.radians(x))

df_test["pickup_latitude"].apply(lambda x: np.radians(x))
df_test["pickup_longitude"].apply(lambda x: np.radians(x))
df_test["dropoff_latitude"].apply(lambda x: np.radians(x))
df_test["dropoff_longitude"].apply(lambda x: np.radians(x))



## === cell 29
pass




## === cell 30
def add_haversine_km(df):
    R = 6371.0  # Earth radius in km
    lat1 = np.radians(df["pickup_latitude"].astype("float64").values)
    lon1 = np.radians(df["pickup_longitude"].astype("float64").values)
    lat2 = np.radians(df["dropoff_latitude"].astype("float64").values)
    lon2 = np.radians(df["dropoff_longitude"].astype("float64").values)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    df["haversine_km"] = (R * c).astype("float32")


add_haversine_km(df_train)
add_haversine_km(df_test)



## === cell 31
df_train = df_train.dropna(
    subset=["pickup_datetime", "hour", "day", "month", "year", "day_of_week"]
)

for c in ["hour", "day", "month", "year", "day_of_week"]:
    if c in df_train.columns:
        df_train[c] = df_train[c].astype("int16")
    if c in df_test.columns:
        df_test[c] = df_test[c].fillna(df_train[c].median()).astype("int16")

for c in [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "distance",
    "direction",
    "haversine_km",
]:
    if c in df_test.columns:
        df_test[c] = df_test[c].fillna(df_train[c].median())

num_cols_train = df_train.select_dtypes(include=[np.number]).columns
df_train[num_cols_train] = df_train[num_cols_train].fillna(
    df_train[num_cols_train].median()
)



## === cell 32
df_train = df_train.sort_values("pickup_datetime").reset_index(drop=True)



## === cell 33
df_train.drop(columns=["pickup_datetime"], inplace=True)

y = df_train["fare_amount"]
df_train = df_train.drop(columns=["fare_amount", "key"])



## === cell 34
df_train.head()



## === cell 35
import lightgbm as lgbm

split_idx = int(len(df_train) * 0.9)
x_train = df_train.iloc[:split_idx].copy()
y_train = y.iloc[:split_idx].copy()
x_test = df_train.iloc[split_idx:].copy()
y_test = y.iloc[split_idx:].copy()



## === cell 36
params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "nthread": 4,
    "num_leaves": 31,
    "learning_rate": 0.1,
    "max_depth": -1,
    "subsample": 0.8,
    "bagging_fraction": 1,
    "max_bin": 10000,
    "bagging_freq": 10,
    "metric": "rmse",
    "zero_as_missing": True,
    "num_rounds": 50000,
}



## === cell 37
cat_feats = ["year", "month", "day", "day_of_week"]

train_set = lgbm.Dataset(
    x_train, y_train, categorical_feature=cat_feats, free_raw_data=False
)
valid_set = lgbm.Dataset(
    x_test, y_test, categorical_feature=cat_feats, free_raw_data=False
)

model = lgbm.train(
    params,
    train_set=train_set,
    num_boost_round=10000,
    valid_sets=[valid_set],
    valid_names=["valid"],
    callbacks=[
        lgbm.early_stopping(stopping_rounds=1000),
        lgbm.log_evaluation(period=500),
    ],
)



## === cell 38
df_train.describe()



## === cell 39
test_key = df_test["key"].copy()

df_test.drop(columns=["pickup_datetime", "key"], axis=1, inplace=True)

df_test = df_test.reindex(columns=df_train.columns, fill_value=0)
df_test = df_test.fillna(0)



## === cell 40
prediction = model.predict(df_test, num_iteration=model.best_iteration)

prediction = np.clip(prediction, 0, None)



## === cell 41
submission = pd.DataFrame({"key": test_key, "fare_amount": prediction})

submission.to_csv("taxi_fare_submission.csv", index=False)
print(submission.head())
print("Wrote taxi_fare_submission.csv with shape:", submission.shape)
assert (
    submission.shape[0] == 9914
), "Submission must have exactly 9914 rows for this dataset."
assert list(submission.columns) == ["key", "fare_amount"]
