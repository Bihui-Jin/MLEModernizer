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

5.68866

# 6. Current score

1097.03975

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 828.35868) has done: 'I fix the pandas errors that currently stop execution (tuple-based column selection in groupby, and correlation failing due to non-numeric columns) while keeping the modeling logic unchanged. I also fix the datetime parsing to correctly handle the train/test `pickup_datetime` format (which includes fractional seconds but no timezone), ensuring the `hour`/`year` features are created reliably so later cells don’t fail. To move RMSE dramatically toward the target (your current score indicates the model is effectively broken/misaligned), I minimally adjust the feature matrix to include the already-engineered `distance_miles` and `passenger_count` (still using the same closed-form least-squares approach) and remove rounding of predictions before scoring/writing. Finally, I keep the same input paths but make them robust to the Kaggle `/kaggle/input/...` layout and ensure a valid `submission.csv` is always written.'
- What this solution (achieved 828.35868) has done: 'Your RMSE is extremely high because the feature engineering you *intend* to use (abs diffs + haversine distance) is computed **before** the train/validation split, but after the split you recompute it only for `val_df`/`test_df`—meaning `train_df` often lacks the engineered columns at fit time (or has inconsistent NaNs), producing a badly-conditioned/incorrect linear fit. I make the smallest change to ensure the exact same feature engineering is applied consistently to `train_df`, `val_df`, and `test_df` **after** the split, without changing the model (still OLS via `lstsq`) or adding new modeling ideas. I also defensively `copy()` before mutating slices to avoid pandas view issues and ensure alignment is stable. This should dramatically reduce RMSE toward the target while keeping the core logic identical.'
- What this solution (achieved 1097.03975) has done: 'Your RMSE is far above the target because the OLS fit is being driven by a huge number of extreme/out-of-range coordinates and fares that weren’t filtered out robustly (and the engineered features can still contain NaNs/inf after re-engineering). I keep your exact core approach (same engineered features + closed-form linear least squares) but (1) apply the same cleaning logic to the *split* train/val data after feature engineering, (2) add standard NYC Taxi Fare competition “sanity bounds” filters (lat/lon in plausible NYC ranges and fare upper bound) to prevent outliers from dominating the linear fit, and (3) ensure we drop any remaining NaN/inf rows right before fitting and scoring. This is a minimal change that typically drops RMSE dramatically toward the target without changing the model class or training method. The script still write a valid `submission.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

INPUT_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "../input"

print("INPUT_DIR =", INPUT_DIR)
try:
    print("Files in INPUT_DIR:", os.listdir(INPUT_DIR)[:20])
except Exception as e:
    print("Could not list input dir:", e)



## === cell 1
train_path = os.path.join(INPUT_DIR, "train.csv")
data = pd.read_csv(train_path, nrows=15_000_000)
print("Loaded train rows:", len(data))




## === cell 2
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(data)



## === cell 3
data["datetime_object"] = pd.to_datetime(
    data["pickup_datetime"], errors="coerce", utc=False
)




## === cell 4
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # miles


data["distance_miles"] = distance(
    data.pickup_latitude,
    data.pickup_longitude,
    data.dropoff_latitude,
    data.dropoff_longitude,
)

_ = data.distance_miles.describe()
_ = data.groupby("passenger_count")[["distance_miles", "fare_amount"]].mean()

print(
    "Average $USD/Mile : {:0.2f}".format(
        data.fare_amount.sum() / data.distance_miles.sum()
    )
)
data["fare_per_mile"] = data.fare_amount / data.distance_miles.replace(0, np.nan)
data["inv_distance_miles"] = 1 / data.distance_miles.replace(0, np.nan)

data["hour"] = data["datetime_object"].dt.hour
data["year"] = data["datetime_object"].dt.year



## === cell 5
import seaborn as sns
import matplotlib.pyplot as plt

numeric_data = data.select_dtypes(include=[np.number])
corrmat = numeric_data.corr()

f, ax = plt.subplots(figsize=(12, 9))
k = 14
cols = corrmat.nlargest(k, "fare_amount")["fare_amount"].index
cm = np.corrcoef(numeric_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(
    cm,
    cbar=True,
    annot=True,
    square=True,
    fmt=".2f",
    annot_kws={"size": 10},
    yticklabels=cols.values,
    xticklabels=cols.values,
)
plt.show()



## === cell 6
print("Old size: %d" % len(data))
data = data.dropna(how="any", axis="rows")
print("New size: %d" % len(data))



## === cell 7
print("Old size: %d" % len(data))
data = data[(data.abs_diff_longitude < 3.0) & (data.abs_diff_latitude < 3.0)]
data = data[data.fare_amount > 0]
data = data[data.passenger_count <= 9]
data = data[(data.distance_miles > 0.0)]
print("New size: %d" % len(data))



## === cell 8
print(data["datetime_object"].iloc[0])
_ = data.iloc[:1000].plot.scatter("hour", "fare_amount")



## === cell 9
from sklearn.model_selection import train_test_split

y = data.fare_amount
X = data.drop("fare_amount", axis=1)
train_df, val_df, train_y, val_y = train_test_split(
    X, y, test_size=0.2, random_state=42
)
train_df.dtypes




## === cell 10
def get_input_matrix(df):
    return np.column_stack(
        (
            df.abs_diff_longitude.to_numpy(),
            df.abs_diff_latitude.to_numpy(),
            df.distance_miles.to_numpy(),
            df.passenger_count.to_numpy(),
            np.ones(len(df)),
        )
    )


def engineer_features(df):
    df = df.copy()
    add_travel_vector_features(df)
    df["datetime_object"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=False
    )
    df["distance_miles"] = distance(
        df.pickup_latitude,
        df.pickup_longitude,
        df.dropoff_latitude,
        df.dropoff_longitude,
    )
    df["fare_per_mile"] = np.nan  # parity with earlier columns; not used in model
    df["inv_distance_miles"] = 1 / df.distance_miles.replace(0, np.nan)
    df["hour"] = df["datetime_object"].dt.hour
    df["year"] = df["datetime_object"].dt.year
    return df


def clean_after_engineering(df, y=None):
    df = df.copy()

    num_cols = [
        "abs_diff_longitude",
        "abs_diff_latitude",
        "distance_miles",
        "passenger_count",
    ]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    coord_mask = (
        df["pickup_longitude"].between(-75.0, -72.0)
        & df["dropoff_longitude"].between(-75.0, -72.0)
        & df["pickup_latitude"].between(40.0, 42.0)
        & df["dropoff_latitude"].between(40.0, 42.0)
    )

    basic_mask = (
        df["abs_diff_longitude"].lt(3.0)
        & df["abs_diff_latitude"].lt(3.0)
        & df["passenger_count"].le(9)
        & df["distance_miles"].gt(0.0)
    )

    keep = coord_mask & basic_mask

    df = df.loc[keep]

    if y is not None:
        y = y.loc[df.index]
        fare_mask = y.gt(0.0) & y.le(250.0)
        df = df.loc[fare_mask]
        y = y.loc[df.index]

    df = df.dropna(subset=num_cols)
    if y is not None:
        y = y.loc[df.index]

    return df, y


train_df = engineer_features(train_df)
val_df = engineer_features(val_df)

train_df, train_y = clean_after_engineering(train_df, train_y)
val_df, val_y = clean_after_engineering(val_df, val_y)

train_X = get_input_matrix(train_df)

print("Train X shape:", train_X.shape)
print("Train y shape:", train_y.shape)
print("Val rows:", len(val_df))



## === cell 11
(w, _, _, _) = np.linalg.lstsq(train_X, train_y, rcond=None)
print(w)



## === cell 12
w_OLS = np.matmul(
    np.matmul(np.linalg.inv(np.matmul(train_X.T, train_X)), train_X.T), train_y
)
print(w_OLS)



## === cell 13
test_path = os.path.join(INPUT_DIR, "test.csv")
test_df = pd.read_csv(test_path)
test_df.dtypes



## === cell 14
from sklearn.metrics import mean_squared_error

test_df = engineer_features(test_df)

test_df, _ = clean_after_engineering(test_df, None)

test_X = get_input_matrix(test_df)
val_X = get_input_matrix(val_df)

test_y_predictions = np.matmul(test_X, w)
val_y_predictions = np.matmul(val_X, w)

print("Validation RMSE:", np.sqrt(mean_squared_error(val_y, val_y_predictions)))

raw_test_df = pd.read_csv(test_path)
raw_test_df = engineer_features(raw_test_df)
raw_test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
for c in [
    "abs_diff_longitude",
    "abs_diff_latitude",
    "distance_miles",
    "passenger_count",
]:
    raw_test_df[c] = raw_test_df[c].fillna(0.0)

raw_test_X = get_input_matrix(raw_test_df)
raw_test_pred = np.matmul(raw_test_X, w)

submission = pd.DataFrame(
    {"key": raw_test_df["key"], "fare_amount": raw_test_pred},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(submission.head())
print("Files:", os.listdir(".")[:20])
