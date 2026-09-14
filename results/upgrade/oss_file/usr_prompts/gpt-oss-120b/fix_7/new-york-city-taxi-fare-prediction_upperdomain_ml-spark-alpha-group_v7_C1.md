# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
scipy==1.15.3
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

# 5. Code solution

## === cell 0
import os, glob
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def find_file(filename):
    """
    Return the first file that matches *filename* anywhere under the current directory.
    """
    if os.path.exists(filename):
        return filename
    candidates = glob.glob(os.path.join("**", filename), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"{filename} not found in any subdirectory.")
    return candidates[0]


train_path = find_file("train.csv")
test_path = find_file("test.csv")

print("Using train path:", train_path)
print("Using test path :", test_path)




## === cell 1
df = pd.read_csv(train_path, nrows=1_000_000)
df.head()




## === cell 2
alpha_ang = 0.506  # constant used in distance computation


def distance_travel(df):
    """
    Compute `distance_travel` and `haversine` using NumPy only.
    This reproduces the original calculations without creating the many
    intermediate DataFrame columns that were later discarded.
    """
    pu_lon = np.radians(df["pickup_longitude"].values)
    pu_lat = np.radians(df["pickup_latitude"].values)
    do_lon = np.radians(df["dropoff_longitude"].values)
    do_lat = np.radians(df["dropoff_latitude"].values)

    abs_diff_longitude = (
        df["dropoff_longitude"] - df["pickup_longitude"]
    ).abs().values * 50
    abs_diff_latitude = (
        df["dropoff_latitude"] - df["pickup_latitude"]
    ).abs().values * 69
    displacement_vector = np.sqrt(abs_diff_latitude**2 + abs_diff_longitude**2)

    angle = np.arctan2(abs_diff_longitude, abs_diff_latitude)

    actual_long = np.abs(displacement_vector * np.sin(angle - alpha_ang))
    actual_lat = np.abs(displacement_vector * np.cos(angle - alpha_ang))
    distance_travel = actual_long + actual_lat

    dlat = do_lat - pu_lat
    dlon = do_lon - pu_lon
    a = np.sin(dlat / 2) ** 2 + np.cos(pu_lat) * np.cos(do_lat) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_miles = 3959
    haversine = earth_radius_miles * c

    df["distance_travel"] = distance_travel
    df["haversine"] = haversine
    return df


df = distance_travel(df)

df["hour"] = pd.to_datetime(df["pickup_datetime"]).dt.hour

mask = (
    (df["passenger_count"] > 0)
    & (df["fare_amount"] > 0)
    & (df["distance_travel"] > 0)
    & (df["distance_travel"] < 30)
    & (df["fare_amount"] < 100)
)
df = df[mask]

df.head()




## === cell 3
l = len(df)
print("Total rows after filtering:", l)

train_df = df.iloc[: int(0.7 * l)]
val_df = df.iloc[int(0.7 * l) :]

train_X = np.column_stack(
    (
        train_df["distance_travel"],
        train_df["haversine"],
        train_df["passenger_count"],
        train_df["hour"],
        np.ones(len(train_df)),  # bias term
    )
)
val_X = np.column_stack(
    (
        val_df["distance_travel"],
        val_df["haversine"],
        val_df["passenger_count"],
        val_df["hour"],
        np.ones(len(val_df)),
    )
)

train_y = np.log1p(train_df["fare_amount"].values)
val_y = np.log1p(val_df["fare_amount"].values)




## === cell 4
imp = SimpleImputer(strategy="mean")
train_X = imp.fit_transform(train_X)
val_X = imp.transform(val_X)

regr = GradientBoostingRegressor(
    random_state=21,
    n_estimators=1200,
    learning_rate=0.05,
)
regr.fit(train_X, train_y)

val_pred_log = regr.predict(val_X)
val_pred = np.expm1(val_pred_log)  # inverse log‑transform
rmse = mean_squared_error(val_df["fare_amount"].values, val_pred, squared=False)
print(f"Validation RMSE: {rmse:.4f}")




## === cell 5
tdf = pd.read_csv(test_path)

tdf = distance_travel(tdf)
tdf["hour"] = pd.to_datetime(tdf["pickup_datetime"]).dt.hour
tdf.head()




## === cell 6
ttrain_X = np.column_stack(
    (
        tdf["distance_travel"],
        tdf["haversine"],
        tdf["passenger_count"],
        tdf["hour"],
        np.ones(len(tdf)),
    )
)
ttrain_X = imp.transform(ttrain_X)
output = np.expm1(regr.predict(ttrain_X))  # back‑transform predictions
print("First 5 predictions:", output[:5])




## === cell 7
submission = pd.DataFrame({"key": tdf["key"], "fare_amount": output})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
submission.head()
