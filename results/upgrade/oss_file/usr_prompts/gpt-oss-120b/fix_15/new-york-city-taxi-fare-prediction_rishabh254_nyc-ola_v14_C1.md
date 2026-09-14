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

# 5. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler  # new import for scaling

data = pd.read_csv("../input/train.csv")




## === cell 1
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()




## === cell 2
add_travel_vector_features(data)

data["datetime_object"] = pd.to_datetime(data["pickup_datetime"], errors="coerce")
data["hour"] = data["datetime_object"].dt.hour
data["year"] = data["datetime_object"].dt.year
data["year_offset"] = data["year"] - data["year"].min()




## === cell 3
def distance(lat1, lon1, lat2, lon2):
    """Haversine distance in miles."""
    p = 0.017453292519943295  # pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))




## === cell 4
data["distance_miles"] = distance(
    data.pickup_latitude,
    data.pickup_longitude,
    data.dropoff_latitude,
    data.dropoff_longitude,
)

group_stats = data.groupby("passenger_count")[["distance_miles", "fare_amount"]].mean()
print(group_stats.head())
print(
    "Average $USD/Mile : {:0.2f}".format(
        data.fare_amount.sum() / data.distance_miles.sum()
    )
)

data["fare_per_mile"] = data.fare_amount / data.distance_miles
data["inv_distance_miles"] = 1 / data.distance_miles


## === cell 5
corrmat = data.select_dtypes(include=[np.number]).corr()
k = 14  # number of variables for heatmap
cols = corrmat.nlargest(k, "fare_amount")["fare_amount"].index
cm = np.corrcoef(data[cols].values.T)

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
print("New size after dropna: %d" % len(data))

nyc = (-74.0063889, 40.7141667)  # (lon, lat) of NYC centre
data = data[
    (data.abs_diff_longitude < 3.0)
    & (data.abs_diff_latitude < 3.0)
    & (data.fare_amount >= 0)
    & (data.passenger_count <= 9)
    & (data.distance_miles > 0.05)
]
data["distance_to_center"] = distance(
    nyc[1], nyc[0], data.dropoff_latitude, data.dropoff_longitude
)
data = data[data.distance_to_center < 15.0]
print("New size after filtering: %d" % len(data))


## === cell 7
y = data.fare_amount
y_log = np.log1p(y)

X = data.drop("fare_amount", axis=1)
train_df, val_df, train_y_log, val_y_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
print("Sample dtypes after split:")
print(train_df.dtypes.head())




## === cell 8
def get_input_matrix(df):
    """
    Build the design matrix used by the linear model.
    Includes distance, passenger count, hour, year offset,
    log‑distance and the geographic helper features.
    """
    log_dist = np.log1p(df.distance_miles)
    abs_lon = df["abs_diff_longitude"] if "abs_diff_longitude" in df else 0
    abs_lat = df["abs_diff_latitude"] if "abs_diff_latitude" in df else 0
    dist_center = df["distance_to_center"] if "distance_to_center" in df else 0
    return np.column_stack(
        (
            df.distance_miles,
            df.passenger_count,
            df.hour,
            df.year_offset,
            log_dist,
            abs_lon,
            abs_lat,
            dist_center,
        )
    )


train_X_raw = get_input_matrix(train_df)
val_X_raw = get_input_matrix(val_df)

scaler = StandardScaler()
train_X = scaler.fit_transform(train_X_raw)
val_X = scaler.transform(val_X_raw)

print("train_X shape:", train_X.shape, "train_y shape:", train_y_log.shape)


## === cell 9
ridge = Ridge(alpha=0.1, fit_intercept=True, random_state=42)
ridge.fit(train_X, train_y_log)
print("Ridge coefficients:", ridge.coef_)
print("Ridge intercept:", ridge.intercept_)


## === cell 10
test_df = pd.read_csv("../input/test.csv")
test_df["distance_miles"] = distance(
    test_df.pickup_latitude,
    test_df.pickup_longitude,
    test_df.dropoff_latitude,
    test_df.dropoff_longitude,
)
test_df["datetime_object"] = pd.to_datetime(test_df["pickup_datetime"], errors="coerce")
test_df["hour"] = test_df["datetime_object"].dt.hour
test_df["year"] = test_df["datetime_object"].dt.year
min_year = data["year"].min()
test_df["year_offset"] = test_df["year"] - min_year
test_df["distance_to_center"] = distance(
    nyc[1], nyc[0], test_df.dropoff_latitude, test_df.dropoff_longitude
)

add_travel_vector_features(test_df)


## === cell 11
val_X = scaler.transform(get_input_matrix(val_df))
test_X = scaler.transform(get_input_matrix(test_df))

val_y_log_pred = ridge.predict(val_X)
test_y_log_pred = ridge.predict(test_X)

val_y_log_pred = np.clip(val_y_log_pred, -5.0, 7.0)
test_y_log_pred = np.clip(test_y_log_pred, -5.0, 7.0)

val_y_pred = np.expm1(val_y_log_pred)
test_y_pred = np.expm1(test_y_log_pred)

val_y_pred = np.clip(val_y_pred, 0, None)
test_y_pred = np.clip(test_y_pred, 0, None)

rmse = np.sqrt(mean_squared_error(np.expm1(val_y_log), val_y_pred))
print("Validation RMSE:", rmse)


## === cell 12
submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": np.round(test_y_pred, 2)},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)
print("Submission file written. Files in current directory:")
print(os.listdir("."))
