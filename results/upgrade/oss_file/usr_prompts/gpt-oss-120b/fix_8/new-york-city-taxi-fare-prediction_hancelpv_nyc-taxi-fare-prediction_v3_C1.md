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

4.31113

# 6. Current score

5.35081

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.30683) has done: 'I fixed the time‑feature creation (using `day_name()` instead of the removed `weekday_name`), dropped the raw `pickup_datetime` column that caused non‑numeric data to be fed to the model, and retained the original workflow otherwise. These changes eliminate the attribute errors, allow the RandomForest to train on numerical data, and produce a proper `submission.csv` with the required columns.'
- What this solution (achieved 5.30434) has done: 'I add a more informative distance feature (haversine distance) to the geographic feature set and include it in the model inputs, then increase the RandomForest estimator count from 200 to 500 to capture more patterns. These targeted changes keep the original workflow intact while providing richer distance information and a slightly stronger learner, which should lower the RMSE toward the target value.'
- What this solution (achieved 5.35081) has done: 'Implemented robust handling for NaN values in the target variable and its log‑transformed form, ensuring the model trains without errors. Added safety checks and median‑imputation for any remaining non‑finite entries, which resolves the earlier `ValueError` and subsequent attribute errors. The workflow now completes end‑to‑end and writes a valid `submission.csv` ready for Kaggle upload.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))




## === cell 1
train = pd.read_csv("../input/train.csv", nrows=10000)
test = pd.read_csv("../input/test.csv")
samp = pd.read_csv("../input/sample_submission.csv")




## === cell 2
train = train.dropna(how="any", axis="rows")




## === cell 3
test.shape




## === cell 4
all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(["fare_amount"], axis=1, inplace=True)

y = pd.to_numeric(train["fare_amount"], errors="coerce")
median_fare = y.median()
y = y.fillna(median_fare).values

y = np.where(np.isfinite(y), y, median_fare)

n_train = len(train)
n_test = len(test)
test_id = test.key




## === cell 5
def week_num(day):
    """Return the week of the month as a categorical string."""
    if day <= 7:
        return "first"
    if day <= 14:
        return "second"
    if day <= 21:
        return "third"
    if day <= 28:
        return "fourth"
    return "fifth"




## === cell 6
def add_time_features(data):
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])

    data["hour"] = data["pickup_datetime"].dt.hour.astype(str)
    data["day_of_week"] = data["pickup_datetime"].dt.day_name()
    data["day_of_month"] = data["pickup_datetime"].dt.day
    data["week_of_month"] = data["day_of_month"].map(week_num)
    data["month"] = data["pickup_datetime"].dt.month.astype(str)
    data["year"] = data["pickup_datetime"].dt.year.astype(str)

    data.drop(["day_of_month", "pickup_datetime"], axis=1, inplace=True)

    return data




## === cell 7
def add_geo_features(data):
    data["abs_diff_longitude"] = (
        data["dropoff_longitude"] - data["pickup_longitude"]
    ).abs()
    data["abs_diff_latitude"] = (
        data["dropoff_latitude"] - data["pickup_latitude"]
    ).abs()

    data["manhattan_distance"] = data["abs_diff_longitude"] + data["abs_diff_latitude"]

    data["squared_long"] = np.power(data["abs_diff_longitude"], 2)
    data["squared_lat"] = np.power(data["abs_diff_latitude"], 2)

    data["euclid_disance"] = np.sqrt(data["squared_long"] + data["squared_lat"])

    lat1 = np.radians(data["pickup_latitude"])
    lat2 = np.radians(data["dropoff_latitude"])
    lon1 = np.radians(data["pickup_longitude"])
    lon2 = np.radians(data["dropoff_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0  # Earth radius in kilometers
    data["haversine_distance"] = R * c

    data["distance_per_passenger"] = (
        data["haversine_distance"] / data["passenger_count"]
    )
    data["distance_per_passenger"] = data["distance_per_passenger"].replace(
        [np.inf, -np.inf], np.nan
    )
    data["distance_per_passenger"] = data["distance_per_passenger"].fillna(0)

    return data




## === cell 8
all_data = add_time_features(all_data)




## === cell 9
all_data = add_geo_features(all_data)




## === cell 10
features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "euclid_disance",
    "manhattan_distance",
    "haversine_distance",
    "distance_per_passenger",
]

all_data = all_data[features]

all_data = pd.get_dummies(all_data)

all_data = all_data.fillna(0)




## === cell 11
x = all_data[:n_train]
x_test = all_data[n_train:]




## === cell 12
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=800,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
)




## === cell 13
y_log = np.log1p(y)
median_log = np.median(y_log[np.isfinite(y_log)])
y_log = np.where(np.isfinite(y_log), y_log, median_log)

model.fit(x, y_log)




## === cell 14
feat_imp = pd.DataFrame(model.feature_importances_, index=x.columns, columns=["imp"])
feat_imp.sort_values("imp", ascending=False, inplace=True)




## === cell 15
pred_log = model.predict(x_test)
test_pred = np.expm1(pred_log)

test_pred = np.where(test_pred < 0, 0, test_pred)




## === cell 16
sub = pd.DataFrame({"key": test_id, "fare_amount": test_pred})
sub.to_csv("submission.csv", index=False)




## === cell 17
sub.head()
