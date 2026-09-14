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

5.51545

# 6. Current score

28.91751

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 786.85271) has done: 'I fix the pandas API breakages (tuple-based column selection in `groupby`, and `corr()` failing due to non-numeric columns) so the notebook runs on pandas 2.2. I also fix the datetime parsing (your format string doesn’t match the dataset) by using `pd.to_datetime`, which is both correct and much faster and ensures `hour`/`year` exist before splitting. Finally, I ensure the same feature engineering is applied consistently to train/val/test before building matrices, and that the script always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 28.87375) has done: 'Your RMSE is exploding because two features in your linear system (`fare_per_mile` and `inv_distance_miles`) create infinities/huge values when `distance_miles` is very small; those rows survive your `distance_miles > 0.0` filter and destabilize OLS, yielding extreme predictions. I make the smallest safe change: compute `fare_per_mile` and `inv_distance_miles` only after clipping `distance_miles` to a small epsilon, and tighten the filter to remove near-zero distances. I also clamp predictions to a reasonable non-negative range (still the same linear model; just prevents invalid negative fares) to reduce RMSE without changing the core approach. Finally, I keep your submission format identical and ensure it always writes `submission.csv`.'
- What this solution (achieved 28.91751) has done: 'Your current RMSE gap to the target is large, so the smallest legitimate improvement is to stabilize and de-bias the same OLS linear model without changing its core form. I (1) stop using the numerically unstable explicit matrix inverse (keep `lstsq` only), (2) add the same basic data cleaning commonly required for this competition (filter obvious out-of-bounds lat/lon and extreme fares) while keeping your existing feature set and training approach, and (3) ensure the exact same feature engineering is applied to train/val/test consistently before matrix building. These changes reduce the influence of bad rows and ill-conditioned fits, which should move RMSE substantially downward toward your target while preserving your model/loop/feature logic and producing the same submission format.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

INPUT_DIR = "../input"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "/kaggle/input"

print(os.listdir(INPUT_DIR))



## === cell 1
data = pd.read_csv(f"{INPUT_DIR}/train.csv", nrows=15_000_000)




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
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


data["distance_miles"] = distance(
    data.pickup_latitude,
    data.pickup_longitude,
    data.dropoff_latitude,
    data.dropoff_longitude,
)

_ = data.groupby("passenger_count")[["distance_miles", "fare_amount"]].mean()

print(
    "Average $USD/Mile : {:0.2f}".format(
        data.fare_amount.sum() / data.distance_miles.sum()
    )
)

_eps = 1e-3  # miles (~1.6 meters)
data["distance_miles_clipped"] = data["distance_miles"].clip(lower=_eps)
data["fare_per_mile"] = data.fare_amount / data["distance_miles_clipped"]
data["inv_distance_miles"] = 1.0 / data["distance_miles_clipped"]

data["hour"] = data["datetime_object"].dt.hour
data["year"] = data["datetime_object"].dt.year



## === cell 5
import seaborn as sns
import matplotlib.pyplot as plt

numeric_data = data.select_dtypes(include=[np.number])
corrmat = numeric_data.corr()
f, ax = plt.subplots(figsize=(12, 9))

k = 14  # number of variables for heatmap
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
data = data[(data.distance_miles > 1e-3)]

data = data[(data.fare_amount <= 250.0)]
data = data[(data.pickup_longitude >= -75) & (data.pickup_longitude <= -72)]
data = data[(data.dropoff_longitude >= -75) & (data.dropoff_longitude <= -72)]
data = data[(data.pickup_latitude >= 40) & (data.pickup_latitude <= 42)]
data = data[(data.dropoff_latitude >= 40) & (data.dropoff_latitude <= 42)]

print("New size: %d" % len(data))



## === cell 8
print(data["datetime_object"].iloc[0])
plot = data.iloc[:1000].plot.scatter("hour", "fare_amount")



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
        (df.distance_miles, df.passenger_count, df.hour, df.year, np.ones(len(df)))
    )


train_X = get_input_matrix(train_df)

print(train_X.shape)
print(train_y.shape)



## === cell 11
(w, _, _, _) = np.linalg.lstsq(train_X, train_y, rcond=None)
print(w)



## === cell 12
w_OLS = w
print(w_OLS)



## === cell 13
test_df = pd.read_csv(f"{INPUT_DIR}/test.csv")

test_df["distance_miles"] = distance(
    test_df.pickup_latitude,
    test_df.pickup_longitude,
    test_df.dropoff_latitude,
    test_df.dropoff_longitude,
)
test_df["datetime_object"] = pd.to_datetime(
    test_df["pickup_datetime"], errors="coerce", utc=False
)
test_df["hour"] = test_df["datetime_object"].dt.hour
test_df["year"] = test_df["datetime_object"].dt.year

val_df = val_df.copy()
val_df["distance_miles"] = distance(
    val_df.pickup_latitude,
    val_df.pickup_longitude,
    val_df.dropoff_latitude,
    val_df.dropoff_longitude,
)
val_df["datetime_object"] = pd.to_datetime(
    val_df["pickup_datetime"], errors="coerce", utc=False
)
val_df["hour"] = val_df["datetime_object"].dt.hour
val_df["year"] = val_df["datetime_object"].dt.year

test_df.dtypes



## === cell 14
add_travel_vector_features(test_df)
test_X = get_input_matrix(test_df)

add_travel_vector_features(val_df)
val_X = get_input_matrix(val_df)

test_y_predictions = np.matmul(test_X, w)
val_y_predictions = np.matmul(val_X, w)

test_y_predictions = np.clip(test_y_predictions, 0.0, 500.0).round(decimals=2)
val_y_predictions = np.clip(val_y_predictions, 0.0, 500.0).round(decimals=2)

from sklearn.metrics import mean_squared_error

print(np.sqrt(mean_squared_error(val_y, val_y_predictions)))

submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": test_y_predictions},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)

print(os.listdir("."))
