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

5.46206

# 6. Current score

886.12289

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 884.45726) has done: 'I fixed the datetime parsing, corrected column selections for group‑by and correlation, added safe numeric‑only handling, and ensured the feature matrix includes the required columns so the linear regression runs without errors and writes a proper `submission.csv`.'
- What this solution (achieved 886.12289) has done: 'The current linear model includes the raw year value, which creates very large coefficients and blows up the predictions, giving an RMSE far above the target. By simplifying the feature set to only the distance and passenger count (and a bias term) we keep the model expressive enough for the fare while avoiding the unstable “year” feature. After training we also clip negative predictions to 0 so the submission stays realistic. These small, targeted edits keep the original workflow intact but move the validation RMSE much closer to the desired 5.46 range.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd

print(os.listdir("../input"))



## === cell 1
data = pd.read_csv("../input/train.csv", nrows=15_000_000)




## === cell 2
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(data)



## === cell 3
data["datetime_object"] = pd.to_datetime(data["pickup_datetime"], errors="coerce")
data["hour"] = data["datetime_object"].dt.hour
data["year"] = data["datetime_object"].dt.year


def haversine_distance(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


data["distance_miles"] = haversine_distance(
    data["pickup_latitude"],
    data["pickup_longitude"],
    data["dropoff_latitude"],
    data["dropoff_longitude"],
)

group_stats = data.groupby("passenger_count")[["distance_miles", "fare_amount"]].mean()
print(group_stats.head())

print(
    "Average $USD/Mile : {:0.2f}".format(
        data["fare_amount"].sum() / data["distance_miles"].sum()
    )
)
data["fare_per_mile"] = data["fare_amount"] / data["distance_miles"]
data["inv_distance_miles"] = 1.0 / data["distance_miles"]



## === cell 4
numeric_cols = data.select_dtypes(include=[np.number]).columns
corrmat = data[numeric_cols].corr()
k = 14
cols = corrmat.nlargest(k, "fare_amount")["fare_amount"].index
cm = np.corrcoef(data[cols].values.T)

import seaborn as sns, matplotlib.pyplot as plt

sns.set(font_scale=1.25)
hm = sns.heatmap(
    cm,
    cbar=True,
    annot=True,
    square=True,
    fmt=".2f",
    annot_kws={"size": 10},
    yticklabels=cols,
    xticklabels=cols,
)
plt.show()



## === cell 5
print("Old size: %d" % len(data))
data = data.dropna(how="any")
print("New size after dropna: %d" % len(data))



## === cell 6
print("Old size: %d" % len(data))
data = data[(data["abs_diff_longitude"] < 3.0) & (data["abs_diff_latitude"] < 3.0)]
data = data[data.fare_amount >= 0]
data = data[data.passenger_count <= 9]
data = data[data.distance_miles > 0.0]

nyc_center = (-74.0063889, 40.7141667)  # (lon, lat)
data["distance_to_center"] = haversine_distance(
    nyc_center[1], nyc_center[0], data["dropoff_latitude"], data["dropoff_longitude"]
)
data = data[data.distance_to_center < 15.0]
print("New size after filtering: %d" % len(data))



## === cell 7
data.iloc[:1000].plot.scatter("year", "fare_amount")
plt.show()



## === cell 8
from sklearn.model_selection import train_test_split

y = data["fare_amount"]
X = data.drop("fare_amount", axis=1)
train_df, val_df, train_y, val_y = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(train_df.dtypes.head())




## === cell 9
def get_input_matrix(df):
    """
    Build a simple design matrix with only the most predictive numeric features:
    distance_miles, passenger_count, and a bias term.
    """
    for c in ["distance_miles", "passenger_count"]:
        if c not in df.columns:
            df[c] = 0.0
    bias = np.ones(len(df))
    return np.column_stack((df["distance_miles"], df["passenger_count"], bias))


train_X = get_input_matrix(train_df)
print("train_X shape:", train_X.shape, "train_y shape:", train_y.shape)



## === cell 10
w, _, _, _ = np.linalg.lstsq(train_X, train_y.values, rcond=None)
print("Learned weights (distance, passenger_count, bias):", w)



## === cell 11
test_df = pd.read_csv("../input/test.csv")
test_df["distance_miles"] = haversine_distance(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
)
test_df["datetime_object"] = pd.to_datetime(test_df["pickup_datetime"], errors="coerce")
test_df["hour"] = test_df["datetime_object"].dt.hour
test_df["year"] = test_df["datetime_object"].dt.year

val_df["distance_miles"] = haversine_distance(
    val_df["pickup_latitude"],
    val_df["pickup_longitude"],
    val_df["dropoff_latitude"],
    val_df["dropoff_longitude"],
)
val_df["datetime_object"] = pd.to_datetime(val_df["pickup_datetime"], errors="coerce")
val_df["hour"] = val_df["datetime_object"].dt.hour
val_df["year"] = val_df["datetime_object"].dt.year

print(test_df.dtypes.head())



## === cell 12
add_travel_vector_features(test_df)
add_travel_vector_features(val_df)

test_X = get_input_matrix(test_df)
val_X = get_input_matrix(val_df)

test_y_predictions = np.matmul(test_X, w)
val_y_predictions = np.matmul(val_X, w)

test_y_predictions = np.clip(test_y_predictions, 0, None).round(2)
val_y_predictions = np.clip(val_y_predictions, 0, None).round(2)

from sklearn.metrics import mean_squared_error

val_rmse = np.sqrt(mean_squared_error(val_y, val_y_predictions))
print("Validation RMSE:", val_rmse)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_y_predictions})
submission.to_csv("submission.csv", index=False)
print("Created files:", os.listdir("."))
