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

5.46166

# 6. Current score

890.53157

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 887.53561) has done: 'Diagnosis: Cell 4 crashes because recent pandas versions disallow selecting multiple columns from a GroupBy object using a tuple-style indexer `['col1', 'col2']` written as `['col1', 'col2']` without wrapping properly; in the notebook export it is being interpreted as a tuple and triggers `ValueError: Cannot subset columns with a tuple... Use a list instead.` This is a compatibility issue with newer pandas, not with your data.  
Patch summary: Change the GroupBy column selection to use an explicit list of columns via double brackets `[['distance_miles','fare_amount']]`, which is the supported syntax across pandas versions. No other logic, features, or downstream variables are altered.  
Updated cells: Only cell 4 is modified.  
Compatibility notes for cell k+1: All variables created in cell 4 (`distance_miles`, `fare_per_mile`, `inv_distance_miles`, `hour`, `year`) remain identical, and `data` remains a DataFrame suitable for `data.corr()` in cell 5.  
Assumptions: The intent of the line is to compute the mean of both `distance_miles` and `fare_amount` grouped by `passenger_count`, and the result is not stored (display-only), so changing only the indexing syntax is sufficient.'
- What this solution (achieved 888.45278) has done: 'Diagnosis: The crash happens in cell 5 when calling `data.corr()`. With pandas 2.2, `DataFrame.corr()` tries to convert the whole frame to float unless you restrict it to numeric columns, and the dataset contains non-numeric columns like `pickup_datetime` (string), causing `ValueError: could not convert string to float`. The plotting logic is fine; only the correlation computation needs to be limited to numeric data.

Patch summary: In cell 5, compute correlations on numeric columns only by using `data.corr(numeric_only=True)`. This preserves the same heatmap logic and avoids attempting to coerce datetime strings to floats.

Updated cells: Only cell 5 is changed.

Compatibility notes for cell k+1: Cell 6 still uses `data` unchanged (no columns dropped from `data` itself), so downstream code remains compatible.

Assumptions: pandas is >= 2.0 (as installed) where `numeric_only` is supported on `DataFrame.corr()`.'
- What this solution (achieved 890.53157) has done: 'Your RMSE is exploding because `pickup_datetime` parsing is currently broken: the format string `'%Y-%m-%d %H:%M:%S %Z'` doesn’t match the dataset (which looks like `YYYY-MM-DD HH:MM:SS UTC` and sometimes includes fractional seconds), so the derived `hour/year` features become incorrect or the code behaves inconsistently. I minimally replace the list-comprehension `datetime.strptime(...)` with robust, vectorized `pd.to_datetime(..., utc=True, errors='coerce')` in both train and test/val, then compute `hour/year` via `.dt`, preserving the same feature definitions and the same linear least-squares model. This should move the score sharply downward (better) toward the 5.46 target without changing the model architecture or training approach. I also keep the submission format unchanged and still write `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



## === cell 1
data = pd.read_csv("../input/train.csv", nrows=15_000_000)




## === cell 2
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(data)



## === cell 3
data["datetime_object"] = pd.to_datetime(
    data["pickup_datetime"], utc=True, errors="coerce"
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

data.distance_miles.describe()

data.groupby("passenger_count")[["distance_miles", "fare_amount"]].mean()

print(
    "Average $USD/Mile : {:0.2f}".format(
        data.fare_amount.sum() / data.distance_miles.sum()
    )
)
data["fare_per_mile"] = data.fare_amount / data.distance_miles
data["inv_distance_miles"] = 1 / data.distance_miles
data["hour"] = data["datetime_object"].dt.hour
data["year"] = data["datetime_object"].dt.year



## === cell 5
import seaborn as sns
import matplotlib.pyplot as plt

corrmat = data.corr(numeric_only=True)

f, ax = plt.subplots(figsize=(12, 9))

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
print("New size: %d" % len(data))



## === cell 7
print("Old size: %d" % len(data))
data = data[(data.abs_diff_longitude < 3.0) & (data.abs_diff_latitude < 3.0)]
data = data[data.fare_amount >= 0]
data = data[data.passenger_count <= 9]
data = data[(data.distance_miles > 0.0) & (data.distance_miles > 0.05)]
nyc = (-74.0063889, 40.7141667)
data["distance_to_center"] = distance(
    nyc[1], nyc[0], data.dropoff_latitude, data.dropoff_longitude
)
data = data[data.distance_to_center < 15.0]
print("New size: %d" % len(data))



## === cell 8
plot = data.iloc[:1000].plot.scatter("year", "fare_amount")



## === cell 9
from sklearn.model_selection import train_test_split

y = data.fare_amount
X = data.drop("fare_amount", axis=1)
train_df, val_df, train_y, val_y = train_test_split(X, y, test_size=0.2)
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
w_OLS = np.matmul(
    np.matmul(np.linalg.inv(np.matmul(train_X.T, train_X)), train_X.T), train_y
)
print(w_OLS)



## === cell 13
test_df = pd.read_csv("../input/test.csv")
test_df["distance_miles"] = distance(
    test_df.pickup_latitude,
    test_df.pickup_longitude,
    test_df.dropoff_latitude,
    test_df.dropoff_longitude,
)

test_df["datetime_object"] = pd.to_datetime(
    test_df["pickup_datetime"], utc=True, errors="coerce"
)
test_df["hour"] = test_df["datetime_object"].dt.hour
test_df["year"] = test_df["datetime_object"].dt.year

val_df["distance_miles"] = distance(
    val_df.pickup_latitude,
    val_df.pickup_longitude,
    val_df.dropoff_latitude,
    val_df.dropoff_longitude,
)
val_df["datetime_object"] = pd.to_datetime(
    val_df["pickup_datetime"], utc=True, errors="coerce"
)
val_df["hour"] = val_df["datetime_object"].dt.hour
val_df["year"] = val_df["datetime_object"].dt.year
test_df.dtypes



## === cell 14
add_travel_vector_features(test_df)
test_X = get_input_matrix(test_df)
add_travel_vector_features(val_df)
val_X = get_input_matrix(val_df)
test_y_predictions = np.matmul(test_X, w).round(decimals=2)
val_y_predictions = np.matmul(val_X, w).round(decimals=2)

from sklearn.metrics import mean_squared_error

print(np.sqrt(mean_squared_error(val_y, val_y_predictions)))
submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": test_y_predictions},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)

print(os.listdir("."))
