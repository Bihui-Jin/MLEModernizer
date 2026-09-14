# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

5.69253

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 113.72277) has done: 'I remove the unsupported `normalize` argument from the LinearRegression constructor and adjust the submission creation so the CSV has the required `key` column (instead of using it as an index). These fixes resolve the runtime errors and ensure a proper submission file is written.'

# 9. Code solution

## === cell 0
train_data = pd.read_csv("../input/train.csv", nrows=10_000_000)
train_data.dtypes




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2702207597.py in <cell line: 0>()
----> 1 train_data = pd.read_csv("../input/train.csv", nrows=10_000_000)
      2 train_data.dtypes
      3 
      4 

NameError: name 'pd' is not defined

## === cell 1
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_data)


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2431142069.py in <cell line: 0>()
      4 
      5 
----> 6 add_travel_vector_features(train_data)

NameError: name 'train_data' is not defined

## === cell 2
print(train_data.isnull().sum())


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2315490603.py in <cell line: 0>()
----> 1 print(train_data.isnull().sum())

NameError: name 'train_data' is not defined

## === cell 3
train_data = train_data.dropna(how="any", axis="rows")


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3650645407.py in <cell line: 0>()
----> 1 train_data = train_data.dropna(how="any", axis="rows")

NameError: name 'train_data' is not defined

## === cell 4
plot = train_data.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1582271480.py in <cell line: 0>()
----> 1 plot = train_data.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")

NameError: name 'train_data' is not defined

## === cell 5
train_data = train_data[
    (train_data.abs_diff_longitude < 5.0) & (train_data.abs_diff_latitude < 5.0)
]


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3597019475.py in <cell line: 0>()
----> 1 train_data = train_data[
      2     (train_data.abs_diff_longitude < 5.0) & (train_data.abs_diff_latitude < 5.0)
      3 ]

NameError: name 'train_data' is not defined

## === cell 6
test_data = pd.read_csv("../input/test.csv")
test_data.dtypes


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2392152822.py in <cell line: 0>()
----> 1 test_data = pd.read_csv("../input/test.csv")
      2 test_data.dtypes

NameError: name 'pd' is not defined

## === cell 7
add_travel_vector_features(test_data)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3962090048.py in <cell line: 0>()
----> 1 add_travel_vector_features(test_data)

NameError: name 'test_data' is not defined

## === cell 8
test_data.dtypes


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/59176276.py in <cell line: 0>()
----> 1 test_data.dtypes

NameError: name 'test_data' is not defined

## === cell 9
train_data.dtypes


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/873271517.py in <cell line: 0>()
----> 1 train_data.dtypes

NameError: name 'train_data' is not defined

## === cell 10
train_data.head()


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1664430853.py in <cell line: 0>()
----> 1 train_data.head()

NameError: name 'train_data' is not defined

## === cell 11
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_data["pickup_time"] = ls1

ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_data["pickup_time"] = ls1


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4277389599.py in <cell line: 0>()
----> 1 ls1 = list(train_data["pickup_datetime"])
      2 for i in range(len(ls1)):
      3     ls1[i] = ls1[i][11:-7:]
      4 train_data["pickup_time"] = ls1
      5 

NameError: name 'train_data' is not defined

## === cell 12
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = pd.Timestamp(ls1[i][:-4:]).weekday()
train_data["weekday"] = ls1

ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = pd.Timestamp(ls1[i][:-4:]).weekday()
test_data["weekday"] = ls1


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1339983278.py in <cell line: 0>()
----> 1 ls1 = list(train_data["pickup_datetime"])
      2 for i in range(len(ls1)):
      3     ls1[i] = pd.Timestamp(ls1[i][:-4:]).weekday()
      4 train_data["weekday"] = ls1
      5 

NameError: name 'train_data' is not defined

## === cell 13
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1969310352.py in <cell line: 0>()
----> 1 train_data.drop("pickup_datetime", inplace=True, axis=1)
      2 test_data.drop("pickup_datetime", inplace=True, axis=1)

NameError: name 'train_data' is not defined

## === cell 14
train_data.head(10)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/745345791.py in <cell line: 0>()
----> 1 train_data.head(10)

NameError: name 'train_data' is not defined

## === cell 15
train_data["weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thrusday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)
test_data["weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thrusday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3012559924.py in <cell line: 0>()
----> 1 train_data["weekday"].replace(
      2     to_replace=[i for i in range(0, 7)],
      3     value=[
      4         "Monday",
      5         "Tuesday",

NameError: name 'train_data' is not defined

## === cell 16
train_one_shot = pd.get_dummies(train_data["weekday"])
test_one_shot = pd.get_dummies(test_data["weekday"])
test_data = pd.concat([test_data, test_one_shot], axis=1)
train_data = pd.concat([train_data, train_one_shot], axis=1)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/706845963.py in <cell line: 0>()
----> 1 train_one_shot = pd.get_dummies(train_data["weekday"])
      2 test_one_shot = pd.get_dummies(test_data["weekday"])
      3 test_data = pd.concat([test_data, test_one_shot], axis=1)
      4 train_data = pd.concat([train_data, train_one_shot], axis=1)

NameError: name 'pd' is not defined

## === cell 17
train_data.drop("weekday", inplace=True, axis=1)
test_data.drop("weekday", inplace=True, axis=1)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3932882589.py in <cell line: 0>()
----> 1 train_data.drop("weekday", inplace=True, axis=1)
      2 test_data.drop("weekday", inplace=True, axis=1)

NameError: name 'train_data' is not defined

## === cell 18
ls1 = list(train_data["pickup_time"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
train_data["pickup_time"] = ls1

ls1 = list(test_data["pickup_time"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
test_data["pickup_time"] = ls1


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1919855524.py in <cell line: 0>()
----> 1 ls1 = list(train_data["pickup_time"])
      2 for i in range(len(ls1)):
      3     z = ls1[i].split(":")
      4     ls1[i] = int(z[0]) * 100 + int(z[1])
      5 train_data["pickup_time"] = ls1

NameError: name 'train_data' is not defined

## === cell 19
R = 6372.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_data["Distance"] = np.asarray(distance) * 0.621

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data["Distance"] = np.asarray(distance) * 0.621


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3552551409.py in <cell line: 0>()
      1 R = 6372.0
----> 2 lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
      3 lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
      4 lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
      5 lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

NameError: name 'np' is not defined

## === cell 20
R = 6372.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

lat3 = np.zeros(len(train_data)) + np.radians(40.641)
lon3 = np.zeros(len(train_data)) + np.radians(-73.778)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
dlon_dropoff = lon3 - lon2
dlat_dropoff = lat3 - lat2
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
train_data["pickup_distance_a"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_data["dropoff_distance_a"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

lat3 = np.zeros(len(test_data)) + np.radians(40.641)
lon3 = np.zeros(len(test_data)) + np.radians(-73.778)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
dlon_dropoff = lon3 - lon2
dlat_dropoff = lat3 - lat2
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
test_data["pickup_distance_a"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_data["dropoff_distance_a"] = np.asarray(distance2) * 0.621


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/503695659.py in <cell line: 0>()
      1 R = 6372.0
----> 2 lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
      3 lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
      4 lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
      5 lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

NameError: name 'np' is not defined

## === cell 21
train_data["Distance"] = np.round(train_data["Distance"], 2)
train_data["pickup_distance_a"] = np.round(train_data["pickup_distance_a"], 2)
train_data["dropoff_distance_a"] = np.round(train_data["dropoff_distance_a"], 2)

test_data["Distance"] = np.round(test_data["Distance"], 2)
test_data["pickup_distance_a"] = np.round(test_data["pickup_distance_a"], 2)
test_data["dropoff_distance_a"] = np.round(test_data["dropoff_distance_a"], 2)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/218600071.py in <cell line: 0>()
----> 1 train_data["Distance"] = np.round(train_data["Distance"], 2)
      2 train_data["pickup_distance_a"] = np.round(train_data["pickup_distance_a"], 2)
      3 train_data["dropoff_distance_a"] = np.round(train_data["dropoff_distance_a"], 2)
      4 
      5 test_data["Distance"] = np.round(test_data["Distance"], 2)

NameError: name 'np' is not defined

## === cell 22
train_data.drop(
    ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"],
    inplace=True,
    axis=1,
)
test_data.drop(
    ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"],
    inplace=True,
    axis=1,
)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3658097374.py in <cell line: 0>()
----> 1 train_data.drop(
      2     ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"],
      3     inplace=True,
      4     axis=1,
      5 )

NameError: name 'train_data' is not defined

## === cell 23
train_data["abs_diff_longitude"] = train_data["abs_diff_longitude"] - np.mean(
    train_data["abs_diff_longitude"]
)
train_data["abs_diff_longitude"] = train_data["abs_diff_longitude"] / np.var(
    train_data["abs_diff_longitude"]
)

train_data["abs_diff_latitude"] = train_data["abs_diff_latitude"] - np.mean(
    train_data["abs_diff_latitude"]
)
train_data["abs_diff_latitude"] = train_data["abs_diff_latitude"] / np.var(
    train_data["abs_diff_latitude"]
)

test_data["abs_diff_longitude"] = test_data["abs_diff_longitude"] - np.mean(
    test_data["abs_diff_longitude"]
)
test_data["abs_diff_longitude"] = test_data["abs_diff_longitude"] / np.var(
    test_data["abs_diff_longitude"]
)

test_data["abs_diff_latitude"] = test_data["abs_diff_latitude"] - np.mean(
    test_data["abs_diff_latitude"]
)
test_data["abs_diff_latitude"] = test_data["abs_diff_latitude"] / np.var(
    test_data["abs_diff_latitude"]
)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/138180799.py in <cell line: 0>()
----> 1 train_data["abs_diff_longitude"] = train_data["abs_diff_longitude"] - np.mean(
      2     train_data["abs_diff_longitude"]
      3 )
      4 train_data["abs_diff_longitude"] = train_data["abs_diff_longitude"] / np.var(
      5     train_data["abs_diff_longitude"]

NameError: name 'train_data' is not defined

## === cell 24
from sklearn.model_selection import train_test_split
import numpy as np

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]
y_log = np.log1p(y)

Xtrain, Xvalid, ytrain_log, yvalid_log = train_test_split(
    X, y_log, test_size=0.01, random_state=80
)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1073908840.py in <cell line: 0>()
      2 import numpy as np
      3 
----> 4 X = train_data.drop(["key", "fare_amount"], axis=1)
      5 y = train_data["fare_amount"]
      6 # Log‑transform the target to reduce skewness

NameError: name 'train_data' is not defined

## === cell 25
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42,
)
rf.fit(Xtrain, ytrain_log)

valid_pred_log = rf.predict(Xvalid)
valid_pred = np.expm1(valid_pred_log)

yvalid = np.expm1(yvalid_log)

rmse = np.sqrt(mean_squared_error(yvalid, valid_pred))
print(f"Validation RMSE: {rmse:.5f}")


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2358491665.py in <cell line: 0>()
     12     random_state=42,
     13 )
---> 14 rf.fit(Xtrain, ytrain_log)
     15 
     16 # Predict on validation and convert back from log scale

NameError: name 'Xtrain' is not defined

## === cell 26
test_pred_log = rf.predict(test_data.drop("key", axis=1))
test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, a_min=0, a_max=None)
test_pred = np.round(test_pred, 2)

submission = pd.DataFrame({"key": test_data["key"], "fare_amount": test_pred})
submission.to_csv("Submission.csv", index=False)
print("Submission file saved as Submission.csv")

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/185401182.py in <cell line: 0>()
      1 # Predict on the test set, convert back from log scale and ensure non‑negative fares
----> 2 test_pred_log = rf.predict(test_data.drop("key", axis=1))
      3 test_pred = np.expm1(test_pred_log)
      4 test_pred = np.clip(test_pred, a_min=0, a_max=None)
      5 test_pred = np.round(test_pred, 2)

NameError: name 'test_data' is not defined
