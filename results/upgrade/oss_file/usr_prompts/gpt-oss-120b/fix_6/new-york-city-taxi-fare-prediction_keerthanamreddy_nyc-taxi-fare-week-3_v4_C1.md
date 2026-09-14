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

5.57556

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 761.7631) has done: 'I remove the obsolete `normalize` argument from the LinearRegression constructor, add an RMSE calculation (the competition metric) to monitor performance, and keep the subsequent prediction and submission steps unchanged so they run correctly and produce a valid Submission.csv file.'
- What this solution (achieved 761.58172) has done: 'The update adds feature scaling and a ridge‑regression model (still a linear model) to improve numerical stability and reduce the validation RMSE, bringing the score closer to the target while keeping the overall workflow unchanged.'
- What this solution (achieved 740.47979) has done: 'I add a modest outlier‑filter on `fare_amount` before the train/validation split, align the test feature columns with the training columns (filling missing dummy columns with 0), and clip any negative predictions to 0. These small data‑cleaning tweaks keep the same Ridge model while moving the validation RMSE much closer to the target 5.58.'
- What this solution (achieved 727.62593) has done: 'I add two simple distance‑based features (Manhattan distance and distance × passenger count) before dropping the raw coordinate columns, and I use a slightly weaker ridge regularization (α = 0.1). These lightweight changes keep the same linear‑model pipeline while giving the model more predictive signal, which should lower the validation RMSE and move it nearer to the target.'

# 9. Code solution

## === cell 0
train_df = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=10_000_000
)
train_df.dtypes


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2890458080.py in <cell line: 0>()
----> 1 train_df = pd.read_csv(
      2     "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=10_000_000
      3 )
      4 train_df.dtypes

NameError: name 'pd' is not defined

## === cell 1
test_df = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
test_df.dtypes




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/227797653.py in <cell line: 0>()
----> 1 test_df = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
      2 test_df.dtypes
      3 
      4 

NameError: name 'pd' is not defined

## === cell 2
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3090621481.py in <cell line: 0>()
      4 
      5 
----> 6 add_travel_vector_features(train_df)
      7 add_travel_vector_features(test_df)

NameError: name 'train_df' is not defined

## === cell 3
test_df.head()


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2239380292.py in <cell line: 0>()
----> 1 test_df.head()

NameError: name 'test_df' is not defined

## === cell 4
print(train_df.isnull().sum())


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3174598079.py in <cell line: 0>()
----> 1 print(train_df.isnull().sum())

NameError: name 'train_df' is not defined

## === cell 5
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4249765067.py in <cell line: 0>()
----> 1 print("Old size: %d" % len(train_df))
      2 train_df = train_df.dropna(how="any", axis="rows")
      3 print("New size: %d" % len(train_df))

NameError: name 'train_df' is not defined

## === cell 6
plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/311890816.py in <cell line: 0>()
----> 1 plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")

NameError: name 'train_df' is not defined

## === cell 7
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/889800582.py in <cell line: 0>()
----> 1 print("Old size: %d" % len(train_df))
      2 train_df = train_df[
      3     (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
      4 ]
      5 print("New size: %d" % len(train_df))

NameError: name 'train_df' is not defined

## === cell 8
def get_input_matrix(df):
    return np.column_stack(
        (df.abs_diff_longitude, df.abs_diff_latitude, np.ones(len(df)))
    )


train_X = get_input_matrix(train_df)
train_y = np.array(train_df["fare_amount"])

print(train_X.shape)
print(train_y.shape)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3521575548.py in <cell line: 0>()
      5 
      6 
----> 7 train_X = get_input_matrix(train_df)
      8 train_y = np.array(train_df["fare_amount"])
      9 

NameError: name 'train_df' is not defined

## === cell 9
train_df.head()


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2663572906.py in <cell line: 0>()
----> 1 train_df.head()

NameError: name 'train_df' is not defined

## === cell 10
ls1 = list(train_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_df["pickuptime"] = ls1


ls1 = list(test_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_df["pickuptime"] = ls1


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/951841258.py in <cell line: 0>()
----> 1 ls1 = list(train_df["pickup_datetime"])
      2 for i in range(len(ls1)):
      3     ls1[i] = ls1[i][11:-7:]
      4 train_df["pickuptime"] = ls1
      5 

NameError: name 'train_df' is not defined

## === cell 11
train_df.head()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2663572906.py in <cell line: 0>()
----> 1 train_df.head()

NameError: name 'train_df' is not defined

## === cell 12
ls1 = list(train_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
train_df["Weekday"] = ls1


ls1 = list(test_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
test_df["Weekday"] = ls1


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3485245489.py in <cell line: 0>()
----> 1 ls1 = list(train_df["pickup_datetime"])
      2 for i in range(len(ls1)):
      3     ls1[i] = ls1[i][:-4:]
      4     ls1[i] = pd.Timestamp(ls1[i])
      5     ls1[i] = ls1[i].weekday()

NameError: name 'train_df' is not defined

## === cell 13
train_df.head()


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2663572906.py in <cell line: 0>()
----> 1 train_df.head()

NameError: name 'train_df' is not defined

## === cell 14
test_df.head()


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2239380292.py in <cell line: 0>()
----> 1 test_df.head()

NameError: name 'test_df' is not defined

## === cell 15
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3439493190.py in <cell line: 0>()
----> 1 train_df.drop("pickup_datetime", inplace=True, axis=1)
      2 test_df.drop("pickup_datetime", inplace=True, axis=1)

NameError: name 'train_df' is not defined

## === cell 16
train_df["Weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)
test_df["Weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/667327334.py in <cell line: 0>()
----> 1 train_df["Weekday"].replace(
      2     to_replace=[i for i in range(0, 7)],
      3     value=[
      4         "Monday",
      5         "Tuesday",

NameError: name 'train_df' is not defined

## === cell 17
train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_df["Weekday"])
train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1158577080.py in <cell line: 0>()
----> 1 train_one_hot = pd.get_dummies(train_df["Weekday"])
      2 test_one_hot = pd.get_dummies(test_df["Weekday"])
      3 train_df = pd.concat([train_df, train_one_hot], axis=1)
      4 test_df = pd.concat([test_df, test_one_hot], axis=1)

NameError: name 'pd' is not defined

## === cell 18
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1056453517.py in <cell line: 0>()
----> 1 train_df.drop("Weekday", axis=1, inplace=True)
      2 test_df.drop("Weekday", axis=1, inplace=True)

NameError: name 'train_df' is not defined

## === cell 19
ls1 = list(train_df["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
train_df["pickuptime"] = ls1


ls1 = list(test_df["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
test_df["pickuptime"] = ls1


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1452285496.py in <cell line: 0>()
----> 1 ls1 = list(train_df["pickuptime"])
      2 for i in range(len(ls1)):
      3     z = ls1[i].split(":")
      4     ls1[i] = int(z[0]) * 100 + int(z[1])
      5 train_df["pickuptime"] = ls1

NameError: name 'train_df' is not defined

## === cell 20
train_df.head()


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2663572906.py in <cell line: 0>()
----> 1 train_df.head()

NameError: name 'train_df' is not defined

## === cell 21
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


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/123003225.py in <cell line: 0>()
      1 R = 6373.0
----> 2 lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
      3 lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
      4 lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
      5 lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

NameError: name 'np' is not defined

## === cell 22
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


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/6987479.py in <cell line: 0>()
      1 R = 6373.0
----> 2 lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
      3 lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
      4 lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
      5 lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

NameError: name 'np' is not defined

## === cell 23
train_df["Distance"] = np.round(train_df["Distance"], 2)
train_df["Pickup_Distance_airport"] = np.round(train_df["Pickup_Distance_airport"], 2)
train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)
test_df["Distance"] = np.round(test_df["Distance"], 2)
test_df["Pickup_Distance_airport"] = np.round(test_df["Pickup_Distance_airport"], 2)
test_df["Dropoff_Distance_airport"] = np.round(test_df["Dropoff_Distance_airport"], 2)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2115963763.py in <cell line: 0>()
----> 1 train_df["Distance"] = np.round(train_df["Distance"], 2)
      2 train_df["Pickup_Distance_airport"] = np.round(train_df["Pickup_Distance_airport"], 2)
      3 train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)
      4 test_df["Distance"] = np.round(test_df["Distance"], 2)
      5 test_df["Pickup_Distance_airport"] = np.round(test_df["Pickup_Distance_airport"], 2)

NameError: name 'np' is not defined

## === cell 24
train_df["Manhattan"] = train_df["abs_diff_longitude"] + train_df["abs_diff_latitude"]
train_df["Dist_pass"] = train_df["Distance"] * train_df["passenger_count"]
test_df["Manhattan"] = test_df["abs_diff_longitude"] + test_df["abs_diff_latitude"]
test_df["Dist_pass"] = test_df["Distance"] * test_df["passenger_count"]

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


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3808210772.py in <cell line: 0>()
----> 1 train_df["Manhattan"] = train_df["abs_diff_longitude"] + train_df["abs_diff_latitude"]
      2 train_df["Dist_pass"] = train_df["Distance"] * train_df["passenger_count"]
      3 test_df["Manhattan"] = test_df["abs_diff_longitude"] + test_df["abs_diff_latitude"]
      4 test_df["Dist_pass"] = test_df["Distance"] * test_df["passenger_count"]
      5 

NameError: name 'train_df' is not defined

## === cell 25
train_df.head()


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2663572906.py in <cell line: 0>()
----> 1 train_df.head()

NameError: name 'train_df' is not defined

## === cell 26
test_df.head()


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2239380292.py in <cell line: 0>()
----> 1 test_df.head()

NameError: name 'test_df' is not defined

## === cell 27
lower_q = train_df["fare_amount"].quantile(0.001)
upper_q = train_df["fare_amount"].quantile(0.999)
train_df = train_df[
    (train_df["fare_amount"] >= lower_q) & (train_df["fare_amount"] <= upper_q)
]

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=42
)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3952508983.py in <cell line: 0>()
----> 1 lower_q = train_df["fare_amount"].quantile(0.001)
      2 upper_q = train_df["fare_amount"].quantile(0.999)
      3 train_df = train_df[
      4     (train_df["fare_amount"] >= lower_q) & (train_df["fare_amount"] <= upper_q)
      5 ]

NameError: name 'train_df' is not defined

## === cell 28
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_log = np.log1p(y_train)
ridge = Ridge(alpha=0.1)  # keep variable name similar to previous code
ridge.fit(X_train_scaled, y_train_log)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/531473777.py in <cell line: 0>()
      3 
      4 scaler = StandardScaler()
----> 5 X_train_scaled = scaler.fit_transform(X_train)
      6 X_test_scaled = scaler.transform(X_test)
      7 

NameError: name 'X_train' is not defined

## === cell 29
from sklearn.metrics import mean_squared_error

y_pred_log = ridge.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Validation RMSE: {rmse:.4f}")


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/913094088.py in <cell line: 0>()
      2 
      3 # Predict on validation, then revert log transformation
----> 4 y_pred_log = ridge.predict(X_test_scaled)
      5 y_pred = np.expm1(y_pred_log)
      6 

NameError: name 'ridge' is not defined

## === cell 30
test_features = test_df.drop("key", axis=1)
test_features = test_features.reindex(columns=X_train.columns, fill_value=0)

test_features_scaled = scaler.transform(test_features)

test_pred_log = ridge.predict(test_features_scaled)
test_pred = np.expm1(test_pred_log)

test_pred = np.clip(test_pred, a_min=0, a_max=None)
test_pred = np.round(test_pred, 2)


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1239516969.py in <cell line: 0>()
----> 1 test_features = test_df.drop("key", axis=1)
      2 test_features = test_features.reindex(columns=X_train.columns, fill_value=0)
      3 
      4 test_features_scaled = scaler.transform(test_features)
      5 

NameError: name 'test_df' is not defined

## === cell 31
Submission = pd.DataFrame(data=test_pred, columns=["fare_amount"])
Submission["key"] = test_df["key"]
Submission = Submission[["key", "fare_amount"]]


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2808171862.py in <cell line: 0>()
----> 1 Submission = pd.DataFrame(data=test_pred, columns=["fare_amount"])
      2 Submission["key"] = test_df["key"]
      3 Submission = Submission[["key", "fare_amount"]]

NameError: name 'pd' is not defined

## === cell 32
Submission.set_index("key", inplace=True)


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2454089214.py in <cell line: 0>()
----> 1 Submission.set_index("key", inplace=True)

NameError: name 'Submission' is not defined

## === cell 33
Submission.head()


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2880856867.py in <cell line: 0>()
----> 1 Submission.head()

NameError: name 'Submission' is not defined

## === cell 34
Submission.to_csv("Submission.csv")

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3260125347.py in <cell line: 0>()
----> 1 Submission.to_csv("Submission.csv")

NameError: name 'Submission' is not defined
