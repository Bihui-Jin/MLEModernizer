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

5.6891

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 936.92806) has done: 'I fix the deprecated `normalize` argument in `LinearRegression`, ensure the test set has exactly the same feature columns as the training set (adding missing one‑hot columns with zeros), and correctly write the prediction dataframe to a CSV file named `submission.csv`. These minimal changes resolve the runtime errors and produce a valid submission while keeping the original modeling approach unchanged.'

# 9. Code solution

## === cell 0
train_df = pd.read_csv("../input/train.csv", nrows=10_000_000)
train_df.dtypes


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2312702329.py in <cell line: 0>()
----> 1 train_df = pd.read_csv("../input/train.csv", nrows=10_000_000)
      2 train_df.dtypes

NameError: name 'pd' is not defined

## === cell 1
train_df.head()


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2663572906.py in <cell line: 0>()
----> 1 train_df.head()

NameError: name 'train_df' is not defined

## === cell 2
train_df.shape


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/85899362.py in <cell line: 0>()
----> 1 train_df.shape

NameError: name 'train_df' is not defined

## === cell 3
train_df.info()


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/765168486.py in <cell line: 0>()
----> 1 train_df.info()

NameError: name 'train_df' is not defined

## === cell 4
test_df = pd.read_csv("../input/test.csv")
test_df.dtypes


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3502887157.py in <cell line: 0>()
----> 1 test_df = pd.read_csv("../input/test.csv")
      2 test_df.dtypes

NameError: name 'pd' is not defined

## === cell 5
test_df.head()


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2239380292.py in <cell line: 0>()
----> 1 test_df.head()

NameError: name 'test_df' is not defined

## === cell 6
test_df.info()


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2306363401.py in <cell line: 0>()
----> 1 test_df.info()

NameError: name 'test_df' is not defined

## === cell 7
test_df.shape


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/578093938.py in <cell line: 0>()
----> 1 test_df.shape

NameError: name 'test_df' is not defined

## === cell 8
train_df.isna().sum()




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/870367625.py in <cell line: 0>()
----> 1 train_df.isna().sum()
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 9
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3090621481.py in <cell line: 0>()
      4 
      5 
----> 6 add_travel_vector_features(train_df)
      7 add_travel_vector_features(test_df)

NameError: name 'train_df' is not defined

## === cell 10
print(f"Before Dropping null values: {len(train_df)}")
train_df.dropna(inplace=True)
print(f"After Dropping null values: {len(train_df)}")


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1364102575.py in <cell line: 0>()
----> 1 print(f"Before Dropping null values: {len(train_df)}")
      2 train_df.dropna(inplace=True)
      3 print(f"After Dropping null values: {len(train_df)}")

NameError: name 'train_df' is not defined

## === cell 11
print(train_df.isnull().sum())


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3174598079.py in <cell line: 0>()
----> 1 print(train_df.isnull().sum())

NameError: name 'train_df' is not defined

## === cell 12
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4249765067.py in <cell line: 0>()
----> 1 print("Old size: %d" % len(train_df))
      2 train_df = train_df.dropna(how="any", axis="rows")
      3 print("New size: %d" % len(train_df))

NameError: name 'train_df' is not defined

## === cell 13
plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/311890816.py in <cell line: 0>()
----> 1 plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")

NameError: name 'train_df' is not defined

## === cell 14
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/889800582.py in <cell line: 0>()
----> 1 print("Old size: %d" % len(train_df))
      2 train_df = train_df[
      3     (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
      4 ]
      5 print("New size: %d" % len(train_df))

NameError: name 'train_df' is not defined

## === cell 15
def creating_time(df):
    ls1 = list(df["pickup_datetime"])
    for i in range(len(ls1)):
        ls1[i] = ls1[i][11:-7:]
    df["pickuptime"] = ls1


creating_time(train_df)
creating_time(test_df)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/380115126.py in <cell line: 0>()
      6 
      7 
----> 8 creating_time(train_df)
      9 creating_time(test_df)

NameError: name 'train_df' is not defined

## === cell 16
train_df.head()


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2663572906.py in <cell line: 0>()
----> 1 train_df.head()

NameError: name 'train_df' is not defined

## === cell 17
test_df.head()




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1899552842.py in <cell line: 0>()
----> 1 test_df.head()
      2 
      3 

NameError: name 'test_df' is not defined

## === cell 18
def creating_weekdays(df):
    ls1 = list(df["pickup_datetime"])
    for i in range(len(ls1)):
        ls1[i] = ls1[i][:-4:]
        ls1[i] = pd.Timestamp(ls1[i])
        ls1[i] = ls1[i].weekday()
    df["Weekday"] = ls1


creating_weekdays(train_df)
creating_weekdays(test_df)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2036159060.py in <cell line: 0>()
      8 
      9 
---> 10 creating_weekdays(train_df)
     11 creating_weekdays(test_df)

NameError: name 'train_df' is not defined

## === cell 19
train_df.head()


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2663572906.py in <cell line: 0>()
----> 1 train_df.head()

NameError: name 'train_df' is not defined

## === cell 20
test_df.head()


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2239380292.py in <cell line: 0>()
----> 1 test_df.head()

NameError: name 'test_df' is not defined

## === cell 21
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/953242425.py in <cell line: 0>()
----> 1 train_df.drop("pickup_datetime", inplace=True, axis=1)
      2 test_df.drop("pickup_datetime", inplace=True, axis=1)
      3 
      4 

NameError: name 'train_df' is not defined

## === cell 22
def replace_weekday(df):
    df["Weekday"].replace(
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


replace_weekday(train_df)
replace_weekday(test_df)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2778786171.py in <cell line: 0>()
     15 
     16 
---> 17 replace_weekday(train_df)
     18 replace_weekday(test_df)

NameError: name 'train_df' is not defined

## === cell 23
train_df.head()


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2663572906.py in <cell line: 0>()
----> 1 train_df.head()

NameError: name 'train_df' is not defined

## === cell 24
test_df.head()


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2239380292.py in <cell line: 0>()
----> 1 test_df.head()

NameError: name 'test_df' is not defined

## === cell 25
train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_df["Weekday"])
train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1158577080.py in <cell line: 0>()
----> 1 train_one_hot = pd.get_dummies(train_df["Weekday"])
      2 test_one_hot = pd.get_dummies(test_df["Weekday"])
      3 train_df = pd.concat([train_df, train_one_hot], axis=1)
      4 test_df = pd.concat([test_df, test_one_hot], axis=1)

NameError: name 'pd' is not defined

## === cell 26
train_df.head()


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2663572906.py in <cell line: 0>()
----> 1 train_df.head()

NameError: name 'train_df' is not defined

## === cell 27
test_df.head()


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2239380292.py in <cell line: 0>()
----> 1 test_df.head()

NameError: name 'test_df' is not defined

## === cell 28
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)




## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2984131749.py in <cell line: 0>()
----> 1 train_df.drop("Weekday", axis=1, inplace=True)
      2 test_df.drop("Weekday", axis=1, inplace=True)
      3 
      4 

NameError: name 'train_df' is not defined

## === cell 29
def creating_pickupdate(df):
    ls1 = list(df["pickuptime"])
    for i in range(len(ls1)):
        z = ls1[i].split(":")
        ls1[i] = int(z[0]) * 100 + int(z[1])
    df["pickuptime"] = ls1


creating_pickupdate(train_df)
creating_pickupdate(test_df)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2003708572.py in <cell line: 0>()
      7 
      8 
----> 9 creating_pickupdate(train_df)
     10 creating_pickupdate(test_df)

NameError: name 'train_df' is not defined

## === cell 30
train_df.head()


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2663572906.py in <cell line: 0>()
----> 1 train_df.head()

NameError: name 'train_df' is not defined

## === cell 31
test_df.head()




## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1899552842.py in <cell line: 0>()
----> 1 test_df.head()
      2 
      3 

NameError: name 'test_df' is not defined

## === cell 32
def finding_distance(df):
    R = 6373.0
    lat1 = np.asarray(np.radians(df["pickup_latitude"]))
    lon1 = np.asarray(np.radians(df["pickup_longitude"]))
    lat2 = np.asarray(np.radians(df["dropoff_latitude"]))
    lon2 = np.asarray(np.radians(df["dropoff_longitude"]))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    df["Distance"] = np.asarray(distance) * 0.621


finding_distance(train_df)
finding_distance(test_df)




## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/555397670.py in <cell line: 0>()
     13 
     14 
---> 15 finding_distance(train_df)
     16 finding_distance(test_df)
     17 

NameError: name 'train_df' is not defined

## === cell 33
def creating_pickup_dropoff_distance(df):
    R = 6373.0
    lat1 = np.asarray(np.radians(df["pickup_latitude"]))
    lon1 = np.asarray(np.radians(df["pickup_longitude"]))
    lat2 = np.asarray(np.radians(df["dropoff_latitude"]))
    lon2 = np.asarray(np.radians(df["dropoff_longitude"]))
    lat3 = np.zeros(len(df)) + np.radians(40.6413111)
    lon3 = np.zeros(len(df)) + np.radians(-73.7781391)
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
    df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621
    a2 = (
        np.sin(d_lat_dropoff / 2) ** 2
        + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
    )
    c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
    distance2 = R * c2
    df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621


creating_pickup_dropoff_distance(train_df)
creating_pickup_dropoff_distance(test_df)


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3675072884.py in <cell line: 0>()
     27 
     28 
---> 29 creating_pickup_dropoff_distance(train_df)
     30 creating_pickup_dropoff_distance(test_df)

NameError: name 'train_df' is not defined

## === cell 34
train_df["Distance"] = np.round(train_df["Distance"], 2)
train_df["Pickup_Distance_airport"] = np.round(train_df["Pickup_Distance_airport"], 2)
train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)
test_df["Distance"] = np.round(test_df["Distance"], 2)
test_df["Pickup_Distance_airport"] = np.round(test_df["Pickup_Distance_airport"], 2)
test_df["Dropoff_Distance_airport"] = np.round(test_df["Dropoff_Distance_airport"], 2)


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2115963763.py in <cell line: 0>()
----> 1 train_df["Distance"] = np.round(train_df["Distance"], 2)
      2 train_df["Pickup_Distance_airport"] = np.round(train_df["Pickup_Distance_airport"], 2)
      3 train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)
      4 test_df["Distance"] = np.round(test_df["Distance"], 2)
      5 test_df["Pickup_Distance_airport"] = np.round(test_df["Pickup_Distance_airport"], 2)

NameError: name 'np' is not defined

## === cell 35
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


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1419223178.py in <cell line: 0>()
----> 1 train_df.drop(
      2     ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
      3     axis=1,
      4     inplace=True,
      5 )

NameError: name 'train_df' is not defined

## === cell 36
train_df["abs_diff_longitude"] = np.abs(
    train_df["abs_diff_longitude"] - np.mean(train_df["abs_diff_longitude"])
)
train_df["abs_diff_longitude"] = train_df["abs_diff_longitude"] / np.var(
    train_df["abs_diff_longitude"]
)


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3677448457.py in <cell line: 0>()
----> 1 train_df["abs_diff_longitude"] = np.abs(
      2     train_df["abs_diff_longitude"] - np.mean(train_df["abs_diff_longitude"])
      3 )
      4 train_df["abs_diff_longitude"] = train_df["abs_diff_longitude"] / np.var(
      5     train_df["abs_diff_longitude"]

NameError: name 'np' is not defined

## === cell 37
train_df["abs_diff_latitude"] = np.abs(
    train_df["abs_diff_latitude"] - np.mean(train_df["abs_diff_latitude"])
)
train_df["abs_diff_latitude"] = train_df["abs_diff_latitude"] / np.var(
    train_df["abs_diff_latitude"]
)


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3476711976.py in <cell line: 0>()
----> 1 train_df["abs_diff_latitude"] = np.abs(
      2     train_df["abs_diff_latitude"] - np.mean(train_df["abs_diff_latitude"])
      3 )
      4 train_df["abs_diff_latitude"] = train_df["abs_diff_latitude"] / np.var(
      5     train_df["abs_diff_latitude"]

NameError: name 'np' is not defined

## === cell 38
test_df["abs_diff_longitude"] = np.abs(
    test_df["abs_diff_longitude"] - np.mean(test_df["abs_diff_longitude"])
)
test_df["abs_diff_longitude"] = test_df["abs_diff_longitude"] / np.var(
    test_df["abs_diff_longitude"]
)
test_df["abs_diff_latitude"] = np.abs(
    test_df["abs_diff_latitude"] - np.mean(test_df["abs_diff_latitude"])
)
test_df["abs_diff_latitude"] = test_df["abs_diff_latitude"] / np.var(
    test_df["abs_diff_latitude"]
)


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/343053348.py in <cell line: 0>()
----> 1 test_df["abs_diff_longitude"] = np.abs(
      2     test_df["abs_diff_longitude"] - np.mean(test_df["abs_diff_longitude"])
      3 )
      4 test_df["abs_diff_longitude"] = test_df["abs_diff_longitude"] / np.var(
      5     test_df["abs_diff_longitude"]

NameError: name 'np' is not defined

## === cell 39
print(train_df.shape)
print(test_df.shape)


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1939175089.py in <cell line: 0>()
----> 1 print(train_df.shape)
      2 print(test_df.shape)

NameError: name 'train_df' is not defined

## === cell 40
from sklearn.model_selection import train_test_split

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]
y_log = np.log1p(y)
X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.01, random_state=80
)


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1070677216.py in <cell line: 0>()
      1 from sklearn.model_selection import train_test_split
      2 
----> 3 X = train_df.drop(["key", "fare_amount"], axis=1)
      4 y = train_df["fare_amount"]
      5 y_log = np.log1p(y)

NameError: name 'train_df' is not defined

## === cell 41
valid_mask = X_train.notnull().all(axis=1) & np.isfinite(y_train_log)
X_train = X_train[valid_mask]
y_train_log = y_train_log[valid_mask]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
lr.fit(X_train, y_train_log)

val_pred_log = lr.predict(X_val)
val_rmse_log = np.sqrt(mean_squared_error(y_val_log, val_pred_log))
print("Validation RMSE on log‑target:", val_rmse_log)


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/429691329.py in <cell line: 0>()
      1 # Remove any rows that still contain NaN or infinite values in features or target
----> 2 valid_mask = X_train.notnull().all(axis=1) & np.isfinite(y_train_log)
      3 X_train = X_train[valid_mask]
      4 y_train_log = y_train_log[valid_mask]
      5 

NameError: name 'X_train' is not defined

## === cell 42
test_features = test_df.drop("key", axis=1)
test_features = test_features.reindex(columns=X.columns, fill_value=0)

pred_log = lr.predict(test_features)
pred = np.expm1(pred_log)
pred = np.clip(pred, a_min=0, a_max=None)
pred = np.round(pred, 2)
print(pred[:5])


## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/950718696.py in <cell line: 0>()
----> 1 test_features = test_df.drop("key", axis=1)
      2 test_features = test_features.reindex(columns=X.columns, fill_value=0)
      3 
      4 pred_log = lr.predict(test_features)
      5 pred = np.expm1(pred_log)

NameError: name 'test_df' is not defined

## === cell 43
Submission = pd.DataFrame({"fare_amount": pred})
Submission["key"] = test_df["key"]
Submission = Submission[["key", "fare_amount"]]


## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2638057651.py in <cell line: 0>()
----> 1 Submission = pd.DataFrame({"fare_amount": pred})
      2 Submission["key"] = test_df["key"]
      3 Submission = Submission[["key", "fare_amount"]]

NameError: name 'pd' is not defined

## === cell 44
Submission.head()


## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2880856867.py in <cell line: 0>()
----> 1 Submission.head()

NameError: name 'Submission' is not defined

## === cell 45
Submission.to_csv("submission.csv", index=False)
print("Submission saved to 'submission.csv'")

## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4030654561.py in <cell line: 0>()
----> 1 Submission.to_csv("submission.csv", index=False)
      2 print("Submission saved to 'submission.csv'")

NameError: name 'Submission' is not defined
