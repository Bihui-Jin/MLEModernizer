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

3.10

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

5.68914

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'I fix the LinearRegression initialization (remove the deprecated `normalize` argument) and add a small step to align one‑hot encoded weekday columns between train and test so the model sees identical feature sets. These changes resolve the runtime errors, ensure a valid prediction array, and allow the script to write a proper `Submission.csv` file without altering the core modeling logic.'
- What this solution (achieved 15.47835) has done: 'I adjust the feature scaling to use standard deviation (more sensible than variance) and switch the linear model to a regularized Ridge regression, then compute the proper RMSE metric and clip extreme predictions to a realistic fare range. These targeted tweaks keep the overall pipeline intact while expectedly moving the RMSE far closer to the target score.'
- What this solution (achieved 19.16724) has done: 'I fix the NaN issue caused by taking the log of negative fares, ensure the model is properly fitted, and make the submission creation use the resulting predictions. This resolves the runtime errors and lets the script produce a valid `Submission.csv` while keeping the core modeling approach unchanged.'
- What this solution (achieved 19.16459) has done: 'I keep the overall pipeline unchanged but stop scaling the one‑hot weekday columns, which were being centered and divided by their standard deviation and hurting the Ridge model. Instead, I use the raw feature values (no StandardScaler) so numeric distances stay on their original scale while dummy variables remain 0/1. This small change should lower the validation RMSE and bring the score closer to the target without altering the core modeling logic.'
- What this solution (achieved 18.41943) has done: 'I keep the overall pipeline unchanged but add the most predictive geographic coordinates back into the model and standardize the distance‑based features. I also lower the Ridge regularisation (α = 0.1) so the linear model can utilise the richer feature set. These small, targeted tweaks should lower the validation RMSE and move the score toward the target without altering the core modelling approach.'

# 9. Code solution

## === cell 0
train_data = pd.read_csv("/kaggle/input/train.csv", nrows=10_000_000)
train_data.head()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2764354390.py in <cell line: 0>()
----> 1 train_data = pd.read_csv("/kaggle/input/train.csv", nrows=10_000_000)
      2 train_data.head()
      3 

NameError: name 'pd' is not defined

## === cell 1
train_data.shape



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4031933074.py in <cell line: 0>()
----> 1 train_data.shape
      2 

NameError: name 'train_data' is not defined

## === cell 2
train_data.info()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4151811981.py in <cell line: 0>()
----> 1 train_data.info()
      2 

NameError: name 'train_data' is not defined

## === cell 3
test_data = pd.read_csv("/kaggle/input/test.csv")
test_data.head()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/818521061.py in <cell line: 0>()
----> 1 test_data = pd.read_csv("/kaggle/input/test.csv")
      2 test_data.head()
      3 

NameError: name 'pd' is not defined

## === cell 4
test_data.info()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1702245187.py in <cell line: 0>()
----> 1 test_data.info()
      2 

NameError: name 'test_data' is not defined

## === cell 5
train_data.isna().sum()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1085632216.py in <cell line: 0>()
----> 1 train_data.isna().sum()
      2 

NameError: name 'train_data' is not defined

## === cell 6
train_data["Difference_longitude"] = np.abs(
    np.asarray(train_data["pickup_longitude"] - train_data["dropoff_longitude"])
)
train_data["Difference_latitude"] = np.abs(
    np.asarray(train_data["pickup_latitude"] - train_data["dropoff_latitude"])
)

test_data["Difference_longitude"] = np.abs(
    np.asarray(test_data["pickup_longitude"] - test_data["dropoff_longitude"])
)
test_data["Difference_latitude"] = np.abs(
    np.asarray(test_data["pickup_latitude"] - test_data["dropoff_latitude"])
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2029277579.py in <cell line: 0>()
----> 1 train_data["Difference_longitude"] = np.abs(
      2     np.asarray(train_data["pickup_longitude"] - train_data["dropoff_longitude"])
      3 )
      4 train_data["Difference_latitude"] = np.abs(
      5     np.asarray(train_data["pickup_latitude"] - train_data["dropoff_latitude"])

NameError: name 'np' is not defined

## === cell 7
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")
train_data = train_data[train_data["fare_amount"] > 0].reset_index(drop=True)
print(f"After removing non‑positive fares: {len(train_data)}")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3530735149.py in <cell line: 0>()
----> 1 print(f"Before Dropping null values: {len(train_data)}")
      2 train_data.dropna(inplace=True)
      3 print(f"After Dropping null values: {len(train_data)}")
      4 train_data = train_data[train_data["fare_amount"] > 0].reset_index(drop=True)
      5 print(f"After removing non‑positive fares: {len(train_data)}")

NameError: name 'train_data' is not defined

## === cell 8
plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1169734720.py in <cell line: 0>()
----> 1 plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")
      2 

NameError: name 'train_data' is not defined

## === cell 9
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/297526157.py in <cell line: 0>()
----> 1 train_data = train_data[
      2     (train_data["Difference_longitude"] < 5.0)
      3     & (train_data["Difference_latitude"] < 5.0)
      4 ]
      5 

NameError: name 'train_data' is not defined

## === cell 10
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_data["pickuptime"] = ls1

ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_data["pickuptime"] = ls1



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3679443842.py in <cell line: 0>()
----> 1 ls1 = list(train_data["pickup_datetime"])
      2 for i in range(len(ls1)):
      3     ls1[i] = ls1[i][11:-7:]
      4 train_data["pickuptime"] = ls1
      5 

NameError: name 'train_data' is not defined

## === cell 11
train_data.head()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3467491927.py in <cell line: 0>()
----> 1 train_data.head()
      2 

NameError: name 'train_data' is not defined

## === cell 12
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
train_data["Weekday"] = ls1

ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
test_data["Weekday"] = ls1



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2444768071.py in <cell line: 0>()
----> 1 ls1 = list(train_data["pickup_datetime"])
      2 for i in range(len(ls1)):
      3     ls1[i] = ls1[i][:-4:]
      4     ls1[i] = pd.Timestamp(ls1[i])
      5     ls1[i] = ls1[i].weekday()

NameError: name 'train_data' is not defined

## === cell 13
train_data.head()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3467491927.py in <cell line: 0>()
----> 1 train_data.head()
      2 

NameError: name 'train_data' is not defined

## === cell 14
test_data.head()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/768455919.py in <cell line: 0>()
----> 1 test_data.head()
      2 

NameError: name 'test_data' is not defined

## === cell 15
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/764264539.py in <cell line: 0>()
----> 1 train_data.drop("pickup_datetime", inplace=True, axis=1)
      2 test_data.drop("pickup_datetime", inplace=True, axis=1)
      3 

NameError: name 'train_data' is not defined

## === cell 16
train_data["Weekday"].replace(
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
test_data["Weekday"].replace(
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
/tmp/ipykernel_11/2022498230.py in <cell line: 0>()
----> 1 train_data["Weekday"].replace(
      2     to_replace=[i for i in range(0, 7)],
      3     value=[
      4         "Monday",
      5         "Tuesday",

NameError: name 'train_data' is not defined

## === cell 17
train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])
train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3579435750.py in <cell line: 0>()
----> 1 train_one_hot = pd.get_dummies(train_data["Weekday"])
      2 test_one_hot = pd.get_dummies(test_data["Weekday"])
      3 train_data = pd.concat([train_data, train_one_hot], axis=1)
      4 test_data = pd.concat([test_data, test_one_hot], axis=1)
      5 

NameError: name 'pd' is not defined

## === cell 18
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4278855193.py in <cell line: 0>()
----> 1 train_data.drop("Weekday", axis=1, inplace=True)
      2 test_data.drop("Weekday", axis=1, inplace=True)
      3 

NameError: name 'train_data' is not defined

## === cell 19
ls1 = list(train_data["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
train_data["pickuptime"] = ls1

ls1 = list(test_data["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
test_data["pickuptime"] = ls1



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3226424968.py in <cell line: 0>()
----> 1 ls1 = list(train_data["pickuptime"])
      2 for i in range(len(ls1)):
      3     z = ls1[i].split(":")
      4     ls1[i] = int(z[0]) * 100 + int(z[1])
      5 train_data["pickuptime"] = ls1

NameError: name 'train_data' is not defined

## === cell 20
train_data.head()



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3467491927.py in <cell line: 0>()
----> 1 train_data.head()
      2 

NameError: name 'train_data' is not defined

## === cell 21
R = 6373.0
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



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2277544786.py in <cell line: 0>()
      1 R = 6373.0
----> 2 lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
      3 lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
      4 lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
      5 lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

NameError: name 'np' is not defined

## === cell 22
R = 6373.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

lat3 = np.zeros(len(train_data)) + np.radians(40.6413111)
lon3 = np.zeros(len(train_data)) + np.radians(-73.7781391)
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
train_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2

train_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621


lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

lat3 = np.zeros(len(test_data)) + np.radians(40.6413111)
lon3 = np.zeros(len(test_data)) + np.radians(-73.7781391)
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
test_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2

test_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4168520821.py in <cell line: 0>()
      1 R = 6373.0
----> 2 lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
      3 lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
      4 lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
      5 lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

NameError: name 'np' is not defined

## === cell 23
train_data["Distance"] = np.round(train_data["Distance"], 2)
train_data["Pickup_Distance_airport"] = np.round(
    train_data["Pickup_Distance_airport"], 2
)
train_data["Dropoff_Distance_airport"] = np.round(
    train_data["Dropoff_Distance_airport"], 2
)
test_data["Distance"] = np.round(test_data["Distance"], 2)
test_data["Pickup_Distance_airport"] = np.round(test_data["Pickup_Distance_airport"], 2)
test_data["Dropoff_Distance_airport"] = np.round(
    test_data["Dropoff_Distance_airport"], 2
)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/257530458.py in <cell line: 0>()
----> 1 train_data["Distance"] = np.round(train_data["Distance"], 2)
      2 train_data["Pickup_Distance_airport"] = np.round(
      3     train_data["Pickup_Distance_airport"], 2
      4 )
      5 train_data["Dropoff_Distance_airport"] = np.round(

NameError: name 'np' is not defined

## === cell 24
num_cols = [
    "Difference_longitude",
    "Difference_latitude",
    "Distance",
    "Pickup_Distance_airport",
    "Dropoff_Distance_airport",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "pickuptime",
]
for col in num_cols:
    mean = train_data[col].mean()
    std = train_data[col].std()
    train_data[col] = (train_data[col] - mean) / std
    test_data[col] = (test_data[col] - mean) / std  # use train stats for test



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1350643098.py in <cell line: 0>()
     12 ]
     13 for col in num_cols:
---> 14     mean = train_data[col].mean()
     15     std = train_data[col].std()
     16     train_data[col] = (train_data[col] - mean) / std

NameError: name 'train_data' is not defined

## === cell 25
train_data.shape



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4031933074.py in <cell line: 0>()
----> 1 train_data.shape
      2 

NameError: name 'train_data' is not defined

## === cell 26
test_data.shape



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1276008450.py in <cell line: 0>()
----> 1 test_data.shape
      2 

NameError: name 'test_data' is not defined

## === cell 27
train_feature_set = set(train_data.columns) - {"key", "fare_amount"}
test_feature_set = set(test_data.columns) - {"key"}

for col in train_feature_set - test_feature_set:
    test_data[col] = 0
for col in test_feature_set - train_feature_set:
    train_data[col] = 0

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]

X_array = X.values
X_test_array = test_data.drop("key", axis=1).values

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_array, y, test_size=0.01, random_state=80
)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2527844382.py in <cell line: 0>()
----> 1 train_feature_set = set(train_data.columns) - {"key", "fare_amount"}
      2 test_feature_set = set(test_data.columns) - {"key"}
      3 
      4 for col in train_feature_set - test_feature_set:
      5     test_data[col] = 0

NameError: name 'train_data' is not defined

## === cell 28
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

alpha_val = 0.01  # slightly less regularisation than before
ridge_raw = Ridge(alpha=alpha_val, random_state=80)
ridge_raw.fit(X_train, y_train)
pred_raw_val = ridge_raw.predict(X_val)
rmse_raw = mean_squared_error(y_val, pred_raw_val, squared=False)

ridge_log = Ridge(alpha=alpha_val, random_state=80)
y_train_log = np.log1p(y_train.clip(lower=0))
ridge_log.fit(X_train, y_train_log)
pred_log_val = np.expm1(ridge_log.predict(X_val))
rmse_log = mean_squared_error(y_val, pred_log_val, squared=False)

if rmse_log < rmse_raw:
    chosen_model = ridge_log
    use_log = True
    best_rmse = rmse_log
else:
    chosen_model = ridge_raw
    use_log = False
    best_rmse = rmse_raw

print(f"Validation RMSE (raw): {rmse_raw:.4f}")
print(f"Validation RMSE (log): {rmse_log:.4f}")
print(
    f"Chosen model uses {'log‑target' if use_log else 'raw'} with RMSE: {best_rmse:.4f}"
)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2583872120.py in <cell line: 0>()
      5 alpha_val = 0.01  # slightly less regularisation than before
      6 ridge_raw = Ridge(alpha=alpha_val, random_state=80)
----> 7 ridge_raw.fit(X_train, y_train)
      8 pred_raw_val = ridge_raw.predict(X_val)
      9 rmse_raw = mean_squared_error(y_val, pred_raw_val, squared=False)

NameError: name 'X_train' is not defined

## === cell 29
if use_log:
    y_full = np.log1p(y.clip(lower=0))
else:
    y_full = y

chosen_model.fit(X_array, y_full)

test_pred = chosen_model.predict(X_test_array)
if use_log:
    test_pred = np.expm1(test_pred)

test_pred = np.clip(test_pred, 0, 300)
test_pred = np.round(test_pred, 2)



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2364458727.py in <cell line: 0>()
      1 # Re‑fit the chosen model on the full training data
----> 2 if use_log:
      3     y_full = np.log1p(y.clip(lower=0))
      4 else:
      5     y_full = y

NameError: name 'use_log' is not defined

## === cell 30
pd.read_csv("/kaggle/input/sample_submission.csv").head()



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2886149637.py in <cell line: 0>()
----> 1 pd.read_csv("/kaggle/input/sample_submission.csv").head()
      2 

NameError: name 'pd' is not defined

## === cell 31
Submission = pd.DataFrame(data=test_pred, columns=["fare_amount"])
Submission["key"] = test_data["key"]
Submission = Submission[["key", "fare_amount"]]



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/629553703.py in <cell line: 0>()
----> 1 Submission = pd.DataFrame(data=test_pred, columns=["fare_amount"])
      2 Submission["key"] = test_data["key"]
      3 Submission = Submission[["key", "fare_amount"]]
      4 

NameError: name 'pd' is not defined

## === cell 32
Submission.set_index("key", inplace=True)



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/109248162.py in <cell line: 0>()
----> 1 Submission.set_index("key", inplace=True)
      2 

NameError: name 'Submission' is not defined

## === cell 33
Submission.to_csv("Submission.csv")

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3260125347.py in <cell line: 0>()
----> 1 Submission.to_csv("Submission.csv")

NameError: name 'Submission' is not defined
