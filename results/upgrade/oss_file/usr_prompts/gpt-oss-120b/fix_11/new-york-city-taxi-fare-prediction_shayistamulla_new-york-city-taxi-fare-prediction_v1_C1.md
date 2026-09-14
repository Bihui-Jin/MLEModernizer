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

5.68914

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'I remove the deprecated `normalize=True` argument from the `LinearRegression` constructor so the model can be instantiated, which resolves the `TypeError`. This also restores the `lr` variable, allowing subsequent prediction, submission creation, and CSV export to run without further errors.'
- What this solution (achieved 11.74409) has done: 'I replace the simple LinearRegression with a more powerful HistGradientBoostingRegressor, which better captures non‑linear relationships in the engineered features while keeping the overall pipeline unchanged. This change is expected to dramatically lower the RMSE from the current ~937 toward the target ~5.7, and the rest of the code (feature engineering, CSV creation) remains intact.'

# 9. Code solution

## === cell 0
import os

data_dir = "/kaggle/input"
assert os.path.isdir(data_dir), "Data directory not found"




## === cell 1
dtype = {
    "key": "category",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
train_data = pd.read_csv(train_path, dtype=dtype)
train_data.head()




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2552699409.py in <cell line: 0>()
      9 }
     10 train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
---> 11 train_data = pd.read_csv(train_path, dtype=dtype)
     12 train_data.head()
     13 

NameError: name 'pd' is not defined

## === cell 2
train_data.shape




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1379133816.py in <cell line: 0>()
----> 1 train_data.shape
      2 
      3 

NameError: name 'train_data' is not defined

## === cell 3
test_data = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    dtype={k: v for k, v in dtype.items() if k != "fare_amount"},
)
test_data.head()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2538646246.py in <cell line: 0>()
----> 1 test_data = pd.read_csv(
      2     "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
      3     dtype={k: v for k, v in dtype.items() if k != "fare_amount"},
      4 )
      5 test_data.head()

NameError: name 'pd' is not defined

## === cell 4
test_data.info()




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/488841587.py in <cell line: 0>()
----> 1 test_data.info()
      2 
      3 

NameError: name 'test_data' is not defined

## === cell 5
train_data.isna().sum()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2624065081.py in <cell line: 0>()
----> 1 train_data.isna().sum()
      2 
      3 

NameError: name 'train_data' is not defined

## === cell 6
train_data["Difference_longitude"] = np.abs(
    train_data["pickup_longitude"] - train_data["dropoff_longitude"]
).astype("float32")
train_data["Difference_latitude"] = np.abs(
    train_data["pickup_latitude"] - train_data["dropoff_latitude"]
).astype("float32")

test_data["Difference_longitude"] = np.abs(
    test_data["pickup_longitude"] - test_data["dropoff_longitude"]
).astype("float32")
test_data["Difference_latitude"] = np.abs(
    test_data["pickup_latitude"] - test_data["dropoff_latitude"]
).astype("float32")




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3028657604.py in <cell line: 0>()
      1 # Compute cheap absolute differences first
----> 2 train_data["Difference_longitude"] = np.abs(
      3     train_data["pickup_longitude"] - train_data["dropoff_longitude"]
      4 ).astype("float32")
      5 train_data["Difference_latitude"] = np.abs(

NameError: name 'np' is not defined

## === cell 7
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2419817112.py in <cell line: 0>()
----> 1 print(f"Before Dropping null values: {len(train_data)}")
      2 train_data.dropna(inplace=True)
      3 print(f"After Dropping null values: {len(train_data)}")
      4 
      5 

NameError: name 'train_data' is not defined

## === cell 8
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2911974954.py in <cell line: 0>()
      1 # Filter out extreme outliers early – this reduces the amount of data that later
      2 # heavy NumPy calculations need to process.
----> 3 train_data = train_data[
      4     (train_data["Difference_longitude"] < 5.0)
      5     & (train_data["Difference_latitude"] < 5.0)

NameError: name 'train_data' is not defined

## === cell 9
train_dt = pd.to_datetime(train_data["pickup_datetime"])
test_dt = pd.to_datetime(test_data["pickup_datetime"])

train_data["pickuptime"] = (train_dt.dt.hour * 100 + train_dt.dt.minute).astype(int)
test_data["pickuptime"] = (test_dt.dt.hour * 100 + test_dt.dt.minute).astype(int)

train_data["Weekday"] = train_dt.dt.weekday
test_data["Weekday"] = test_dt.dt.weekday

train_data.drop("pickup_datetime", axis=1, inplace=True)
test_data.drop("pickup_datetime", axis=1, inplace=True)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/10331687.py in <cell line: 0>()
----> 1 train_dt = pd.to_datetime(train_data["pickup_datetime"])
      2 test_dt = pd.to_datetime(test_data["pickup_datetime"])
      3 
      4 train_data["pickuptime"] = (train_dt.dt.hour * 100 + train_dt.dt.minute).astype(int)
      5 test_data["pickuptime"] = (test_dt.dt.hour * 100 + test_dt.dt.minute).astype(int)

NameError: name 'pd' is not defined

## === cell 10
pass




## === cell 11
pass




## === cell 12
train_one_hot = pd.get_dummies(train_data["Weekday"], prefix="Weekday")
test_one_hot = pd.get_dummies(test_data["Weekday"], prefix="Weekday")
train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/603067700.py in <cell line: 0>()
----> 1 train_one_hot = pd.get_dummies(train_data["Weekday"], prefix="Weekday")
      2 test_one_hot = pd.get_dummies(test_data["Weekday"], prefix="Weekday")
      3 train_data = pd.concat([train_data, train_one_hot], axis=1)
      4 test_data = pd.concat([test_data, test_one_hot], axis=1)
      5 

NameError: name 'pd' is not defined

## === cell 13
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/927199483.py in <cell line: 0>()
----> 1 train_data.drop("Weekday", axis=1, inplace=True)
      2 test_data.drop("Weekday", axis=1, inplace=True)
      3 
      4 

NameError: name 'train_data' is not defined

## === cell 14
train_data["pickuptime"] = train_data["pickuptime"].astype("int16")
test_data["pickuptime"] = test_data["pickuptime"].astype("int16")




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3115627582.py in <cell line: 0>()
----> 1 train_data["pickuptime"] = train_data["pickuptime"].astype("int16")
      2 test_data["pickuptime"] = test_data["pickuptime"].astype("int16")
      3 
      4 

NameError: name 'train_data' is not defined

## === cell 15
R = 6373.0
lat1 = np.radians(train_data["pickup_latitude"].astype("float32"))
lon1 = np.radians(train_data["pickup_longitude"].astype("float32"))
lat2 = np.radians(train_data["dropoff_latitude"].astype("float32"))
lon2 = np.radians(train_data["dropoff_longitude"].astype("float32"))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_data["Distance"] = (distance * 0.621).astype("float32")

lat1 = np.radians(test_data["pickup_latitude"].astype("float32"))
lon1 = np.radians(test_data["pickup_longitude"].astype("float32"))
lat2 = np.radians(test_data["dropoff_latitude"].astype("float32"))
lon2 = np.radians(test_data["dropoff_longitude"].astype("float32"))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data["Distance"] = (distance * 0.621).astype("float32")

del lat1, lon1, lat2, lon2, dlon, dlat, a, c, distance




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3879877638.py in <cell line: 0>()
      1 # Haversine distance – performed after the outlier filter to work on the smaller set.
      2 R = 6373.0
----> 3 lat1 = np.radians(train_data["pickup_latitude"].astype("float32"))
      4 lon1 = np.radians(train_data["pickup_longitude"].astype("float32"))
      5 lat2 = np.radians(train_data["dropoff_latitude"].astype("float32"))

NameError: name 'np' is not defined

## === cell 16
R = 6373.0
lat1 = np.radians(train_data["pickup_latitude"].astype("float32"))
lon1 = np.radians(train_data["pickup_longitude"].astype("float32"))
lat2 = np.radians(train_data["dropoff_latitude"].astype("float32"))
lon2 = np.radians(train_data["dropoff_longitude"].astype("float32"))

lat3 = np.radians(40.6413111)
lon3 = np.radians(-73.7781391)

dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
train_data["Pickup_Distance_airport"] = (R * c1 * 0.621).astype("float32")

dlon_dropoff = lon3 - lon2
dlat_dropoff = lat3 - lat2
a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
train_data["Dropoff_Distance_airport"] = (R * c2 * 0.621).astype("float32")

lat1 = np.radians(test_data["pickup_latitude"].astype("float32"))
lon1 = np.radians(test_data["pickup_longitude"].astype("float32"))
lat2 = np.radians(test_data["dropoff_latitude"].astype("float32"))
lon2 = np.radians(test_data["dropoff_longitude"].astype("float32"))

dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
test_data["Pickup_Distance_airport"] = (R * c1 * 0.621).astype("float32")

dlon_dropoff = lon3 - lon2
dlat_dropoff = lat3 - lat2
a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
test_data["Dropoff_Distance_airport"] = (R * c2 * 0.621).astype("float32")

del (
    lat1,
    lon1,
    lat2,
    lon2,
    dlon_pickup,
    dlat_pickup,
    dlon_dropoff,
    dlat_dropoff,
    a1,
    a2,
    c1,
    c2,
)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1642673804.py in <cell line: 0>()
      1 R = 6373.0
----> 2 lat1 = np.radians(train_data["pickup_latitude"].astype("float32"))
      3 lon1 = np.radians(train_data["pickup_longitude"].astype("float32"))
      4 lat2 = np.radians(train_data["dropoff_latitude"].astype("float32"))
      5 lon2 = np.radians(train_data["dropoff_longitude"].astype("float32"))

NameError: name 'np' is not defined

## === cell 17
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




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2473229134.py in <cell line: 0>()
----> 1 train_data["Distance"] = np.round(train_data["Distance"], 2)
      2 train_data["Pickup_Distance_airport"] = np.round(
      3     train_data["Pickup_Distance_airport"], 2
      4 )
      5 train_data["Dropoff_Distance_airport"] = np.round(

NameError: name 'np' is not defined

## === cell 18
train_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/618874771.py in <cell line: 0>()
----> 1 train_data.drop(
      2     ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
      3     axis=1,
      4     inplace=True,
      5 )

NameError: name 'train_data' is not defined

## === cell 19
mean_dl = train_data["Difference_longitude"].mean()
var_dl = train_data["Difference_longitude"].var()
train_data["Difference_longitude"] = (
    np.abs(train_data["Difference_longitude"] - mean_dl) / var_dl
)




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1715537136.py in <cell line: 0>()
----> 1 mean_dl = train_data["Difference_longitude"].mean()
      2 var_dl = train_data["Difference_longitude"].var()
      3 train_data["Difference_longitude"] = (
      4     np.abs(train_data["Difference_longitude"] - mean_dl) / var_dl
      5 )

NameError: name 'train_data' is not defined

## === cell 20
mean_dlat = train_data["Difference_latitude"].mean()
var_dlat = train_data["Difference_latitude"].var()
train_data["Difference_latitude"] = (
    np.abs(train_data["Difference_latitude"] - mean_dlat) / var_dlat
)




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1445125177.py in <cell line: 0>()
----> 1 mean_dlat = train_data["Difference_latitude"].mean()
      2 var_dlat = train_data["Difference_latitude"].var()
      3 train_data["Difference_latitude"] = (
      4     np.abs(train_data["Difference_latitude"] - mean_dlat) / var_dlat
      5 )

NameError: name 'train_data' is not defined

## === cell 21
mean_dl_test = test_data["Difference_longitude"].mean()
var_dl_test = test_data["Difference_longitude"].var()
test_data["Difference_longitude"] = (
    np.abs(test_data["Difference_longitude"] - mean_dl_test) / var_dl_test
)

mean_dlat_test = test_data["Difference_latitude"].mean()
var_dlat_test = test_data["Difference_latitude"].var()
test_data["Difference_latitude"] = (
    np.abs(test_data["Difference_latitude"] - mean_dlat_test) / var_dlat_test
)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1681919800.py in <cell line: 0>()
----> 1 mean_dl_test = test_data["Difference_longitude"].mean()
      2 var_dl_test = test_data["Difference_longitude"].var()
      3 test_data["Difference_longitude"] = (
      4     np.abs(test_data["Difference_longitude"] - mean_dl_test) / var_dl_test
      5 )

NameError: name 'test_data' is not defined

## === cell 22
from sklearn.model_selection import train_test_split

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]

valid_mask = X.notnull().all(axis=1) & np.isfinite(X).all(axis=1) & y.notnull()
X = X[valid_mask]
y = y[valid_mask]

test_features = test_data.drop("key", axis=1).copy()
test_features = test_features.fillna(test_features.median())

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=80
)




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/452023205.py in <cell line: 0>()
      1 from sklearn.model_selection import train_test_split
      2 
----> 3 X = train_data.drop(["key", "fare_amount"], axis=1)
      4 y = train_data["fare_amount"]
      5 

NameError: name 'train_data' is not defined

## === cell 23
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

y_train = y_train.clip(lower=0)
y_train_log = np.log1p(y_train)

hgb = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.05,
    max_depth=10,
    max_bins=255,
    random_state=42,
)
hgb.fit(X_train, y_train_log)

valid_pred_log = hgb.predict(X_valid)
valid_pred = np.expm1(valid_pred_log)

rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
print(f"Validation RMSE (original scale): {rmse:.5f}")




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3008192725.py in <cell line: 0>()
      3 import numpy as np
      4 
----> 5 y_train = y_train.clip(lower=0)
      6 y_train_log = np.log1p(y_train)
      7 

NameError: name 'y_train' is not defined

## === cell 24
test_features = test_features.reindex(columns=X_train.columns, fill_value=0)

test_pred_log = hgb.predict(test_features)
test_pred = np.round(np.expm1(test_pred_log), 2)




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1806526643.py in <cell line: 0>()
----> 1 test_features = test_features.reindex(columns=X_train.columns, fill_value=0)
      2 
      3 test_pred_log = hgb.predict(test_features)
      4 test_pred = np.round(np.expm1(test_pred_log), 2)
      5 

NameError: name 'test_features' is not defined

## === cell 25
Submission = pd.DataFrame({"key": test_data["key"], "fare_amount": test_pred})
Submission = Submission[["key", "fare_amount"]]




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1645295844.py in <cell line: 0>()
----> 1 Submission = pd.DataFrame({"key": test_data["key"], "fare_amount": test_pred})
      2 Submission = Submission[["key", "fare_amount"]]
      3 
      4 

NameError: name 'pd' is not defined

## === cell 26
Submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1428118525.py in <cell line: 0>()
----> 1 Submission.to_csv("submission.csv", index=False)

NameError: name 'Submission' is not defined
