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

3.12

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

4.12819

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
data = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=100_000
)



## === cell 2
data.head()



## === cell 3
data.shape



## === cell 4
data.info()



## === cell 5
print(data.isnull().sum())



## === cell 6
mean_dropoff_longitude = data["dropoff_longitude"].mean()
mean_dropoff_latitude = data["dropoff_latitude"].mean()
mean_pickup_longitude = data["pickup_longitude"].mean()
mean_pickup_latitude = data["pickup_latitude"].mean()

data["dropoff_longitude"] = data["dropoff_longitude"].fillna(mean_dropoff_longitude)
data["dropoff_latitude"] = data["dropoff_latitude"].fillna(mean_dropoff_latitude)
data["pickup_longitude"] = data["pickup_longitude"].fillna(mean_pickup_longitude)
data["pickup_latitude"] = data["pickup_latitude"].fillna(mean_pickup_latitude)



## === cell 7
print(data.isnull().sum())



## === cell 8
import seaborn as s

s.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap="cool")



## === cell 9
testData = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 10
testData.head()



## === cell 11
testData.shape



## === cell 12
print(testData.isnull().sum())



## === cell 13
testData["dropoff_longitude"] = testData["dropoff_longitude"].fillna(
    mean_dropoff_longitude
)
testData["dropoff_latitude"] = testData["dropoff_latitude"].fillna(
    mean_dropoff_latitude
)
testData["pickup_longitude"] = testData["pickup_longitude"].fillna(
    mean_pickup_longitude
)
testData["pickup_latitude"] = testData["pickup_latitude"].fillna(mean_pickup_latitude)



## === cell 14
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(data["fare_amount"], kde=True)
plt.title("Distribution of Fare Amount")
plt.show()



## === cell 15
data.shape



## === cell 16
data = data[data["fare_amount"] <= 500]



## === cell 17
data.shape



## === cell 18
plt.scatter(data["pickup_longitude"], data["fare_amount"])
plt.xlabel("Pickup Longitude")
plt.ylabel("Fare Amount")
plt.show()



## === cell 19
data.shape



## === cell 20
data = data[(data["passenger_count"] >= 1) & (data["passenger_count"] <= 7)]



## === cell 21
data.shape



## === cell 22
data.shape



## === cell 23
ny_lat_min, ny_lat_max = 40.4774, 40.9176
ny_lon_min, ny_lon_max = -74.2591, -73.7004

data = data[
    (data["pickup_latitude"].between(ny_lat_min, ny_lat_max))
    & (data["pickup_longitude"].between(ny_lon_min, ny_lon_max))
    & (data["dropoff_latitude"].between(ny_lat_min, ny_lat_max))
    & (data["dropoff_longitude"].between(ny_lon_min, ny_lon_max))
]



## === cell 24
data.shape




## === cell 25
def haversine_distance_vec(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    r = 6371.0  # km
    return c * r


data["haversine_distance"] = haversine_distance_vec(
    data["pickup_latitude"].values,
    data["pickup_longitude"].values,
    data["dropoff_latitude"].values,
    data["dropoff_longitude"].values,
)

testData["haversine_distance"] = haversine_distance_vec(
    testData["pickup_latitude"].values,
    testData["pickup_longitude"].values,
    testData["dropoff_latitude"].values,
    testData["dropoff_longitude"].values,
)



## === cell 26
data = data[(data["haversine_distance"] > 0.1) & (data["haversine_distance"] < 500)]



## === cell 27
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.regplot(
    x="haversine_distance",
    y="fare_amount",
    data=data,
    fit_reg=True,
    scatter_kws={"alpha": 0.5},
)
plt.title("Correlation between Haversine Distance and Fare Amount")
plt.xlabel("Haversine Distance (km)")
plt.ylabel("Fare Amount ($)")
plt.show()



## === cell 28
plt.figure(figsize=(10, 6))
sns.histplot(
    data.loc[data["haversine_distance"] <= 100, "haversine_distance"], kde=True, bins=50
)
plt.xlim(0, 100)
plt.title("Distribution of Haversine Distance (0 to 100)")
plt.xlabel("Haversine Distance (km)")
plt.ylabel("Count")
plt.show()



## === cell 29
data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
data["is_night_time"] = data["pickup_datetime"].apply(
    lambda x: 1 if (x.hour >= 21 or x.hour < 6) else 0
)

testData["pickup_datetime"] = pd.to_datetime(testData["pickup_datetime"])
testData["is_night_time"] = testData["pickup_datetime"].apply(
    lambda x: 1 if (x.hour >= 21 or x.hour < 6) else 0
)



## === cell 30
sns.boxplot(x="is_night_time", y="fare_amount", data=data)
plt.title("Fare Amount by Time of Day")
plt.xlabel("Is Night Time (0 = No, 1 = Yes)")
plt.ylabel("Fare Amount ($)")
plt.show()



## === cell 31
data["year"] = data["pickup_datetime"].dt.year
data["month"] = data["pickup_datetime"].dt.month
data["hour_of_day"] = data["pickup_datetime"].dt.hour

testData["year"] = testData["pickup_datetime"].dt.year
testData["month"] = testData["pickup_datetime"].dt.month
testData["hour_of_day"] = testData["pickup_datetime"].dt.hour



## === cell 32
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.lineplot(x="year", y="fare_amount", data=data, estimator="mean", errorbar=None)
plt.title("Average Fare Amount by Year")
plt.xlabel("Year")
plt.ylabel("Average Fare Amount ($)")
plt.show()



## === cell 33
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.lineplot(x="month", y="fare_amount", data=data, estimator="mean", errorbar=None)
plt.title("Average Fare Amount by Month")
plt.xlabel("Month")
plt.ylabel("Average Fare Amount ($)")
plt.show()



## === cell 34
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.lineplot(
    x="hour_of_day", y="fare_amount", data=data, estimator="mean", errorbar=None
)
plt.title("Average Fare Amount by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Fare Amount ($)")
plt.show()



## === cell 35
airports = [(-73.7789, 40.6413), (-73.8740, 40.7769), (-74.1811, 40.6925)]  # (lon, lat)


def near_airport_vec(lat, lon, airports_lon_lat, threshold_km=5.0):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    near = np.zeros(lat.shape[0], dtype=bool)
    for ap_lon, ap_lat in airports_lon_lat:
        d = haversine_distance_vec(lat, lon, ap_lat, ap_lon)
        near |= d < threshold_km
    return near.astype(int)


data["pickup_near_airport"] = near_airport_vec(
    data["pickup_latitude"].values, data["pickup_longitude"].values, airports
)
data["dropoff_near_airport"] = near_airport_vec(
    data["dropoff_latitude"].values, data["dropoff_longitude"].values, airports
)

testData["pickup_near_airport"] = near_airport_vec(
    testData["pickup_latitude"].values, testData["pickup_longitude"].values, airports
)
testData["dropoff_near_airport"] = near_airport_vec(
    testData["dropoff_latitude"].values, testData["dropoff_longitude"].values, airports
)



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/803130948.py in <cell line: 0>()
     13 
     14 
---> 15 data["pickup_near_airport"] = near_airport_vec(
     16     data["pickup_latitude"].values, data["pickup_longitude"].values, airports
     17 )

/tmp/ipykernel_11/803130948.py in near_airport_vec(lat, lon, airports_lon_lat, threshold_km)
      8     near = np.zeros(lat.shape[0], dtype=bool)
      9     for ap_lon, ap_lat in airports_lon_lat:
---> 10         d = haversine_distance_vec(lat, lon, ap_lat, ap_lon)
     11         near |= d < threshold_km
     12     return near.astype(int)

/tmp/ipykernel_11/3345823197.py in haversine_distance_vec(lat1, lon1, lat2, lon2)
      3     lat1 = np.radians(lat1.astype(float))
      4     lon1 = np.radians(lon1.astype(float))
----> 5     lat2 = np.radians(lat2.astype(float))
      6     lon2 = np.radians(lon2.astype(float))
      7 

AttributeError: 'float' object has no attribute 'astype'

## === cell 36
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.violinplot(x="pickup_near_airport", y="fare_amount", data=data, inner="quartile")
plt.title("Fare Amount by Pickup Proximity to Airport - Violin Plot")
plt.xlabel("Pickup Near Airport (0 = No, 1 = Yes)")
plt.ylabel("Fare Amount ($)")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x="dropoff_near_airport", y="fare_amount", data=data, inner="quartile")
plt.title("Fare Amount by Dropoff Proximity to Airport - Violin Plot")
plt.xlabel("Dropoff Near Airport (0 = No, 1 = Yes)")
plt.ylabel("Fare Amount ($)")
plt.show()



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2903351835.py in <cell line: 0>()
      3 
      4 plt.figure(figsize=(10, 6))
----> 5 sns.violinplot(x="pickup_near_airport", y="fare_amount", data=data, inner="quartile")
      6 plt.title("Fare Amount by Pickup Proximity to Airport - Violin Plot")
      7 plt.xlabel("Pickup Near Airport (0 = No, 1 = Yes)")

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in violinplot(data, x, y, hue, order, hue_order, bw, cut, scale, scale_hue, gridsize, width, inner, split, dodge, orient, linewidth, color, palette, saturation, ax, **kwargs)
   2303 ):
   2304 
-> 2305     plotter = _ViolinPlotter(x, y, hue, data, order, hue_order,
   2306                              bw, cut, scale, scale_hue, gridsize,
   2307                              width, inner, split, dodge, orient, linewidth,

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in __init__(self, x, y, hue, data, order, hue_order, bw, cut, scale, scale_hue, gridsize, width, inner, split, dodge, orient, linewidth, color, palette, saturation)
    899                  color, palette, saturation):
    900 
--> 901         self.establish_variables(x, y, hue, data, orient, order, hue_order)
    902         self.establish_colors(color, palette, saturation)
    903         self.estimate_densities(bw, cut, scale, scale_hue, gridsize)

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in establish_variables(self, x, y, hue, data, orient, order, hue_order, units)
    539                 if isinstance(var, str):
    540                     err = f"Could not interpret input '{var}'"
--> 541                     raise ValueError(err)
    542 
    543             # Figure out the plotting orientation

ValueError: Could not interpret input 'pickup_near_airport'

## === cell 37
public_holidays = [
    (1, 1),
    (1, 15),
    (2, 12),
    (2, 19),
    (5, 27),
    (6, 19),
    (7, 4),
    (9, 2),
    (10, 14),
    (11, 5),
    (11, 11),
    (11, 28),
    (12, 25),
]


def is_public_holiday(date):
    return (date.month, date.day) in public_holidays


data["is_public_holiday"] = data["pickup_datetime"].apply(is_public_holiday).astype(int)
testData["is_public_holiday"] = (
    testData["pickup_datetime"].apply(is_public_holiday).astype(int)
)



## === cell 38
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.boxplot(x="is_public_holiday", y="fare_amount", data=data)
plt.title("Fare Amount on Public Holidays vs. Regular Days")
plt.xlabel("Is Public Holiday (0 = No, 1 = Yes)")
plt.ylabel("Fare Amount ($)")
plt.show()



## === cell 39
public_holiday_data = data[data["is_public_holiday"] == 1]

plt.figure(figsize=(10, 6))
sns.histplot(public_holiday_data["fare_amount"], kde=True)
plt.title("Distribution of Fare Amount on Public Holidays")
plt.xlabel("Fare Amount ($)")
plt.ylabel("Frequency")
plt.show()



## === cell 40
features = [
    "passenger_count",
    "haversine_distance",
    "pickup_near_airport",
    "dropoff_near_airport",
    "hour_of_day",
    "year",
    "month",
    "is_public_holiday",
]



## === cell 41
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = data[features]
y = data["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/748947601.py in <cell line: 0>()
      2 from sklearn.linear_model import LinearRegression
      3 
----> 4 X = data[features]
      5 y = data["fare_amount"]
      6 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['pickup_near_airport', 'dropoff_near_airport'] not in index"

## === cell 42
from sklearn.metrics import mean_squared_error

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

XTEST = testData[features].values
y_pred_final = linear_model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred_final, squared=False)
print("RMSE: ", rmse)



## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/230925111.py in <cell line: 0>()
      2 
      3 linear_model = LinearRegression()
----> 4 linear_model.fit(X_train, y_train)
      5 
      6 XTEST = testData[features].values

NameError: name 'X_train' is not defined

## === cell 43
from sklearn.ensemble import RandomForestRegressor

random_forest_model = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1
)

random_forest_model.fit(X_train, y_train)

y_pred_rf = random_forest_model.predict(X_test)

rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
print("Random Forest RMSE: ", rmse_rf)

y_pred_rf_test = random_forest_model.predict(XTEST)
y_pred_rf_test = np.clip(y_pred_rf_test, 0.0, None)

submission_rf = pd.DataFrame(
    {"key": testData["key"], "fare_amount": y_pred_rf_test},
    columns=["key", "fare_amount"],
)

submission_rf.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_rf.shape)

## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2899699402.py in <cell line: 0>()
      5 )
      6 
----> 7 random_forest_model.fit(X_train, y_train)
      8 
      9 y_pred_rf = random_forest_model.predict(X_test)

NameError: name 'X_train' is not defined
