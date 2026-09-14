# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
data = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=500_000
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

data["dropoff_longitude"] = data["dropoff_longitude"].fillna(mean_dropoff_longitude)
data["dropoff_latitude"] = data["dropoff_latitude"].fillna(mean_dropoff_latitude)



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
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(data["fare_amount"], kde=True)
plt.title("Distribution of Fare Amount")
plt.show()



## === cell 14
data.shape



## === cell 15
data = data[data["fare_amount"] <= 500]



## === cell 16
data.shape



## === cell 17
plt.scatter(data["pickup_longitude"], data["fare_amount"])
plt.xlabel("Pickup Longitude")
plt.ylabel("Fare Amount")
plt.show()



## === cell 18
data.shape



## === cell 19
data = data[(data["passenger_count"] >= 1) & (data["passenger_count"] <= 7)]



## === cell 20
data.shape



## === cell 21
data.shape



## === cell 22
ny_lat_min, ny_lat_max = 40.4774, 40.9176
ny_lon_min, ny_lon_max = -74.2591, -73.7004

data = data[
    (data["pickup_latitude"].between(ny_lat_min, ny_lat_max))
    & (data["pickup_longitude"].between(ny_lon_min, ny_lon_max))
    & (data["dropoff_latitude"].between(ny_lat_min, ny_lat_max))
    & (data["dropoff_longitude"].between(ny_lon_min, ny_lon_max))
]



## === cell 23
data.shape




## === cell 24
def haversine_distance_vec(lat1, lon1, lat2, lon2):
    """
    Compute haversine distance (km) for array‑like inputs.
    lat/lon are in degrees.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371.0  # Earth radius in km
    return c * r


data["haversine_distance"] = haversine_distance_vec(
    data["pickup_latitude"].values,
    data["pickup_longitude"].values,
    data["dropoff_latitude"].values,
    data["dropoff_longitude"].values,
)

data["log_haversine_distance"] = np.log1p(data["haversine_distance"])

testData["haversine_distance"] = haversine_distance_vec(
    testData["pickup_latitude"].values,
    testData["pickup_longitude"].values,
    testData["dropoff_latitude"].values,
    testData["dropoff_longitude"].values,
)

testData["log_haversine_distance"] = np.log1p(testData["haversine_distance"])



## === cell 25
data = data[(data["haversine_distance"] > 0.1) & (data["haversine_distance"] < 500)]



## === cell 26
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



## === cell 27
plt.figure(figsize=(10, 6))
sns.histplot(
    data.loc[data["haversine_distance"] <= 100, "haversine_distance"], kde=True, bins=50
)
plt.xlim(0, 100)  # Set the x-axis limit from 0 to 100
plt.title("Distribution of Haversine Distance (0 to 100)")
plt.xlabel("Haversine Distance (km)")
plt.ylabel("Count")
plt.show()



## === cell 28
data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
data["is_night_time"] = data["pickup_datetime"].apply(
    lambda x: 1 if (x.hour >= 21 or x.hour < 6) else 0
)



## === cell 29
sns.boxplot(x="is_night_time", y="fare_amount", data=data)
plt.title("Fare Amount by Time of Day")
plt.xlabel("Is Night Time (0 = No, 1 = Yes)")
plt.ylabel("Fare Amount")
plt.show()



## === cell 30
data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
data["year"] = data["pickup_datetime"].dt.year
data["month"] = data["pickup_datetime"].dt.month
data["hour_of_day"] = data["pickup_datetime"].dt.hour

testData["pickup_datetime"] = pd.to_datetime(testData["pickup_datetime"])
testData["year"] = testData["pickup_datetime"].dt.year
testData["month"] = testData["pickup_datetime"].dt.month
testData["hour_of_day"] = testData["pickup_datetime"].dt.hour

testData["dropoff_longitude"] = testData["dropoff_longitude"].fillna(
    mean_dropoff_longitude
)
testData["dropoff_latitude"] = testData["dropoff_latitude"].fillna(
    mean_dropoff_latitude
)

testData["is_night_time"] = testData["pickup_datetime"].apply(
    lambda x: 1 if (x.hour >= 21 or x.hour < 6) else 0
)



## === cell 31
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.lineplot(x="year", y="fare_amount", data=data, estimator="mean", ci=None)
plt.title("Average Fare Amount by Year")
plt.xlabel("Year")
plt.ylabel("Average Fare Amount ($)")
plt.show()



## === cell 32
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.lineplot(x="month", y="fare_amount", data=data, estimator="mean", ci=None)
plt.title("Average Fare Amount by Month")
plt.xlabel("Month")
plt.ylabel("Average Fare Amount ($)")
plt.show()



## === cell 33
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.lineplot(x="hour_of_day", y="fare_amount", data=data, estimator="mean", ci=None)
plt.title("Average Fare Amount by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Fare Amount ($)")
plt.show()



## === cell 34
airports = [(-73.7789, 40.6413), (-73.8740, 40.7769), (-74.1811, 40.6925)]


def compute_near_airport(df, lat_col, lon_col, airports, threshold_km=5):
    """
    Returns an integer array where 1 indicates the point is within threshold_km
    of any airport in the list.
    """
    lat = df[lat_col].values
    lon = df[lon_col].values
    near = np.zeros(df.shape[0], dtype=int)

    for lon_a, lat_a in airports:
        dist = haversine_distance_vec(
            lat,
            lon,
            np.full(df.shape[0], lat_a),
            np.full(df.shape[0], lon_a),
        )
        near = np.where(dist < threshold_km, 1, near)
    return near


data["pickup_near_airport"] = compute_near_airport(
    data, "pickup_latitude", "pickup_longitude", airports
)
data["dropoff_near_airport"] = compute_near_airport(
    data, "dropoff_latitude", "dropoff_longitude", airports
)

testData["pickup_near_airport"] = compute_near_airport(
    testData, "pickup_latitude", "pickup_longitude", airports
)
testData["dropoff_near_airport"] = compute_near_airport(
    testData, "dropoff_latitude", "dropoff_longitude", airports
)



## === cell 35
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



## === cell 36
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



## === cell 37
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.boxplot(x="is_public_holiday", y="fare_amount", data=data)
plt.title("Fare Amount on Public Holidays vs. Regular Days")
plt.xlabel("Is Public Holiday (0 = No, 1 = Yes)")
plt.ylabel("Fare Amount ($)")
plt.show()



## === cell 38
features = [
    "passenger_count",
    "haversine_distance",
    "log_haversine_distance",
    "pickup_near_airport",
    "dropoff_near_airport",
    "hour_of_day",
    "year",
    "month",
    "is_public_holiday",
    "is_night_time",
]



## === cell 39
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = data[features]
y = data["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)



## === cell 40
from sklearn.metrics import mean_squared_error

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

rmse_linear = mean_squared_error(y_test, y_pred_linear, squared=False)
print("Linear Regression RMSE: ", rmse_linear)



## === cell 41
from sklearn.ensemble import RandomForestRegressor

random_forest_model = RandomForestRegressor(
    n_estimators=200,  # lowered from 500
    max_depth=20,
    random_state=42,
    n_jobs=-1,
)

random_forest_model.fit(X_train, y_train)

y_pred_rf = random_forest_model.predict(X_test)

rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
print("Random Forest RMSE: ", rmse_rf)



## === cell 42
from sklearn.ensemble import GradientBoostingRegressor

gbr_model = GradientBoostingRegressor(
    n_estimators=500,  # lowered from 2000
    learning_rate=0.01,
    max_depth=6,
    random_state=42,
)

gbr_model.fit(X_train, y_train)

y_pred_gbr = gbr_model.predict(X_test)

rmse_gbr = mean_squared_error(y_test, y_pred_gbr, squared=False)
print("Gradient Boosting RMSE: ", rmse_gbr)



## === cell 43
y_pred_avg = (y_pred_rf + y_pred_gbr) / 2
rmse_avg = mean_squared_error(y_test, y_pred_avg, squared=False)
print("Average RF+GBR RMSE: ", rmse_avg)

best_rmse = min(rmse_rf, rmse_gbr, rmse_avg)
if best_rmse == rmse_rf:
    best_model = random_forest_model
    best_name = "Random Forest"
elif best_rmse == rmse_gbr:
    best_model = gbr_model
    best_name = "Gradient Boosting"
else:

    class AvgModel:
        def predict(self, X):
            return (random_forest_model.predict(X) + gbr_model.predict(X)) / 2

    best_model = AvgModel()
    best_name = "Average of RF and GBR"

print(f"Selected model for final prediction: {best_name}")

test_predictions = np.clip(best_model.predict(testData[features]), a_min=0, a_max=None)

submission = pd.DataFrame(
    {"key": testData["key"], "fare_amount": test_predictions},
    columns=["key", "fare_amount"],
)

submission.to_csv("submission.csv", index=False)
