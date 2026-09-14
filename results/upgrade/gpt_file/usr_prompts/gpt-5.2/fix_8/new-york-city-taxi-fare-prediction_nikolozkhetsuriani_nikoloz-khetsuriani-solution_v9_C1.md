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

3.12

# 3. Installed packages

No external packages required in the script and installed.

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

4.330098025036998

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.11554) has done: 'I remove the stray `testData.` line that causes the SyntaxError so the notebook runs end-to-end and always writes `submission.csv`. I also make the feature engineering steps robust by applying the same NaN filling and datetime parsing to both train and test (your current code only fills NaNs for train dropoff coords), which prevents hidden NaNs from breaking model training/prediction and typically improves RMSE. Finally, I keep your model choices and training approach intact, but I use a fixed `random_state` in `train_test_split` to make the validation RMSE reproducible and avoid accidental score instability.'
- What this solution (achieved 8.11554) has done: 'Your current score (8.11554 RMSE) is far worse than the target (4.33), so we should improve generalization with minimal changes while keeping your overall approach (feature engineering + sklearn models) intact. The biggest issue is that several engineered boolean/time features can still contain missing values in both train and test (e.g., `hour_of_day/year/month` if datetime parsing fails), and scikit-learn models don’t handle NaNs—this silently hurts fit quality or can cause inconsistencies. I add a small, consistent “final NaN cleanup” for exactly the `features` columns in both train and test, and I clamp negative predictions to 0 (fares can’t be negative), which typically reduces RMSE without changing model class or training loop. I also fix a metric/evaluation bug where `XTEST` was created but unused for the linear model, and keep the RandomForest submission writing as-is (still `submission.csv`).'
- What this solution (achieved 6.9661) has done: 'To move RMSE down toward your 4.33 target without changing your overall modeling approach, I make two minimal, high-impact fixes: (1) compute Haversine distance and airport proximity with vectorized NumPy instead of per-row `apply` (same feature, but far faster and allows safely using more training rows within the 600s budget), and (2) increase the training sample size from 100k to 1M while keeping the exact same feature set and RandomForest training flow. Using more data typically reduces generalization error substantially for this competition, and the vectorization ensures the notebook still completes in time. I also apply the already-computed `is_night_time` feature consistently to both train and test and include it in `features` (it’s already part of your core feature engineering intent and is a small, legitimate signal), while keeping all existing filters/cleanup and the same RandomForest model/hyperparameters.'
- What this solution (achieved 9.97385) has done: 'To reduce RMSE from 6.97 toward your 4.33 target while keeping your exact feature set and RandomForest approach intact, I (1) add the standard NYC Taxi “cleaning” filters (remove non-positive fares and unreasonable coordinate values) that are currently missing and are known to heavily hurt this competition if left in, (2) slightly widen the NYC bounding box filter you already use (your current one is too tight and likely removes many valid trips, hurting generalization), and (3) make the `train_test_split` deterministic and representative by using a fixed `test_size` (no change to the training loop/model). These are minimal semantic changes that mostly remove obvious label noise/outliers rather than changing the model itself, and they should move your public RMSE down meaningfully without rewriting core logic. The script still run end-to-end and write `submission.csv` with the required columns.'
- What this solution (achieved 10.87062) has done: 'Your RMSE (9.97) is far above the 4.33 target, so we should improve legitimately with the smallest high-impact changes while keeping your same feature engineering + RandomForest training flow. The biggest issue is that the model is learning from many remaining bad/noisy training rows because only basic filters are applied; in this competition, tightening standard NYC cleanup (coordinates, passenger_count, and especially “fare vs distance” sanity) typically yields a large RMSE drop without changing the model or features. I add a few well-known, conservative filters (NYC bbox, distance bounds, passenger bounds, and a mild “fare_per_km” sanity band) and make sure the exact same cleaning-relevant NaN handling is applied before computing distance/time features. The script still trains the same RandomForestRegressor and writes a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 10.87703) has done: 'Your current RMSE (10.87, lower is better) is far above the 4.33 target, so the smallest legitimate way to close the gap without changing the model/training loop is to fix data quality issues that strongly dominate error in this competition. I keep your exact feature set and RandomForestRegressor approach, but (1) read the training sample with explicit dtypes and parse dates at read-time to avoid silent string/object issues, (2) apply the standard NYC cleaning filters more safely (remove NaNs, restrict bbox, remove extreme coordinate/fare/distance inconsistencies) to reduce label noise, and (3) make feature engineering consistent and fully numeric for both train and test before fitting/predicting. These changes typically drop RMSE substantially while preserving your core logic and still writing a valid `submission.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

dtypes = {
    "fare_amount": "float64",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "float64",
}
data = pd.read_csv(
    TRAIN_PATH,
    nrows=1_000_000,
    dtype=dtypes,
    parse_dates=["pickup_datetime"],
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
mean_passenger_count = data["passenger_count"].median()

data["dropoff_longitude"] = data["dropoff_longitude"].fillna(mean_dropoff_longitude)
data["dropoff_latitude"] = data["dropoff_latitude"].fillna(mean_dropoff_latitude)
data["pickup_longitude"] = data["pickup_longitude"].fillna(mean_pickup_longitude)
data["pickup_latitude"] = data["pickup_latitude"].fillna(mean_pickup_latitude)
data["passenger_count"] = data["passenger_count"].fillna(mean_passenger_count)



## === cell 7
print(data.isnull().sum())



## === cell 8
import seaborn as s

s.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap="cool")



## === cell 9
testData = pd.read_csv(
    TEST_PATH,
    dtype={k: v for k, v in dtypes.items() if k != "fare_amount"},
    parse_dates=["pickup_datetime"],
)



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
testData["passenger_count"] = testData["passenger_count"].fillna(mean_passenger_count)



## === cell 14
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(data["fare_amount"], kde=True)
plt.title("Distribution of Fare Amount")
plt.show()



## === cell 15
data.shape



## === cell 16
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(
    subset=[
        "fare_amount",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "pickup_datetime",
    ]
)

data = data[data["fare_amount"] > 0]
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
data["passenger_count"] = np.round(data["passenger_count"]).astype(
    "int64", errors="ignore"
)
data = data[(data["passenger_count"] >= 1) & (data["passenger_count"] <= 6)]

testData["passenger_count"] = np.round(testData["passenger_count"]).astype(
    "int64", errors="ignore"
)
testData["passenger_count"] = testData["passenger_count"].clip(lower=1, upper=6)



## === cell 21
data.shape



## === cell 22
data.shape



## === cell 23
data = data[
    data["pickup_latitude"].between(-90, 90)
    & data["dropoff_latitude"].between(-90, 90)
    & data["pickup_longitude"].between(-180, 180)
    & data["dropoff_longitude"].between(-180, 180)
]

ny_lat_min, ny_lat_max = 40.40, 41.05
ny_lon_min, ny_lon_max = -74.50, -73.50

data = data[
    (data["pickup_latitude"].between(ny_lat_min, ny_lat_max))
    & (data["pickup_longitude"].between(ny_lon_min, ny_lon_max))
    & (data["dropoff_latitude"].between(ny_lat_min, ny_lat_max))
    & (data["dropoff_longitude"].between(ny_lon_min, ny_lon_max))
]

testData = testData[
    testData["pickup_latitude"].between(-90, 90)
    & testData["dropoff_latitude"].between(-90, 90)
    & testData["pickup_longitude"].between(-180, 180)
    & testData["dropoff_longitude"].between(-180, 180)
]
testData = testData[
    (testData["pickup_latitude"].between(ny_lat_min, ny_lat_max))
    & (testData["pickup_longitude"].between(ny_lon_min, ny_lon_max))
    & (testData["dropoff_latitude"].between(ny_lat_min, ny_lat_max))
    & (testData["dropoff_longitude"].between(ny_lon_min, ny_lon_max))
]

eps = 1e-6
data = data[
    ~(
        (data["pickup_latitude"].abs() < eps)
        | (data["pickup_longitude"].abs() < eps)
        | (data["dropoff_latitude"].abs() < eps)
        | (data["dropoff_longitude"].abs() < eps)
    )
]
testData = testData[
    ~(
        (testData["pickup_latitude"].abs() < eps)
        | (testData["pickup_longitude"].abs() < eps)
        | (testData["dropoff_latitude"].abs() < eps)
        | (testData["dropoff_longitude"].abs() < eps)
    )
]



## === cell 24
data.shape



## === cell 25
from math import radians, cos, sin, asin, sqrt


def haversine_distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of the Earth in kilometers
    return c * r


def haversine_distance_vec(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype("float64"))
    lon1 = np.radians(lon1.astype("float64"))
    lat2 = np.radians(lat2.astype("float64"))
    lon2 = np.radians(lon2.astype("float64"))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    r = 6371.0
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
same_point = (data["pickup_latitude"] == data["dropoff_latitude"]) & (
    data["pickup_longitude"] == data["dropoff_longitude"]
)
data = data[~same_point]

data = data[(data["haversine_distance"] > 0.05) & (data["haversine_distance"] < 120)]

fare_per_km = data["fare_amount"] / (data["haversine_distance"] + 1e-3)
data = data[fare_per_km.between(1.0, 50.0)]

data = data[~((data["haversine_distance"] > 10) & (data["fare_amount"] < 5))]

data = data[data["fare_amount"] <= (30.0 * data["haversine_distance"] + 20.0)]



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
plt.xlim(0, 100)  # Set the x-axis limit from 0 to 100
plt.title("Distribution of Haversine Distance (0 to 100)")
plt.xlabel("Haversine Distance (km)")
plt.ylabel("Count")
plt.show()



## === cell 29
data["pickup_datetime"] = pd.to_datetime(
    data["pickup_datetime"], errors="coerce", utc=True
)
testData["pickup_datetime"] = pd.to_datetime(
    testData["pickup_datetime"], errors="coerce", utc=True
)

train_dt_mode = data["pickup_datetime"].dropna().mode()
fill_dt = (
    train_dt_mode.iloc[0]
    if len(train_dt_mode)
    else pd.Timestamp("2010-01-01", tz="UTC")
)
data["pickup_datetime"] = data["pickup_datetime"].fillna(fill_dt)
testData["pickup_datetime"] = testData["pickup_datetime"].fillna(fill_dt)

data_hour = data["pickup_datetime"].dt.hour
test_hour = testData["pickup_datetime"].dt.hour
data["is_night_time"] = ((data_hour >= 21) | (data_hour < 6)).astype(int)
testData["is_night_time"] = ((test_hour >= 21) | (test_hour < 6)).astype(int)



## === cell 30
sns.boxplot(x="is_night_time", y="fare_amount", data=data)
plt.title("Fare Amount by Time of Day")
plt.xlabel("Is Night Time (0 = No, 1 = Yes)")
plt.ylabel("Fare Amount")
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
def near_airport(lat, lon, airports):
    return any(
        haversine_distance(lat, lon, airport[1], airport[0]) < 5 for airport in airports
    )


airports = [(-73.7789, 40.6413), (-73.8740, 40.7769), (-74.1811, 40.6925)]


def near_airport_vec(lat, lon, airports, thresh_km=5.0):
    lat = lat.astype("float64")
    lon = lon.astype("float64")
    near = np.zeros(lat.shape[0], dtype=bool)
    for airport_lon, airport_lat in airports:
        d = haversine_distance_vec(
            lat, lon, np.full_like(lat, airport_lat), np.full_like(lon, airport_lon)
        )
        near |= d < thresh_km
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
plt.ylabel("Fare Amount")
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
    "is_night_time",
]



## === cell 41
train_feature_medians = data[features].median(numeric_only=True)
data[features] = data[features].apply(pd.to_numeric, errors="coerce")
testData[features] = testData[features].apply(pd.to_numeric, errors="coerce")

data[features] = data[features].fillna(train_feature_medians)
testData[features] = testData[features].fillna(train_feature_medians)



## === cell 42
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = data[features]
y = data["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 43
from sklearn.metrics import mean_squared_error

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

XTEST = testData[features]
y_pred_linear = linear_model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred_linear, squared=False)
print("RMSE: ", rmse)



## === cell 44
from sklearn.ensemble import RandomForestRegressor

random_forest_model = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1
)

random_forest_model.fit(X_train, y_train)

y_pred_rf = random_forest_model.predict(X_test)

rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
print("Random Forest RMSE: ", rmse_rf)

y_pred_rf_test = random_forest_model.predict(XTEST)
y_pred_rf_test = np.clip(y_pred_rf_test, 0, None)

submission_rf = pd.DataFrame(
    {"key": testData["key"], "fare_amount": y_pred_rf_test},
    columns=["key", "fare_amount"],
)

submission_rf.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_rf.shape)
print(submission_rf.head())
