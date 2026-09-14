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

# 5. Code solution

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

NROWS = 1_000_000
RANDOM_SEED = 42


def read_uniform_sample_csv(
    path: str,
    nrows: int,
    dtypes: dict,
    parse_dates: list[str],
    seed: int,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    Change (score-relevant): switch to true reservoir sampling over the full file.
    Your previous per-chunk sampling heuristic is biased toward early chunks and can
    materially worsen generalization -> higher public RMSE. Reservoir sampling keeps
    the same "train on NROWS" core logic but makes the sample distribution correct.
    """
    rng = np.random.default_rng(seed)

    reservoir = None
    filled = 0
    seen = 0

    for chunk in pd.read_csv(
        path, dtype=dtypes, parse_dates=parse_dates, chunksize=chunksize
    ):
        m = len(chunk)
        if m == 0:
            continue

        if reservoir is None:
            reservoir = chunk.iloc[0:0].copy()

        if filled < nrows:
            take = min(nrows - filled, m)
            if take > 0:
                reservoir = pd.concat([reservoir, chunk.iloc[:take]], ignore_index=True)
                filled += take
            seen += m
            if take == m:
                continue
            chunk = chunk.iloc[take:]
            m = len(chunk)
            if m == 0:
                continue

        start_i = seen + 1
        end_i = seen + m
        i = np.arange(start_i, end_i + 1, dtype=np.int64)
        j = rng.integers(low=0, high=i, size=m, dtype=np.int64)  # in [0, i-1]
        mask = j < nrows
        if np.any(mask):
            replace_pos = j[mask]
            reservoir.iloc[replace_pos] = chunk.loc[mask].to_numpy()

        seen += m

    if reservoir is None:
        return pd.DataFrame()
    return reservoir.reset_index(drop=True)


data = read_uniform_sample_csv(
    TRAIN_PATH,
    nrows=NROWS,
    dtypes=dtypes,
    parse_dates=["pickup_datetime"],
    seed=RANDOM_SEED,
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

test_keys = testData["key"].copy()



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
data = data[
    data["pickup_latitude"].between(-90, 90)
    & data["dropoff_latitude"].between(-90, 90)
    & data["pickup_longitude"].between(-180, 180)
    & data["dropoff_longitude"].between(-180, 180)
]

ny_lat_min, ny_lat_max = 40.25, 41.15
ny_lon_min, ny_lon_max = -74.65, -73.35

data = data[
    (data["pickup_latitude"].between(ny_lat_min, ny_lat_max))
    & (data["pickup_longitude"].between(ny_lon_min, ny_lon_max))
    & (data["dropoff_latitude"].between(ny_lat_min, ny_lat_max))
    & (data["dropoff_longitude"].between(ny_lon_min, ny_lon_max))
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

for col, mn, mx, fill in [
    ("pickup_latitude", -90, 90, mean_pickup_latitude),
    ("dropoff_latitude", -90, 90, mean_dropoff_latitude),
    ("pickup_longitude", -180, 180, mean_pickup_longitude),
    ("dropoff_longitude", -180, 180, mean_dropoff_longitude),
]:
    testData[col] = pd.to_numeric(testData[col], errors="coerce")
    bad = (
        (~testData[col].between(mn, mx))
        | (testData[col].abs() < eps)
        | testData[col].isna()
    )
    testData.loc[bad, col] = fill



## === cell 23
data.shape



## === cell 24
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



## === cell 25
same_point = (data["pickup_latitude"] == data["dropoff_latitude"]) & (
    data["pickup_longitude"] == data["dropoff_longitude"]
)
data = data[~same_point]

data = data[(data["haversine_distance"] > 0.02) & (data["haversine_distance"] < 200)]

fare_per_km = data["fare_amount"] / (data["haversine_distance"] + 1e-3)

data = data[fare_per_km.between(0.5, 40.0)]

data = data[~((data["haversine_distance"] > 10) & (data["fare_amount"] < 4))]

data = data[data["fare_amount"] <= (35.0 * data["haversine_distance"] + 25.0)]



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
data["pickup_datetime"] = pd.to_datetime(
    data["pickup_datetime"], errors="coerce", utc=True
)
testData["pickup_datetime"] = pd.to_datetime(
    testData["pickup_datetime"], errors="coerce", utc=True
)

data = data[data["pickup_datetime"].dt.year.between(2009, 2016)]

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



## === cell 29
sns.boxplot(x="is_night_time", y="fare_amount", data=data)
plt.title("Fare Amount by Time of Day")
plt.xlabel("Is Night Time (0 = No, 1 = Yes)")
plt.ylabel("Fare Amount ($)")
plt.show()



## === cell 30
data["year"] = data["pickup_datetime"].dt.year
data["month"] = data["pickup_datetime"].dt.month
data["hour_of_day"] = data["pickup_datetime"].dt.hour

testData["year"] = testData["pickup_datetime"].dt.year
testData["month"] = testData["pickup_datetime"].dt.month
testData["hour_of_day"] = testData["pickup_datetime"].dt.hour



## === cell 31
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.lineplot(x="year", y="fare_amount", data=data, estimator="mean", errorbar=None)
plt.title("Average Fare Amount by Year")
plt.xlabel("Year")
plt.ylabel("Average Fare Amount ($)")
plt.show()



## === cell 32
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.lineplot(x="month", y="fare_amount", data=data, estimator="mean", errorbar=None)
plt.title("Average Fare Amount by Month")
plt.xlabel("Month")
plt.ylabel("Average Fare Amount ($)")
plt.show()



## === cell 33
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.lineplot(
    x="hour_of_day", y="fare_amount", data=data, estimator="mean", errorbar=None
)
plt.title("Average Fare Amount by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Fare Amount ($)")
plt.show()




## === cell 34
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
plt.ylabel("Fare Amount")
plt.show()



## === cell 38
public_holiday_data = data[data["is_public_holiday"] == 1]

plt.figure(figsize=(10, 6))
sns.histplot(public_holiday_data["fare_amount"], kde=True)
plt.title("Distribution of Fare Amount on Public Holidays")
plt.xlabel("Fare Amount ($)")
plt.ylabel("Frequency")
plt.show()



## === cell 39
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



## === cell 40
train_feature_medians = data[features].median(numeric_only=True)
data[features] = data[features].apply(pd.to_numeric, errors="coerce")
testData[features] = testData[features].apply(pd.to_numeric, errors="coerce")

data[features] = data[features].fillna(train_feature_medians)
testData[features] = testData[features].fillna(train_feature_medians)



## === cell 41
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = data[features]
y = data["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 42
from sklearn.metrics import mean_squared_error

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

XTEST = testData[features]
y_pred_linear = linear_model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred_linear, squared=False)
print("RMSE: ", rmse)



## === cell 43
from sklearn.ensemble import RandomForestRegressor

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=3,
    max_features=0.7,
)

random_forest_model.fit(X_train, y_train)

y_pred_rf = random_forest_model.predict(X_test)

rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
print("Random Forest RMSE: ", rmse_rf)

y_pred_rf_test = random_forest_model.predict(XTEST)
y_pred_rf_test = np.clip(y_pred_rf_test, 0, None)

submission_rf = pd.DataFrame(
    {"key": test_keys.values, "fare_amount": y_pred_rf_test},
    columns=["key", "fare_amount"],
)

submission_rf.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_rf.shape)
print(submission_rf.head())
