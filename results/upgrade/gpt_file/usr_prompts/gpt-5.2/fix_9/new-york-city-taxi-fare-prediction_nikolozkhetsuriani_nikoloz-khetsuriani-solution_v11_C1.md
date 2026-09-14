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

7.61555

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.98305) has done: 'I fix the airport-proximity feature bug by making the Haversine function accept scalar airport coordinates (broadcasting them to arrays), which unblocks feature creation and prevents downstream KeyErrors. Then I ensure those engineered columns exist before plotting/modeling, and I keep the existing LinearRegression/RandomForest core training logic intact. I also correct the RMSE computation bug (it was predicting on X_test with a variable named y_pred_final built from X_test but mislabeled) and ensure the final submission is written as `submission.csv` with exactly `key,fare_amount` columns. All changes are minimal and only to restore end-to-end execution and produce a valid CSV submission.'
- What this solution (achieved 7.44726) has done: 'I make the smallest changes that are most likely to improve RMSE without changing your core feature set or model choices: (1) avoid dropping any test rows during geographic/distance filtering (dropping rows harms Kaggle scoring because predictions go missing), and instead only filter the training data; (2) replace slow/variable `.apply()` datetime feature extraction with vectorized `.dt` logic to keep semantics identical but more reliable; and (3) ensure the submission rows are in exactly the same order/length as `test.csv` (9914) with no accidental row loss. This should move you meaningfully toward the target RMSE by fixing a major evaluation-impacting issue while preserving your LinearRegression/RandomForest approach. The script still write `submission.csv` with exactly `key,fare_amount`.'
- What this solution (achieved 7.80516) has done: 'Your current RMSE (7.44726) is much worse than the target (4.12819), so we should make the smallest legitimate improvements that reduce error without changing your overall approach (same features, same LinearRegression + RandomForest training/predict flow). The biggest low-risk gain here is to stop training on a tiny, potentially unrepresentative first 100k rows by switching to a larger *random* sample from the full train file (still within time) using `skiprows`, which preserves the same model/feature logic but improves generalization substantially. I also add a minimal sanity filter to drop rows with missing/invalid datetimes (since your time features become NaN otherwise and can silently degrade training), while keeping test rows intact. Everything else (feature engineering, models, submission format, file name) stays the same.'
- What this solution (achieved 8.31775) has done: 'Your current RMSE (7.80516) is much worse than the target (4.12819), so we need small, legitimate changes that improve generalization without changing your overall feature set or model choices. The biggest issue is that `skiprows` with `choice` is effectively unstable and can accidentally include the CSV header as a “data row” (corrupting dtypes) and also costs huge memory/time; switching to a deterministic chunked random sample keeps the same training approach but makes the sampled 1M rows valid and representative. Next, we keep test rows intact but ensure datetime-derived features are not NaN in test (fill with train medians) so the model never sees missing values at prediction time. Finally, we add a minimal, standard cleaning step for obviously invalid coordinates/fare values (only on train) to reduce noise, which typically moves RMSE toward the ~4–5 range with the same LinearRegression/RandomForest core logic.'
- What this solution (achieved 8.00098) has done: 'Your RMSE is far worse than the target, so we should make a small, legitimate improvement that keeps your model/feature logic intact but reduces noise in training. The biggest low-risk issue is that you compute `haversine_distance` in kilometers while NYC taxi fares scale more closely with miles; switching this one feature to miles keeps the same feature and same models, just on a more appropriate scale, which typically improves generalization. I also vectorize the public-holiday feature (same semantics) to avoid slow `.apply()` and ensure no unexpected dtype issues, while keeping the rest of the pipeline unchanged. Everything still runs end-to-end and writes `submission.csv` with exactly `key,fare_amount` and all test rows.'
- What this solution (achieved 7.61555) has done: 'You’re currently far worse than the target (RMSE 8.00 vs 4.13, lower is better), so we should make small, legitimate fixes that reduce label noise without changing your model choices or feature set. The biggest high-impact/low-risk issue is that your training sample still contains many “bad” NYC taxi rows (0,0 coordinates, identical pickup/dropoff, extreme distances) that your current bounds don’t reliably remove; tightening train-only cleaning around coordinates and distance typically drops RMSE a lot while preserving core logic. I also fix a subtle train/test imputation mismatch by using medians from the *final cleaned training frame* (not the earlier pre-filter means) so the model sees consistent distributions. Everything else (features, LinearRegression + RandomForest, training loop, submission format) stays the same and still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
nrows = 1_000_000
chunk_size = 200_000
frac = nrows / 55_423_856  # approximate sampling fraction
rng = np.random.RandomState(42)

usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

chunks = []
kept = 0
for chunk in pd.read_csv(train_path, usecols=usecols, chunksize=chunk_size):
    mask = rng.rand(len(chunk)) < frac
    sub = chunk.loc[mask]
    if len(sub) == 0:
        continue
    chunks.append(sub)
    kept += len(sub)
    if kept >= nrows:
        break

data = pd.concat(chunks, ignore_index=True)
if len(data) > nrows:
    data = data.sample(n=nrows, random_state=42).reset_index(drop=True)

print("Loaded train sample:", data.shape)



## === cell 2
data.head()



## === cell 3
data.shape



## === cell 4
data.info()



## === cell 5
print(data.isnull().sum())



## === cell 6
num_cols = [
    "fare_amount",
    "dropoff_longitude",
    "dropoff_latitude",
    "pickup_longitude",
    "pickup_latitude",
    "passenger_count",
]
for c in num_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")

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
for c in [
    "dropoff_longitude",
    "dropoff_latitude",
    "pickup_longitude",
    "pickup_latitude",
    "passenger_count",
]:
    testData[c] = pd.to_numeric(testData[c], errors="coerce")

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

testData["passenger_count"] = testData["passenger_count"].fillna(
    data["passenger_count"].median()
)



## === cell 14
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(data["fare_amount"], kde=True)
plt.title("Distribution of Fare Amount")
plt.show()



## === cell 15
data.shape



## === cell 16
data = data[(data["fare_amount"] > 0) & (data["fare_amount"] <= 500)].copy()



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
data = data[(data["passenger_count"] >= 1) & (data["passenger_count"] <= 7)].copy()



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
].copy()

testData = testData.copy()



## === cell 24
data.shape




## === cell 25
def haversine_distance_vec(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)

    if lat2.ndim == 0:
        lat2 = np.full_like(lat1, lat2, dtype=float)
    if lon2.ndim == 0:
        lon2 = np.full_like(lon1, lon2, dtype=float)

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    r_miles = 3958.7613  # Earth radius in miles
    return c * r_miles


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
data = data[
    (data["haversine_distance"] >= 0.1) & (data["haversine_distance"] <= 60.0)
].copy()



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
plt.xlabel("Haversine Distance (miles)")
plt.ylabel("Fare Amount ($)")
plt.show()



## === cell 28
plt.figure(figsize=(10, 6))
sns.histplot(
    data.loc[data["haversine_distance"] <= 62.137, "haversine_distance"],
    kde=True,
    bins=50,
)
plt.xlim(0, 62.137)
plt.title("Distribution of Haversine Distance (0 to 62 miles)")
plt.xlabel("Haversine Distance (miles)")
plt.ylabel("Count")
plt.show()



## === cell 29
data["pickup_datetime"] = pd.to_datetime(
    data["pickup_datetime"], errors="coerce", utc=True
)
testData["pickup_datetime"] = pd.to_datetime(
    testData["pickup_datetime"], errors="coerce", utc=True
)

data = data[data["pickup_datetime"].notna()].copy()

data["is_night_time"] = (
    (data["pickup_datetime"].dt.hour >= 21) | (data["pickup_datetime"].dt.hour < 6)
).astype(int)
testData["is_night_time"] = (
    (testData["pickup_datetime"].dt.hour >= 21)
    | (testData["pickup_datetime"].dt.hour < 6)
).astype(int)



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

for c in ["year", "month", "hour_of_day"]:
    testData[c] = testData[c].fillna(int(data[c].median()))



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
    threshold_miles = threshold_km * 0.621371
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    near = np.zeros(lat.shape[0], dtype=bool)
    for ap_lon, ap_lat in airports_lon_lat:
        d = haversine_distance_vec(lat, lon, ap_lat, ap_lon)
        near |= d < threshold_miles
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



## === cell 38
holidays_set = set(public_holidays)

data_md = list(
    zip(data["pickup_datetime"].dt.month.values, data["pickup_datetime"].dt.day.values)
)
data["is_public_holiday"] = (
    pd.Series(data_md, index=data.index).isin(holidays_set).astype(int)
)

test_md = list(
    zip(
        testData["pickup_datetime"].dt.month.values,
        testData["pickup_datetime"].dt.day.values,
    )
)
testData["is_public_holiday"] = (
    pd.Series(test_md, index=testData.index).isin(holidays_set).astype(int)
)



## === cell 39
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.boxplot(x="is_public_holiday", y="fare_amount", data=data)
plt.title("Fare Amount on Public Holidays vs. Regular Days")
plt.xlabel("Is Public Holiday (0 = No, 1 = Yes)")
plt.ylabel("Fare Amount ($)")
plt.show()



## === cell 40
public_holiday_data = data[data["is_public_holiday"] == 1]

plt.figure(figsize=(10, 6))
sns.histplot(public_holiday_data["fare_amount"], kde=True)
plt.title("Distribution of Fare Amount on Public Holidays")
plt.xlabel("Fare Amount ($)")
plt.ylabel("Frequency")
plt.show()



## === cell 41
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



## === cell 42
train_medians = data[
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
].median(numeric_only=True)

for c in [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]:
    if c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(train_medians[c])
    if c in testData.columns:
        testData[c] = pd.to_numeric(testData[c], errors="coerce").fillna(
            train_medians[c]
        )



## === cell 43
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = data[features]
y = data["fare_amount"]

train_ok = X.notna().all(axis=1) & y.notna()
X = X.loc[train_ok]
y = y.loc[train_ok]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 44
from sklearn.metrics import mean_squared_error

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_lr = linear_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred_lr, squared=False)
print("Linear Regression RMSE: ", rmse)

XTEST = testData[features].copy()

for c in features:
    if XTEST[c].isna().any():
        XTEST[c] = XTEST[c].fillna(X_train[c].median())



## === cell 45
from sklearn.ensemble import RandomForestRegressor

random_forest_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_depth=18,
    min_samples_leaf=2,
)

random_forest_model.fit(X_train, y_train)

y_pred_rf = random_forest_model.predict(X_test)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
print("Random Forest RMSE: ", rmse_rf)

y_pred_rf_test = random_forest_model.predict(XTEST)
y_pred_rf_test = np.clip(y_pred_rf_test, 0.0, None)

submission_rf = pd.DataFrame(
    {"key": testData["key"].values, "fare_amount": y_pred_rf_test.astype(float)},
    columns=["key", "fare_amount"],
)

assert (
    submission_rf.shape[0] == testData.shape[0]
), "Submission row count must match test.csv"
assert submission_rf["key"].isna().sum() == 0, "Keys must not be missing"

submission_rf.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_rf.shape)
print(submission_rf.head())
