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

5.689

# 6. Current score

15.35702

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.94064) has done: 'I fix the runtime error caused by the deprecated `normalize` argument in `LinearRegression` (removed in scikit-learn 1.2) by replacing it with an equivalent `Pipeline(StandardScaler, LinearRegression)` so the training logic remains linear regression but runs correctly. I also make the one-hot encoding robust by aligning train/test dummy columns to prevent feature mismatch if any weekday category is missing in either split. Finally, I ensure the submission file is written with the required header and columns (`key,fare_amount`) and a `.csv` suffix to the working directory.'
- What this solution (achieved 750.77933) has done: 'Your RMSE is exploding because the model is being trained on many invalid/outlier target rows (negative/huge fares, bad passenger counts) and because the “standardization” of `Difference_longitude/latitude` incorrectly uses the test-set mean/variance, creating a train/test distribution mismatch. I make two minimal, score-relevant fixes: (1) filter clearly invalid training rows using simple, standard NYC taxi-fare constraints; (2) compute the mean/variance for those two engineered features on the training data and apply the same parameters to the test data. Everything else (feature set, LinearRegression via Pipeline(StandardScaler, LinearRegression), and submission format) stays the same, but these fixes should move RMSE drastically down toward the 5.689 target.'
- What this solution (achieved 866.93631) has done: 'Your current RMSE is still extremely high for this competition, which usually indicates remaining invalid rows/features causing the linear model to extrapolate wildly. I make two minimal, score-relevant fixes without changing your model or feature set: (1) enforce standard NYC coordinate bounds and a sane maximum trip distance to remove GPS glitches that dominate least-squares; (2) clip negative predictions to 0 (fares can’t be negative), which typically reduces RMSE substantially when a linear model produces invalid negatives. Everything else (feature engineering, Pipeline(StandardScaler, LinearRegression), train/test handling, and submission schema) stays the same and it still write `Submission.csv`.'
- What this solution (achieved 988.91786) has done: 'Your RMSE is still massive for this competition, which strongly suggests that the linear model is being distorted by a remaining pocket of bad rows and by a feature-scale bug. I make two minimal, score-relevant fixes while keeping the same feature set and the same LinearRegression-in-a-StandardScaler pipeline: (1) change your “variance” normalization to use standard deviation (you currently divide by variance, which squashes the feature by ~1/std and harms generalization), and (2) filter a few more clearly invalid rows (zero-distance rides and unrealistic speed) that least-squares is very sensitive to. Everything else stays the same, and it still writes a valid `Submission.csv` with `key,fare_amount`.'
- What this solution (achieved 15.35418) has done: 'Your RMSE is still far from the 5.689 target (lower is better), and the biggest remaining issue is that the linear regression is being fit with plain least-squares, which is extremely sensitive to the heavy-tailed/label-noise remainder even after filtering—this typically yields wild coefficients and huge test errors. To move the score sharply down toward the target without changing the “linear regression” core approach, I keep the same Pipeline(StandardScaler → linear linear-model) but switch the estimator to `HuberRegressor`, which is still a linear regression model yet robust to outliers. I also make one minimal, metric-aligned post-processing fix: clip predictions to a realistic upper bound (250, matching your training filter) to prevent a handful of extreme predictions from dominating RMSE. Everything else (features, training loop, submission schema/path) stays the same and it still writes `Submission.csv`.'
- What this solution (achieved 15.35665) has done: 'Your current RMSE (15.35) is far above the 5.689 target (lower is better), and the most likely cause is that the robust linear model is still being distorted by a remaining set of bad training rows (GPS glitches / implausible routes) and by unhandled invalid values in the test features. I keep your exact feature set and the same Pipeline(StandardScaler → HuberRegressor), but add a minimal, standard NYC-taxi cleaning step: remove rides with clearly impossible “fare per mile” and airport-distance inconsistencies that least-squares/Huber still struggles with. I also ensure we drop/replace any inf/NaN created during feature engineering in both train and test (without changing the feature logic), so the model doesn’t extrapolate wildly on a few corrupted rows. Finally, I keep your submission schema and file name unchanged while adding a key-order sanity check to avoid accidental misalignment.'
- What this solution (achieved 15.35665) has done: 'To move your RMSE down toward the 5.689 target without changing the model type or feature set, the highest-impact minimal fix is to make the HuberRegressor actually converge to a robust solution; with its defaults it often stops too early on this dataset and behaves like a poorly-fit linear model. I keep the exact same Pipeline(StandardScaler → HuberRegressor) and same engineered features/cleaning, but set a higher `max_iter` and a stricter `tol`, plus fix `epsilon` to a standard robust value to stabilize the fit (still Huber loss, same semantics). I also ensure train/test columns are aligned after all filtering/feature engineering (to avoid any silent column-order/category issues impacting predictions). The submission writing remains identical (`Submission.csv` with `key,fare_amount`).'
- What this solution (achieved 15.35702) has done: 'Your RMSE is far above the 5.689 target (lower is better), so we should make the smallest changes that legitimately improve generalization without changing the model family or feature set. The biggest remaining score killer here is that `HuberRegressor.score()` reports R² (not RMSE) and, more importantly, the Huber solver is sensitive to feature scale; your `Distance`/airport distances are in miles (0–60) while standardized diffs are z-scaled, and rounding those distance features to 2 decimals throws away signal the linear model needs. I keep the same engineered features and the same `Pipeline(StandardScaler → HuberRegressor)`, but remove the distance rounding (keep full float precision), and add a minimal, metric-aligned training-target cleanup by removing unrealistically high $/mile outliers more tightly (still the same idea you already use, just less permissive). Finally, I compute and print a proper validation RMSE (for debugging only) while keeping submission generation identical.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
td = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=10_000_000
)
td.head()  # td:train data # ted:test data



## === cell 2
td.shape



## === cell 3
td.info()



## === cell 4
ted = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
ted.head()



## === cell 5
ted.info()



## === cell 6
td.isna().sum()



## === cell 7
td["Difference_longitude"] = np.abs(
    np.asarray(td["pickup_longitude"] - td["dropoff_longitude"])
)
td["Difference_latitude"] = np.abs(
    np.asarray(td["pickup_latitude"] - td["dropoff_latitude"])
)

ted["Difference_longitude"] = np.abs(
    np.asarray(ted["pickup_longitude"] - ted["dropoff_longitude"])
)
ted["Difference_latitude"] = np.abs(
    np.asarray(ted["pickup_latitude"] - ted["dropoff_latitude"])
)



## === cell 8
print(f"Before Dropping null values: {len(td)}")
td.dropna(inplace=True)
print(f"After Dropping null values: {len(td)}")



## === cell 9
try:
    plot = td[:2000].plot.scatter("Difference_longitude", "Difference_latitude")
except Exception as _:
    plot = None



## === cell 10
td = td[(td["Difference_longitude"] < 5.0) & (td["Difference_latitude"] < 5.0)]



## === cell 11
td = td[
    (td["fare_amount"] > 0)  # non-negative fares
    & (td["fare_amount"] < 250)  # remove extreme outliers
    & (td["passenger_count"] >= 1)  # valid passenger counts
    & (td["passenger_count"] <= 6)
].copy()



## === cell 12
nyc_bounds = {
    "lon_min": -74.5,
    "lon_max": -72.8,
    "lat_min": 40.0,
    "lat_max": 41.8,
}

td = td[
    (td["pickup_longitude"].between(nyc_bounds["lon_min"], nyc_bounds["lon_max"]))
    & (td["dropoff_longitude"].between(nyc_bounds["lon_min"], nyc_bounds["lon_max"]))
    & (td["pickup_latitude"].between(nyc_bounds["lat_min"], nyc_bounds["lat_max"]))
    & (td["dropoff_latitude"].between(nyc_bounds["lat_min"], nyc_bounds["lat_max"]))
].copy()

td.shape



## === cell 13
l1 = list(td["pickup_datetime"])
for i in range(len(l1)):
    l1[i] = l1[i][11:-7:]
td["pickuptime"] = l1

l1 = list(ted["pickup_datetime"])
for i in range(len(l1)):
    l1[i] = l1[i][11:-7:]
ted["pickuptime"] = l1



## === cell 14
td.head()



## === cell 15
l1 = list(td["pickup_datetime"])
for i in range(len(l1)):
    l1[i] = l1[i][:-4:]
    l1[i] = pd.Timestamp(l1[i])
    l1[i] = l1[i].weekday()
td["Weekday"] = l1

l1 = list(ted["pickup_datetime"])
for i in range(len(l1)):
    l1[i] = l1[i][:-4:]
    l1[i] = pd.Timestamp(l1[i])
    l1[i] = l1[i].weekday()
ted["Weekday"] = l1

td.head()



## === cell 16
ted.head()



## === cell 17
td.drop("pickup_datetime", inplace=True, axis=1)
ted.drop("pickup_datetime", inplace=True, axis=1)

td["Weekday"].replace(
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
ted["Weekday"].replace(
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

th = pd.get_dummies(td["Weekday"])
teh = pd.get_dummies(ted["Weekday"])

th, teh = th.align(teh, join="outer", axis=1, fill_value=0)

td = pd.concat([td, th], axis=1)
ted = pd.concat([ted, teh], axis=1)

td.drop("Weekday", axis=1, inplace=True)
ted.drop("Weekday", axis=1, inplace=True)

l1 = list(td["pickuptime"])
for i in range(len(l1)):
    z = l1[i].split(":")
    l1[i] = int(z[0]) * 100 + int(z[1])
td["pickuptime"] = l1

l1 = list(ted["pickuptime"])
for i in range(len(l1)):
    z = l1[i].split(":")
    l1[i] = int(z[0]) * 100 + int(z[1])
ted["pickuptime"] = l1

td.head()



## === cell 18
R = 6373.0
lat1 = np.asarray(np.radians(td["pickup_latitude"]))
lon1 = np.asarray(np.radians(td["pickup_longitude"]))
lat2 = np.asarray(np.radians(td["dropoff_latitude"]))
lon2 = np.asarray(np.radians(td["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
td["Distance"] = np.asarray(distance) * 0.621

lat1 = np.asarray(np.radians(ted["pickup_latitude"]))
lon1 = np.asarray(np.radians(ted["pickup_longitude"]))
lat2 = np.asarray(np.radians(ted["dropoff_latitude"]))
lon2 = np.asarray(np.radians(ted["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
ted["Distance"] = np.asarray(distance) * 0.621



## === cell 19
td = td[td["Distance"].between(0.05, 100)].copy()
td.shape



## === cell 20
R = 6373.0
lat1 = np.asarray(np.radians(td["pickup_latitude"]))
lon1 = np.asarray(np.radians(td["pickup_longitude"]))
lat2 = np.asarray(np.radians(td["dropoff_latitude"]))
lon2 = np.asarray(np.radians(td["dropoff_longitude"]))

lat3 = np.zeros(len(td)) + np.radians(40.6413111)
lon3 = np.zeros(len(td)) + np.radians(-73.7781391)
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
td["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
td["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(ted["pickup_latitude"]))
lon1 = np.asarray(np.radians(ted["pickup_longitude"]))
lat2 = np.asarray(np.radians(ted["dropoff_latitude"]))
lon2 = np.asarray(np.radians(ted["dropoff_longitude"]))

lat3 = np.zeros(len(ted)) + np.radians(40.6413111)
lon3 = np.zeros(len(ted)) + np.radians(-73.7781391)
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
ted["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
ted["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

pickup_minutes = (td["pickuptime"] // 100) * 60 + (td["pickuptime"] % 100)
valid_time = pickup_minutes.between(0, 24 * 60 - 1)
td = td[valid_time].copy()

td = td[td["Distance"] <= 60].copy()

fare_per_mile = td["fare_amount"] / np.maximum(td["Distance"], 0.1)
td = td[fare_per_mile.between(1.0, 30.0)].copy()

far_from_jfk = (td["Pickup_Distance_airport"] > 25) & (
    td["Dropoff_Distance_airport"] > 25
)
near_jfk = (td["Pickup_Distance_airport"] < 2) & (td["Dropoff_Distance_airport"] < 2)
td = td[~(far_from_jfk & (td["fare_amount"] > 150))].copy()
td = td[~(near_jfk & (td["fare_amount"] > 150))].copy()


td.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
ted.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)

dl_mean = float(np.mean(td["Difference_longitude"]))
dl_std = float(np.std(td["Difference_longitude"]))
if dl_std == 0.0:
    dl_std = 1.0

dlat_mean = float(np.mean(td["Difference_latitude"]))
dlat_std = float(np.std(td["Difference_latitude"]))
if dlat_std == 0.0:
    dlat_std = 1.0

td["Difference_longitude"] = np.abs(td["Difference_longitude"] - dl_mean) / dl_std
td["Difference_latitude"] = np.abs(td["Difference_latitude"] - dlat_mean) / dlat_std

ted["Difference_longitude"] = np.abs(ted["Difference_longitude"] - dl_mean) / dl_std
ted["Difference_latitude"] = np.abs(ted["Difference_latitude"] - dlat_mean) / dlat_std

for df in (td, ted):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
td.dropna(inplace=True)
ted.fillna(0.0, inplace=True)

feature_cols = [c for c in td.columns if c not in ["key", "fare_amount"]]
ted = ted[["key"] + feature_cols].copy()

td.shape



## === cell 21
ted.shape



## === cell 22
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error

X = td.drop(["key", "fare_amount"], axis=1)
y = td["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)

lr = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", HuberRegressor(epsilon=1.35, max_iter=1000, tol=1e-5)),
    ]
)

lr.fit(X_train, y_train)

val_pred = lr.predict(X_test)
val_pred = np.clip(val_pred, 0, 250)
rmse = float(np.sqrt(mean_squared_error(y_test, val_pred)))
print("Validation RMSE (lower is better):", rmse)



## === cell 23
pred = lr.predict(ted.drop("key", axis=1))

pred = np.clip(pred, 0, 250)
pred = np.round(pred, 2)

pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 24
Submission = pd.DataFrame({"key": ted["key"].values, "fare_amount": pred})
Submission.to_csv("Submission.csv", index=False)

print(Submission.head())
print("Wrote submission to:", os.path.abspath("Submission.csv"))
print("Submission rows:", len(Submission), " Test rows:", len(ted))
