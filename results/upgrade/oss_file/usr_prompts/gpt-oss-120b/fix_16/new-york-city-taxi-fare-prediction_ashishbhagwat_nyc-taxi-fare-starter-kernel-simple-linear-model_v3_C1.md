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

3.10

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

6.54475

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'I fixed the `LinearRegression` initialization by removing the deprecated `normalize` argument, aligned the test‑set features with the training columns (adding missing dummy columns and reordering), and updated the prediction step to use this aligned feature frame. These changes resolve the runtime errors and ensure a correctly formatted submission CSV is produced, while keeping the original modeling approach unchanged.'
- What this solution (achieved 26.64384) has done: 'We slightly adjust the ridge regularisation (α = 0.5) and clip predictions to a realistic range (0‑200 $) to bring the RMSE closer to the target without altering the overall modelling pipeline. These minimal tweaks keep the core logic unchanged while improving calibration and preventing extreme out‑liers that hurt the score.'
- What this solution (achieved 12.66637) has done: 'I remove the unnecessary log‑transform of the target (training and prediction use the original `fare_amount`), and increase the ridge regularization slightly (α = 5) to lower variance. These minimal tweaks keep the overall pipeline unchanged while expected to reduce the RMSE and move the score closer to the target.'
- What this solution (achieved 6.44246) has done: 'I add a second‑order polynomial feature expansion before scaling and set the Ridge regularisation to a modest α=1.0. This keeps the linear‑model core unchanged while giving the model more expressive power, which should reduce the RMSE and move the score closer to the target 5.6891.'
- What this solution (achieved 6.438) has done: 'I increase the ridge regularisation strength slightly (α = 5.0) which is a minimal change to the existing linear model and is expected to reduce variance on the validation split, moving the RMSE closer to the target 5.6891 while keeping the overall pipeline unchanged.'
- What this solution (achieved 6.44185) has done: 'I add a simple “pickup_hour” feature extracted from the existing `pickuptime` column (hour = HHMM // 100) for both train and test data, which gives the model a clearer sense of time‑of‑day effects. Then I lower the Ridge regularisation strength from α = 5.0 to α = 2.0, a modest change expected to reduce the validation RMSE and move the score closer to the target while preserving the overall pipeline.'
- What this solution (achieved 6.44343) has done: 'I slightly reduce the Ridge regularisation (α = 1.0) which should let the model fit the data a bit better and move the RMSE closer to the target. Additionally, I output the submission file with the required columns (no index) to ensure a valid Kaggle‑compatible CSV.'
- What this solution (achieved 6.53083) has done: 'I add one‑hot encoding for the `pickup_hour` feature to give the linear model richer categorical information, and increase the Ridge regularisation strength slightly (α = 7.0) to reduce variance. These minimal adjustments keep the overall pipeline unchanged while aiming to lower the RMSE toward the target.'
- What this solution (achieved 6.54475) has done: 'I added the missing imports, corrected the workflow so that all DataFrames are defined before they are used, switched the model to train on the original target (removing the log‑transform) and lowered the Ridge regularisation slightly (α = 0.5) to improve RMSE while keeping the core feature‑engineering unchanged. The script now runs end‑to‑end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 12.66516) has done: 'I reduce model complexity by using only linear (degree‑1) polynomial features and increase the Ridge regularisation strength slightly (α = 2.0). This keeps the overall pipeline identical while expected to lower over‑fitting on the validation split, moving the RMSE closer to the target 5.6891.'
- What this solution (achieved 6.54475) has done: 'I lower the ridge regularisation (α = 0.5) and enable second‑order polynomial features (degree = 2). Both tweaks keep the linear‑model pipeline unchanged while giving the model a bit more flexibility and reducing bias, which should lower the validation RMSE and move the score closer to the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error




## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=10_000_000)




## === cell 2
test_df = pd.read_csv("../input/test.csv")




## === cell 3
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()




## === cell 4
add_travel_vector_features(train_df)
add_travel_vector_features(test_df)




## === cell 5
print(f"Before Dropping null values: {len(train_df)}")
train_df.dropna(inplace=True)
print(f"After Dropping null values: {len(train_df)}")




## === cell 6
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]




## === cell 7
def creating_time(df):
    ls1 = list(df["pickup_datetime"])
    for i in range(len(ls1)):
        ls1[i] = ls1[i][11:-7:]  # extract HH:MM:SS
    df["pickuptime"] = ls1


creating_time(train_df)
creating_time(test_df)




## === cell 8
def creating_weekdays(df):
    ls1 = list(df["pickup_datetime"])
    for i in range(len(ls1)):
        ls1[i] = ls1[i][:-4:]  # strip seconds
        ls1[i] = pd.Timestamp(ls1[i])
        ls1[i] = ls1[i].weekday()
    df["Weekday"] = ls1


creating_weekdays(train_df)
creating_weekdays(test_df)




## === cell 9
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)




## === cell 10
def replace_weekday(df):
    df["Weekday"].replace(
        to_replace=list(range(7)),
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




## === cell 11
train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_df["Weekday"])
train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)




## === cell 12
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)




## === cell 13
def creating_pickupdate(df):
    ls1 = list(df["pickuptime"])
    for i in range(len(ls1)):
        z = ls1[i].split(":")
        ls1[i] = int(z[0]) * 100 + int(z[1])
    df["pickuptime"] = ls1
    df["pickup_hour"] = df["pickuptime"] // 100


creating_pickupdate(train_df)
creating_pickupdate(test_df)




## === cell 14
def finding_distance(df):
    R = 6373.0
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    df["Distance"] = distance * 0.621  # miles


finding_distance(train_df)
finding_distance(test_df)




## === cell 15
def creating_pickup_dropoff_distance(df):
    R = 6373.0
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])

    lat3 = np.radians(40.6413111)
    lon3 = np.radians(-73.7781391)

    dlon_pickup = lon3 - lon1
    dlat_pickup = lat3 - lat1
    a1 = (
        np.sin(dlat_pickup / 2) ** 2
        + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
    )
    c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
    df["Pickup_Distance_airport"] = R * c1 * 0.621

    dlon_dropoff = lon3 - lon2
    dlat_dropoff = lat3 - lat2
    a2 = (
        np.sin(dlat_dropoff / 2) ** 2
        + np.cos(lat2) * np.cos(lat3) * np.sin(dlon_dropoff / 2) ** 2
    )
    c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
    df["Dropoff_Distance_airport"] = R * c2 * 0.621


creating_pickup_dropoff_distance(train_df)
creating_pickup_dropoff_distance(test_df)




## === cell 16
for col in ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]:
    train_df[col] = np.round(train_df[col], 2)
    test_df[col] = np.round(test_df[col], 2)




## === cell 17
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




## === cell 18
for df in (train_df, test_df):
    df["abs_diff_longitude"] = np.abs(
        df["abs_diff_longitude"] - df["abs_diff_longitude"].mean()
    )
    df["abs_diff_longitude"] = df["abs_diff_longitude"] / df["abs_diff_longitude"].var()
    df["abs_diff_latitude"] = np.abs(
        df["abs_diff_latitude"] - df["abs_diff_latitude"].mean()
    )
    df["abs_diff_latitude"] = df["abs_diff_latitude"] / df["abs_diff_latitude"].var()




## === cell 19
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)




## === cell 20
hour_dummies_train = pd.get_dummies(train_df["pickup_hour"], prefix="hour")
hour_dummies_test = pd.get_dummies(test_df["pickup_hour"], prefix="hour")
train_df = pd.concat([train_df, hour_dummies_train], axis=1)
test_df = pd.concat([test_df, hour_dummies_test], axis=1)
train_df.drop("pickup_hour", axis=1, inplace=True)
test_df.drop("pickup_hour", axis=1, inplace=True)

train_df = train_df.dropna(subset=["fare_amount"])
train_df = train_df[train_df["fare_amount"] <= 200]

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]

mask = ~np.isnan(y)
X = X[mask]
y = y[mask]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=80)

test_features = test_df.drop("key", axis=1)
missing_cols = set(X.columns) - set(test_features.columns)
for col in missing_cols:
    test_features[col] = 0
extra_cols = set(test_features.columns) - set(X.columns)
if extra_cols:
    test_features.drop(columns=extra_cols, inplace=True)
test_features = test_features[X.columns]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_val_poly = poly.transform(X_val)
test_features_poly = poly.transform(test_features)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_val_scaled = scaler.transform(X_val_poly)
test_features_scaled = scaler.transform(test_features_poly)




## === cell 21
ridge = Ridge(alpha=0.5, random_state=80)
ridge.fit(X_train_scaled, y_train)

val_pred = ridge.predict(X_val_scaled)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print("Validation RMSE (original scale):", rmse)




## === cell 22
test_pred = ridge.predict(test_features_scaled)
test_pred = np.clip(test_pred, 0, 200)  # enforce realistic bounds
test_pred = np.round(test_pred, 2)




## === cell 23
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission.head()




## === cell 24
submission.to_csv("submission.csv", index=False)
