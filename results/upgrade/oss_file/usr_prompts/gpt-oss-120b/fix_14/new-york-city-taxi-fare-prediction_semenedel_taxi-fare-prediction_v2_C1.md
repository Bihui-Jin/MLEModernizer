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

5.69073

# 6. Current score

12.95379

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 937.15562) has done: 'I fix the failing LinearRegression initialization by removing the deprecated `normalize` argument, and adjust the submission creation so the CSV is written correctly without an extra index column. These minimal changes resolve the runtime errors and ensure a valid `submission.csv` is produced, allowing the model to be evaluated toward the target RMSE.'
- What this solution (achieved 19593.63828) has done: 'I keep the overall pipeline but remove features that dominate the scale (the raw pick‑up time and the oddly normalised coordinate‑difference columns) before fitting the linear model. By dropping `pickuptime`, `abs_diff_longitude` and `abs_diff_latitude` from the training and test matrices, the regression focuses on distance‑based variables and one‑hot weekdays, which should bring the RMSE much closer to the target (around 5‑6) while preserving the original model type and overall workflow.'
- What this solution (achieved 719.73503) has done: 'I fixed the huge RMSE by correcting the test‑set distance calculation, which previously used an incorrect latitude difference (`lat2 - lon1`). The new implementation computes the haversine distance with the proper `lat2 - lat1` term and rounds it like the training set, preserving the original linear‑regression workflow and all other feature engineering steps.'
- What this solution (achieved 717.84208) has done: 'I tighten the preprocessing to avoid mismatched weekday columns and filter out unrealistic fare amounts, then switch to a regularized linear model (Ridge) which is still a linear regression but more stable. These small changes keep the original workflow while expectedly lowering the RMSE toward the target.'
- What this solution (achieved 717.65875) has done: 'I fix the column‑dropping logic so that the `pickuptime` feature is kept for training, validation, and test scaling. This removes the KeyError caused by trying to scale a missing column and allows the pipeline to produce a valid `submission.csv` with the correct columns.'
- What this solution (achieved 759.44504) has done: 'We keep the overall pipeline but expand the feature set: retain the longitude/latitude difference columns (`abs_diff_longitude`, `abs_diff_latitude`) and include them among the numeric variables that are standard‑scaled. This gives the linear model more relevant information and is expected to drop the validation RMSE from the huge 717 + range toward the target ~5.7 while preserving the original workflow.'
- What this solution (achieved 759.4979) has done: 'I add the passenger count as a numeric feature (it is strongly predictive) and drop the raw absolute coordinate‑difference columns, which add noise and were causing the model to predict wildly off values. The numeric column list and the DataFrame preparation are updated accordingly, keeping the rest of the pipeline intact.'
- What this solution (achieved 758.12857) has done: 'The fix updates the file paths for the training and test CSVs to the correct directory (`new-york-city-taxi-fare-prediction`). This resolves the `FileNotFoundError` and allows the entire preprocessing, model training, and submission generation pipeline to run, producing a valid `submission.csv` that can be evaluated for RMSE.'
- What this solution (achieved 12.95379) has done: 'I added a few low‑impact tweaks that keep the original pipeline but improve robustness and push the RMSE toward the target: after building all features I drop any rows containing NaNs (so the model isn’t trained on incomplete data), raise the Ridge regularisation slightly (α = 10) to curb over‑fitting, and clip the final fare predictions to the realistic range 0‑200 before saving. These changes preserve the core logic while helping the validation score move closer to the desired 5.69 and still produce a correct submission.csv.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=10_000_000
)
train_df.dtypes



## === cell 2
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
test_df.dtypes



## === cell 3
print("Train DF shape ", train_df.shape)
print("Test DF Shape: ", test_df.shape)




## === cell 4
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)



## === cell 5
test_df.head()



## === cell 6
print(train_df.isnull().sum())



## === cell 7
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 8
print(test_df.isnull().sum())



## === cell 9
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))




## === cell 10
def get_input_matrix(df):
    return np.column_stack(
        (df.abs_diff_longitude, df.abs_diff_latitude, np.ones(len(df)))
    )


train_X = get_input_matrix(train_df)
train_y = np.array(train_df["fare_amount"])

print(train_X.shape)
print(train_y.shape)



## === cell 11
ls1 = list(train_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_df["pickuptime"] = ls1


ls1 = list(test_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_df["pickuptime"] = ls1



## === cell 12
ls1 = list(train_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
train_df["Weekday"] = ls1


ls1 = list(test_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
test_df["Weekday"] = ls1



## === cell 13
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)



## === cell 14
train_df["Weekday"].replace(
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
test_df["Weekday"].replace(
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

train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_df["Weekday"])

test_one_hot = test_one_hot.reindex(columns=train_one_hot.columns, fill_value=0)

train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)

train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)

ls1 = list(train_df["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
train_df["pickuptime"] = ls1


ls1 = list(test_df["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
test_df["pickuptime"] = ls1



## === cell 15
train_df.head()



## === cell 16
R = 6373.0
lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_df["Distance"] = np.round(distance * 0.621, 2)

lat1_t = np.asarray(np.radians(test_df["pickup_latitude"]))
lon1_t = np.asarray(np.radians(test_df["pickup_longitude"]))
lat2_t = np.asarray(np.radians(test_df["dropoff_latitude"]))
lon2_t = np.asarray(np.radians(test_df["dropoff_longitude"]))

dlon_t = lon2_t - lon1_t
dlat_t = lat2_t - lat1_t  # corrected computation
a_t = (
    np.sin(dlat_t / 2) ** 2 + np.cos(lat1_t) * np.cos(lat2_t) * np.sin(dlon_t / 2) ** 2
)
c_t = 2 * np.arctan2(np.sqrt(a_t), np.sqrt(1 - a_t))
distance_t = R * c_t
test_df["Distance"] = np.round(distance_t * 0.621, 2)



## === cell 17
R = 6373.0
lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

lat3 = np.zeros(len(train_df)) + np.radians(40.6413111)
lon3 = np.zeros(len(train_df)) + np.radians(-73.7781391)
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
train_df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2

train_df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(test_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_df["dropoff_longitude"]))

lat3 = np.zeros(len(test_df)) + np.radians(40.6413111)
lon3 = np.zeros(len(test_df)) + np.radians(-73.7781391)
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
test_df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2

test_df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## === cell 18
train_df["Distance"] = np.round(train_df["Distance"], 2)
train_df["Pickup_Distance_airport"] = np.round(train_df["Pickup_Distance_airport"], 2)
train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)
test_df["Distance"] = np.round(test_df["Distance"], 2)
test_df["Pickup_Distance_airport"] = np.round(test_df["Pickup_Distance_airport"], 2)
test_df["Dropoff_Distance_airport"] = np.round(test_df["Dropoff_Distance_airport"], 2)



## === cell 19
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

train_df = train_df.dropna()
test_df = test_df.dropna()



## === cell 20
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import math

fare_mask = (train_df["fare_amount"] > 0) & (train_df["fare_amount"] < 200)

drop_cols = [
    "key",
    "fare_amount",
]
X = train_df.drop(drop_cols, axis=1)[fare_mask]
y = train_df["fare_amount"][fare_mask]

X = X.dropna()
y = y.loc[X.index]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=42)

numeric_cols = [
    "Distance",
    "Pickup_Distance_airport",
    "Dropoff_Distance_airport",
    "pickuptime",
    "passenger_count",
    "abs_diff_longitude",
    "abs_diff_latitude",
]

X_train = X_train.copy()
X_val = X_val.copy()

scaler = StandardScaler()
X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train.loc[:, numeric_cols])
X_val.loc[:, numeric_cols] = scaler.transform(X_val.loc[:, numeric_cols])



## === cell 21
from sklearn.linear_model import Ridge

y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

lr = Ridge(alpha=10.0, fit_intercept=True)
lr.fit(X_train, y_train_log)

val_pred_log = lr.predict(X_val)
val_pred = np.expm1(val_pred_log)

rmse = math.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {rmse:.4f}")



## === cell 22
test_features = test_df.drop(["key"], axis=1)

test_features = test_features.copy()
test_features.loc[:, numeric_cols] = scaler.transform(
    test_features.loc[:, numeric_cols]
)

pred_log = lr.predict(test_features)
pred = np.expm1(pred_log)

pred = np.clip(pred, 0, 200)
pred = np.round(pred, 2)

Submission = pd.DataFrame({"key": test_df["key"], "fare_amount": pred})
Submission.head()



## === cell 23
Submission.to_csv("submission.csv", index=False)
