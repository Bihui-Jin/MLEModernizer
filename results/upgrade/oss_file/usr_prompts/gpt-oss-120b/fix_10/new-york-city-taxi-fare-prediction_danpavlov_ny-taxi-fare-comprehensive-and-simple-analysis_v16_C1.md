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

3.7

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
xgboost==2.0.3

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

3.623

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 8.50328) has done: 'I remove the deprecated use of `best_ntree_limit` and the old `reg:linear` objective, updating the XGBoost training function to return a Booster that can predict directly. I also guard the Jupyter magic line so the script runs in a plain Python environment. These fixes eliminate the runtime errors, allow predictions to be generated, and ensure a proper CSV submission file is written.'
- What this solution (achieved 7.39037) has done: 'I keep the overall pipeline unchanged but improve the XGBoost model by using a richer set of hyper‑parameters (deeper trees, learning rate, subsampling) and a slightly higher early‑stopping patience. After predicting I also clip any negative fares to zero. These small tweaks are expected to lower the RMSE toward the target without altering the core logic.'
- What this solution (achieved 9.58233) has done: 'I increase the training sample size, add cyclic hour/weekday/month features (sin / cos), and slightly strengthen the XGBoost model (deeper trees, lower learning rate) so the validation RMSE moves closer to the target while leaving the overall pipeline unchanged.'
- What this solution (achieved 5.16198) has done: 'The changes vectorize the costly haversine loops, replacing `iterrows` with fast NumPy operations for both the pickup‑dropoff distance and distances to the airports/Manhattan. This keeps the exact same feature values while cutting runtime from minutes to seconds. Additionally, the XGBoost parameters now include `tree_method="hist"` which speeds up training without altering the model’s logic or predictions. All other code and paths remain unchanged.'
- What this solution (achieved 5.04284) has done: 'I added robust handling for missing and infinite values after feature engineering, filling NaNs with column medians so both LinearRegression and XGBoost can operate without errors. I also ensured that predicted fares are non‑negative by clipping LinearRegression outputs. No core modeling logic was changed.'
- What this solution (achieved 6.79145) has done: 'I train the XGBoost model directly on the original fare amounts instead of on a log‑transformed target. This small change keeps the overall pipeline unchanged but aligns the loss function with the competition’s RMSE metric, which should reduce the validation error and move the score closer to the target. I also adjust the prediction step to use the raw model output (no exp‑transform) and keep the existing clipping and rounding.'

# 9. Code solution

## === cell 0
print(os.listdir("../input"))


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1644877966.py in <cell line: 0>()
----> 1 print(os.listdir("../input"))

NameError: name 'os' is not defined

## === cell 1
test = pd.read_csv("../input/test.csv")


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1536470077.py in <cell line: 0>()
----> 1 test = pd.read_csv("../input/test.csv")

NameError: name 'pd' is not defined

## === cell 2
test.dtypes


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1869611232.py in <cell line: 0>()
----> 1 test.dtypes

NameError: name 'test' is not defined

## === cell 3
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
train = pd.read_csv("../input/train.csv", nrows=2000000, dtype=types)


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3763961558.py in <cell line: 0>()
      7     "passenger_count": "uint8",
      8 }
----> 9 train = pd.read_csv("../input/train.csv", nrows=2000000, dtype=types)

NameError: name 'pd' is not defined

## === cell 4
train.head()


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3450903571.py in <cell line: 0>()
----> 1 train.head()

NameError: name 'train' is not defined

## === cell 5
train.describe()


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2359111164.py in <cell line: 0>()
----> 1 train.describe()

NameError: name 'train' is not defined

## === cell 6
sns.distplot(train["fare_amount"])


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2974934209.py in <cell line: 0>()
----> 1 sns.distplot(train["fare_amount"])

NameError: name 'sns' is not defined

## === cell 7
sns.distplot(train["passenger_count"])


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3339604200.py in <cell line: 0>()
----> 1 sns.distplot(train["passenger_count"])

NameError: name 'sns' is not defined

## === cell 8
train.isnull().sum()


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/102480753.py in <cell line: 0>()
----> 1 train.isnull().sum()

NameError: name 'train' is not defined

## === cell 9
train.dropna(inplace=True)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3333548266.py in <cell line: 0>()
----> 1 train.dropna(inplace=True)

NameError: name 'train' is not defined

## === cell 10
train = train[train["fare_amount"] > 0]
train = train[train["pickup_longitude"] < -72]
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
train = train[train["dropoff_longitude"] < -72]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2412499310.py in <cell line: 0>()
----> 1 train = train[train["fare_amount"] > 0]
      2 train = train[train["pickup_longitude"] < -72]
      3 train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
      4 train = train[train["dropoff_longitude"] < -72]
      5 train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]

NameError: name 'train' is not defined

## === cell 11
train.describe()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/293065314.py in <cell line: 0>()
----> 1 train.describe()
      2 
      3 

NameError: name 'train' is not defined

## === cell 12
def quick_dist_calc(df):
    R = 6373.0
    lat1 = np.radians(df["pickup_latitude"].values)
    lon1 = np.radians(df["pickup_longitude"].values)
    lat2 = np.radians(df["dropoff_latitude"].values)
    lon2 = np.radians(df["dropoff_longitude"].values)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["distance"] = R * c




## === cell 13
def quick_dist_calc_loc(df, c1, c2, cname):
    R = 6373.0
    latc = np.radians(c1)
    lonc = np.radians(c2)

    lat1 = np.radians(df["pickup_latitude"].values)
    lon1 = np.radians(df["pickup_longitude"].values)
    lat2 = np.radians(df["dropoff_latitude"].values)
    lon2 = np.radians(df["dropoff_longitude"].values)

    dlat1 = latc - lat1
    dlon1 = lonc - lon1
    a1 = np.sin(dlat1 / 2) ** 2 + np.cos(latc) * np.cos(lat1) * np.sin(dlon1 / 2) ** 2
    c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
    dist_pickup = R * c1

    dlat2 = latc - lat2
    dlon2 = lonc - lon2
    a2 = np.sin(dlat2 / 2) ** 2 + np.cos(latc) * np.cos(lat2) * np.sin(dlon2 / 2) ** 2
    c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
    dist_dropoff = R * c2

    df[cname + "_pickup_dist"] = dist_pickup
    df[cname + "_dropoff_dist"] = dist_dropoff




## === cell 14
quick_dist_calc(train)
quick_dist_calc(test)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/579453514.py in <cell line: 0>()
----> 1 quick_dist_calc(train)
      2 quick_dist_calc(test)

NameError: name 'train' is not defined

## === cell 15
jfk_airport = (-73.785193, 40.645972)
laguardia_airport = (-73.872925, 40.773335)
newark_airport = (-74.184156, 40.692764)
manhattan = (-73.983132, 40.759006)

quick_dist_calc_loc(train, jfk_airport[1], jfk_airport[0], "jfk_airport")
quick_dist_calc_loc(
    train, laguardia_airport[1], laguardia_airport[0], "laguardia_airport"
)
quick_dist_calc_loc(train, newark_airport[1], newark_airport[0], "newark_airport")
quick_dist_calc_loc(train, manhattan[1], manhattan[0], "manhattan")

quick_dist_calc_loc(test, jfk_airport[1], jfk_airport[0], "jfk_airport")
quick_dist_calc_loc(
    test, laguardia_airport[1], laguardia_airport[0], "laguardia_airport"
)
quick_dist_calc_loc(test, newark_airport[1], newark_airport[0], "newark_airport")
quick_dist_calc_loc(test, manhattan[1], manhattan[0], "manhattan")


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4237882410.py in <cell line: 0>()
      4 manhattan = (-73.983132, 40.759006)
      5 
----> 6 quick_dist_calc_loc(train, jfk_airport[1], jfk_airport[0], "jfk_airport")
      7 quick_dist_calc_loc(
      8     train, laguardia_airport[1], laguardia_airport[0], "laguardia_airport"

NameError: name 'train' is not defined

## === cell 16
train["jfk_distance"] = pd.concat(
    [train["jfk_airport_pickup_dist"], train["jfk_airport_dropoff_dist"]], axis=1
).min(axis=1)
train["laguardia_distance"] = pd.concat(
    [train["laguardia_airport_pickup_dist"], train["laguardia_airport_dropoff_dist"]],
    axis=1,
).min(axis=1)
train["newark_distance"] = pd.concat(
    [train["newark_airport_pickup_dist"], train["newark_airport_dropoff_dist"]], axis=1
).min(axis=1)
train["manhattan_distance"] = pd.concat(
    [train["manhattan_pickup_dist"], train["manhattan_dropoff_dist"]], axis=1
).min(axis=1)

test["jfk_distance"] = pd.concat(
    [test["jfk_airport_pickup_dist"], test["jfk_airport_dropoff_dist"]], axis=1
).min(axis=1)
test["laguardia_distance"] = pd.concat(
    [test["laguardia_airport_pickup_dist"], test["laguardia_airport_dropoff_dist"]],
    axis=1,
).min(axis=1)
test["newark_distance"] = pd.concat(
    [test["newark_airport_pickup_dist"], test["newark_airport_dropoff_dist"]], axis=1
).min(axis=1)
test["manhattan_distance"] = pd.concat(
    [test["manhattan_pickup_dist"], test["manhattan_dropoff_dist"]], axis=1
).min(axis=1)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2340575956.py in <cell line: 0>()
----> 1 train["jfk_distance"] = pd.concat(
      2     [train["jfk_airport_pickup_dist"], train["jfk_airport_dropoff_dist"]], axis=1
      3 ).min(axis=1)
      4 train["laguardia_distance"] = pd.concat(
      5     [train["laguardia_airport_pickup_dist"], train["laguardia_airport_dropoff_dist"]],

NameError: name 'pd' is not defined

## === cell 17
train.drop(
    [
        "jfk_airport_pickup_dist",
        "jfk_airport_dropoff_dist",
        "laguardia_airport_pickup_dist",
        "laguardia_airport_dropoff_dist",
        "newark_airport_pickup_dist",
        "newark_airport_dropoff_dist",
        "manhattan_pickup_dist",
        "manhattan_dropoff_dist",
    ],
    axis=1,
    inplace=True,
)

test.drop(
    [
        "jfk_airport_pickup_dist",
        "jfk_airport_dropoff_dist",
        "laguardia_airport_pickup_dist",
        "laguardia_airport_dropoff_dist",
        "newark_airport_pickup_dist",
        "newark_airport_dropoff_dist",
        "manhattan_pickup_dist",
        "manhattan_dropoff_dist",
    ],
    axis=1,
    inplace=True,
)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/743615425.py in <cell line: 0>()
----> 1 train.drop(
      2     [
      3         "jfk_airport_pickup_dist",
      4         "jfk_airport_dropoff_dist",
      5         "laguardia_airport_pickup_dist",

NameError: name 'train' is not defined

## === cell 18
train["pickup_datetime"] = train["pickup_datetime"].str.replace(" UTC", "")
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)

test["pickup_datetime"] = test["pickup_datetime"].str.replace(" UTC", "")
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)

train["hour"] = train.pickup_datetime.dt.hour
train["weekday"] = train.pickup_datetime.dt.weekday
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year

test["hour"] = test.pickup_datetime.dt.hour
test["weekday"] = test.pickup_datetime.dt.weekday
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year

train["hour_sin"] = np.sin(2 * np.pi * train["hour"] / 24)
train["hour_cos"] = np.cos(2 * np.pi * train["hour"] / 24)
train["weekday_sin"] = np.sin(2 * np.pi * train["weekday"] / 7)
train["weekday_cos"] = np.cos(2 * np.pi * train["weekday"] / 7)
train["month_sin"] = np.sin(2 * np.pi * train["month"] / 12)
train["month_cos"] = np.cos(2 * np.pi * train["month"] / 12)

test["hour_sin"] = np.sin(2 * np.pi * test["hour"] / 24)
test["hour_cos"] = np.cos(2 * np.pi * test["hour"] / 24)
test["weekday_sin"] = np.sin(2 * np.pi * test["weekday"] / 7)
test["weekday_cos"] = np.cos(2 * np.pi * test["weekday"] / 7)
test["month_sin"] = np.sin(2 * np.pi * test["month"] / 12)
test["month_cos"] = np.cos(2 * np.pi * test["month"] / 12)

train["log_distance"] = np.log1p(train["distance"])
test["log_distance"] = np.log1p(test["distance"])

train["distance_per_passenger"] = train["distance"] / train["passenger_count"]
test["distance_per_passenger"] = test["distance"] / test["passenger_count"]

train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)

median_vals = train.median(numeric_only=True)
train.fillna(median_vals, inplace=True)
test.fillna(median_vals, inplace=True)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1514969207.py in <cell line: 0>()
----> 1 train["pickup_datetime"] = train["pickup_datetime"].str.replace(" UTC", "")
      2 train["pickup_datetime"] = pd.to_datetime(
      3     train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
      4 )
      5 

NameError: name 'train' is not defined

## === cell 19
train.head()


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3450903571.py in <cell line: 0>()
----> 1 train.head()

NameError: name 'train' is not defined

## === cell 20
test.head()


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3938725978.py in <cell line: 0>()
----> 1 test.head()

NameError: name 'test' is not defined

## === cell 21
plt.figure(figsize=(20, 12))
sns.heatmap(
    train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1134696804.py in <cell line: 0>()
----> 1 plt.figure(figsize=(20, 12))
      2 sns.heatmap(
      3     train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
      4 )

NameError: name 'plt' is not defined

## === cell 22
X = train.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = train["fare_amount"]


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/140956807.py in <cell line: 0>()
----> 1 X = train.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
      2 y = train["fare_amount"]

NameError: name 'train' is not defined

## === cell 23
X.head()


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2424826063.py in <cell line: 0>()
----> 1 X.head()

NameError: name 'X' is not defined

## === cell 24
y.head()


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3007166867.py in <cell line: 0>()
----> 1 y.head()

NameError: name 'y' is not defined

## === cell 25
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1462313936.py in <cell line: 0>()
----> 1 X_train, X_test, y_train, y_test = train_test_split(
      2     X, y, test_size=0.2, random_state=42
      3 )

NameError: name 'train_test_split' is not defined

## === cell 26
test_pred = test.drop(["key", "pickup_datetime"], axis=1)


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1219563936.py in <cell line: 0>()
----> 1 test_pred = test.drop(["key", "pickup_datetime"], axis=1)

NameError: name 'test' is not defined

## === cell 27
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_test, y_test))


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/314666892.py in <cell line: 0>()
----> 1 lm = LinearRegression()
      2 lm.fit(X_train, y_train)
      3 print(lm.score(X_train, y_train))
      4 print(lm.score(X_test, y_test))

NameError: name 'LinearRegression' is not defined

## === cell 28
y_pred = lm.predict(X)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y))
print(lrmse)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2780606501.py in <cell line: 0>()
----> 1 y_pred = lm.predict(X)
      2 lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y))
      3 print(lrmse)

NameError: name 'lm' is not defined

## === cell 29
LinearPredictions = lm.predict(test_pred)
LinearPredictions = np.maximum(LinearPredictions, 0)  # no negative fares
LinearPredictions = np.round(LinearPredictions, 2)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3437515495.py in <cell line: 0>()
----> 1 LinearPredictions = lm.predict(test_pred)
      2 LinearPredictions = np.maximum(LinearPredictions, 0)  # no negative fares
      3 LinearPredictions = np.round(LinearPredictions, 2)

NameError: name 'lm' is not defined

## === cell 30
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)




## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1807615474.py in <cell line: 0>()
----> 1 linear_submission = pd.DataFrame(
      2     {"key": test["key"], "fare_amount": LinearPredictions},
      3     columns=["key", "fare_amount"],
      4 )
      5 

NameError: name 'pd' is not defined

## === cell 31
def XGBoost(X_train, X_test, y_train, y_test):
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    dtrain = xgb.DMatrix(X_train, label=y_train_log)
    dtest = xgb.DMatrix(X_test, label=y_test_log)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 42,
        "max_depth": 10,
        "eta": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 0,
        "gamma": 0,
        "lambda": 1.0,
        "alpha": 0.0,
        "tree_method": "hist",
    }

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        early_stopping_rounds=200,
        evals=[(dtest, "test")],
        verbose_eval=False,
    )
    return model




## === cell 32
xgbm = XGBoost(X_train, X_test, y_train, y_test)


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2738206094.py in <cell line: 0>()
----> 1 xgbm = XGBoost(X_train, X_test, y_train, y_test)

NameError: name 'X_train' is not defined

## === cell 33
XGBPredictions = np.expm1(xgbm.predict(xgb.DMatrix(test_pred)))
XGBPredictions = np.maximum(XGBPredictions, 0)
XGBPredictions = np.round(XGBPredictions, 2)


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/882885120.py in <cell line: 0>()
      1 # Convert predictions back from log scale, then clip and round
----> 2 XGBPredictions = np.expm1(xgbm.predict(xgb.DMatrix(test_pred)))
      3 XGBPredictions = np.maximum(XGBPredictions, 0)
      4 XGBPredictions = np.round(XGBPredictions, 2)

NameError: name 'np' is not defined

## === cell 34
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2418489850.py in <cell line: 0>()
----> 1 XGB_submission = pd.DataFrame(
      2     {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
      3 )

NameError: name 'pd' is not defined

## === cell 35
submission = XGB_submission


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2616470590.py in <cell line: 0>()
----> 1 submission = XGB_submission

NameError: name 'XGB_submission' is not defined

## === cell 36
submission.to_csv("XGBSubmission23082018_2M.csv", index=False)

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3583591807.py in <cell line: 0>()
----> 1 submission.to_csv("XGBSubmission23082018_2M.csv", index=False)

NameError: name 'submission' is not defined
