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
geopy==2.4.1
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

3.4594

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 13.37112) has done: 'The changes remove the invalid `%matplotlib inline` line, fix the XGBoost prediction by using the correct Booster API (no `best_ntree_limit` attribute) and update the training parameters to the current objective name. These fixes let the notebook run end‑to‑end and correctly write a `submission.csv` file with the required columns.'
- What this solution (achieved 431.76956) has done: 'I keep the overall workflow and feature set but average the LinearRegression and XGBoost predictions (both already computed) before creating the submission. Averaging usually reduces variance and brings the RMSE down, moving the score closer to the target while leaving the core modeling logic unchanged. The final submission file is still written as `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
print(os.listdir("../input"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3967235659.py in <cell line: 0>()
----> 1 print(os.listdir("../input"))
      2 

NameError: name 'os' is not defined

## === cell 1
test = pd.read_csv("../input/test.csv")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/465273792.py in <cell line: 0>()
----> 1 test = pd.read_csv("../input/test.csv")
      2 

NameError: name 'pd' is not defined

## === cell 2
test.dtypes



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1872366253.py in <cell line: 0>()
----> 1 test.dtypes
      2 

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



## === cell 4
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
train = pd.read_csv(
    "../input/train.csv",
    dtype=types,
    usecols=usecols,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/163691665.py in <cell line: 0>()
     10     "passenger_count",
     11 ]
---> 12 train = pd.read_csv(
     13     "../input/train.csv",
     14     dtype=types,

NameError: name 'pd' is not defined

## === cell 5
train.head()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2745801949.py in <cell line: 0>()
----> 1 train.head()
      2 

NameError: name 'train' is not defined

## === cell 6
train.describe()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/199662553.py in <cell line: 0>()
----> 1 train.describe()
      2 

NameError: name 'train' is not defined

## === cell 7
sns.distplot(train["fare_amount"])



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/915830171.py in <cell line: 0>()
----> 1 sns.distplot(train["fare_amount"])
      2 

NameError: name 'sns' is not defined

## === cell 8
sns.distplot(train["passenger_count"])



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2644906479.py in <cell line: 0>()
----> 1 sns.distplot(train["passenger_count"])
      2 

NameError: name 'sns' is not defined

## === cell 9
train.isnull().sum()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2969740008.py in <cell line: 0>()
----> 1 train.isnull().sum()
      2 

NameError: name 'train' is not defined

## === cell 10
train.dropna(inplace=True)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1783675776.py in <cell line: 0>()
----> 1 train.dropna(inplace=True)
      2 

NameError: name 'train' is not defined

## === cell 11
train = train[train["fare_amount"] > 0]
train = train[train["pickup_longitude"] < -72]
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
train = train[train["dropoff_longitude"] < -72]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/259571751.py in <cell line: 0>()
----> 1 train = train[train["fare_amount"] > 0]
      2 train = train[train["pickup_longitude"] < -72]
      3 train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
      4 train = train[train["dropoff_longitude"] < -72]
      5 train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]

NameError: name 'train' is not defined

## === cell 12
train.describe()




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/293065314.py in <cell line: 0>()
----> 1 train.describe()
      2 
      3 

NameError: name 'train' is not defined

## === cell 13
def quick_dist_calc(df):
    """
    Vectorized Haversine distance calculation.
    Adds a 'distance' column (km) to the dataframe.
    """
    R = 6373.0  # Earth radius in km
    lat1 = np.radians(df["pickup_latitude"].astype("float64"))
    lon1 = np.radians(df["pickup_longitude"].astype("float64"))
    lat2 = np.radians(df["dropoff_latitude"].astype("float64"))
    lon2 = np.radians(df["dropoff_longitude"].astype("float64"))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["distance"] = (R * c).astype("float32")




## === cell 14
quick_dist_calc(train)
quick_dist_calc(test)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2540011164.py in <cell line: 0>()
----> 1 quick_dist_calc(train)
      2 quick_dist_calc(test)
      3 

NameError: name 'train' is not defined

## === cell 15
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S"
)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S"
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1846944890.py in <cell line: 0>()
      1 # Faster datetime handling: slice the trailing " UTC" instead of regex replace.
----> 2 train["pickup_datetime"] = pd.to_datetime(
      3     train["pickup_datetime"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S"
      4 )
      5 test["pickup_datetime"] = pd.to_datetime(

NameError: name 'pd' is not defined

## === cell 16
train["hour"] = train.pickup_datetime.dt.hour
train["weekday"] = train.pickup_datetime.dt.weekday
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year

test["hour"] = test.pickup_datetime.dt.hour
test["weekday"] = test.pickup_datetime.dt.weekday
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2030255826.py in <cell line: 0>()
----> 1 train["hour"] = train.pickup_datetime.dt.hour
      2 train["weekday"] = train.pickup_datetime.dt.weekday
      3 train["month"] = train.pickup_datetime.dt.month
      4 train["year"] = train.pickup_datetime.dt.year
      5 

NameError: name 'train' is not defined

## === cell 17
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    R_earth = 6371
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(
        np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    )
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon / 2.0) ** 2
    )
    return 2 * R_earth * np.arcsin(np.sqrt(a))




## === cell 18
def add_airport_dist(dataset):
    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)

    pickup_lat = dataset["pickup_latitude"]
    dropoff_lat = dataset["dropoff_latitude"]
    pickup_lon = dataset["pickup_longitude"]
    dropoff_lon = dataset["dropoff_longitude"]

    pickup_jfk = sphere_dist(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1])
    dropoff_jfk = sphere_dist(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon)
    pickup_ewr = sphere_dist(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])
    dropoff_ewr = sphere_dist(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon)
    pickup_lga = sphere_dist(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1])
    dropoff_lga = sphere_dist(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon)

    dataset["jfk_dist"] = pd.concat([pickup_jfk, dropoff_jfk], axis=1).min(axis=1)
    dataset["ewr_dist"] = pd.concat([pickup_ewr, dropoff_ewr], axis=1).min(axis=1)
    dataset["lga_dist"] = pd.concat([pickup_lga, dropoff_lga], axis=1).min(axis=1)

    return dataset




## === cell 19
train = add_airport_dist(train)
test = add_airport_dist(test)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1207635751.py in <cell line: 0>()
----> 1 train = add_airport_dist(train)
      2 test = add_airport_dist(test)
      3 

NameError: name 'train' is not defined

## === cell 20
train.head()



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2745801949.py in <cell line: 0>()
----> 1 train.head()
      2 

NameError: name 'train' is not defined

## === cell 21
test.head()



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3214727096.py in <cell line: 0>()
----> 1 test.head()
      2 

NameError: name 'test' is not defined

## === cell 22
plt.figure(figsize=(15, 8))
sns.heatmap(
    train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3582551848.py in <cell line: 0>()
----> 1 plt.figure(figsize=(15, 8))
      2 sns.heatmap(
      3     train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
      4 )
      5 

NameError: name 'plt' is not defined

## === cell 23
X = train.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = train["fare_amount"]
y_log = np.log1p(y)  # transform target to log scale



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3224458697.py in <cell line: 0>()
----> 1 X = train.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
      2 y = train["fare_amount"]
      3 y_log = np.log1p(y)  # transform target to log scale
      4 

NameError: name 'train' is not defined

## === cell 24
X.head()



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/896322502.py in <cell line: 0>()
----> 1 X.head()
      2 

NameError: name 'X' is not defined

## === cell 25
y.head()



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/442728915.py in <cell line: 0>()
----> 1 y.head()
      2 

NameError: name 'y' is not defined

## === cell 26
np.random.seed(42)
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2463407307.py in <cell line: 0>()
      1 # Fixed random seed for reproducibility.
----> 2 np.random.seed(42)
      3 X_train, X_test, y_train_log, y_test_log = train_test_split(
      4     X, y_log, test_size=0.2, random_state=42
      5 )

NameError: name 'np' is not defined

## === cell 27
test_pred = test.drop(["key", "pickup_datetime"], axis=1)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3108275306.py in <cell line: 0>()
----> 1 test_pred = test.drop(["key", "pickup_datetime"], axis=1)
      2 

NameError: name 'test' is not defined

## === cell 28
lm = LinearRegression()
lm.fit(X_train, y_train_log)
print("Linear R^2 (log target) train:", lm.score(X_train, y_train_log))
print("Linear R^2 (log target) test :", lm.score(X_test, y_test_log))



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4261799053.py in <cell line: 0>()
----> 1 lm = LinearRegression()
      2 lm.fit(X_train, y_train_log)
      3 print("Linear R^2 (log target) train:", lm.score(X_train, y_train_log))
      4 print("Linear R^2 (log target) test :", lm.score(X_test, y_test_log))
      5 

NameError: name 'LinearRegression' is not defined

## === cell 29
log_rmse = np.sqrt(metrics.mean_squared_error(lm.predict(X_test), y_test_log))
print("Linear RMSE on log target:", log_rmse)



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1480015012.py in <cell line: 0>()
----> 1 log_rmse = np.sqrt(metrics.mean_squared_error(lm.predict(X_test), y_test_log))
      2 print("Linear RMSE on log target:", log_rmse)
      3 

NameError: name 'np' is not defined

## === cell 30
LinearPredictions = np.expm1(lm.predict(test_pred))  # back‑transform
LinearPredictions = np.round(LinearPredictions, 2)




## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2906291999.py in <cell line: 0>()
----> 1 LinearPredictions = np.expm1(lm.predict(test_pred))  # back‑transform
      2 LinearPredictions = np.round(LinearPredictions, 2)
      3 
      4 

NameError: name 'np' is not defined

## === cell 31
def XGBoost(X_tr, X_va, y_tr, y_va):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 42,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",  # fast histogram algorithm
        "max_bin": 256,  # limit bins for speed
        "nthread": -1,  # use all cores
    }

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=300,  # fewer rounds; early stopping will keep best
        evals=[(dvalid, "valid")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    return bst




## === cell 32
xgbm = XGBoost(X_train, X_test, y_train_log, y_test_log)



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3105877328.py in <cell line: 0>()
----> 1 xgbm = XGBoost(X_train, X_test, y_train_log, y_test_log)
      2 

NameError: name 'X_train' is not defined

## === cell 33
XGBPredictions = np.expm1(xgbm.predict(xgb.DMatrix(test_pred)))  # back‑transform
XGBPredictions = np.round(XGBPredictions, 2)



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/284582933.py in <cell line: 0>()
----> 1 XGBPredictions = np.expm1(xgbm.predict(xgb.DMatrix(test_pred)))  # back‑transform
      2 XGBPredictions = np.round(XGBPredictions, 2)
      3 

NameError: name 'np' is not defined

## === cell 34
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3413601982.py in <cell line: 0>()
----> 1 XGB_submission = pd.DataFrame(
      2     {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
      3 )
      4 

NameError: name 'pd' is not defined

## === cell 35
combined_pred = np.round(((LinearPredictions + XGBPredictions) / 2), 2)
combined_pred = np.clip(combined_pred, a_min=0, a_max=None)

submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": combined_pred}, columns=["key", "fare_amount"]
)



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1499938232.py in <cell line: 0>()
----> 1 combined_pred = np.round(((LinearPredictions + XGBPredictions) / 2), 2)
      2 combined_pred = np.clip(combined_pred, a_min=0, a_max=None)
      3 
      4 submission = pd.DataFrame(
      5     {"key": test["key"], "fare_amount": combined_pred}, columns=["key", "fare_amount"]

NameError: name 'np' is not defined

## === cell 36
submission.to_csv("submission.csv", index=False)



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/799702916.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 

NameError: name 'submission' is not defined

## === cell 37
del train, X, y, X_train, X_test, y_train_log, y_test_log, test

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3266421125.py in <cell line: 0>()
      1 # Release memory that is no longer needed.
----> 2 del train, X, y, X_train, X_test, y_train_log, y_test_log, test

NameError: name 'train' is not defined
