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

5.48885

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 800.26126) has done: 'The changes enable Intel‑optimized scikit‑learn (sklearnex) and parallel execution of the RandomForest, which dramatically cuts training time without altering the model’s hyper‑parameters or prediction logic. Only the import of `sklearnex` and the `n_jobs=-1` argument are added; all other code and data handling remain identical, preserving exact results while staying under the 600‑second limit.'
- What this solution (achieved 16.9774) has done: 'The fix adds a safe import for the Intel‑optimized scikit‑learn patch (using a try/except fallback) so the notebook can run, and clips model predictions to non‑negative values to avoid wildly erroneous fares that hurt RMSE. No core modeling logic is changed.'
- What this solution (achieved 16.9774) has done: 'I keep the existing cleaning, feature engineering, and model code unchanged and add a new, simple distance‑based prediction rule (fare ≈ 2.5 + 1.56 × H_Distance) which is known to be a strong baseline for this dataset. This rule is applied after the models are trained, and its predictions are written to a third submission file (`submission_3.csv`). Using this deterministic estimate should move the RMSE much closer to the target score while preserving the core workflow.'
- What this solution (achieved 797.51326) has done: 'I replace the simple distance‑based rule with a tiny linear fit of fare ≈ a + b·H_Distance computed on the cleaned training data. This keeps the same features and preprocessing, only calibrates the two coefficients to better match the target RMSE, and writes the final predictions to submission.csv (the required output file).'

# 9. Code solution

## === cell 0
dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
parse_dates = ["pickup_datetime"]
train = pd.read_csv(
    "../input/train.csv",
    usecols=list(dtypes.keys()) + ["pickup_datetime"],
    dtype=dtypes,
    parse_dates=parse_dates,
    infer_datetime_format=True,
    memory_map=True,
)
test = pd.read_csv(
    "../input/test.csv",
    usecols=list(dtypes.keys()) + ["pickup_datetime"],
    dtype=dtypes,
    parse_dates=parse_dates,
    infer_datetime_format=True,
    memory_map=True,
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1831019358.py in <cell line: 0>()
      9 }
     10 parse_dates = ["pickup_datetime"]
---> 11 train = pd.read_csv(
     12     "../input/train.csv",
     13     usecols=list(dtypes.keys()) + ["pickup_datetime"],

NameError: name 'pd' is not defined

## === cell 1
train.dropna(inplace=True)
train = train[train["fare_amount"] >= 0]
train = train[train["passenger_count"] <= 8]

lat_cond = (train["pickup_latitude"].between(-90, 90)) & (
    train["dropoff_latitude"].between(-90, 90)
)
lon_cond = (train["pickup_longitude"].between(-180, 180)) & (
    train["dropoff_longitude"].between(-180, 180)
)
train = train[lat_cond & lon_cond]




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2081548015.py in <cell line: 0>()
      1 # In‑place cleaning to avoid creating new DataFrames.
----> 2 train.dropna(inplace=True)
      3 train = train[train["fare_amount"] >= 0]
      4 train = train[train["passenger_count"] <= 8]
      5 

NameError: name 'train' is not defined

## === cell 2
def haversine_distance(df):
    r = 6371.0  # Earth radius in km
    phi1 = np.radians(df["pickup_latitude"].values)
    phi2 = np.radians(df["dropoff_latitude"].values)
    dphi = np.radians(df["dropoff_latitude"].values - df["pickup_latitude"].values)
    dlambda = np.radians(df["dropoff_longitude"].values - df["pickup_longitude"].values)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


train["H_Distance"] = haversine_distance(train)
test["H_Distance"] = haversine_distance(test)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3430540706.py in <cell line: 0>()
     13 
     14 
---> 15 train["H_Distance"] = haversine_distance(train)
     16 test["H_Distance"] = haversine_distance(test)
     17 

NameError: name 'train' is not defined

## === cell 3
for df in (train, test):
    dt = df["pickup_datetime"].dt
    df["Year"] = dt.year
    df["Month"] = dt.month
    df["Date"] = dt.day
    df["Day_of_Week"] = dt.dayofweek
    df["Hour"] = dt.hour



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2555460118.py in <cell line: 0>()
      1 # Extract datetime features in a single pass (still vectorised) – no extra copies.
----> 2 for df in (train, test):
      3     dt = df["pickup_datetime"].dt
      4     df["Year"] = dt.year
      5     df["Month"] = dt.month

NameError: name 'train' is not defined

## === cell 4
cond_zero = (train["H_Distance"] == 0) & (train["fare_amount"] == 0)
train.drop(train[cond_zero].index, inplace=True)

high_dist = (train["H_Distance"] > 200) & (train["fare_amount"] != 0)
train.loc[high_dist, "H_Distance"] = (train.loc[high_dist, "fare_amount"] - 2.5) / 1.56

sc3 = (train["H_Distance"] != 0) & (train["fare_amount"] == 0)
train.loc[sc3, "fare_amount"] = train.loc[sc3, "H_Distance"] * 1.56 + 2.5

sc4 = (train["H_Distance"] == 0) & (train["fare_amount"] != 0)
train.loc[sc4, "H_Distance"] = (train.loc[sc4, "fare_amount"] - 2.5) / 1.56



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2130061102.py in <cell line: 0>()
      1 # Clean inconsistent distance/price rows – all operations are in‑place.
----> 2 cond_zero = (train["H_Distance"] == 0) & (train["fare_amount"] == 0)
      3 train.drop(train[cond_zero].index, inplace=True)
      4 
      5 high_dist = (train["H_Distance"] > 200) & (train["fare_amount"] != 0)

NameError: name 'train' is not defined

## === cell 5
train.drop(columns=["pickup_datetime"], inplace=True)
test.drop(columns=["pickup_datetime"], inplace=True)

x_train = train.drop(columns=["fare_amount"])
y_train = train["fare_amount"].values
x_test = test.copy()

median_vals = x_train.median()
x_train.fillna(median_vals, inplace=True)
x_test.fillna(median_vals, inplace=True)

x_train = x_train.astype(np.float32, copy=False)
x_test = x_test.astype(np.float32, copy=False)

del train, test
import gc

gc.collect()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2563368437.py in <cell line: 0>()
----> 1 train.drop(columns=["pickup_datetime"], inplace=True)
      2 test.drop(columns=["pickup_datetime"], inplace=True)
      3 
      4 x_train = train.drop(columns=["fare_amount"])
      5 y_train = train["fare_amount"].values

NameError: name 'train' is not defined

## === cell 6
rf = RandomForestRegressor(
    n_estimators=10,
    max_depth=15,
    random_state=42,
    n_jobs=-1,
)
rf.fit(x_train.values, y_train)  # use underlying ndarray to avoid pandas overhead
rf_predict = np.clip(rf.predict(x_test.values), 0, None)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3837652973.py in <cell line: 0>()
----> 1 rf = RandomForestRegressor(
      2     n_estimators=10,
      3     max_depth=15,
      4     random_state=42,
      5     n_jobs=-1,

NameError: name 'RandomForestRegressor' is not defined

## === cell 7
submission = pd.read_csv("../input/sample_submission.csv")
submission["fare_amount"] = rf_predict
submission.to_csv("submission_1.csv", index=False)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1854623679.py in <cell line: 0>()
----> 1 submission = pd.read_csv("../input/sample_submission.csv")
      2 submission["fare_amount"] = rf_predict
      3 submission.to_csv("submission_1.csv", index=False)
      4 

NameError: name 'pd' is not defined

## === cell 8
lr = LinearRegression()
lr.fit(x_train.values, y_train)
lr_predict = np.clip(lr.predict(x_test.values), 0, None)

submission = pd.read_csv("../input/sample_submission.csv")
submission["fare_amount"] = lr_predict
submission.to_csv("submission_2.csv", index=False)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2551913804.py in <cell line: 0>()
----> 1 lr = LinearRegression()
      2 lr.fit(x_train.values, y_train)
      3 lr_predict = np.clip(lr.predict(x_test.values), 0, None)
      4 
      5 submission = pd.read_csv("../input/sample_submission.csv")

NameError: name 'LinearRegression' is not defined

## === cell 9
final_submission = pd.read_csv("../input/sample_submission.csv")
final_submission["fare_amount"] = rf_predict
final_submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1189556027.py in <cell line: 0>()
----> 1 final_submission = pd.read_csv("../input/sample_submission.csv")
      2 final_submission["fare_amount"] = rf_predict
      3 final_submission.to_csv("submission.csv", index=False)

NameError: name 'pd' is not defined
