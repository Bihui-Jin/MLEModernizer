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

3.8

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
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

3.09737

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.01119) has done: 'I fix the feature mismatch that caused LightGBM to error by adding the missing `direction` column to the training data (it was only added to the test set) and then keep both datasets aligned before model training and prediction. This resolves the “different number of features” error and allows the script to produce a valid `taxi_fare_submission.csv` file.'
- What this solution (achieved 3.9356) has done: 'I increase the training sample size to 5 million rows and adjust LightGBM parameters (more leaves, lower learning rate, and a feature fraction) to obtain a model that is slightly more expressive while keeping the same overall pipeline. These minimal changes should reduce the RMSE toward the target without altering the core logic or the submission format.'

# 9. Code solution

## === cell 0
pass



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("dark_background")
sns.set_style("darkgrid")



## === cell 2
train_path = "../input/train.csv"

traintypes = {
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

cols = list(traintypes.keys())
train_df = pd.read_csv(
    train_path,
    usecols=cols,
    dtype=traintypes,
    nrows=5_000_000,
    memory_map=True,
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/743919177.py in <cell line: 0>()
     13 cols = list(traintypes.keys())
     14 # Enable memory‑mapping for faster reading of the large file.
---> 15 train_df = pd.read_csv(
     16     train_path,
     17     usecols=cols,

NameError: name 'pd' is not defined

## === cell 3
df_train = train_df



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2957985749.py in <cell line: 0>()
----> 1 df_train = train_df
      2 

NameError: name 'train_df' is not defined

## === cell 4
pass



## === cell 5
df_train.dtypes



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1723727466.py in <cell line: 0>()
----> 1 df_train.dtypes
      2 

NameError: name 'df_train' is not defined

## === cell 6
df_train.describe()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1262383953.py in <cell line: 0>()
----> 1 df_train.describe()
      2 

NameError: name 'df_train' is not defined

## === cell 7
len(df_train[df_train.fare_amount > 0])



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2727958426.py in <cell line: 0>()
----> 1 len(df_train[df_train.fare_amount > 0])
      2 

NameError: name 'df_train' is not defined

## === cell 8
df_train = df_train[df_train.fare_amount >= 0]



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3917201610.py in <cell line: 0>()
----> 1 df_train = df_train[df_train.fare_amount >= 0]
      2 

NameError: name 'df_train' is not defined

## === cell 9
pass



## === cell 10
df_train.isnull().sum()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4110665702.py in <cell line: 0>()
----> 1 df_train.isnull().sum()
      2 

NameError: name 'df_train' is not defined

## === cell 11
df_train = df_train.dropna(how="any", axis="rows")



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2066447606.py in <cell line: 0>()
----> 1 df_train = df_train.dropna(how="any", axis="rows")
      2 

NameError: name 'df_train' is not defined

## === cell 12
df_test = pd.read_csv("../input/test.csv")
df_test.head(5)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/880897196.py in <cell line: 0>()
----> 1 df_test = pd.read_csv("../input/test.csv")
      2 df_test.head(5)
      3 

NameError: name 'pd' is not defined

## === cell 13
df_test.describe()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4278432816.py in <cell line: 0>()
----> 1 df_test.describe()
      2 

NameError: name 'df_test' is not defined

## === cell 14
df_train["pickup_datetime"] = pd.to_datetime(
    df_train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S UTC"
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1090646201.py in <cell line: 0>()
----> 1 df_train["pickup_datetime"] = pd.to_datetime(
      2     df_train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S UTC"
      3 )
      4 

NameError: name 'pd' is not defined

## === cell 15
df_train["pickup_datetime"]



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3484439915.py in <cell line: 0>()
----> 1 df_train["pickup_datetime"]
      2 

NameError: name 'df_train' is not defined

## === cell 16
df_test["pickup_datetime"] = pd.to_datetime(
    df_test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S UTC"
)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/265290871.py in <cell line: 0>()
----> 1 df_test["pickup_datetime"] = pd.to_datetime(
      2     df_test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S UTC"
      3 )
      4 
      5 

NameError: name 'pd' is not defined

## === cell 17
def add_new_date_time_features(dataset):
    dataset["hour"] = dataset.pickup_datetime.dt.hour.astype("uint8")
    dataset["day"] = dataset.pickup_datetime.dt.day.astype("uint8")
    dataset["month"] = dataset.pickup_datetime.dt.month.astype("uint8")
    dataset["year"] = dataset.pickup_datetime.dt.year.astype("uint16")
    dataset["day_of_week"] = dataset.pickup_datetime.dt.dayofweek.astype("uint8")
    return dataset




## === cell 18
df_train = add_new_date_time_features(df_train)
df_test = add_new_date_time_features(df_test)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2876843891.py in <cell line: 0>()
----> 1 df_train = add_new_date_time_features(df_train)
      2 df_test = add_new_date_time_features(df_test)
      3 

NameError: name 'df_train' is not defined

## === cell 19
df_train.describe()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1262383953.py in <cell line: 0>()
----> 1 df_train.describe()
      2 

NameError: name 'df_train' is not defined

## === cell 20
lon_min, lon_max = df_test["pickup_longitude"].min(), df_test["pickup_longitude"].max()
lat_min, lat_max = df_test["pickup_latitude"].min(), df_test["pickup_latitude"].max()
drop_lon_min, drop_lon_max = (
    df_test["dropoff_longitude"].min(),
    df_test["dropoff_longitude"].max(),
)
drop_lat_min, drop_lat_max = (
    df_test["dropoff_latitude"].min(),
    df_test["dropoff_latitude"].max(),
)

mask = (
    df_train["pickup_longitude"].between(lon_min, lon_max)
    & df_train["pickup_latitude"].between(lat_min, lat_max)
    & df_train["dropoff_longitude"].between(drop_lon_min, drop_lon_max)
    & df_train["dropoff_latitude"].between(drop_lat_min, drop_lat_max)
)
df_train = df_train[mask].copy()



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/12479933.py in <cell line: 0>()
----> 1 lon_min, lon_max = df_test["pickup_longitude"].min(), df_test["pickup_longitude"].max()
      2 lat_min, lat_max = df_test["pickup_latitude"].min(), df_test["pickup_latitude"].max()
      3 drop_lon_min, drop_lon_max = (
      4     df_test["dropoff_longitude"].min(),
      5     df_test["dropoff_longitude"].max(),

NameError: name 'df_test' is not defined

## === cell 21
df_train.shape




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1522788542.py in <cell line: 0>()
----> 1 df_train.shape
      2 
      3 

NameError: name 'df_train' is not defined

## === cell 22
def calculate_abs_different(df):
    df["abs_diff_longitude"] = (
        (df.dropoff_longitude - df.pickup_longitude).abs().astype("float32")
    )
    df["abs_diff_latitude"] = (
        (df.dropoff_latitude - df.pickup_latitude).abs().astype("float32")
    )


calculate_abs_different(df_train)
calculate_abs_different(df_test)




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2129740480.py in <cell line: 0>()
      8 
      9 
---> 10 calculate_abs_different(df_train)
     11 calculate_abs_different(df_test)
     12 

NameError: name 'df_train' is not defined

## === cell 23
def convert_different_miles(df):
    df["abs_diff_longitude"] = (df.abs_diff_longitude * 50).astype(
        "float32"
    )  # approx. miles per degree longitude at NYC latitude
    df["abs_diff_latitude"] = (df.abs_diff_latitude * 69).astype(
        "float32"
    )  # approx. miles per degree latitude


convert_different_miles(df_train)
convert_different_miles(df_test)




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/260976631.py in <cell line: 0>()
      8 
      9 
---> 10 convert_different_miles(df_train)
     11 convert_different_miles(df_test)
     12 

NameError: name 'df_train' is not defined

## === cell 24
def add_distance(df):
    df["Euclidean"] = np.sqrt(
        df.abs_diff_latitude**2 + df.abs_diff_longitude**2
    ).astype("float32")
    df["delta_manh_long"] = (
        (
            df.Euclidean
            * np.sin(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude) - meas_ang)
        )
        .abs()
        .astype("float32")
    )
    df["delta_manh_lat"] = (
        (
            df.Euclidean
            * np.cos(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude) - meas_ang)
        )
        .abs()
        .astype("float32")
    )
    df["distance"] = (df.delta_manh_long + df.delta_manh_lat).astype("float32")

    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 3959  # Earth radius in miles
    df["haversine"] = (R * c).astype("float32")

    df.drop(
        [
            "abs_diff_longitude",
            "abs_diff_latitude",
            "Euclidean",
            "delta_manh_long",
            "delta_manh_lat",
        ],
        axis=1,
        inplace=True,
    )


meas_ang = 0.506  # 29 degrees = 0.506 radians (Commissioners' Plan of 1811)

add_distance(df_train)
add_distance(df_test)




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3121040687.py in <cell line: 0>()
     48 meas_ang = 0.506  # 29 degrees = 0.506 radians (Commissioners' Plan of 1811)
     49 
---> 50 add_distance(df_train)
     51 add_distance(df_test)
     52 

NameError: name 'df_train' is not defined

## === cell 25
def calculate_direction(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Approximate bearing between pickup and dropoff points.
    """
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(
        np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    )
    dlat = dropoff_lat - pickup_lat
    dlon = pickup_lon - dropoff_lon
    a = np.arctan2(
        np.sin(dlon * np.cos(dropoff_lat)),
        np.cos(pickup_lat) * np.sin(dropoff_lat)
        - np.sin(pickup_lat) * np.cos(dropoff_lat) * np.cos(dlon),
    )
    return a.astype("float32")




## === cell 26
df_train["direction"] = calculate_direction(
    df_train["pickup_latitude"].values,
    df_train["pickup_longitude"].values,
    df_train["dropoff_latitude"].values,
    df_train["dropoff_longitude"].values,
)

df_test["direction"] = calculate_direction(
    df_test["pickup_latitude"].values,
    df_test["pickup_longitude"].values,
    df_test["dropoff_latitude"].values,
    df_test["dropoff_longitude"].values,
)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1894868521.py in <cell line: 0>()
      1 df_train["direction"] = calculate_direction(
----> 2     df_train["pickup_latitude"].values,
      3     df_train["pickup_longitude"].values,
      4     df_train["dropoff_latitude"].values,
      5     df_train["dropoff_longitude"].values,

NameError: name 'df_train' is not defined

## === cell 27
pass



## === cell 28
df_train.drop(columns=["pickup_datetime"], inplace=True)

y = df_train["fare_amount"]
df_train = df_train.drop(columns=["fare_amount"])



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1803808913.py in <cell line: 0>()
----> 1 df_train.drop(columns=["pickup_datetime"], inplace=True)
      2 
      3 y = df_train["fare_amount"]
      4 df_train = df_train.drop(columns=["fare_amount"])
      5 

NameError: name 'df_train' is not defined

## === cell 29
df_train.head()



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2063835329.py in <cell line: 0>()
----> 1 df_train.head()
      2 

NameError: name 'df_train' is not defined

## === cell 30
from sklearn.model_selection import train_test_split
import lightgbm as lgbm

x_train, x_test, y_train, y_test = train_test_split(
    df_train, y, random_state=123, test_size=0.1
)



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2397063093.py in <cell line: 0>()
      3 
      4 x_train, x_test, y_train, y_test = train_test_split(
----> 5     df_train, y, random_state=123, test_size=0.1
      6 )
      7 

NameError: name 'df_train' is not defined

## === cell 31
params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "nthread": 4,
    "num_leaves": 255,
    "learning_rate": 0.03,
    "max_depth": -1,
    "subsample": 0.8,
    "bagging_fraction": 0.9,
    "feature_fraction": 0.9,
    "max_bin": 255,
    "bagging_freq": 10,
    "metric": "rmse",
    "zero_as_missing": True,
    "min_data_in_leaf": 20,
    "verbose": -1,
}

train_set = lgbm.Dataset(
    x_train, label=y_train, categorical_feature=["year", "month", "day", "day_of_week"]
)
valid_set = lgbm.Dataset(
    x_test, label=y_test, categorical_feature=["year", "month", "day", "day_of_week"]
)

callbacks = [lgbm.early_stopping(stopping_rounds=500, verbose=500)]

model = lgbm.train(
    params,
    train_set=train_set,
    num_boost_round=5000,
    valid_sets=[valid_set],
    callbacks=callbacks,
)



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3314187692.py in <cell line: 0>()
     18 
     19 train_set = lgbm.Dataset(
---> 20     x_train, label=y_train, categorical_feature=["year", "month", "day", "day_of_week"]
     21 )
     22 valid_set = lgbm.Dataset(

NameError: name 'x_train' is not defined

## === cell 32
df_train.describe()



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1262383953.py in <cell line: 0>()
----> 1 df_train.describe()
      2 

NameError: name 'df_train' is not defined

## === cell 33
test_key = df_test["key"]

df_test.drop(columns=["pickup_datetime", "key"], axis=1, inplace=True)



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/884115703.py in <cell line: 0>()
----> 1 test_key = df_test["key"]
      2 
      3 df_test.drop(columns=["pickup_datetime", "key"], axis=1, inplace=True)
      4 

NameError: name 'df_test' is not defined

## === cell 34
prediction = model.predict(df_test, num_iteration=model.best_iteration)



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1510488328.py in <cell line: 0>()
----> 1 prediction = model.predict(df_test, num_iteration=model.best_iteration)
      2 

NameError: name 'model' is not defined

## === cell 35
submission = pd.DataFrame({"key": test_key, "fare_amount": prediction})
submission.to_csv("taxi_fare_submission.csv", index=False)

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4258256167.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"key": test_key, "fare_amount": prediction})
      2 submission.to_csv("taxi_fare_submission.csv", index=False)

NameError: name 'pd' is not defined
