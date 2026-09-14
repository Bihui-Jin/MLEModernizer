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

8.39274

# 6. Current score

5.65799

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.50323) has done: 'I fix the pipeline-breaking pandas/seaborn incompatibilities (notably deprecated `.ix`, deprecated `dt.week`, and seaborn’s `size=` argument), and I ensure `pickup_datetime` is converted into numeric time features so scikit-learn scalers/models no longer see `Timestamp` objects. I also correct a longitude-clipping loop that mistakenly clips latitude twice, and I make the plotting/EDA cells non-fatal (so the notebook runs end-to-end in a Kaggle script environment). Finally, I ensure the prediction is inverse-transformed consistently (`exp(x) - 1`) and that a valid `my_submission.csv` with `key,fare_amount` is always written.'
- What this solution (achieved 5.65799) has done: 'To move RMSE down toward your target (8.39274) with minimal disruption, I’m keeping the exact same feature set, transformations (log1p target, sqrt distance), scaler, and GradientBoostingRegressor setup. The main score lever here is data quality: your current pipeline clips coordinates to global bounds (±90/±180), which leaves many NYC-taxi-invalid coordinates and bad rides that inflate error. I’m adding a small, standard NYC bounding-box filter on the training rows only (no leakage; test untouched) and also removing implausibly large fares (a common cleanup for this competition) while keeping everything else identical. Finally, I recompute `ride_distance_km` after coordinate cleanup (still the same haversine feature) so the distance feature is consistent with the cleaned coordinates.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from math import sin, cos, sqrt, atan2, radians
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn import ensemble
from sklearn.preprocessing import RobustScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

import os

INPUT_DIR = "/kaggle/input"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "../input"

print("Using INPUT_DIR =", INPUT_DIR)
print(os.listdir(INPUT_DIR)[:20])



## === cell 1
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
    TRAIN_PATH = os.path.join(
        INPUT_DIR, "new-york-city-taxi-fare-prediction", "train.csv"
    )
    TEST_PATH = os.path.join(
        INPUT_DIR, "new-york-city-taxi-fare-prediction", "test.csv"
    )

taxi_ride_train = pd.read_csv(
    TRAIN_PATH,
    sep=",",
    index_col="key",
    header=0,
    parse_dates=["pickup_datetime"],
    nrows=99999,
)
taxi_ride_test = pd.read_csv(
    TEST_PATH,
    sep=",",
    index_col="key",
    header=0,
    parse_dates=["pickup_datetime"],
)
taxi_ride_train.head()



## === cell 2
print("The shape train data are {0}".format((taxi_ride_train.shape)))
print("The shape test data are {0}".format((taxi_ride_test.shape)))



## === cell 3
taxi_ride_train.info()



## === cell 4
taxi_ride_test.info()



## === cell 5
taxi_ride_train.dtypes.value_counts().reset_index()



## === cell 6
taxi_ride_train.isnull().sum().sum()



## === cell 7
taxi_ride_test.isnull().sum().sum()



## === cell 8
taxi_ride_train = taxi_ride_train.dropna(axis=0)
taxi_ride_test = taxi_ride_test.dropna(axis=0)
print(taxi_ride_train.isnull().sum().sum())
print(taxi_ride_test.isnull().sum().sum())




## === cell 9
def calculate_distance(row):
    R = 6373.0  # approximate radius of earth in km
    lat1 = radians(row[0])
    lon1 = radians(row[1])
    lat2 = radians(row[2])
    lon2 = radians(row[3])
    longitude_distance = lon2 - lon1
    latitude_distance = lat2 - lat1
    a = (
        sin(latitude_distance / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(longitude_distance / 2) ** 2
    )
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance




## === cell 10
taxi_ride_train["ride_distance_km"] = taxi_ride_train[
    ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
].apply(calculate_distance, axis=1)
taxi_ride_test["ride_distance_km"] = taxi_ride_test[
    ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
].apply(calculate_distance, axis=1)



## === cell 11
taxi_ride_train["ride_distance_km"].describe()



## === cell 12
try:
    sns.boxplot(x=taxi_ride_train["ride_distance_km"])
    plt.show()
except Exception as e:
    print("Skipping plot (cell 13):", repr(e))



## === cell 13
IQR = taxi_ride_train.ride_distance_km.quantile(
    0.75
) - taxi_ride_train.ride_distance_km.quantile(0.25)
Lower_fence = taxi_ride_train.ride_distance_km.quantile(0.25) - (IQR * 3)
Upper_fence = taxi_ride_train.ride_distance_km.quantile(0.75) + (IQR * 3)
print(
    "Distance outliers are values < {lowerboundary} or > {upperboundary}".format(
        lowerboundary=Lower_fence, upperboundary=Upper_fence
    )
)



## === cell 14
distance_outlier_train = len(taxi_ride_train[taxi_ride_train["ride_distance_km"] >= 30])
distance_outlier_test = len(taxi_ride_test[taxi_ride_test["ride_distance_km"] >= 30])
print(
    "There are {0} trains rows and {1} test rows that have distance value more than 30km".format(
        distance_outlier_train, distance_outlier_test
    )
)



## === cell 15
taxi_ride_train["ride_distance_km"] = np.where(
    taxi_ride_train["ride_distance_km"].astype("float64") <= 30.0,
    taxi_ride_train["ride_distance_km"],
    30.0,
)
taxi_ride_train["ride_distance_km"] = np.where(
    taxi_ride_train["ride_distance_km"].astype("float64") >= 0.0,
    taxi_ride_train["ride_distance_km"],
    0.0,
)

taxi_ride_test["ride_distance_km"] = np.where(
    taxi_ride_test["ride_distance_km"].astype("float64") <= 30.0,
    taxi_ride_test["ride_distance_km"],
    30.0,
)
taxi_ride_test["ride_distance_km"] = np.where(
    taxi_ride_test["ride_distance_km"].astype("float64") >= 0.0,
    taxi_ride_test["ride_distance_km"],
    0.0,
)



## === cell 16
try:
    sns.boxplot(x=taxi_ride_train["ride_distance_km"])
    plt.show()
except Exception as e:
    print("Skipping plot (cell 17):", repr(e))



## === cell 17
try:
    sns.jointplot(x="ride_distance_km", y="fare_amount", data=taxi_ride_train)
    plt.show()
except Exception as e:
    print("Skipping plot (cell 18):", repr(e))



## === cell 18
pick_up_date_train = taxi_ride_train.loc[:, "pickup_datetime"]
pick_up_date_test = taxi_ride_test.loc[:, "pickup_datetime"]

temp_df_train = pd.DataFrame(
    {
        "year": pick_up_date_train.dt.year,
        "month": pick_up_date_train.dt.month,
        "day": pick_up_date_train.dt.day,
        "hour": pick_up_date_train.dt.hour,
        "dayofyear": pick_up_date_train.dt.dayofyear,
        "week": pick_up_date_train.dt.isocalendar().week.astype(int),
        "weekday": pick_up_date_train.dt.weekday,
        "quarter": pick_up_date_train.dt.quarter,
    },
    index=taxi_ride_train.index,
)

temp_df_test = pd.DataFrame(
    {
        "year": pick_up_date_test.dt.year,
        "month": pick_up_date_test.dt.month,
        "day": pick_up_date_test.dt.day,
        "hour": pick_up_date_test.dt.hour,
        "dayofyear": pick_up_date_test.dt.dayofyear,
        "week": pick_up_date_test.dt.isocalendar().week.astype(int),
        "weekday": pick_up_date_test.dt.weekday,
        "quarter": pick_up_date_test.dt.quarter,
    },
    index=taxi_ride_test.index,
)

taxi_ride_train = pd.concat([taxi_ride_train, temp_df_train], axis=1)
taxi_ride_test = pd.concat([taxi_ride_test, temp_df_test], axis=1)

taxi_ride_train.drop("pickup_datetime", inplace=True, axis=1)
taxi_ride_test.drop("pickup_datetime", inplace=True, axis=1)

taxi_ride_train.head()



## === cell 19
taxi_ride_train.dtypes.value_counts().reset_index()



## === cell 20
print(
    "The new dataset contains {0} null entries ".format(
        taxi_ride_train.isnull().sum().sum()
    )
)



## === cell 21
try:
    sns.histplot(taxi_ride_train["fare_amount"], kde=True)
    plt.show()
except Exception as e:
    print("Skipping plot (cell 22):", repr(e))



## === cell 22
taxi_ride_train["fare_amount"].describe()



## === cell 23
length_before = len(taxi_ride_train)
taxi_ride_train = taxi_ride_train[taxi_ride_train.fare_amount >= 0.0]
length_after = len(taxi_ride_train)
print("No of rows removed {0}".format(length_before - length_after))



## === cell 24
length_before = len(taxi_ride_train)
taxi_ride_train = taxi_ride_train[taxi_ride_train["fare_amount"] <= 250.0]
length_after = len(taxi_ride_train)
print(
    "No of rows removed due to very large fare_amount (>250) {0}".format(
        length_before - length_after
    )
)



## === cell 25
print("Skweness before transformation {0}".format(taxi_ride_train.fare_amount.skew()))
try:
    sns.histplot(np.log1p(taxi_ride_train["fare_amount"]), kde=True)
    plt.show()
except Exception:
    pass
taxi_ride_train["fare_amount"] = np.log1p(taxi_ride_train["fare_amount"])
print("Skweness after transformation {0}".format(taxi_ride_train.fare_amount.skew()))



## === cell 26
Y_train = taxi_ride_train.fare_amount
X_train = taxi_ride_train.drop("fare_amount", axis=1)
X_test = taxi_ride_test
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X_train, Y_train, test_size=0.33, random_state=42
)
print("Shape of training set is {0}".format(X_train.shape))
print("Shape of Validation set is {0}".format(X_valid.shape))
print("Shape of testing set is {0}".format(X_test.shape))



## === cell 27
discrete_col_list = []
continous_col_list = []
for col in X_train.columns.tolist():
    if (taxi_ride_train[col].value_counts().count() / len(taxi_ride_train)) < 0.1:
        discrete_col_list.append(col)
    else:
        continous_col_list.append(col)
print("The descrete column in our data are {0}".format(discrete_col_list))
print("The continous column in our data are {0}".format(continous_col_list))



## === cell 28
for var in continous_col_list:
    if var not in taxi_ride_train.columns:
        continue
    try:
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        fig = taxi_ride_train.boxplot(column=var)
        fig.set_title("")

        plt.subplot(1, 2, 2)
        fig = taxi_ride_train[var].hist(bins=20)
        fig.set_xlabel(var)
        plt.show()
    except Exception as e:
        print(f"Skipping plot for {var} (cell 28):", repr(e))



## === cell 29
latitude_upper_range = 90.0
latitude_lower_range = -90.0
for var in ["pickup_latitude", "dropoff_latitude"]:
    taxi_ride_train[var] = np.where(
        taxi_ride_train[var].astype("float64") <= latitude_upper_range,
        taxi_ride_train[var],
        latitude_upper_range,
    )
    taxi_ride_train[var] = np.where(
        taxi_ride_train[var].astype("float64") >= latitude_lower_range,
        taxi_ride_train[var],
        latitude_lower_range,
    )

    taxi_ride_test[var] = np.where(
        taxi_ride_test[var].astype("float64") <= latitude_upper_range,
        taxi_ride_test[var],
        latitude_upper_range,
    )
    taxi_ride_test[var] = np.where(
        taxi_ride_test[var].astype("float64") >= latitude_lower_range,
        taxi_ride_test[var],
        latitude_lower_range,
    )

longitude_upper_range = 180.0
longitude_lower_range = -180.0
for var in ["pickup_longitude", "dropoff_longitude"]:
    taxi_ride_train[var] = np.where(
        taxi_ride_train[var].astype("float64") <= longitude_upper_range,
        taxi_ride_train[var],
        longitude_upper_range,
    )
    taxi_ride_train[var] = np.where(
        taxi_ride_train[var].astype("float64") >= longitude_lower_range,
        taxi_ride_train[var],
        longitude_lower_range,
    )

    taxi_ride_test[var] = np.where(
        taxi_ride_test[var].astype("float64") <= longitude_upper_range,
        taxi_ride_test[var],
        longitude_upper_range,
    )
    taxi_ride_test[var] = np.where(
        taxi_ride_test[var].astype("float64") >= longitude_lower_range,
        taxi_ride_test[var],
        longitude_lower_range,
    )



## === cell 30
nyc_lat_min, nyc_lat_max = 40.0, 42.0
nyc_lon_min, nyc_lon_max = -75.0, -72.0

length_before = len(taxi_ride_train)
coord_mask = (
    taxi_ride_train["pickup_latitude"].between(nyc_lat_min, nyc_lat_max)
    & taxi_ride_train["dropoff_latitude"].between(nyc_lat_min, nyc_lat_max)
    & taxi_ride_train["pickup_longitude"].between(nyc_lon_min, nyc_lon_max)
    & taxi_ride_train["dropoff_longitude"].between(nyc_lon_min, nyc_lon_max)
)

taxi_ride_train = taxi_ride_train.loc[coord_mask].copy()
length_after = len(taxi_ride_train)
print(
    "No of rows removed due to non-NYC coordinates {0}".format(
        length_before - length_after
    )
)

taxi_ride_train["ride_distance_km"] = taxi_ride_train[
    ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
].apply(calculate_distance, axis=1)
taxi_ride_train["ride_distance_km"] = np.where(
    taxi_ride_train["ride_distance_km"].astype("float64") <= 30.0,
    taxi_ride_train["ride_distance_km"],
    30.0,
)
taxi_ride_train["ride_distance_km"] = np.where(
    taxi_ride_train["ride_distance_km"].astype("float64") >= 0.0,
    taxi_ride_train["ride_distance_km"],
    0.0,
)



## === cell 31
for var in continous_col_list:
    if var not in taxi_ride_train.columns:
        continue
    try:
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        fig = taxi_ride_train.boxplot(column=var)
        fig.set_title("")

        plt.subplot(1, 2, 2)
        fig = taxi_ride_train[var].hist(bins=20)
        fig.set_xlabel(var)
        plt.show()
    except Exception as e:
        print(f"Skipping plot for {var} (cell 30):", repr(e))



## === cell 32
try:
    sns.histplot(np.sqrt(taxi_ride_train["ride_distance_km"]), kde=True)
    plt.show()
except Exception:
    pass
taxi_ride_train["ride_distance_km"] = np.sqrt(taxi_ride_train["ride_distance_km"])
taxi_ride_test["ride_distance_km"] = np.sqrt(taxi_ride_test["ride_distance_km"])



## === cell 33
for i, var in enumerate(discrete_col_list):
    if var not in taxi_ride_train.columns:
        continue
    try:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        sns.countplot(x=taxi_ride_train[var], ax=ax)
        plt.show()
    except Exception as e:
        print(f"Skipping countplot for {var} (cell 32):", repr(e))



## === cell 34
try:
    sns.pairplot(
        taxi_ride_train.sample(n=min(2000, len(taxi_ride_train)), random_state=42),
        x_vars=[
            c
            for c in continous_col_list
            if c in taxi_ride_train.columns and c != "fare_amount"
        ],
        y_vars="fare_amount",
        height=3,
        aspect=1.0,
        kind="reg",
    )
    plt.show()
except Exception as e:
    print("Skipping pairplot (cell 33):", repr(e))



## === cell 35
try:
    corr = X_train.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr)
    plt.show()
except Exception as e:
    print("Skipping heatmap (cell 34):", repr(e))



## === cell 36
try:
    taxi_ride_train.groupby("hour")["fare_amount"].sum().plot()
    plt.show()
except Exception as e:
    print("Skipping groupby plot (cell 35):", repr(e))



## === cell 37
try:
    taxi_ride_train.groupby("weekday")["fare_amount"].sum().plot()
    plt.show()
except Exception as e:
    print("Skipping groupby plot (cell 36):", repr(e))



## === cell 38
try:
    taxi_ride_train.groupby("passenger_count")["fare_amount"].sum().plot()
    plt.show()
except Exception as e:
    print("Skipping groupby plot (cell 37):", repr(e))



## === cell 39
try:
    taxi_ride_train.groupby("month")["fare_amount"].sum().plot()
    plt.show()
except Exception as e:
    print("Skipping groupby plot (cell 38):", repr(e))



## === cell 40
try:
    taxi_ride_train.groupby("year")["fare_amount"].sum().plot()
    plt.show()
except Exception as e:
    print("Skipping groupby plot (cell 39):", repr(e))



## === cell 41
try:
    pd.crosstab(
        taxi_ride_train["quarter"],
        np.ones(len(taxi_ride_train), dtype=int),
        margins=True,
    )
except Exception as e:
    print("Skipping crosstab (cell 40):", repr(e))



## === cell 42
constant_features = [
    feat for feat in taxi_ride_train.columns if taxi_ride_train[feat].std() == 0
]
print(constant_features)



## === cell 43
Y_train = taxi_ride_train.fare_amount
X_train = taxi_ride_train.drop("fare_amount", axis=1)
X_test = taxi_ride_test

X_train, X_valid, Y_train, Y_valid = train_test_split(
    X_train, Y_train, test_size=0.33, random_state=42
)

Y_train = taxi_ride_train.loc[X_train.index, "fare_amount"]
Y_valid = taxi_ride_train.loc[X_valid.index, "fare_amount"]

sel_ = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
sel_.fit(X_train, Y_train)
selected_feat = X_train.columns[(sel_.get_support())]
print(
    "So the feature that holds highest importance are {0}".format(list(selected_feat))
)




## === cell 44
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr(numeric_only=True)
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:  # absolute coeff
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


corr_features = correlation(X_train, 0.8)
print("The features that are corelated with each other are {0}".format(corr_features))
X_train.drop(labels=corr_features, axis=1, inplace=True)
X_valid.drop(labels=corr_features, axis=1, inplace=True)
X_test.drop(labels=corr_features, axis=1, inplace=True)
print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)



## === cell 45
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)



## === cell 46
regr = linear_model.LinearRegression()
regr.fit(X_train_scaled, Y_train)
Y_valid_pred = regr.predict(X_valid_scaled)
Y_test_pred = regr.predict(X_test_scaled)
print("Coefficients: \n", regr.coef_)
print("Mean squared error: %.4f" % mean_squared_error(Y_valid, Y_valid_pred))
print("Variance score: %.4f" % r2_score(Y_valid, Y_valid_pred))




## === cell 47
def generate_residual_plot(label, prediction, type):
    plt.scatter(prediction, np.subtract(label, prediction))
    title = "Residual plot for predicting " + type
    plt.title(title)
    plt.xlabel("Fitted Value")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.hlines(
        y=0,
        xmin=float(np.min(prediction)),
        xmax=float(np.max(prediction)),
        colors="orange",
        linewidth=3,
    )




## === cell 48
def generate_actual_vs_predicted_plot(label, prediction, type):
    plt.scatter(prediction, label, s=30, c="r", marker="+", zorder=10)
    title = "Actual vs Predicted values for " + type
    plt.title(title)
    plt.xlabel("Predicted Values from model")
    plt.ylabel("Actual Values")
    plt.tight_layout()




## === cell 49
try:
    generate_residual_plot(Y_valid, Y_valid_pred, "Taxi fares")
    plt.show()
except Exception as e:
    print("Skipping residual plot (cell 48):", repr(e))



## === cell 50
try:
    generate_actual_vs_predicted_plot(Y_valid, Y_valid_pred, "Taxi fares")
    plt.show()
except Exception as e:
    print("Skipping actual-vs-pred plot (cell 49):", repr(e))



## === cell 51
params = {
    "n_estimators": 700,
    "max_depth": 2,
    "min_samples_split": 2,
    "learning_rate": 0.01,
    "loss": "squared_error",
    "random_state": 42,
}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train_scaled, Y_train)
mse = mean_squared_error(Y_valid, clf.predict(X_valid_scaled))
print("MSE: %.4f" % mse)
print("Variance score: %.4f" % r2_score(Y_valid, clf.predict(X_valid_scaled)))



## === cell 52
test_pred = pd.DataFrame(clf.predict(X_test_scaled), index=X_test.index)
test_pred.columns = ["fare_amount"]
test_pred["fare_amount"] = np.expm1(test_pred["fare_amount"])
test_pred["fare_amount"] = test_pred["fare_amount"].clip(lower=0.0)

submission = test_pred.reset_index().rename(columns={"index": "key"})
submission = submission[["key", "fare_amount"]]
submission.to_csv("my_submission.csv", index=False)
print("Wrote my_submission.csv with shape:", submission.shape)
submission.head()



## === cell 53
try:
    feature_importance = clf.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    plt.figure(figsize=(10, 6))
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, X_train.columns[sorted_idx])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Skipping feature importance plot (cell 52):", repr(e))



## === cell 54
try:
    generate_actual_vs_predicted_plot(
        Y_valid, clf.predict(X_valid_scaled), "Taxi fares"
    )
    plt.show()
except Exception as e:
    print("Skipping actual-vs-pred plot (cell 53):", repr(e))



## === cell 55
try:
    generate_residual_plot(Y_valid, clf.predict(X_valid_scaled), "Taxi fares")
    plt.show()
except Exception as e:
    print("Skipping residual plot (cell 54):", repr(e))
