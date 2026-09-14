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

3.8987

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5.15099) has done: 'I fix the runtime errors by updating deprecated dtype checks (`np.object`), correcting the datetime feature function (it currently returns `None` due to `inplace=True`), and ensuring scikit-learn parameters like `max_iter` are integers. I also fix the Haversine calculation bugs (wrong lambda and swapped diffs) because they directly degrade feature quality and therefore RMSE, while keeping the same overall feature approach and RandomForest training logic. To avoid crashing in notebook-to-script execution, I remove the IPython magic and make the plotting optional. Finally, I ensure the submission is written with the required columns (`key`, `fare_amount`) and a `.csv` suffix.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

CANDIDATE_INPUT_DIRS = [
    "../input",
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/data/new-york-city-taxi-fare-prediction",
    "/kaggle/input/new-york-city-taxi-fare-prediction",
]

INPUT_DIR = None
for d in CANDIDATE_INPUT_DIRS:
    if (
        os.path.exists(d)
        and os.path.isfile(os.path.join(d, "train.csv"))
        and os.path.isfile(os.path.join(d, "test.csv"))
    ):
        INPUT_DIR = d
        break

if INPUT_DIR is None:
    INPUT_DIR = "../input" if os.path.exists("../input") else "/kaggle/input"
    raise FileNotFoundError(
        f"Could not find train.csv/test.csv in any of: {CANDIDATE_INPUT_DIRS}. "
        f"Last tried INPUT_DIR={INPUT_DIR} with files={os.listdir(INPUT_DIR)[:20] if os.path.exists(INPUT_DIR) else 'MISSING'}"
    )

print("Input dir:", INPUT_DIR)
print("Files:", sorted(os.listdir(INPUT_DIR))[:50])



## === cell 1
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"), nrows=1_000_000)
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

train.head()



## === cell 2
test.head()



## === cell 3
train.shape



## === cell 4
test.shape



## === cell 5
train.dtypes.value_counts()



## === cell 6
test.dtypes.value_counts()



## === cell 7
train.isnull().sum()



## === cell 8
train = train.dropna()
train.isnull().sum()



## === cell 9
(train == 0).astype(int).sum()



## === cell 10
train = train[
    (train["fare_amount"] > 0)
    & (train["fare_amount"] < 500)
    & (train["passenger_count"] >= 1)
    & (train["passenger_count"] <= 6)
    & (train["pickup_longitude"].between(-74.3, -73.6))
    & (train["dropoff_longitude"].between(-74.3, -73.6))
    & (train["pickup_latitude"].between(40.4, 41.0))
    & (train["dropoff_latitude"].between(40.4, 41.0))
].copy()



## === cell 11
test = test[
    (test["passenger_count"] >= 1)
    & (test["passenger_count"] <= 6)
    & (test["pickup_longitude"].between(-74.3, -73.6))
    & (test["dropoff_longitude"].between(-74.3, -73.6))
    & (test["pickup_latitude"].between(40.4, 41.0))
    & (test["dropoff_latitude"].between(40.4, 41.0))
].copy()



## === cell 12
(train == 0).astype(int).sum()



## === cell 13
train.shape



## === cell 14
train.describe()



## === cell 15
train.describe()



## === cell 16
train.dtypes.value_counts()



## === cell 17
object_data = train.dtypes == object
categoricals = train.columns[object_data]
categoricals



## === cell 18
train.drop("key", axis=1, inplace=True)
train.head()



## === cell 19
import datetime as dt


def date_extraction(data: pd.DataFrame) -> pd.DataFrame:
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
    data["year"] = data["pickup_datetime"].dt.year
    data["month"] = data["pickup_datetime"].dt.month
    data["weekday"] = data[
        "pickup_datetime"
    ].dt.day  # keep original behavior (day-of-month)
    data["hour"] = data["pickup_datetime"].dt.hour
    data.drop("pickup_datetime", axis=1, inplace=True)
    return data


date_extraction(train)



## === cell 20
train.head()



## === cell 21
date_extraction(test)
test.head()




## === cell 22
def long_lat_distance(x: pd.DataFrame) -> pd.DataFrame:
    x["Longitude_distance"] = np.radians(x["pickup_longitude"] - x["dropoff_longitude"])
    x["Latitude_distance"] = np.radians(x["pickup_latitude"] - x["dropoff_latitude"])
    x["distance_travelled/10e3"] = (
        (x["Longitude_distance"] ** 2 + x["Latitude_distance"] ** 2) ** 0.5
    ) * 1000
    return x




## === cell 23
for x in [train, test]:
    long_lat_distance(x)

train.head()




## === cell 24
def harvesine(x: pd.DataFrame) -> pd.DataFrame:
    r = 6371000.0
    theta_1 = np.radians(x["dropoff_latitude"])
    theta_2 = np.radians(x["pickup_latitude"])
    lambda_1 = np.radians(x["dropoff_longitude"])
    lambda_2 = np.radians(x["pickup_longitude"])

    theta_diff = theta_2 - theta_1
    lambda_diff = lambda_2 - lambda_1

    a = (
        np.sin(theta_diff / 2.0) ** 2
        + np.cos(theta_1) * np.cos(theta_2) * np.sin(lambda_diff / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(a**0.5, (1.0 - a) ** 0.5)
    x["harvesine/km"] = (r * c) / 1000.0
    return x




## === cell 25
for x in [train, test]:
    harvesine(x)

train.head()



## === cell 26
train.dtypes.value_counts()



## === cell 27
train.head()



## === cell 28
test.head()



## === cell 29
train.describe()



## === cell 30
print("Are there any nulls in the train data:")
print(train.isnull().sum())

print("\nAre there any nulls in the test data:")
print(test.isnull().sum())



## === cell 31
train["harvesine/km"] = train["harvesine/km"].fillna(train["harvesine/km"].median())
test["harvesine/km"] = test["harvesine/km"].fillna(train["harvesine/km"].median())



## === cell 32
from sklearn.ensemble import RandomForestRegressor

feature_cols = [c for c in train.columns if c != "fare_amount"]
X = train[feature_cols]
y = train["fare_amount"]



## === cell 33
correlations = X.corrwith(y)
correlations = abs(correlations * 100)
correlations.sort_values(ascending=False, inplace=True)
correlations



## === cell 34
try:
    import matplotlib.pyplot as plt

    ax = correlations.plot(kind="bar")
    ax.set(ylim=[-1, 100], ylabel="abs(pearson correlation)*100")
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 35
train.head()



## === cell 36
train_1 = train.drop(
    [
        "pickup_longitude",
        "dropoff_longitude",
        "pickup_latitude",
        "dropoff_latitude",
        "Longitude_distance",
        "Latitude_distance",
    ],
    axis=1,
)
train_1.head()



## === cell 37
train_1.head()



## === cell 38
train_1.describe()



## === cell 39
test_1 = test.drop(
    [
        "pickup_longitude",
        "dropoff_longitude",
        "pickup_latitude",
        "dropoff_latitude",
        "Longitude_distance",
        "Latitude_distance",
    ],
    axis=1,
)
test_1.head()



## === cell 40
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoCV, RidgeCV


def rmse(ytrue, ypredicted):
    return np.sqrt(mean_squared_error(ytrue, ypredicted))




## === cell 41
feat_cols = [c for c in train_1.columns if c != "fare_amount"]
X_1 = train_1[feat_cols]
y_1 = train_1["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X_1, y_1, test_size=0.25, random_state=42
)



## === cell 42
lr = LinearRegression().fit(X_train, y_train)
lr_rmse = rmse(y_test, lr.predict(X_test))
print(lr_rmse)



## === cell 43
try:
    import matplotlib.pyplot as plt

    f = plt.figure(figsize=(6, 6))
    ax = plt.axes()
    ax.plot(y_test, lr.predict(X_test), marker="o", ls="")
    lim = (0, y_test.max())
    ax.set(
        xlabel="actual fare amount",
        ylabel="predicted_amount",
        xlim=lim,
        ylim=(0, 100),
        title="Linear regression results",
    )
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 44
alphas = [0.05, 0.03, 0.01, 0.5, 0.3, 0.1, 1, 3, 5]
rr = RidgeCV(alphas=alphas, cv=4).fit(X_train, y_train)
rr_rmse = rmse(y_test, rr.predict(X_test))
print(rr.alpha_, rr_rmse)



## === cell 45
alphas = np.array([0.05, 0.03, 0.01, 0.5, 0.3, 0.1, 1, 3, 5])
la = LassoCV(alphas=alphas, max_iter=int(5e4), cv=4).fit(X_train, y_train)
la_rmse = rmse(y_test, la.predict(X_test))
print(la.alpha_, la_rmse)



## === cell 46
l1_ratios = np.linspace(0.1, 0.5, 5)
alphas = np.array([0.05, 0.03, 0.01, 0.5, 0.3, 0.1, 1, 3, 5])

en = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, max_iter=int(1e4), cv=4).fit(
    X_train, y_train
)
en_rmse = rmse(y_test, en.predict(X_test))
print(en.alpha_, en.l1_ratio_, en_rmse)



## === cell 47
rf = RandomForestRegressor(n_estimators=100, max_features=5, random_state=42, n_jobs=-1)
rf = rf.fit(X_train, y_train)



## === cell 48
labels = ["Linear", "lasso", "Ridge", "Elastic-Net"]
models_rmse = [lr_rmse, la_rmse, rr_rmse, en_rmse]
rmse_df = pd.Series(models_rmse, index=labels).to_frame()
rmse_df.rename(columns={0: "Errors"}, inplace=True)
rmse_df



## === cell 49
test.head()



## === cell 50
test_1 = test_1.copy()
test_keys = test_1["key"].copy()
test_1.drop("key", axis=1, inplace=True)
test_1.head()



## === cell 51
final_prediction = rf.predict(test_1)

final_prediction = np.clip(final_prediction, 0.0, None)

NYCtaxiFare_submission = pd.DataFrame(
    {"key": test_keys, "fare_amount": final_prediction}
)

NYCtaxiFare_submission.to_csv("NYCtaxiFare_prediction.csv", index=False)

NYCtaxiFare_submission.head()
