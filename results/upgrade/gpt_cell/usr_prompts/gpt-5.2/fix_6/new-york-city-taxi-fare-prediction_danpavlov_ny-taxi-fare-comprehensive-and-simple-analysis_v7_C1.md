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

9.1952

# 6. Current score

608.86617

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 33.57496) has done: 'Diagnosis: The crash happens because `seaborn.lmplot` in seaborn 0.12+ no longer accepts `x`, `y`, and `data` as positional arguments; they must be passed as keyword arguments. In cell 21, three positional arguments are provided, triggering the `TypeError`. This is an API change, not a data issue. The fix is to call `lmplot` with `x=`, `y=`, and `data=` explicitly while keeping the exact same plotting semantics.

Patch summary: Update the `sns.lmplot` call in cell 21 to use keyword arguments (`x`, `y`, `data`) so it works with seaborn 0.12.2. No other logic or variables are changed.

Updated cells: (cell 21 only)

Compatibility notes for cell k+1: Cell 22 is unaffected because it uses `sns.heatmap` and relies only on `train` existing; this patch does not modify `train` or any downstream variables.

Assumptions: Seaborn version is 0.12.2 as listed, and the intent of cell 21 is only exploratory visualization (no variables from the plot are used later).'
- What this solution (achieved 18.87877) has done: 'Your score is far above the target (RMSE 33.57 vs 9.1952, lower is better), so we should improve it with minimal, core-logic-preserving changes. The biggest issue is the distance feature: it’s computed via a slow row-iteration loop and (as written) can also leave many rows with missing/incorrect distances depending on pandas’ behavior; replacing it with a vectorized haversine distance keeps the same feature intent but makes it correct and consistent for all rows. Next, align evaluation and prediction sanity by computing RMSE with correct argument order and clipping negative fare predictions to zero (fares can’t be negative), which typically reduces RMSE without changing the model. Finally, keep the same LinearRegression approach and same feature set, but add a fixed random_state for stable splits and write a properly named `.csv` submission.'
- What this solution (achieved 608.86617) has done: 'Your current RMSE (18.88) is much worse than the target (9.20, lower is better), so we should improve accuracy with very small, metric-aligned changes while keeping the same LinearRegression core. The main issue is the feature set: you’re dropping strong location/time predictors and training only on `passenger_count`, `distance`, `month`, `year`, which typically yields weak performance; keeping the same model but adding back the raw lat/lon and simple time parts (dayofweek/hour) is a minimal change that usually drops RMSE substantially. We also add a couple of standard data-cleaning filters (cap extreme fares and remove zero-distance trips) that reduce outliers without changing the approach. Finally, we keep submission formatting identical but ensure the same feature columns are used for train and test.'
- What this solution (achieved 608.86617) has done: 'Your current RMSE (608.87, lower is better) is far worse than the target (9.20), which strongly suggests a submission alignment/format issue rather than a pure modeling issue. The smallest high-impact fix is to ensure the test `key` and predictions are aligned 1:1 in the exact same order as `sample_submission.csv` (some evaluation pipelines are sensitive to ordering/misalignment). To make this robust while keeping the same LinearRegression + feature set, we reindex predictions to the `sample_submission` key order and also guard against any NaNs/infs in engineered features by filling them with train medians. These changes preserve the core logic and typically move the score dramatically closer to what the model actually achieves.'
- What this solution (achieved 608.86617) has done: 'Your RMSE is catastrophically worse than the model’s own validation RMSE, which strongly indicates the submitted predictions are not aligned to the right `key`s (or many keys ended up with NaN and got filled with a bad fallback). I make the submission generation fully alignment-safe by building predictions indexed by `key`, aggregating any duplicate keys, and then reindexing exactly to the `sample_submission` key order (no merge ambiguity). I also ensure the test feature engineering cannot introduce NaNs/infs that cascade into fallback fills by filling test NaNs with training medians consistently and coercing datetime parsing identically. These are minimal changes that keep the same LinearRegression, features, and training logic, but should move the Kaggle score dramatically closer to the true model performance (toward the 9.1952 target).'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from geopy.distance import great_circle
from sklearn import metrics, ensemble, linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 1
print(os.listdir("../input"))



## === cell 2
test = pd.read_csv("../input/test.csv")



## === cell 3
test.dtypes



## === cell 4
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}



## === cell 5
train = pd.read_csv("../input/train.csv", nrows=500000, dtype=types)



## === cell 6
train.head()



## === cell 7
train.describe()



## === cell 8
sns.distplot(train["fare_amount"])



## === cell 9
sns.distplot(train["passenger_count"])



## === cell 10
train.isnull().sum()



## === cell 11
train.dropna(inplace=True)



## === cell 12
train = train[train["fare_amount"] > 0]
train = train[train["fare_amount"] < 250]  # cap extreme fares (common cleanup)
train = train[train["pickup_longitude"] < -72]
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
train = train[train["dropoff_longitude"] < -72]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]



## === cell 13
train.describe()




## === cell 14
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088  # mean Earth radius in km
    lat1 = np.radians(lat1.astype("float64"))
    lon1 = np.radians(lon1.astype("float64"))
    lat2 = np.radians(lat2.astype("float64"))
    lon2 = np.radians(lon2.astype("float64"))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))


def dist_calc(df):
    df["distance"] = haversine_km(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
    ).astype("float32")




## === cell 15
dist_calc(train)
dist_calc(test)



## === cell 16
train = train[train["distance"] > 0]



## === cell 17
train["pickup_datetime"] = train["pickup_datetime"].str.replace(" UTC", "", regex=False)
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)



## === cell 18
test["pickup_datetime"] = test["pickup_datetime"].str.replace(" UTC", "", regex=False)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)



## === cell 19
train.dropna(subset=["pickup_datetime"], inplace=True)

train["dayofweek"] = train.pickup_datetime.dt.dayofweek
train["hour"] = train.pickup_datetime.dt.hour
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year

test["dayofweek"] = test.pickup_datetime.dt.dayofweek
test["hour"] = test.pickup_datetime.dt.hour
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year



## === cell 20
test.head()



## === cell 21
sns.lmplot(x="year", y="fare_amount", data=train[["year", "fare_amount"]])



## === cell 22
sns.heatmap(
    train.drop(
        [
            "key",
            "pickup_datetime",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
        ],
        axis=1,
    ).corr()
)



## === cell 23
feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance",
    "dayofweek",
    "hour",
    "month",
    "year",
]
X = train[feature_cols].copy()
y = train["fare_amount"]



## === cell 24
X.head()



## === cell 25
y.head()



## === cell 26
X = X.replace([np.inf, -np.inf], np.nan)
feat_medians = X.median(numeric_only=True)
X = X.fillna(feat_medians)



## === cell 27
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 28
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_test, y_test))



## === cell 29
y_pred = lm.predict(X_test)
y_pred = np.clip(y_pred, 0, None)
lrmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
lrmse




## === cell 30
def get_score(prediction, lables):
    print("R2: {}".format(r2_score(lables, prediction)))
    print("RMSE: {}".format(np.sqrt(mean_squared_error(lables, prediction))))


def train_test(estimator, x_trn, x_tst, y_trn, y_tst):
    prediction_train = estimator.predict(x_trn)
    prediction_train = np.clip(prediction_train, 0, None)
    print(estimator)
    get_score(prediction_train, y_trn)
    prediction_test = estimator.predict(x_tst)
    prediction_test = np.clip(prediction_test, 0, None)
    print("Test")
    get_score(prediction_test, y_tst)




## === cell 31
train_test(lm, X_train, X_test, y_train, y_test)



## === cell 32
Xtest = test[feature_cols].copy()



## === cell 33
Xtest = Xtest.replace([np.inf, -np.inf], np.nan)
Xtest = Xtest.fillna(feat_medians)



## === cell 34
LinearPredictions = lm.predict(Xtest)
LinearPredictions = np.clip(LinearPredictions, 0, None)
LinearPredictions = np.round(LinearPredictions, decimals=2)
LinearPredictions



## === cell 35
LinearPredictions.size



## === cell 36
sample_sub = pd.read_csv("../input/sample_submission.csv")

pred_by_key = pd.DataFrame(
    {"key": test["key"].values, "fare_amount": LinearPredictions}
)
pred_by_key = pred_by_key.groupby("key", as_index=True)["fare_amount"].mean()

submission = sample_sub.copy()
submission["fare_amount"] = submission["key"].map(pred_by_key)

fallback_fare = float(train["fare_amount"].median())
submission["fare_amount"] = (
    submission["fare_amount"].fillna(fallback_fare).astype("float32")
)

submission["fare_amount"] = (
    submission["fare_amount"]
    .replace([np.inf, -np.inf], fallback_fare)
    .fillna(fallback_fare)
)

submission



## === cell 37
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("NaN fare_amount count:", int(submission["fare_amount"].isna().sum()))
print("Key match with sample_submission:", submission["key"].equals(sample_sub["key"]))
