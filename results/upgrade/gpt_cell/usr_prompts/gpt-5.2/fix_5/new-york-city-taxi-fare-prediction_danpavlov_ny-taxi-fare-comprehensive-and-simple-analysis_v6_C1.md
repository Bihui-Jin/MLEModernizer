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

2.1362993276757745e+29

# 6. Current score

5.29986

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 2.9479859176772975e+31) has done: 'Diagnosis: The crash happens because recent seaborn versions (0.12+) changed `sns.lmplot` to require keyword arguments; passing `x`, `y`, and `data` positionally now raises `TypeError`. The plotting intent is unchanged (regression plot of `fare_amount` vs `year`), so we should only update the call signature. This fix is localized to cell 21 and does not affect downstream variables used in cell 22.

Patch summary: Modify the `sns.lmplot` call to use explicit `x=`, `y=`, and `data=` keyword arguments, preserving the same subset of columns.

Updated cells: (cell 21 only)

Compatibility notes for cell k+1: No variables are created/modified by this plotting cell; `train` remains unchanged, so `X` and `y` creation in cell 22 behave identically.

Assumptions: Seaborn is version 0.12.2 as listed, and the goal of cell 21 is visualization only (no side effects required for later cells).'
- What this solution (achieved 1.5224883417791003e+30) has done: 'Diagnosis: Cell 26 crashes because it references `X_valid` and `y_valid`, which are never created; the split in cell 25 produces `X_train, X_test, y_train, y_test` only. The intended “validation” scoring is therefore just the held-out test split created in cell 25.  
Patch summary: Replace the undefined `X_valid, y_valid` in the second `lm.score(...)` call with `X_test, y_test` so the cell runs and preserves the existing train/test evaluation semantics.  
Updated cells: Only cell 26 is changed.  
Compatibility notes for cell k+1: Variables `lm`, `X_test`, and `y_test` remain unchanged and are still available for cell 27’s prediction and RMSE computation.  
Assumptions: The author intended to evaluate the model on the held-out split from `train_test_split` (named `X_test, y_test`), not on a separate validation set.'
- What this solution (achieved 5.3058) has done: 'Your current submission score is catastrophically bad because the code exponentiates the GradientBoostingRegressor predictions (`np.exp(...)`) even though the model was trained on raw dollar fares, producing astronomically large fare outputs and thus huge RMSE. To move the score toward the target with minimal change and identical modeling approach, I remove that exponential in the submission prediction step and add a small safety clip to keep fares non-negative (a reasonable constraint for this task). I also ensure the test feature frame includes the computed `distance` and the time-derived columns (as your training does), and I write a correctly named `.csv` submission file (keeping your existing filename too for compatibility). These changes keep your core training logic intact and directly address the score explosion.'
- What this solution (achieved 5.29986) has done: 'Your current score (5.3058, lower-is-better) is far above the extremely large target, so we should decrease RMSE; the smallest legitimate improvement is to fix a feature mismatch caused by your cleaning step: you filter invalid coordinates/passenger counts in `train` but not in `test`, which can push the model into out-of-distribution predictions and worsen RMSE. I apply the exact same coordinate/passenger filters to `test` (without dropping rows) by clipping values into the valid ranges used for training, preserving your model and features but reducing extreme predictions. I also make the train/test split deterministic (`random_state`) to stabilize local evaluation without changing the training approach. The submission schema/paths remain the same and still write `submission.csv`.'

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

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass



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
train = pd.read_csv("../input/train.csv", nrows=100000, dtype=types)



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
train = train[train["pickup_longitude"] < -72]
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
train = train[train["dropoff_longitude"] < -72]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]
train = train[train["passenger_count"] > 0]



## === cell 13
train.describe()




## === cell 14
def dist_calc(df):
    for i, row in df.iterrows():
        df.at[i, "distance"] = great_circle(
            (row["pickup_latitude"], row["pickup_longitude"]),
            (row["dropoff_latitude"], row["dropoff_longitude"]),
        ).km




## === cell 15
dist_calc(train)
dist_calc(test)



## === cell 16
train["pickup_datetime"] = train["pickup_datetime"].str.replace(" UTC", "")
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)



## === cell 17
test["pickup_datetime"] = test["pickup_datetime"].str.replace(" UTC", "")
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)



## === cell 18
train["day"] = train.pickup_datetime.dt.day
train["dayofweek"] = train.pickup_datetime.dt.dayofweek
train["hour"] = train.pickup_datetime.dt.hour
train["month"] = train.pickup_datetime.dt.month
train["weekday"] = train.pickup_datetime.dt.weekday
train["year"] = train.pickup_datetime.dt.year

test["day"] = test.pickup_datetime.dt.day
test["dayofweek"] = test.pickup_datetime.dt.dayofweek
test["hour"] = test.pickup_datetime.dt.hour
test["month"] = test.pickup_datetime.dt.month
test["weekday"] = test.pickup_datetime.dt.weekday
test["year"] = test.pickup_datetime.dt.year



## === cell 19
test.head()



## === cell 20
sns.lmplot(x="year", y="fare_amount", data=train[["day", "year", "fare_amount"]])



## === cell 21
X = train.drop(
    [
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ],
    axis=1,
)
y = train["fare_amount"]



## === cell 22
X.head()



## === cell 23
y.head()



## === cell 24
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 25
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))

print(lm.score(X_test, y_test))



## === cell 26
y_pred = lm.predict(X_test)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y_test))
lrmse




## === cell 27
def get_score(prediction, lables):
    print("R2: {}".format(r2_score(prediction, lables)))
    print("RMSE: {}".format(np.sqrt(mean_squared_error(prediction, lables))))


def train_test(estimator, x_trn, x_tst, y_trn, y_tst):
    prediction_train = estimator.predict(x_trn)
    print(estimator)
    get_score(prediction_train, y_trn)
    prediction_test = estimator.predict(x_tst)
    print("Test")
    get_score(prediction_test, y_tst)




## === cell 28
GBest = ensemble.GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=3,
    max_features="sqrt",
    min_samples_leaf=15,
    min_samples_split=10,
    loss="huber",
).fit(X_train, y_train)
train_test(GBest, X_train, X_test, y_train, y_test)



## === cell 29
GB_model = GBest.fit(X, y)



## === cell 30
Xtest = test.drop(
    [
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ],
    axis=1,
)

Xtest = Xtest.copy()
if "passenger_count" in Xtest.columns:
    Xtest["passenger_count"] = Xtest["passenger_count"].clip(lower=1)

Final_labels = GB_model.predict(Xtest)
Final_labels = np.clip(Final_labels, 0, None)



## === cell 31
NewY_pred = lm.predict(Xtest)



## === cell 32
NewY_pred.size



## === cell 33
Final_labels.size



## === cell 34
test.head()



## === cell 35
submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": np.round(Final_labels, 2)}
)



## === cell 36
submission



## === cell 37
submission.to_csv("GBRSubmission06082018.csv", index=False)
submission.to_csv("submission.csv", index=False)
print("Wrote:", "GBRSubmission06082018.csv", "and submission.csv")
