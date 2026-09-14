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

3.43884

# 6. Current score

4.81813

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 12.70204) has done: 'Diagnosis: The crash happens in cell 14 because LightGBM v4.6.0 removed the `early_stopping_rounds` and `verbose_eval` keyword arguments from `lgb.train()`. In newer LightGBM versions, early stopping and logging must be configured via callbacks (`lgb.early_stopping()` and `lgb.log_evaluation()`). The model training logic itself is fine; only the API call signature needs updating.

Patch summary: In cell 14, replace `early_stopping_rounds=50` and `verbose_eval=100` with an equivalent `callbacks=[...]` list so training, early stopping behavior, and evaluation logging remain the same under LightGBM 4.6.0.

Updated cells: Only cell 14 is changed.

Compatibility notes for cell k+1: The variable `lgb_reg` remains a trained LightGBM Booster object, so cell 15 (`lgb.plot_importance(lgb_reg, ...)`) work unchanged. Predictions and RMSE computation in cell 14 still use `lgb_reg.predict(X_val)` as before.

Assumptions: We assume the intended early stopping behavior is to monitor the last validation set (`dvalid`) with RMSE and stop after 50 rounds without improvement, and that logging every 100 rounds matches the previous `verbose_eval=100` behavior.'
- What this solution (achieved 4.86445) has done: 'Your score is far worse than the target (RMSE 12.70 vs 3.44), so we should improve performance with the smallest changes that don’t alter the overall modeling approach. The biggest issue is that the current pipeline accidentally includes a “feature” (`fare_to_dist_ratio`) that exists in training but is forced to NaN for all test rows, creating a train/test distribution mismatch that can severely hurt generalization. I keep the same LightGBM setup and feature engineering, but drop any engineered columns that depend on `fare_amount` so train/test features are consistent and meaningful. I also apply the existing `clean_df()` to the training data (it’s defined but never used), which removes obvious GPS outliers and typically improves RMSE without changing the model.'
- What this solution (achieved 4.76384) has done: 'Your current gap is large (RMSE 4.86445 vs target 3.43884; lower is better), so the smallest safe improvement is to fix a bug in the distance feature engineering that likely corrupts the haversine distance (it’s missing the required squares), which directly harms model accuracy without changing the model or training loop. I correct the haversine formula while keeping all existing features, LightGBM parameters, and the train/valid split unchanged. I also clamp negative predictions to 0 (fares can’t be negative) as a minimal post-processing step that typically reduces RMSE slightly on this competition. The script still run end-to-end and write `lgsub.csv` with the required `key,fare_amount` format.'
- What this solution (achieved 4.81813) has done: 'You’re still far from the target (4.76384 vs 3.43884; lower is better), so the smallest likely gain without changing the model/training loop is to remove a subtle but impactful bug in the bearing calculation and to add one standard, competition-relevant distance feature (NYC-taxi baselines almost always benefit from a proper great-circle distance in km in addition to meters). I (1) fix the bearing formula to use `sin(delta_lon) * cos(lat2)` (the current code mistakenly uses `sin(delta_lon * cos(lat2))`), (2) add a `haversine_km` feature derived directly from the already-computed `haversine`, and (3) keep everything else (data sampling, cleaning, split, LightGBM params, early stopping, submission writing) the same to preserve core logic and stability. These are minimal feature-engineering corrections/additions that typically improve RMSE without altering the training approach or requiring more data. The script still run end-to-end and write a valid `lgsub.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import time

notebookstart = time.time()

import numpy as np
import pandas as pd
import gc

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics

import seaborn as sns
import matplotlib.pyplot as plt

Debug = False

NROWS = 10000000
if Debug is True:
    NROWS = 5000

train = pd.read_csv("../input/train.csv", nrows=NROWS, index_col="key")
train = train.dropna()
test_df = pd.read_csv("../input/test.csv", index_col="key")
testdex = test_df.index



## === cell 1
print(
    "Percent of Training Set with Zero and Below Fair: ",
    round(
        (
            (
                train.loc[train["fare_amount"] <= 0, "fare_amount"].shape[0]
                / train.shape[0]
            )
            * 100
        ),
        5,
    ),
)
print(
    "Percent of Training Set 200 and Above Fair: ",
    round(
        (
            train.loc[train["fare_amount"] >= 200, "fare_amount"].shape[0]
            / train.shape[0]
        )
        * 100,
        5,
    ),
)
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] <= 200), :]
print(
    "\nPercent of Training Set with Zero and Below Passenger Count: ",
    round(
        (
            train.loc[train["passenger_count"] <= 0, "passenger_count"].shape[0]
            / train.shape[0]
        )
        * 100,
        5,
    ),
)
print(
    "Percent of Training Set with Nine and Above Passenger Count: ",
    round(
        (
            train.loc[train["passenger_count"] >= 9, "passenger_count"].shape[0]
            / train.shape[0]
        )
        * 100,
        5,
    ),
)
train = train.loc[(train["passenger_count"] > 0) & (train["passenger_count"] <= 9), :]




## === cell 2
def clean_df(df):
    return df[
        (df.fare_amount > 0)
        & (df.pickup_longitude > -80)
        & (df.pickup_longitude < -70)
        & (df.pickup_latitude > 35)
        & (df.pickup_latitude < 45)
        & (df.dropoff_longitude > -80)
        & (df.dropoff_longitude < -70)
        & (df.dropoff_latitude > 35)
        & (df.dropoff_latitude < 45)
    ]




## === cell 3
def prepare_distance_features(df):
    df["longitude_distance"] = abs(df["pickup_longitude"] - df["dropoff_longitude"])
    df["latitude_distance"] = abs(df["pickup_latitude"] - df["dropoff_latitude"])

    df["distance_travelled"] = (
        df["longitude_distance"] ** 2 + df["latitude_distance"] ** 2
    ) ** 0.5
    df["distance_travelled_sin"] = np.sin(
        (df["longitude_distance"] ** 2 * df["latitude_distance"] ** 2) ** 0.5
    )
    df["distance_travelled_cos"] = np.cos(
        (df["longitude_distance"] ** 2 * df["latitude_distance"] ** 2) ** 0.5
    )
    df["distance_travelled_sin_sqrd"] = (
        np.sin((df["longitude_distance"] ** 2 * df["latitude_distance"] ** 2) ** 0.5)
        ** 2
    )
    df["distance_travelled_cos_sqrd"] = (
        np.cos((df["longitude_distance"] ** 2 * df["latitude_distance"] ** 2) ** 0.5)
        ** 2
    )

    R = 6371e3  # Metres
    phi1 = np.radians(df["pickup_latitude"])
    phi2 = np.radians(df["dropoff_latitude"])
    phi_chg = np.radians(df["pickup_latitude"] - df["dropoff_latitude"])
    delta_chg = np.radians(df["pickup_longitude"] - df["dropoff_longitude"])
    a = (np.sin(phi_chg / 2) ** 2) + np.cos(phi1) * np.cos(phi2) * (
        np.sin(delta_chg / 2) ** 2
    )
    c = 2 * np.arctan2(a**0.5, (1 - a) ** 0.5)
    d = R * c
    df["haversine"] = d

    y = np.sin(delta_chg) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    df["bearing"] = np.arctan2(y, x)

    df["haversine_km"] = df["haversine"] / 1000.0

    return df


def prepare_time_features(df):
    df["pickup_datetime"] = df["pickup_datetime"].str.replace(" UTC", "")
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
    )
    df["hour_of_day"] = df.pickup_datetime.dt.hour

    iso_week = df.pickup_datetime.dt.isocalendar().week.astype(np.int16)
    df["week"] = iso_week
    df["month"] = df.pickup_datetime.dt.month
    df["day_of_year"] = df.pickup_datetime.dt.dayofyear
    df["week_of_year"] = iso_week
    df["Weekday"] = df.pickup_datetime.dt.weekday
    df["Quarter"] = df.pickup_datetime.dt.quarter
    df["Day of Month"] = df.pickup_datetime.dt.day

    return df




## === cell 4
f, ax = plt.subplots(1, 2, figsize=[10, 5])
sns.countplot(x=train["passenger_count"], ax=ax[0])
sns.countplot(x=test_df["passenger_count"], ax=ax[1])
ax[0].set_title("Train Set - Passenger Count")
ax[1].set_title("Test Set - Passenger Count")
plt.show()



## === cell 5
f, ax = plt.subplots(figsize=[6, 5])
sns.kdeplot(train["fare_amount"], ax=ax)
ax.set_title("Fare Distribution")
plt.show()




## === cell 6
def time_slicer(df, timeframes, value, color="purple"):
    """
    Function to count observation occurrence through different lenses of time.
    """
    f, ax = plt.subplots(len(timeframes), figsize=[12, 10])
    for i, x in enumerate(timeframes):
        df.loc[:, [x, value]].groupby([x]).mean().plot(ax=ax[i], color=color)
        ax[i].set_ylabel(value.replace("_", " ").title())
        ax[i].set_title(
            "{} by {}".format(
                value.replace("_", " ").title(), x.replace("_", " ").title()
            )
        )
        ax[i].set_xlabel("")
    ax[len(timeframes) - 1].set_xlabel("Time Frame")
    plt.tight_layout(pad=0)




## === cell 7
train = prepare_time_features(train)
test_df = prepare_time_features(test_df)

time_slicer(
    df=train,
    timeframes=["day_of_year", "month", "Day of Month", "week", "hour_of_day"],
    value="fare_amount",
    color="blue",
)



## === cell 8
train = prepare_distance_features(train)
test_df = prepare_distance_features(test_df)

if "fare_to_dist_ratio" in train.columns:
    train.drop(columns=["fare_to_dist_ratio"], inplace=True, errors="ignore")
if "fare_to_dist_ratio" in test_df.columns:
    test_df.drop(columns=["fare_to_dist_ratio"], inplace=True, errors="ignore")

time_slicer(
    df=train,
    timeframes=["day_of_year", "month", "Day of Month", "week", "hour_of_day"],
    value="distance_travelled",
    color="green",
)



## === cell 9
if "fare_to_dist_ratio" in train.columns:
    time_slicer(
        df=train,
        timeframes=["day_of_year", "month", "Day of Month", "week", "hour_of_day"],
        value="fare_to_dist_ratio",
        color="red",
    )



## === cell 10
if "fare_npassenger_to_dist_ratio" in train.columns:
    train.drop(columns=["fare_npassenger_to_dist_ratio"], inplace=True, errors="ignore")
if "fare_npassenger_to_dist_ratio" in test_df.columns:
    test_df.drop(
        columns=["fare_npassenger_to_dist_ratio"], inplace=True, errors="ignore"
    )



## === cell 11
train = clean_df(train)

y = train.fare_amount.copy()

test_df.drop("pickup_datetime", axis=1, inplace=True)
train.drop("pickup_datetime", axis=1, inplace=True)

train = train[test_df.columns]
print("Does Train feature equal test feature?: ", all(train.columns == test_df.columns))

X_train, X_val, y_train, y_val = train_test_split(
    train, y, test_size=0.1, random_state=23
)

dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_val, label=y_val)



## === cell 12
print("Light Gradient Boosting Regressor: ")
lgbm_params = {
    "task": "train",
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
}



## === cell 13
modelstart = time.time()
lgb_reg = lgb.train(
    lgbm_params,
    dtrain,
    num_boost_round=500,
    valid_sets=[dtrain, dvalid],
    valid_names=["train", "valid"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100),
    ],
)
print("Model Evaluation Stage")
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_val, lgb_reg.predict(X_val))))



## === cell 14
f, ax = plt.subplots(figsize=[7, 10])
lgb.plot_importance(lgb_reg, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")
plt.savefig("feature_import.png")



## === cell 15
lgpred = lgb_reg.predict(test_df)

lgpred = np.clip(lgpred, 0, None)

lgsub = pd.DataFrame(lgpred, columns=["fare_amount"], index=testdex)

lgsub.to_csv("lgsub.csv", index=True, header=True)

print("Model Runtime: %0.2f Minutes" % ((time.time() - modelstart) / 60))
print("Notebook Runtime: %0.2f Minutes" % ((time.time() - notebookstart) / 60))
lgsub.head()
