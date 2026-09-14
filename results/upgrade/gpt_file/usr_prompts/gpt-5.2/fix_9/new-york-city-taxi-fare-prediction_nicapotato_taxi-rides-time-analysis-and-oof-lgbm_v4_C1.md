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

4.8101

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.86445) has done: 'I fix the Pandas datetime feature extraction to work with Pandas 2.2 (replacing deprecated `.dt.week`/`.dt.weekofyear`), which currently stops feature generation and causes downstream KeyErrors. I also update the LightGBM training call to use the v4 callback-based early stopping API (since `early_stopping_rounds` is no longer a valid keyword), so the model actually trains. Finally, I ensure the submission file matches the required format (`key,fare_amount`) and is written as a `.csv` in the working directory, while keeping the model and feature logic otherwise unchanged.'
- What this solution (achieved 4.27813) has done: 'Your current gap is 4.86445 − 3.43884 = 1.42561 (about 41% worse than target), so we need a legitimate quality improvement while keeping the same LightGBM training approach. The biggest score issue is a bug in the haversine computation (missing squares in the formula), which makes the key distance feature incorrect and hurts RMSE; fixing it preserves the same feature idea but corrects the math. I also make the datetime parsing robust to the dataset’s actual format (it includes microseconds) to avoid silent `NaT` values, and I apply the same geographic cleaning bounds to the test set (without fare filter) to reduce distribution mismatch while keeping row order and restoring predictions for any filtered-out rows. These are minimal, directly score-relevant changes and still write a valid `submission.csv`.'
- What this solution (achieved 4.33371) has done: 'To move RMSE down toward the 3.43884 target with minimal disruption, I keep your exact feature set and LightGBM training loop, but improve data correctness and alignment where it directly affects RMSE. Specifically: (1) ensure datetime parsing is consistent and doesn’t silently produce `NaT` (which can create missing/garbage time features), (2) apply the same passenger_count validity filter to training that you already intended (your `clean_df()` currently ignores passenger_count), and (3) make the bearing calculation correct (it currently uses `sin(delta * cos(phi2))` instead of `sin(delta) * cos(phi2)`), which improves a core geometric feature without changing the approach. These are small, legitimate fixes that typically reduce error on this competition while preserving your model/feature logic and producing the same submission format.'
- What this solution (achieved 4.33371) has done: 'You’re currently above the target RMSE (4.33371 vs 3.43884, lower is better), so we need a modest real quality gain while keeping the same LightGBM approach and the same overall feature set idea. The biggest score drag left in this script is that you train with two leakage features (`fare_to_dist_ratio` and `fare_npassenger_to_dist_ratio`) that use `fare_amount` (the target) but are not available for test, and then you silently drop them by aligning to `test_df_valid.columns`; this mismatch harms training usefulness and generalization. I remove those target-derived features entirely (rather than partially using/dropping) so train/test feature generation is consistent, and I also ensure `clean_df()` doesn’t inadvertently depend on `fare_amount` when used for test-like frames. These are minimal, directly score-relevant fixes that preserve the same model/training loop and should move RMSE down toward the target.'
- What this solution (achieved 4.36969) has done: 'We need to reduce RMSE from 4.33371 toward 3.43884 (lower is better), so we make small, legitimate quality improvements without changing the overall LightGBM approach. The biggest gain with minimal disruption is to add a couple of standard NYC Taxi engineered features that are consistent with your existing geometry/time feature logic: absolute delta in lon/lat, Manhattan distance, and a simple “airport-ish” flag to help with long trips. We also add a light, RMSE-relevant data cleaning step to drop implausible coordinate pairs (0,0) and extremely tiny/huge trip distances in training only, which reduces label noise while keeping your core pipeline intact. Finally, we keep train/test feature alignment explicit and still produce the required `submission.csv`.'
- What this solution (achieved 4.63104) has done: 'Your pipeline doesn’t currently yield a Kaggle score because the environment errors out before grading; the safest way to move toward the target RMSE is to (a) make the run stable within the 600s constraint and (b) keep your existing feature/model logic intact while training on a cleaner, representative subset. I keep the same LightGBM training approach and the same engineered feature set, but switch to reading a deterministic random sample from the full `train.csv` (instead of the first N rows, which are distribution-shifted) using `skiprows`, which typically lowers RMSE without changing semantics. I also make the input paths robust to either `../input/*.csv` or `/kaggle/input/*.csv` so it runs in your provided filesystem, and I keep the submission writing exactly `submission.csv` with `key,fare_amount` aligned to the original test order.'
- What this solution (achieved 4.8101) has done: 'To move RMSE down toward the 3.43884 target (lower is better) with minimal disruption, I keep your exact LightGBM training loop and existing feature set, but fix one data/feature alignment issue that hurts generalization: you currently train on a subset sampled from `train.csv`, which is noisier than the already-provided cleaned `labels.csv`; switching the sampled source to `labels.csv` (same schema) typically reduces label noise and improves RMSE without changing the model approach. I also make datetime parsing explicitly `utc=True` to avoid occasional `NaT`/timezone inconsistencies that can degrade time features, while keeping the same extracted fields. Finally, I add LightGBM’s `extra_trees` and `lambda_l2` regularization knobs (same GBDT, same objective/metric) to modestly improve stability/fit without changing your architecture or training semantics, and still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import time

notebookstart = time.time()

import numpy as np
import pandas as pd
import gc
import os

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics

import seaborn as sns
import matplotlib.pyplot as plt

Debug = False

NROWS = 2_000_000
if Debug is True:
    NROWS = 50_000


def _resolve_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {candidates}")


train_path = _resolve_path(
    [
        "/kaggle/data/labels.csv",
        "/kaggle/input/labels.csv",
        "../input/labels.csv",
        "/kaggle/data/new-york-city-taxi-fare-prediction/labels.csv",
        "/kaggle/input/new-york-city-taxi-fare-prediction/labels.csv",
        "../input/new-york-city-taxi-fare-prediction/labels.csv",
        "/kaggle/data/train.csv",
        "/kaggle/input/train.csv",
        "../input/train.csv",
        "/kaggle/data/new-york-city-taxi-fare-prediction/train.csv",
        "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
        "../input/new-york-city-taxi-fare-prediction/train.csv",
    ]
)
test_path = _resolve_path(
    [
        "../input/test.csv",
        "/kaggle/input/test.csv",
        "/kaggle/data/test.csv",
        "../input/new-york-city-taxi-fare-prediction/test.csv",
        "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
        "/kaggle/data/new-york-city-taxi-fare-prediction/test.csv",
    ]
)


def _sample_train_csv(path, nrows, seed=23):
    try:
        with open(path, "rb") as f:
            n_lines = sum(1 for _ in f)
        n_data = max(0, n_lines - 1)
        if n_data <= nrows:
            return pd.read_csv(path, nrows=nrows, index_col="key")
        rng = np.random.RandomState(seed)
        keep = set(rng.choice(np.arange(1, n_data + 1), size=nrows, replace=False))
        skip = lambda i: (i != 0) and (i not in keep)
        return pd.read_csv(path, skiprows=skip, index_col="key")
    except Exception as e:
        print(
            "Sampling via skiprows failed; falling back to first N rows. Error:",
            repr(e),
        )
        return pd.read_csv(path, nrows=nrows, index_col="key")


train = _sample_train_csv(train_path, NROWS, seed=23)
train = train.dropna()

test_df = pd.read_csv(test_path, index_col="key")
testdex = test_df.index

gc.collect()



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
    cond = (
        (df.passenger_count > 0)
        & (df.passenger_count <= 9)
        & (df.pickup_longitude > -80)
        & (df.pickup_longitude < -70)
        & (df.pickup_latitude > 35)
        & (df.pickup_latitude < 45)
        & (df.dropoff_longitude > -80)
        & (df.dropoff_longitude < -70)
        & (df.dropoff_latitude > 35)
        & (df.dropoff_latitude < 45)
    )

    cond = cond & ~(
        ((df.pickup_longitude == 0) & (df.pickup_latitude == 0))
        | ((df.dropoff_longitude == 0) & (df.dropoff_latitude == 0))
    )

    if "fare_amount" in df.columns:
        cond = cond & (df.fare_amount > 0) & (df.fare_amount <= 200)
    return df[cond]




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

    R_km = 6371.0  # Kilometers
    phi1 = np.radians(df["pickup_latitude"])
    phi2 = np.radians(df["dropoff_latitude"])
    phi_chg = np.radians(df["pickup_latitude"] - df["dropoff_latitude"])
    delta_chg = np.radians(df["pickup_longitude"] - df["dropoff_longitude"])

    a = (np.sin(phi_chg / 2) ** 2) + np.cos(phi1) * np.cos(phi2) * (
        np.sin(delta_chg / 2) ** 2
    )
    c = 2 * np.arctan2(a**0.5, (1 - a) ** 0.5)
    d_km = R_km * c
    df["haversine"] = d_km

    y = np.sin(delta_chg) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    df["bearing"] = np.arctan2(y, x)

    df["manhattan_distance"] = df["longitude_distance"] + df["latitude_distance"]
    df["lon_lat_sum"] = df["longitude_distance"] + df["latitude_distance"]
    df["lon_lat_diff"] = df["longitude_distance"] - df["latitude_distance"]

    return df


def prepare_time_features(df):
    if df["pickup_datetime"].dtype != "datetime64[ns]":
        s = df["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
        df["pickup_datetime"] = pd.to_datetime(
            s, errors="coerce", utc=True
        ).dt.tz_convert(None)

    df["hour_of_day"] = df.pickup_datetime.dt.hour
    df["month"] = df.pickup_datetime.dt.month
    df["day_of_year"] = df.pickup_datetime.dt.dayofyear

    iso = df.pickup_datetime.dt.isocalendar()
    df["week"] = iso.week.astype("int16")
    df["week_of_year"] = iso.week.astype("int16")

    df["Weekday"] = df.pickup_datetime.dt.weekday
    df["Quarter"] = df.pickup_datetime.dt.quarter
    df["Day of Month"] = df.pickup_datetime.dt.day

    df["is_weekend"] = (df["Weekday"] >= 5).astype("int8")

    return df




## === cell 4
train = clean_df(train)

test_df_valid = clean_df(test_df.copy())

train = prepare_distance_features(train)
test_df_valid = prepare_distance_features(test_df_valid)

train = prepare_time_features(train)
test_df_valid = prepare_time_features(test_df_valid)

if "haversine" in train.columns:
    train = train.loc[
        (train["haversine"] >= 0.05) & (train["haversine"] <= 200.0)
    ].copy()


def add_airport_flags(df):
    jfk = (-73.7781, 40.6413)
    lga = (-73.8740, 40.7769)
    ewr = (-74.1745, 40.6895)

    def approx_deg_dist(lon, lat, center):
        return np.sqrt((lon - center[0]) ** 2 + (lat - center[1]) ** 2)

    pu_jfk = approx_deg_dist(df["pickup_longitude"], df["pickup_latitude"], jfk)
    do_jfk = approx_deg_dist(df["dropoff_longitude"], df["dropoff_latitude"], jfk)
    pu_lga = approx_deg_dist(df["pickup_longitude"], df["pickup_latitude"], lga)
    do_lga = approx_deg_dist(df["dropoff_longitude"], df["dropoff_latitude"], lga)
    pu_ewr = approx_deg_dist(df["pickup_longitude"], df["pickup_latitude"], ewr)
    do_ewr = approx_deg_dist(df["dropoff_longitude"], df["dropoff_latitude"], ewr)

    df["near_airport"] = (
        (pu_jfk < 0.05)
        | (do_jfk < 0.05)
        | (pu_lga < 0.05)
        | (do_lga < 0.05)
        | (pu_ewr < 0.05)
        | (do_ewr < 0.05)
    ).astype("int8")
    return df


train = add_airport_flags(train)
test_df_valid = add_airport_flags(test_df_valid)

gc.collect()



## === cell 5
try:
    f, ax = plt.subplots(1, 2, figsize=[10, 5])
    sns.countplot(x=train["passenger_count"], ax=ax[0])
    sns.countplot(x=test_df["passenger_count"], ax=ax[1])
    ax[0].set_title("Train Set - Passenger Count")
    ax[1].set_title("Test Set - Passenger Count")
    plt.tight_layout()
    plt.close()
except Exception as e:
    print("Plotting skipped:", repr(e))



## === cell 6
try:
    f, ax = plt.subplots(figsize=[6, 5])
    sns.kdeplot(train["fare_amount"], ax=ax)
    ax.set_title("Fare Distribution")
    plt.tight_layout()
    plt.close()
except Exception as e:
    print("Plotting skipped:", repr(e))




## === cell 7
def time_slicer(df, timeframes, value, color="purple"):
    """
    Function to count observation occurrence through different lenses of time.
    """
    f, ax = plt.subplots(len(timeframes), figsize=[12, 10])
    if len(timeframes) == 1:
        ax = [ax]
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
    plt.close()




## === cell 8
try:
    time_slicer(
        df=train,
        timeframes=["day_of_year", "month", "Day of Month", "week", "hour_of_day"],
        value="fare_amount",
        color="blue",
    )
except Exception as e:
    print("time_slicer skipped:", repr(e))



## === cell 9
try:
    time_slicer(
        df=train,
        timeframes=["day_of_year", "month", "Day of Month", "week", "hour_of_day"],
        value="distance_travelled",
        color="green",
    )
except Exception as e:
    print("time_slicer skipped:", repr(e))



## === cell 10
try:
    pass
except Exception as e:
    print("time_slicer skipped:", repr(e))



## === cell 11
try:
    pass
except Exception as e:
    print("time_slicer skipped:", repr(e))



## === cell 12
y = train.fare_amount.copy()

test_df_valid.drop("pickup_datetime", axis=1, inplace=True)
train = train.drop("pickup_datetime", axis=1)

train = train[test_df_valid.columns]
print(
    "Does Train feature equal test feature?: ",
    all(train.columns == test_df_valid.columns),
)

X_train, X_val, y_train, y_val = train_test_split(
    train, y, test_size=0.1, random_state=23
)

dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_val, label=y_val)



## === cell 13
print("Light Gradient Boosting Regressor: ")
lgbm_params = {
    "task": "train",
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 23,
    "feature_fraction_seed": 23,
    "bagging_seed": 23,
    "extra_trees": True,
    "lambda_l2": 0.2,
}



## === cell 14
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
val_pred = lgb_reg.predict(X_val, num_iteration=lgb_reg.best_iteration)
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_val, val_pred)))



## === cell 15
try:
    f, ax = plt.subplots(figsize=[7, 10])
    lgb.plot_importance(lgb_reg, max_num_features=50, ax=ax)
    plt.title("Light GBM Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_import.png")
    plt.close()
except Exception as e:
    print("Feature importance plot skipped:", repr(e))



## === cell 16
lgpred_valid = lgb_reg.predict(test_df_valid, num_iteration=lgb_reg.best_iteration)

lgpred_full = pd.Series(index=testdex, dtype="float64")
lgpred_full.loc[test_df_valid.index] = lgpred_valid
fill_value = (
    float(np.nanmean(lgpred_valid)) if len(lgpred_valid) > 0 else float(y_train.mean())
)
lgpred_full = lgpred_full.fillna(fill_value).values

submission = pd.DataFrame({"key": testdex, "fare_amount": lgpred_full})

submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print("Model Runtime: %0.2f Minutes" % ((time.time() - modelstart) / 60))
print("Notebook Runtime: %0.2f Minutes" % ((time.time() - notebookstart) / 60))

submission.head()
