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

4.97992

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.04274) has done: 'We remove the unsupported `verbose_eval` argument from the LightGBM `train` call and add a proper logging callback. This fixes the `TypeError`, allowing the model to train, so subsequent cells can generate the prediction file without errors. No other logic is changed, preserving the original feature engineering and evaluation.'
- What this solution (achieved 5.97412) has done: 'Implemented two focused adjustments to move the RMSE closer to the target:

1. Fixed the Haversine distance calculation in `prepare_distance_features` by applying the correct squared‑sin formulation, providing a more accurate travel‑distance feature.
2. Refined LightGBM hyper‑parameters (learning_rate, num_leaves, feature_fraction, bagging_fraction, bagging_freq) to improve model capacity while preserving the original training workflow and early stopping.

These minimal changes keep the core pipeline intact while delivering a better‑calibrated model and expected lower RMSE.'
- What this solution (achieved 12.39503) has done: 'I replace the distance‑based ratio features with haversine‑based ones (a more accurate travel distance) and slightly increase model capacity while keeping the same training workflow. These adjustments should improve the RMSE toward the target without altering the overall pipeline.'
- What this solution (achieved 12.45414) has done: 'I clean up the distance‑feature engineering by removing the unnecessary sine/cosine columns, add a simple Manhattan‑distance column, and normalise the haversine distance to kilometres. I also fill any remaining NaNs (the ratio features are NaN in the test set) with ‑1 before training. Finally, I slightly increase LightGBM capacity (more leaves, a few more boosting rounds) to give the model a better chance of lowering the RMSE toward the target.'
- What this solution (achieved 6.1507) has done: 'Implemented fixes to unblock the pipeline and improve model performance:
- Corrected the plotting call that referenced a non‑existent column by using the available `haversine_km` feature.
- Removed target‑leaking ratio features (`fare_to_dist_ratio`, `fare_npassenger_to_dist_ratio`) which harmed generalisation.
- Adjusted LightGBM hyper‑parameters (more leaves, lower learning rate) for better capacity while keeping the original training workflow.
- Updated column alignment after feature engineering to ensure train and test feature sets match.'
- What this solution (achieved 6.23463) has done: 'We enrich the time‑based features with cyclic sin/cos encodings (hour, month, day‑of‑year) so the model can capture periodic patterns more naturally, which usually lowers RMSE. We also clip the final fare predictions to the realistic range (0 – 200) to avoid large out‑liers that hurt the score. These small, targeted changes keep the original pipeline intact while moving the metric toward the target.'
- What this solution (achieved 6.16177) has done: 'I increase the training sample size, add a few informative distance‑based features, and slightly enlarge the LightGBM tree capacity. These tweaks keep the original pipeline intact while providing the model with more data and richer signals, which should lower the RMSE toward the target.'
- What this solution (achieved 6.16249) has done: 'I added a few simple, non‑intrusive features that capture passenger‑count scale, weekend effect and a log‑scaled distance‑per‑passenger metric, which are known to help fare prediction without altering the core model. I also introduced a mild L2 regularisation term in the LightGBM parameters to improve generalisation. All other logic stays the same, and the script still writes a correct `lgsub.csv` submission file.'
- What this solution (achieved 5.56965) has done: 'I increase the training sample size, add two cyclic weekday features and a night‑time flag to give the model more useful signal, and modestly enlarge the LightGBM capacity (more leaves, lower learning rate, small L1 term). These adjustments keep the original pipeline intact while providing richer features and a slightly stronger model, which should lower the RMSE toward the target.'
- What this solution (achieved 4.97992) has done: 'Implemented a log‑transform of the fare target so the model learns on a more symmetric distribution, then inverse‑transformed predictions for proper RMSE calculation and submission. Also nudged LightGBM hyper‑parameters (higher learning rate, fewer leaves) to improve generalisation while keeping the original pipeline unchanged. These minimal adjustments are expected to reduce the validation RMSE toward the target value.'

# 9. Code solution

## === cell 0
import time

notebookstart = time.time()

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import gc

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics

import seaborn as sns
import matplotlib.pyplot as plt

Debug = False

NROWS = 2_000_000  # previously 1_000_000
if Debug:
    NROWS = 5_000

train = pd.read_csv("../input/train.csv", nrows=NROWS, index_col="key")
train = train.dropna()
test_df = pd.read_csv("../input/test.csv", index_col="key")
testdex = test_df.index




## === cell 1
print(
    "Percent of Training Set with Zero and Below Fare: ",
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
    "Percent of Training Set 200 and Above Fare: ",
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

    df["manhattan_distance"] = df["longitude_distance"] + df["latitude_distance"]

    R = 6371e3  # metres
    phi1 = np.radians(df["pickup_latitude"])
    phi2 = np.radians(df["dropoff_latitude"])
    dphi = np.radians(df["dropoff_latitude"] - df["pickup_latitude"])
    dlambda = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["haversine"] = R * c  # metres
    df["haversine_km"] = df["haversine"] / 1000.0

    df["log_haversine"] = np.log1p(df["haversine_km"])
    df["haversine_per_passenger"] = df["haversine_km"] / df["passenger_count"].replace(
        0, np.nan
    )

    df["passenger_count_log"] = np.log1p(df["passenger_count"])

    df["distance_per_passenger_log"] = np.log1p(
        df["haversine_km"] / df["passenger_count"].replace(0, np.nan)
    )

    y = np.sin(dlambda * np.cos(phi2))
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    df["bearing"] = np.arctan2(y, x)

    cols_to_drop = [
        "distance_travelled",
        "distance_travelled_sin",
        "distance_travelled_cos",
        "distance_travelled_sin_sqrd",
        "distance_travelled_cos_sqrd",
    ]
    df.drop(
        columns=[c for c in cols_to_drop if c in df.columns],
        inplace=True,
        errors="ignore",
    )
    return df


def prepare_time_features(df):
    df["pickup_datetime"] = df["pickup_datetime"].str.replace(" UTC", "", regex=False)
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
    )

    df["hour_of_day"] = df.pickup_datetime.dt.hour
    df["week"] = df.pickup_datetime.dt.isocalendar().week.astype(int)
    df["month"] = df.pickup_datetime.dt.month
    df["day_of_year"] = df.pickup_datetime.dt.dayofyear
    df["weekday"] = df.pickup_datetime.dt.weekday
    df["quarter"] = df.pickup_datetime.dt.quarter
    df["Day of Month"] = df.pickup_datetime.dt.day

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["dayofyear_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    df["hour_distance"] = df["hour_of_day"] * df["haversine_km"]

    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    df["is_night"] = (df["hour_of_day"] < 6).astype(int)

    return df




## === cell 4
train = clean_df(train)

train = prepare_distance_features(train)
test_df = prepare_distance_features(test_df)

train = prepare_time_features(train)
test_df = prepare_time_features(test_df)




## === cell 5
f, ax = plt.subplots(1, 2, figsize=[10, 5])
sns.countplot(train["passenger_count"], ax=ax[0])
sns.countplot(test_df["passenger_count"], ax=ax[1])
ax[0].set_title("Train Set - Passenger Count")
ax[1].set_title("Test Set - Passenger Count")
plt.show()




## === cell 6
f, ax = plt.subplots(figsize=[6, 5])
sns.kdeplot(train["fare_amount"], ax=ax)
ax.set_title("Fare Distribution")
plt.show()




## === cell 7
def time_slicer(df, timeframes, value, color="purple"):
    """
    Plot mean of `value` over each column listed in `timeframes`.
    """
    f, ax = plt.subplots(len(timeframes), figsize=[12, 10])
    for i, x in enumerate(timeframes):
        df.loc[:, [x, value]].groupby([x]).mean().plot(ax=ax[i], color=color)
        ax[i].set_ylabel(value.replace("_", " ").title())
        ax[i].set_title(
            f"{value.replace('_', ' ').title()} by {x.replace('_', ' ').title()}"
        )
        ax[i].set_xlabel("")
    ax[-1].set_xlabel("Time Frame")
    plt.tight_layout(pad=0)




## === cell 8
time_slicer(
    df=train,
    timeframes=["day_of_year", "month", "Day of Month", "week", "hour_of_day"],
    value="fare_amount",
    color="blue",
)




## === cell 9
time_slicer(
    df=train,
    timeframes=["day_of_year", "month", "Day of Month", "week", "hour_of_day"],
    value="haversine_km",
    color="green",
)




## === cell 10
y_original = train.fare_amount.copy()
y_log = np.log1p(y_original)

test_df.drop("pickup_datetime", axis=1, inplace=True)

train = train[test_df.columns]
print("Do Train and Test feature sets match? ", all(train.columns == test_df.columns))

train = train.fillna(-1)
test_df = test_df.fillna(-1)

X_train, X_val, y_train_log, y_val_log = train_test_split(
    train, y_log, test_size=0.1, random_state=23
)

dtrain = lgb.Dataset(X_train, label=y_train_log)
dvalid = lgb.Dataset(X_val, label=y_val_log)




## === cell 11
print("Light Gradient Boosting Regressor:")
lgbm_params = {
    "task": "train",
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 1023,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "lambda_l2": 0.1,
    "lambda_l1": 0.1,
}




## === cell 12
modelstart = time.time()
lgb_reg = lgb.train(
    params=lgbm_params,
    train_set=dtrain,
    num_boost_round=15000,
    valid_sets=[dtrain, dvalid],
    valid_names=["train", "valid"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=120, verbose=False),
        lgb.log_evaluation(period=100, show_stdv=False),
    ],
)

print("Model Evaluation Stage")
val_pred_log = lgb_reg.predict(X_val, num_iteration=lgb_reg.best_iteration)
val_pred = np.expm1(val_pred_log)
y_val_original = np.expm1(y_val_log)

rmse = np.sqrt(metrics.mean_squared_error(y_val_original, val_pred))
print("RMSE:", rmse)




## === cell 13
f, ax = plt.subplots(figsize=[7, 10])
lgb.plot_importance(lgb_reg, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")
plt.savefig("feature_import.png")
plt.show()




## === cell 14
test_pred_log = lgb_reg.predict(test_df, num_iteration=lgb_reg.best_iteration)
lgpred = np.expm1(test_pred_log)

lgpred = np.clip(lgpred, 0, 200)

lgsub = pd.DataFrame(lgpred, columns=["fare_amount"], index=testdex)
lgsub.to_csv("lgsub.csv", index=True, header=True)

print(f"Model Runtime: {(time.time() - modelstart) / 60:.2f} Minutes")
print(f"Notebook Runtime: {(time.time() - notebookstart) / 60:.2f} Minutes")
lgsub.head()
