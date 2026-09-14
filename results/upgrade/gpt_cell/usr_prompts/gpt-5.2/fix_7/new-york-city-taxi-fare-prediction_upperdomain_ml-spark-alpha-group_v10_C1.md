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
scipy==1.15.3
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

4.11442

# 6. Current score

17.10661

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.3125) has done: 'Diagnosis: The crash happens when constructing `SimpleImputer` with the deprecated/removed `axis` argument; in scikit-learn 1.2.2 `SimpleImputer` no longer accepts `axis`. This raises a `TypeError` before the training loop can run. The fix is to remove the `axis` parameter (the default behavior is already column-wise imputation) while keeping all other logic unchanged.

Patch summary: In cell 6, update the `Imputer(...)` instantiation to drop `axis=0` and use `missing_values=np.nan` (equivalent intent) so the code is compatible with the installed scikit-learn version. No other training, feature, or model logic is modified.

Updated cells: (cell 6 only)

Compatibility notes for cell k+1: This change preserves the `imp` variable and does not alter `regr`, `gen`, or any dataframe/feature interfaces used later; subsequent cells still run the same.

Assumptions: Missing values, if any, are represented as actual `np.nan` values by pandas (standard behavior), so `missing_values=np.nan` matches the intended imputation behavior.'
- What this solution (achieved 5.73226) has done: 'Your current score (5.3125 RMSE) is worse than the target (4.11442), so we should make small, legitimate improvements without changing the overall approach (same features: distance + passenger_count + bias; same GradientBoostingRegressor). The biggest issue is that the intended “incremental” training isn’t actually happening: `GradientBoostingRegressor` does not support continuing training via `warm_start` unless you increase `n_estimators`, so your loop effectively just refits 100 trees repeatedly on the last chunk. I change the loop to truly accumulate trees by increasing `n_estimators` each iteration (keeping the same model family and training loop style), and I also fix the train/test split to avoid temporal leakage by shuffling rows within each chunk before splitting. Finally, I keep the submission generation identical but ensure the imputer is actually applied during training (so train/predict pipelines match), which can slightly stabilize predictions and reduce RMSE.'
- What this solution (achieved 6.35466) has done: 'Your score is worse than the target (RMSE 5.73226 vs 4.11442), so we should make small, legitimate improvements without changing the overall model family or feature set. The biggest correctness issue is that your “incremental training” loop unintentionally re-fits almost from scratch each chunk, because you only add 1 estimator per iteration but call `fit`, so the model barely grows while being repeatedly overwritten; we instead increase `n_estimators` by a fixed step per chunk so trees truly accumulate (still `GradientBoostingRegressor`, same warm_start approach). We also apply the same imputer transformation to the held-out split inside each chunk (to keep train/validation semantics consistent) and clamp negative predictions to 0 at submission time (fares are non-negative and this typically reduces RMSE). These changes are minimal, keep your core distance/passenger_count/bias features, and preserve the same training loop structure and loss/metric semantics.'
- What this solution (achieved 8.92916) has done: 'Your current RMSE (6.35466) is worse than the target (4.11442), so we should make small, legitimate improvements without changing your core model or feature set. The biggest score drag is that the imputer is being re-fit on every chunk and then only the *last* chunk’s imputer is used at test time, which creates a train/test preprocessing mismatch; I fit the imputer once on an initial cleaned sample and then keep it fixed for all chunks and test. Second, your current “warm_start” loop is still only learning from the most recent chunk because `.fit()` overwrites previous trees; I switch to the supported incremental pattern for `GradientBoostingRegressor` by calling `.fit()` once, then using `.staged_predict()` to select how many trees to keep and growing `n_estimators` while keeping previous trees (same model family and loop concept, but now it truly accumulates). Finally, I make the distance computation numerically safer (avoid division by zero) without changing the feature definition.'
- What this solution (achieved 17.10661) has done: 'Your current RMSE (8.92916) is far worse than the target (4.11442), so we should make a small but high-impact correction without changing the core model/features. The biggest score issue is that your generator parses the wrong column as datetime (`parse_dates=[1]`), which currently tries to parse `fare_amount` as dates and leaves `pickup_datetime` as an object; this breaks the usual NYC Taxi baseline signal from time-of-day/weekday and can also corrupt reading. I fix the generator to parse `pickup_datetime` (by name) and then add a minimal, standard set of datetime-derived features (hour, weekday, month) while keeping your existing distance/passenger_count/bias features and the same `GradientBoostingRegressor` training loop. Finally, I ensure the exact same feature columns are used for imputer fit, training, and test prediction to avoid train/test feature mismatch.'

# 9. Code solution

## === cell 0
from sklearn.impute import SimpleImputer as Imputer
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import os

print(os.listdir("../input"))


def chunck_generator(filename, header=False, chunk_size=10**5):
    for chunk in pd.read_csv(
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        parse_dates=["pickup_datetime"],
    ):
        yield chunk




## === cell 1
alpha_ang = 0.506


def distance_travel(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs() * 50
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs() * 69
    df["displacement_vector"] = (
        df.abs_diff_latitude**2 + df.abs_diff_longitude**2
    ) ** 0.5

    ang = np.arctan2(df["abs_diff_longitude"].values, df["abs_diff_latitude"].values)
    df["actual_long"] = (
        df["displacement_vector"].values * np.sin(ang - alpha_ang)
    ).astype(np.float64)
    df["actual_lat"] = (
        df["displacement_vector"].values * np.cos(ang - alpha_ang)
    ).astype(np.float64)

    df["actual_long"] = np.abs(df["actual_long"])
    df["actual_lat"] = np.abs(df["actual_lat"])
    df["distance_travel"] = df["actual_long"] + df["actual_lat"]
    return df




## === cell 2
def add_time_features(df):
    dt = df["pickup_datetime"]
    df["pickup_hour"] = dt.dt.hour.astype(np.float64)
    df["pickup_weekday"] = dt.dt.weekday.astype(np.float64)
    df["pickup_month"] = dt.dt.month.astype(np.float64)
    return df


def make_features(df):
    distance_travel(df)
    add_time_features(df)
    return df




## === cell 3
def data_clean(df):
    df = df[df.passenger_count > 0]
    df.fare_amount = df.fare_amount.astype(np.float64)
    df = df[df.fare_amount > 0]
    make_features(df)
    df = df[df.distance_travel > 0]
    return df




## === cell 4
def remove_outliers(df):
    df = df[df.distance_travel < 30]
    df = df[df.fare_amount < 100]
    return df




## === cell 5
def graph_presesnt(df):
    test = df[df.passenger_count == 1]
    plot = test.iloc[: len(test)].plot.scatter("distance_travel", "fare_amount")
    plot = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")




## === cell 6
def incremental_training(train_X, train_y, regr, is_first_fit=False):
    if is_first_fit:
        regr.fit(train_X, train_y)
    else:
        regr.fit(train_X, train_y)
    return regr




## === cell 7
filename = r"../input/train.csv"
gen = chunck_generator(filename=filename)

regr = GradientBoostingRegressor(n_estimators=100, warm_start=True, random_state=42)

imp = Imputer(missing_values=np.nan, strategy="mean")

rng = np.random.RandomState(42)


def to_X(df):
    return np.column_stack(
        (
            df.distance_travel.values,
            df.passenger_count.values,
            df.pickup_hour.values,
            df.pickup_weekday.values,
            df.pickup_month.values,
            np.ones(len(df)),
        )
    )


imp_fit_rows_target = 300_000
imp_fit_rows = 0
imp_fit_X_parts = []

while imp_fit_rows < imp_fit_rows_target:
    df0 = next(gen)
    make_features(df0)
    df0 = data_clean(df0)
    df0 = remove_outliers(df0)
    if len(df0) == 0:
        continue
    take = min(len(df0), imp_fit_rows_target - imp_fit_rows)
    df0 = df0.iloc[:take]
    X0 = to_X(df0)
    imp_fit_X_parts.append(X0)
    imp_fit_rows += take

imp_fit_X = np.vstack(imp_fit_X_parts)
imp.fit(imp_fit_X)

n_chunks = 100
trees_per_chunk = 10  # total trees grows gradually
t = n_chunks

is_first_fit = True
while t > 0:
    df = next(gen)
    make_features(df)
    df = data_clean(df)
    df = remove_outliers(df)

    if len(df) == 0:
        t -= 1
        continue

    df = df.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    l = len(df)
    df_train = df[: int(0.7 * l)]
    df_test = df[int(0.7 * l) :]

    train_X = to_X(df_train)
    train_y = np.array(df_train.fare_amount.values)

    train_X_imp = imp.transform(train_X)

    if is_first_fit:
        regr = incremental_training(train_X_imp, train_y, regr, is_first_fit=True)
        is_first_fit = False
    else:
        regr.set_params(n_estimators=regr.n_estimators + trees_per_chunk)
        regr = incremental_training(train_X_imp, train_y, regr, is_first_fit=False)

    t -= 1



## === cell 8
tdf = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"], nrows=10_00_000)
make_features(tdf)
tdf.head()



## === cell 9
ttrain_X = to_X(tdf)
ttrain_X = imp.transform(ttrain_X)
output = regr.predict(ttrain_X)

output = np.maximum(output, 0.0)

print(output)



## === cell 10
my_submission = pd.DataFrame({"key": tdf.key, "fare_amount": output})
my_submission.to_csv("submission.csv", index=False)
my_submission.head()
