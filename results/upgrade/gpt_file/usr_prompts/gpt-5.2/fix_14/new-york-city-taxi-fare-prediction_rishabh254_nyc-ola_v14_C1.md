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

5.46166

# 6. Current score

6.61359

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 887.48032) has done: 'I fix the pandas API breakages (tuple column selection in groupby and `DataFrame.corr()` failing due to non-numeric columns) so the notebook runs without stopping early. I also fix the datetime parsing so it works with the dataset’s `pickup_datetime` format (which includes fractional seconds but no timezone), and ensure `hour`/`year` are created before the train/validation split so later cells can access them. Finally, I keep the same OLS (least-squares) core model and feature set, but make the pipeline deterministic and guarantee a valid `submission.csv` with the required columns is written end-to-end.'
- What this solution (achieved 988.75049) has done: 'Your current RMSE (887) is dominated by a small number of wildly wrong predictions; the simplest way to move toward the target is to make the OLS fit more numerically stable and remove a clear data-quality issue that creates extreme outliers. I (1) ensure the core features used in the regression (`distance_miles`, `hour`, `year`) are computed before the train/validation split so the model is trained on the same feature definitions you use at inference, (2) add minimal coordinate sanity filters (NYC bounding box + fare cap) to prevent pathological rows from exploding the least-squares solution, and (3) stop rounding predictions before RMSE evaluation/submission and clip negative fares to 0.0 (rounding/clipping is post-processing and doesn’t change the model). These are small, metric-aligned fixes that should dramatically reduce RMSE while keeping your same OLS core logic and feature set.'
- What this solution (achieved 674.69369) has done: 'Your RMSE is still being blown up by a small number of bad rows (typically corrupted coordinates or implausible fares/distances) that slip through and create extreme leverage for an OLS fit. I keep the same OLS least-squares model and the same feature matrix, but (1) add one additional minimal, standard NYC-taxi cleaning rule (`fare_amount <= 3 * distance_miles + 7`, which removes only implausible outliers), and (2) make the OLS coefficients use the numerically-stable `np.linalg.lstsq` solution consistently (removing the explicit matrix inverse path that can become unstable). This should sharply reduce validation RMSE and move it toward your target while preserving the core logic and producing the same required `submission.csv`. All paths and the submission schema stay unchanged.'
- What this solution (achieved 670.27732) has done: 'Your current RMSE is still dominated by a handful of high-leverage training rows that survive cleaning and destabilize an OLS fit; the smallest change that should move you dramatically toward the target is to make the training set cleaning slightly stricter in the two places most correlated with blow-ups: tiny-distance rides and implausibly high fare-per-mile. I keep the same OLS least-squares core model and the same feature matrix, but (1) add a minimal additional filter on `fare_per_mile` (already computed) and (2) tighten the minimum distance threshold a bit to reduce extreme ratios; both are standard NYC Taxi Fare competition hygiene and don’t alter your model logic. I also ensure the validation/test feature engineering mirrors training exactly by computing `distance_to_center` for test/val as well (even though it’s not used as a feature, it keeps preprocessing consistent and avoids accidental future leakage/bugs). The submission format and paths remain unchanged and a valid `submission.csv` is still written.'
- What this solution (achieved 727.69691) has done: 'Your current RMSE (670) is far from the target (5.46), so the least invasive way to move toward the target is to keep the exact same OLS model/features but make the data cleaning consistent and remove the remaining high-leverage outliers that can explode least-squares. I (1) apply the same “plausible fare” rule already present but tighten it slightly and (2) add a minimal max-distance cap plus a small passenger_count floor to remove corrupted/rare rows; these are standard NYC Taxi Fare hygiene steps and don’t change the model architecture. I also ensure `fare_per_mile` is recomputed after filtering to avoid stale NaNs/infs affecting later filters. The submission writing stays identical and still produces `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 728.20524) has done: 'Your RMSE is still massively inflated by a relatively small number of corrupted/high‑leverage rows that survive the current filters and destabilize the OLS fit. To move strongly toward the target while keeping the exact same OLS model and the same 5-column feature matrix, I add two standard NYC Taxi Fare hygiene filters that remove the worst remaining leverage points: (1) drop rows with near-zero coordinate movement (often bad GPS with nonzero fare) and (2) cap unrealistically high fare-per-mile more strictly after recomputing it post-filtering. I also ensure we don’t accidentally carry any inf/NaN from inverse-distance into the regression dataset by dropping non-finite rows before splitting (this doesn’t change the model, it just prevents numeric poison). These are minimal, metric-aligned data-quality changes and keep submission generation unchanged.'
- What this solution (achieved 6.61334) has done: 'Your RMSE is still extremely far from the target (lower is better), and the main reason is that the OLS model is being fit on a heavily biased subset (after aggressive cleaning) and without the biggest missing “core-logic-preserving” signal: the NYC starting fare and the typical per-distance/ per-time structure. Keeping the same OLS least-squares approach and the same training loop, the smallest high-impact change is to expand the feature matrix modestly with standard, directly metric-aligned linear features (trip distance in both linear/quadratic form, time-of-day effects, and a constant + passenger_count) while keeping `np.linalg.lstsq` unchanged. I also make training/test preprocessing consistent by applying the same NaN/inf dropping and by recomputing engineered columns after datetime parsing in both sets, which prevents silent NaNs from producing wildly wrong predictions. The submission writing remains identical (`submission.csv` with `key,fare_amount`).'
- What this solution (achieved 6.61334) has done: 'You’re already close-ish to the target band (6.613 → target 5.462, lower is better), so the smallest reliable improvement is to remove a clear train/test preprocessing mismatch that injects extra noise: you currently re-parse `pickup_datetime` on `val_df` (after it was already parsed) and then fill missing engineered values with 0.0, which creates invalid `hour/year/dist` patterns the model never learned. I keep the exact same OLS (`np.linalg.lstsq`) and the exact same feature matrix, but (1) do feature engineering once via a shared function for train/val/test, (2) drop non-finite rows in train/val instead of zero-filling them (test still gets safe imputation), and (3) ensure `val_df` retains `pickup_datetime` by preventing it from being dropped in `X` so the validation features are consistent with training. These are minimal, metric-aligned consistency fixes that should reduce RMSE toward your target without changing the core model. The script still write a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.61334) has done: 'I fix the crash in feature engineering by replacing the fragile `np.issubdtype(..., np.datetime64)` check (which breaks on timezone-aware `datetime64[ns, UTC]`) with Pandas’ robust `is_datetime64_any_dtype`. I also ensure `val_df` retains `pickup_datetime` during the train/validation split (it was dropped from `X`), so the shared feature engineering function can run consistently. These changes are execution-blocking bug fixes and also remove a train/val preprocessing mismatch that was likely inflating RMSE; the OLS least-squares model and feature matrix remain unchanged. The script run end-to-end and write a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.71942) has done: 'I keep your exact OLS least-squares model and the same feature matrix, and make only two score-relevant fixes that typically move NYC Taxi RMSE down: (1) train on a log-transformed target (fit OLS on `log1p(fare_amount)` and invert with `expm1` at prediction time), which is still plain least-squares but reduces the influence of long-tail fares that inflate RMSE, and (2) use a safe test-time imputation that matches training distributions (median per feature) instead of filling NaNs with 0.0, which creates unrealistic feature patterns. I also ensure the validation path uses the same missing-value handling as training (drop non-finite rows, as you already intended) while keeping submission format and paths unchanged. These are minimal, metric-aligned adjustments aimed at improving from 6.61 toward your 5.46 target without changing your overall approach.'
- What this solution (achieved 6.74688) has done: 'Your current RMSE (6.719) is above the target (5.462), so we should make a small, metric-aligned improvement without changing the OLS/log1p core. The biggest low-risk win here is to make the feature distribution closer between train and test by applying the same basic “valid NYC taxi row” filters to the training data in a way that doesn’t over-clean, specifically focusing on removing remaining high-leverage fare outliers that OLS is sensitive to. I add one additional standard outlier rule based on fare-per-mile (after recomputation) and a simple lower bound on fare to remove zero/near-zero anomalous rows, keeping your existing feature matrix and lstsq unchanged. I also make the validation imputation consistent with test (median impute) so the printed local RMSE better reflects the inference path, while preserving submission generation exactly.'
- What this solution (achieved 6.61359) has done: 'We need to move RMSE down from 6.74688 toward 5.46166 (lower is better), so we make the smallest metric-aligned improvements without changing your OLS/log1p core model or the feature matrix shape. The highest-leverage, low-risk fix is to correct a train/test preprocessing mismatch: you currently impute only 4 raw columns, but your model uses transforms (dist², sqrt(dist), hour²), so any NaNs in `distance_miles/hour/year/passenger_count` propagate into the matrix and silently degrade predictions. I add a tiny, shared imputation step that enforces valid ranges and fills those four base columns consistently for train/val/test before building matrices, plus clip extreme distances at inference to the same training-filter range to avoid extrapolation blow-ups. Submission writing remains identical (key,fare_amount) and still produces `submission.csv`.'
- What this solution (achieved 6.61359) has done: 'To move RMSE down from 6.61359 toward the 5.46166 target (lower is better) with minimal disruption, I keep your exact OLS/log1p least-squares core and the same feature matrix structure, but fix one train/validation inconsistency that’s currently adding noise: you train `w_OLS` on `train_df` before applying the same “constrain + median-impute” step you use for validation/test. I apply the identical constraint/imputation to the training base features before building `train_X`, then refit with `np.linalg.lstsq` as before; this typically reduces RMSE by removing feature-distribution mismatch without changing the modeling approach. I also ensure the medians are computed from the constrained/imputed training data (so they’re consistent), and keep submission formatting/paths unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))



## === cell 1
data = pd.read_csv("../input/train.csv", nrows=15_000_000)




## === cell 2
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(data)



## === cell 3
data["pickup_datetime"] = pd.to_datetime(
    data["pickup_datetime"], errors="coerce", utc=False
)




## === cell 4
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


data["distance_miles"] = distance(
    data.pickup_latitude,
    data.pickup_longitude,
    data.dropoff_latitude,
    data.dropoff_longitude,
)

_ = data.distance_miles.describe()
_ = data.groupby("passenger_count")[["distance_miles", "fare_amount"]].mean()

print(
    "Average $USD/Mile : {:0.2f}".format(
        data.fare_amount.sum() / data.distance_miles.replace(0, np.nan).sum()
    )
)

data["fare_per_mile"] = data.fare_amount / data.distance_miles.replace(0, np.nan)
data["inv_distance_miles"] = 1 / data.distance_miles.replace(0, np.nan)

data["hour"] = data["pickup_datetime"].dt.hour
data["year"] = data["pickup_datetime"].dt.year



## === cell 5
import seaborn as sns
import matplotlib.pyplot as plt

corrmat = data.corr(numeric_only=True)
f, ax = plt.subplots(figsize=(12, 9))

k = 14  # number of variables for heatmap
cols = corrmat.nlargest(k, "fare_amount")["fare_amount"].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(
    cm,
    cbar=True,
    annot=True,
    square=True,
    fmt=".2f",
    annot_kws={"size": 10},
    yticklabels=cols.values,
    xticklabels=cols.values,
)
plt.show()



## === cell 6
print("Old size: %d" % len(data))
data = data.dropna(how="any", axis="rows")
print("New size: %d" % len(data))



## === cell 7
print("Old size: %d" % len(data))

data = data[(data.abs_diff_longitude < 3.0) & (data.abs_diff_latitude < 3.0)]
data = data[data.fare_amount >= 0]

data = data[data.fare_amount >= 2.5]

data = data[data.fare_amount <= 250]  # cap extreme leverage points
data = data[(data.passenger_count >= 1) & (data.passenger_count <= 9)]

data = data[(data.distance_miles > 0.15) & (data.distance_miles < 60.0)]

data = data[
    (data.pickup_longitude.between(-75, -72))
    & (data.dropoff_longitude.between(-75, -72))
    & (data.pickup_latitude.between(40, 42))
    & (data.dropoff_latitude.between(40, 42))
]

nyc = (-74.0063889, 40.7141667)
data["distance_to_center"] = distance(
    nyc[1], nyc[0], data.dropoff_latitude, data.dropoff_longitude
)
data = data[data.distance_to_center < 15.0]

data = data[data.fare_amount <= (2.75 * data.distance_miles + 8.0)]

data = data[(data.abs_diff_longitude + data.abs_diff_latitude) > 1e-4]

data["fare_per_mile"] = data.fare_amount / data.distance_miles.replace(0, np.nan)

data = data[(data.fare_per_mile.isna()) | (data.fare_per_mile <= 25.0)]

data = data[np.isfinite(data["distance_miles"].values)]
data = data[np.isfinite(data["fare_amount"].values)]
data = data[np.isfinite(data["hour"].values)]
data = data[np.isfinite(data["year"].values)]
data = data[np.isfinite(data["passenger_count"].values)]

print("New size: %d" % len(data))



## === cell 8
plot = data.iloc[:1000].plot.scatter("year", "fare_amount")



## === cell 9
from sklearn.model_selection import train_test_split

y = data.fare_amount
X = data.drop("fare_amount", axis=1)

train_df, val_df, train_y, val_y = train_test_split(
    X, y, test_size=0.2, random_state=42
)
train_df.dtypes




## === cell 10
def get_input_matrix(df):
    dist = df.distance_miles.to_numpy()
    hour = df.hour.to_numpy()
    year = df.year.to_numpy()
    pc = df.passenger_count.to_numpy()

    return np.column_stack(
        (
            dist,
            dist**2,
            np.sqrt(dist + 1e-6),
            pc,
            hour,
            hour**2,
            year,
            np.ones(len(df)),
        )
    )


train_X = get_input_matrix(train_df)

print(train_X.shape)
print(train_y.shape)



## === cell 11
from sklearn.metrics import mean_squared_error

val_features = ["distance_miles", "hour", "year", "passenger_count"]


def constrain_and_impute_base_features(df, medians):
    df = df.copy()

    if "distance_miles" in df.columns:
        df["distance_miles"] = df["distance_miles"].clip(lower=0.15, upper=60.0)
    if "passenger_count" in df.columns:
        df["passenger_count"] = df["passenger_count"].clip(lower=1.0, upper=9.0)
    if "hour" in df.columns:
        df["hour"] = df["hour"].clip(lower=0.0, upper=23.0)

    for c in val_features:
        df[c] = df[c].fillna(medians[c])

    df = df.replace([np.inf, -np.inf], np.nan)
    for c in val_features:
        df[c] = df[c].fillna(medians[c])

    return df


train_df_constrained = train_df.copy()
train_df_constrained["distance_miles"] = train_df_constrained["distance_miles"].clip(
    lower=0.15, upper=60.0
)
train_df_constrained["passenger_count"] = train_df_constrained["passenger_count"].clip(
    lower=1.0, upper=9.0
)
train_df_constrained["hour"] = train_df_constrained["hour"].clip(lower=0.0, upper=23.0)

train_feature_medians = train_df_constrained[val_features].median(numeric_only=True)

train_df_imp = constrain_and_impute_base_features(train_df, train_feature_medians)
train_ok = np.isfinite(train_df_imp[val_features].to_numpy()).all(axis=1)
train_df_final = train_df_imp.loc[train_ok].copy()
train_y_final = train_y.loc[train_ok].copy()

train_X = get_input_matrix(train_df_final)
train_y_log = np.log1p(train_y_final.to_numpy())

(w, _, _, _) = np.linalg.lstsq(train_X, train_y_log, rcond=None)
print(w)



## === cell 12
w_OLS = w
print(w_OLS)



## === cell 13
from datetime import datetime
from pandas.api.types import is_datetime64_any_dtype


def engineer_features(df, nyc_center):
    df = df.copy()
    add_travel_vector_features(df)

    if not is_datetime64_any_dtype(df["pickup_datetime"]):
        df["pickup_datetime"] = pd.to_datetime(
            df["pickup_datetime"], errors="coerce", utc=False
        )

    df["hour"] = df["pickup_datetime"].dt.hour
    df["year"] = df["pickup_datetime"].dt.year

    df["distance_miles"] = distance(
        df.pickup_latitude,
        df.pickup_longitude,
        df.dropoff_latitude,
        df.dropoff_longitude,
    )
    df["distance_to_center"] = distance(
        nyc_center[1], nyc_center[0], df.dropoff_latitude, df.dropoff_longitude
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


test_df = pd.read_csv("../input/test.csv")
test_df = engineer_features(test_df, nyc)

val_df = engineer_features(val_df, nyc)

test_df.dtypes



## === cell 14
val_df_imp = constrain_and_impute_base_features(val_df, train_feature_medians)
test_df_imp = constrain_and_impute_base_features(test_df, train_feature_medians)

val_ok = np.isfinite(val_df_imp[val_features].to_numpy()).all(axis=1)
val_df_final = val_df_imp.loc[val_ok].copy()
val_y_final = val_y.loc[val_ok].copy()

test_ok = np.isfinite(test_df_imp[val_features].to_numpy()).all(axis=1)
test_df_final = test_df_imp.loc[test_ok].copy()

test_X = get_input_matrix(test_df_final)
val_X = get_input_matrix(val_df_final)

test_y_pred_log = np.matmul(test_X, w_OLS)
val_y_pred_log = np.matmul(val_X, w_OLS)

test_y_predictions = np.expm1(test_y_pred_log)
val_y_predictions = np.expm1(val_y_pred_log)

test_y_predictions = np.clip(test_y_predictions, 0.0, None)
val_y_predictions = np.clip(val_y_predictions, 0.0, None)

print(np.sqrt(mean_squared_error(val_y_final, val_y_predictions)))

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": np.nan})
submission.loc[test_ok, "fare_amount"] = test_y_predictions
submission["fare_amount"] = submission["fare_amount"].fillna(
    float(np.median(test_y_predictions))
)

submission = submission[["key", "fare_amount"]]
submission.to_csv("submission.csv", index=False)

print(os.listdir("."))
