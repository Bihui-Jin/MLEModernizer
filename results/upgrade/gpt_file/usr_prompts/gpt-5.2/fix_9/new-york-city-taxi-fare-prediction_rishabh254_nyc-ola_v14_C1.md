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

6.61334

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 887.48032) has done: 'I fix the pandas API breakages (tuple column selection in groupby and `DataFrame.corr()` failing due to non-numeric columns) so the notebook runs without stopping early. I also fix the datetime parsing so it works with the dataset’s `pickup_datetime` format (which includes fractional seconds but no timezone), and ensure `hour`/`year` are created before the train/validation split so later cells can access them. Finally, I keep the same OLS (least-squares) core model and feature set, but make the pipeline deterministic and guarantee a valid `submission.csv` with the required columns is written end-to-end.'
- What this solution (achieved 988.75049) has done: 'Your current RMSE (887) is dominated by a small number of wildly wrong predictions; the simplest way to move toward the target is to make the OLS fit more numerically stable and remove a clear data-quality issue that creates extreme outliers. I (1) ensure the core features used in the regression (`distance_miles`, `hour`, `year`) are computed before the train/validation split so the model is trained on the same feature definitions you use at inference, (2) add minimal coordinate sanity filters (NYC bounding box + fare cap) to prevent pathological rows from exploding the least-squares solution, and (3) stop rounding predictions before RMSE evaluation/submission and clip negative fares to 0.0 (rounding/clipping is post-processing and doesn’t change the model). These are small, metric-aligned fixes that should dramatically reduce RMSE while keeping your same OLS core logic and feature set.'
- What this solution (achieved 674.69369) has done: 'Your RMSE is still being blown up by a small number of bad rows (typically corrupted coordinates or implausible fares/distances) that slip through and create extreme leverage for an OLS fit. I keep the same OLS least-squares model and the same feature matrix, but (1) add one additional minimal, standard NYC-taxi cleaning rule (`fare_amount <= 3 * distance_miles + 7`, which removes only implausible outliers), and (2) make the OLS coefficients use the numerically-stable `np.linalg.lstsq` solution consistently (removing the explicit matrix inverse path that can become unstable). This should sharply reduce validation RMSE and move it toward your target while preserving the core logic and producing the same required `submission.csv`. All paths and the submission schema stay unchanged.'
- What this solution (achieved 670.27732) has done: 'Your current RMSE is still dominated by a handful of high-leverage training rows that survive cleaning and destabilize an OLS fit; the smallest change that should move you dramatically toward the target is to make the training set cleaning slightly stricter in the two places most correlated with blow-ups: tiny-distance rides and implausibly high fare-per-mile. I keep the same OLS least-squares core model and the same feature matrix, but (1) add a minimal additional filter on `fare_per_mile` (already computed) and (2) tighten the minimum distance threshold a bit to reduce extreme ratios; both are standard NYC Taxi Fare competition hygiene and don’t alter your model logic. I also ensure the validation/test feature engineering mirrors training exactly by computing `distance_to_center` for test/val as well (even though it’s not used as a feature, it keeps preprocessing consistent and avoids accidental future leakage/bugs). The submission format and paths remain unchanged and a valid `submission.csv` is still written.'
- What this solution (achieved 727.69691) has done: 'Your current RMSE (670) is far from the target (5.46), so the least invasive way to move toward the target is to keep the exact same OLS model/features but make the data cleaning consistent and remove the remaining high-leverage outliers that can explode least-squares. I (1) apply the same “plausible fare” rule already present but tighten it slightly and (2) add a minimal max-distance cap plus a small passenger_count floor to remove corrupted/rare rows; these are standard NYC Taxi Fare hygiene steps and don’t change the model architecture. I also ensure `fare_per_mile` is recomputed after filtering to avoid stale NaNs/infs affecting later filters. The submission writing stays identical and still produces `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 728.20524) has done: 'Your RMSE is still massively inflated by a relatively small number of corrupted/high‑leverage rows that survive the current filters and destabilize the OLS fit. To move strongly toward the target while keeping the exact same OLS model and the same 5-column feature matrix, I add two standard NYC Taxi Fare hygiene filters that remove the worst remaining leverage points: (1) drop rows with near-zero coordinate movement (often bad GPS with nonzero fare) and (2) cap unrealistically high fare-per-mile more strictly after recomputing it post-filtering. I also ensure we don’t accidentally carry any inf/NaN from inverse-distance into the regression dataset by dropping non-finite rows before splitting (this doesn’t change the model, it just prevents numeric poison). These are minimal, metric-aligned data-quality changes and keep submission generation unchanged.'
- What this solution (achieved 6.61334) has done: 'Your RMSE is still extremely far from the target (lower is better), and the main reason is that the OLS model is being fit on a heavily biased subset (after aggressive cleaning) and without the biggest missing “core-logic-preserving” signal: the NYC starting fare and the typical per-distance/ per-time structure. Keeping the same OLS least-squares approach and the same training loop, the smallest high-impact change is to expand the feature matrix modestly with standard, directly metric-aligned linear features (trip distance in both linear/quadratic form, time-of-day effects, and a constant + passenger_count) while keeping `np.linalg.lstsq` unchanged. I also make training/test preprocessing consistent by applying the same NaN/inf dropping and by recomputing engineered columns after datetime parsing in both sets, which prevents silent NaNs from producing wildly wrong predictions. The submission writing remains identical (`submission.csv` with `key,fare_amount`).'
- What this solution (achieved 6.61334) has done: 'You’re already close-ish to the target band (6.613 → target 5.462, lower is better), so the smallest reliable improvement is to remove a clear train/test preprocessing mismatch that injects extra noise: you currently re-parse `pickup_datetime` on `val_df` (after it was already parsed) and then fill missing engineered values with 0.0, which creates invalid `hour/year/dist` patterns the model never learned. I keep the exact same OLS (`np.linalg.lstsq`) and the exact same feature matrix, but (1) do feature engineering once via a shared function for train/val/test, (2) drop non-finite rows in train/val instead of zero-filling them (test still gets safe imputation), and (3) ensure `val_df` retains `pickup_datetime` by preventing it from being dropped in `X` so the validation features are consistent with training. These are minimal, metric-aligned consistency fixes that should reduce RMSE toward your target without changing the core model. The script still write a valid `submission.csv` with `key,fare_amount`.'

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
data = data[(data.fare_per_mile.isna()) | (data.fare_per_mile <= 35.0)]

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
(w, _, _, _) = np.linalg.lstsq(train_X, train_y, rcond=None)
print(w)



## === cell 12
w_OLS = w
print(w_OLS)



## === cell 13
from datetime import datetime


def engineer_features(df, nyc_center):
    df = df.copy()
    add_travel_vector_features(df)

    if not np.issubdtype(df["pickup_datetime"].dtype, np.datetime64):
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



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2028503570.py in <cell line: 0>()
     34 test_df = engineer_features(test_df, nyc)
     35 
---> 36 val_df = engineer_features(val_df, nyc)
     37 
     38 test_df.dtypes

/tmp/ipykernel_11/2028503570.py in engineer_features(df, nyc_center)
      9 
     10     # Parse only if not already datetime64; this avoids creating extra NaTs in val_df
---> 11     if not np.issubdtype(df["pickup_datetime"].dtype, np.datetime64):
     12         df["pickup_datetime"] = pd.to_datetime(
     13             df["pickup_datetime"], errors="coerce", utc=False

/usr/local/lib/python3.11/dist-packages/numpy/core/numerictypes.py in issubdtype(arg1, arg2)
    415     """
    416     if not issubclass_(arg1, generic):
--> 417         arg1 = dtype(arg1).type
    418     if not issubclass_(arg2, generic):
    419         arg2 = dtype(arg2).type

TypeError: Cannot interpret 'datetime64[ns, UTC]' as a data type

## === cell 14
val_features = ["distance_miles", "hour", "year", "passenger_count"]
val_ok = np.isfinite(val_df[val_features].to_numpy()).all(axis=1)
val_df_clean = val_df.loc[val_ok].copy()
val_y_clean = val_y.loc[val_ok].copy()

test_X = get_input_matrix(test_df.fillna(0.0))
val_X = get_input_matrix(val_df_clean)

test_y_predictions = np.matmul(test_X, w)
val_y_predictions = np.matmul(val_X, w)

test_y_predictions = np.clip(test_y_predictions, 0.0, None)
val_y_predictions = np.clip(val_y_predictions, 0.0, None)

from sklearn.metrics import mean_squared_error

print(np.sqrt(mean_squared_error(val_y_clean, val_y_predictions)))

submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": test_y_predictions},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)

print(os.listdir("."))
