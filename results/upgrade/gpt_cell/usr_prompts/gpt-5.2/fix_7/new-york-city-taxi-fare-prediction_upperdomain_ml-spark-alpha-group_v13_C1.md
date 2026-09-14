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

4.22828

# 6. Current score

5.51267

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.38379) has done: 'Diagnosis: The crash occurs because `SimpleImputer` is instantiated with `missing_values="NaN"` (a string), but `train_X` is a float64 numeric array; in scikit-learn this requires `missing_values` to be numeric (typically `np.nan`). The `fit()` call in cell 6 triggers `_check_inputs_dtype`, raising a `ValueError` due to the type mismatch. The minimal fix is to change the imputer’s `missing_values` from the string `"NaN"` to `np.nan`, keeping all downstream logic unchanged.

Patch summary: In cell 6, update the `Imputer` initialization to use `missing_values=np.nan` so `imp.fit(train_X)` accepts the numeric matrix. No other behavior (model, data flow, loop) is modified.

Updated cells: Only cell 6 is changed.

Compatibility notes for cell k+1: Cell 7 does not depend on `imp`, `regr`, or any variables from cell 6; the change does not affect `distance_travel(test_df)` usage and preserves all existing names and semantics.

Assumptions: Missing values in `train_X` (if any) are represented as actual `np.nan` values rather than the literal string `"NaN"`.'
- What this solution (achieved 6.4395) has done: 'You’re currently training on only the first ~56M rows (56 chunks × 1M), which unintentionally overrepresents early-in-file data; a minimal way to move RMSE toward your target is to train on a small, more representative sample across the whole dataset by randomizing chunk order deterministically. I keep your exact feature engineering and `GradientBoostingRegressor` setup, but I (1) avoid double-applying `distance_travel` during cleaning, (2) actually use the imputer by transforming train/test matrices before fitting/scoring, and (3) ensure the same preprocessing is applied in train and test. These are small correctness fixes that typically improve RMSE without changing the core approach. The script still write `submission.csv` with `key,fare_amount` and finish within the time limit by using fewer, well-chosen chunks.'
- What this solution (achieved 6.33872) has done: 'Diagnosis: The `StopIteration` arises because `pd.read_csv("../input/train.csv", chunksize=...)` exhausts the generator before the loop has consumed all indices in `chosen_chunks`. This happens because `max_chunks_est=60` assumes at least 60 chunks exist, but the actual number of chunks in the file (given the true row count and `chunk_size`) can be smaller, so some randomly chosen chunk indices are out of range. The fix is to bound the sampled chunk indices by the true number of chunks (computed from the file row count) so the generator is never advanced past EOF. This preserves the same training logic while making chunk selection valid and deterministic.

Patch summary: In cell 6 only, compute `n_chunks_total` from the CSV row count (minus header) and `chunk_size`, then sample `chosen_chunks` from `range(n_chunks_total)` and cap `n_chunks_to_use` accordingly.

Updated cells / Compatibility notes for cell k+1 / Assumptions are included inline below.'
- What this solution (achieved 5.51267) has done: 'To move your RMSE down toward 4.228, the smallest reliable gain (without changing your model or features) is to (1) train on a bit more data while keeping the same random-chunk sampling strategy, and (2) avoid the within-chunk time-order split by using a deterministic shuffle before the 90/10 split so the validation slice is less biased. These changes keep your core logic intact (same `distance_travel`, same outlier rules, same `GradientBoostingRegressor`, same imputer usage) while typically improving generalization. I also ensure the test-time `distance_travel` call assigns back to `test_df` to avoid any potential chained-assignment edge cases. The submission format and path stay the same, and runtime remains bounded by reading only a small number of 1M-row chunks.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

from sklearn.impute import SimpleImputer as Imputer

from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor

print(os.listdir("../input"))


def chunck_generator(filename, header=False, chunk_size=10**6):
    for chunk in pd.read_csv(
        filename, delimiter=",", iterator=True, chunksize=chunk_size, parse_dates=[1]
    ):
        yield (chunk)




## === cell 1
alpha_ang = 0.506


def distance_travel(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs() * 50
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs() * 69
    df["displacement_vector"] = (
        df.abs_diff_latitude**2 + df.abs_diff_longitude**2
    ) ** 0.5  ### as the crow flies
    df["actual_long"] = (
        df.displacement_vector
        * np.sin(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude) - alpha_ang)
    ).abs()
    df["actual_lat"] = (
        df.displacement_vector
        * np.cos(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude) - alpha_ang)
    ).abs()
    df["distance_travel"] = df.actual_long + df.actual_lat
    return df




## === cell 2
def data_clean(df):
    df = df[df.passenger_count > 0]
    df.fare_amount = df.fare_amount.astype(np.float64)
    df = df[df.fare_amount > 0]
    df = df[df.distance_travel > 0]
    return df




## === cell 3
def remove_outliers(df):
    df = df[df.distance_travel < 30]
    df = df[df.fare_amount < 100]
    return df




## === cell 4
def graph_presesnt(df):
    test = df[df.passenger_count == 1]
    plot = test.iloc[: len(test)].plot.scatter("distance_travel", "fare_amount")
    plot = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")




## === cell 5
def incremental_training(train_X, train_y, regr):
    regr.fit(train_X, train_y)
    return regr




## === cell 6
filename = r"../input/train.csv"
chunk_size = 10**6

rng = np.random.RandomState(42)

with open(filename, "r", encoding="utf-8") as f:
    n_rows = sum(1 for _ in f) - 1  # subtract header row
n_chunks_total = int(math.ceil(n_rows / float(chunk_size)))

n_chunks_to_use = 20
n_chunks_to_use = min(n_chunks_to_use, n_chunks_total)

chosen_chunks = sorted(
    rng.choice(np.arange(n_chunks_total), size=n_chunks_to_use, replace=False).tolist()
)
print("Total chunks available:", n_chunks_total)
print("Chosen chunk indices:", chosen_chunks)

regr = GradientBoostingRegressor(n_estimators=100, warm_start=True)
imp = Imputer(missing_values=np.nan, strategy="mean")

gen = chunck_generator(filename=filename, chunk_size=chunk_size)

current_chunk_idx = 0
used = 0
while used < len(chosen_chunks):
    df = next(gen)
    if current_chunk_idx != chosen_chunks[used]:
        current_chunk_idx += 1
        continue

    print("Training on chunk:", current_chunk_idx)
    df = distance_travel(df)
    df = data_clean(df)
    df = remove_outliers(df)

    l = len(df)
    if l < 1000:
        current_chunk_idx += 1
        used += 1
        continue

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    df_train = df[: int(0.9 * l)]
    df_test = df[int(0.9 * l) :]

    train_X = np.column_stack(
        (df_train.distance_travel, df_train.passenger_count, np.ones(len(df_train)))
    )
    test_X = np.column_stack(
        (df_test.distance_travel, df_test.passenger_count, np.ones(len(df_test)))
    )

    train_y = np.array(df_train.fare_amount)
    test_y = np.array(df_test.fare_amount)

    imp = imp.fit(train_X)
    train_X = imp.transform(train_X)
    test_X = imp.transform(test_X)

    regr = incremental_training(train_X, train_y, regr)
    print(regr.score(test_X, test_y))

    current_chunk_idx += 1
    used += 1



## === cell 7
test_df = pd.read_csv("../input/test.csv")

test_df = distance_travel(test_df)
test_df.head()



## === cell 8
test_X = np.column_stack(
    (test_df.distance_travel, test_df.passenger_count, np.ones(len(test_df)))
)
test_X = imp.transform(test_X)
predicted_fare = regr.predict(test_X)
print(predicted_fare)



## === cell 9
my_submission = pd.DataFrame({"key": test_df.key, "fare_amount": predicted_fare})
my_submission.to_csv("submission.csv", index=False)
my_submission.head()
