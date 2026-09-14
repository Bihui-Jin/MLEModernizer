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

No external packages required in the script and installed.

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

4.13939939564504

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 10.02934) has done: 'I replace the failing TensorFlow‑based pipeline with a minimal, reliable solution: load the training data, compute the overall mean fare, and use this constant prediction for every test row. This removes the TF 1.x incompatibilities, fixes the date‑part feature errors, and guarantees a valid `submission.csv` file with the required columns. The change is small, keeps the overall workflow (load data → generate predictions → save submission) and moves the script from “no output” to a working baseline submission.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd

print("Input directories:", os.listdir("../input"))




## === cell 1
test_path = "../input/new-york-city-taxi-fare-prediction/test.csv"
df_test = pd.read_csv(
    test_path,
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    parse_dates=["pickup_datetime"],
)
print("Test rows:", df_test.shape[0])




## === cell 2
train_path = "../input/new-york-city-taxi-fare-prediction/train.csv"

sample_n = 1000000  # 1 M rows ≈ 60 MB, acceptable for the environment
df_train = pd.read_csv(
    train_path,
    usecols=[
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    parse_dates=["pickup_datetime"],
    nrows=sample_n,
)


def haversine(lon1, lat1, lon2, lat2):
    """km distance between two lon/lat points (vectorised)."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km


def build_features(df):
    """Return a 2‑D array with intercept and engineered features."""
    dist = haversine(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    )
    hour = df["pickup_datetime"].dt.hour.values
    dow = df["pickup_datetime"].dt.dayofweek.values
    month = df["pickup_datetime"].dt.month.values
    pax = df["passenger_count"].values.astype(float)

    X = np.column_stack((np.ones_like(dist), dist, pax, hour, dow, month))
    return X


X_train = build_features(df_train)
y_train = df_train["fare_amount"].values

beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
print("Linear model coefficients:", beta)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
LinAlgError                               Traceback (most recent call last)
/tmp/ipykernel_11/146055927.py in <cell line: 0>()
     55 # Solve ordinary least squares: (X^T X) β = X^T y
     56 # Using lstsq for numerical stability
---> 57 beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
     58 print("Linear model coefficients:", beta)
     59 

/usr/local/lib/python3.11/dist-packages/numpy/linalg/linalg.py in lstsq(a, b, rcond)
   2324         # lapack can't handle n_rhs = 0 - so allocate the array one larger in that axis
   2325         b = zeros(b.shape[:-2] + (m, n_rhs + 1), dtype=b.dtype)
-> 2326     x, resids, rank, s = gufunc(a, b, rcond, signature=signature, extobj=extobj)
   2327     if m == 0:
   2328         x[...] = 0

/usr/local/lib/python3.11/dist-packages/numpy/linalg/linalg.py in _raise_linalgerror_lstsq(err, flag)
    122 
    123 def _raise_linalgerror_lstsq(err, flag):
--> 124     raise LinAlgError("SVD did not converge in Linear Least Squares")
    125 
    126 def _raise_linalgerror_qr(err, flag):

LinAlgError: SVD did not converge in Linear Least Squares

## === cell 3
X_test = build_features(df_test)

preds = X_test @ beta

preds = np.clip(preds, a_min=0.0, a_max=None)

submission = pd.DataFrame({"key": df_test["key"], "fare_amount": preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1421838062.py in <cell line: 0>()
      3 
      4 # Predict fare amounts
----> 5 preds = X_test @ beta
      6 
      7 # Clip predictions to a reasonable range (avoid negative fares)

NameError: name 'beta' is not defined
