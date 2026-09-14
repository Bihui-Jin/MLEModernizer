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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

5.91416

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 28.8042) has done: 'Your very high RMSE is mainly coming from (1) predicting negative/near-zero fares for some rides (no non-negativity constraint) and (2) rounding predictions to cents before scoring, which adds avoidable error. I keep your exact OLS-with-two-features approach, but make three minimal fixes: set a fixed `random_state` for stable validation, stop rounding predictions before evaluation/submission, and clip predictions to a reasonable non-negative range to avoid catastrophic outliers. These changes preserve your core logic (same features, same linear least squares fit) while moving the score substantially toward the target. The script still writes a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 28.8139) has done: 'Your current RMSE is far above the target, so we should make small, score-relevant fixes without changing the core “OLS on abs diffs + bias” logic. The biggest gain (still within the same modeling approach) is to remove obviously bad training rows (invalid coordinates, unrealistic passenger counts, and extreme/invalid fares) that distort the least-squares fit and produce large errors. We also ensure the features are computed after cleaning and keep the existing non-negativity clipping for predictions (it reduces catastrophic outliers under RMSE). Finally, we keep the same split/random_state and still write a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 15.14804) has done: 'We keep your exact “OLS on abs coordinate diffs + bias” approach, but make small, score-relevant fixes that typically reduce RMSE a lot: (1) compute the same feature matrix, but fit with `lstsq` only (dropping the unstable explicit inverse that can amplify noise), (2) add one more minimal, competition-standard cleaning step to remove extreme long trips (using haversine distance only for filtering, not as a model feature), and (3) clip predictions to a tighter, more realistic max fare (reducing RMSE blow-ups from a few large over-predictions). These changes don’t alter the model class or training loop; they just improve conditioning and remove distortive outliers so the same linear model fits better. The script still runs end-to-end within time limits and writes a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 15.14776) has done: 'Your current RMSE (15.148) is far above the target (5.914), so the smallest safe way to move toward the target is to keep the exact same linear OLS-on-abs-diffs model but fix the single biggest remaining source of error: you’re not using `passenger_count` at all, even though it’s already in the data and is a standard helpful linear feature. I add `passenger_count` as an additional regressor (still the same least-squares fit, no new model/training loop), and I also remove the unused explicit-inverse OLS computation to avoid any accidental divergence from the `lstsq` weights you actually use. Everything else (cleaning, feature extraction, clipping, submission format/path) stays the same, and the script still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))



## === cell 1
data = pd.read_csv("../input/train.csv", nrows=20_000_000)




## === cell 2
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()




## === cell 3
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * (2.0 * np.arcsin(np.sqrt(a)))




## === cell 4
print(data.isnull().sum())



## === cell 5
print("Old size: %d" % len(data))
data = data.dropna(how="any", axis="rows")
print("New size: %d" % len(data))



## === cell 6
data = data[(data["passenger_count"] >= 1) & (data["passenger_count"] <= 6)]
data = data[(data["fare_amount"] > 0.0) & (data["fare_amount"] <= 250.0)]

data = data[
    (data["pickup_longitude"].between(-75.0, -72.0))
    & (data["dropoff_longitude"].between(-75.0, -72.0))
    & (data["pickup_latitude"].between(40.0, 42.0))
    & (data["dropoff_latitude"].between(40.0, 42.0))
]

trip_km = haversine_km(
    data["pickup_longitude"].values,
    data["pickup_latitude"].values,
    data["dropoff_longitude"].values,
    data["dropoff_latitude"].values,
)
data = data[
    trip_km < 60.0
]  # conservative NYC-centric cutoff; reduces catastrophic RMSE contributors

print("After passenger/fare/coord/distance cleaning size: %d" % len(data))



## === cell 7
data["trip_distance_km"] = trip_km



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3692930852.py in <cell line: 0>()
      1 # Change (score-relevant, minimal): add a physically meaningful linear feature (haversine distance)
      2 # instead of relying on abs coordinate diffs, while keeping the same linear lstsq training approach.
----> 3 data["trip_distance_km"] = trip_km
      4 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (19508557) does not match length of index (19505282)

## === cell 8
add_travel_vector_features(data)



## === cell 9
try:
    plot = data.iloc[:10000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 10
print("Old size: %d" % len(data))
data = data[(data.abs_diff_longitude < 0.5) & (data.abs_diff_latitude < 0.5)]
print("New size: %d" % len(data))



## === cell 11
from sklearn.model_selection import train_test_split

y = data.fare_amount
X = data.drop("fare_amount", axis=1)

train_df, val_df, train_y, val_y = train_test_split(
    X, y, test_size=0.2, random_state=42
)
train_df.dtypes




## === cell 12
def get_input_matrix(df):
    return np.column_stack(
        (
            df.trip_distance_km.values.astype(float),
            df.passenger_count.values.astype(float),
            np.ones(len(df)),
        )
    )


train_X = get_input_matrix(train_df)

print(train_X.shape)
print(train_y.shape)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/835185948.py in <cell line: 0>()
     11 
     12 
---> 13 train_X = get_input_matrix(train_df)
     14 
     15 print(train_X.shape)

/tmp/ipykernel_11/835185948.py in get_input_matrix(df)
      4     return np.column_stack(
      5         (
----> 6             df.trip_distance_km.values.astype(float),
      7             df.passenger_count.values.astype(float),
      8             np.ones(len(df)),

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'trip_distance_km'

## === cell 13
(w, _, _, _) = np.linalg.lstsq(train_X, train_y, rcond=None)
print(w)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3062393645.py in <cell line: 0>()
----> 1 (w, _, _, _) = np.linalg.lstsq(train_X, train_y, rcond=None)
      2 print(w)
      3 

NameError: name 'train_X' is not defined

## === cell 14
print("Skipped explicit inverse OLS; using np.linalg.lstsq weights.")



## === cell 15
test_df = pd.read_csv("../input/test.csv")
test_df.dtypes



## === cell 16
test_df = test_df.copy()
test_df["passenger_count"] = test_df["passenger_count"].clip(lower=1, upper=6)

test_df["trip_distance_km"] = haversine_km(
    test_df["pickup_longitude"].values,
    test_df["pickup_latitude"].values,
    test_df["dropoff_longitude"].values,
    test_df["dropoff_latitude"].values,
)

add_travel_vector_features(test_df)

test_X = get_input_matrix(test_df)
val_X = get_input_matrix(val_df)

test_y_predictions = np.matmul(test_X, w)
val_y_predictions = np.matmul(val_X, w)

test_y_predictions = np.clip(test_y_predictions, 0.0, 250.0)
val_y_predictions = np.clip(val_y_predictions, 0.0, 250.0)

from sklearn.metrics import mean_squared_error

print(np.sqrt(mean_squared_error(val_y, val_y_predictions)))

submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": test_y_predictions},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)

print(os.listdir("."))

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2304274982.py in <cell line: 0>()
     14 
     15 test_X = get_input_matrix(test_df)
---> 16 val_X = get_input_matrix(val_df)
     17 
     18 test_y_predictions = np.matmul(test_X, w)

/tmp/ipykernel_11/835185948.py in get_input_matrix(df)
      4     return np.column_stack(
      5         (
----> 6             df.trip_distance_km.values.astype(float),
      7             df.passenger_count.values.astype(float),
      8             np.ones(len(df)),

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'trip_distance_km'
