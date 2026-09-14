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
sklearn-pandas==2.2.0
xgboost==2.0.3

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

3.35291

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 6.1605) has done: 'I apply a log‑transform to the target variable during training and inference, which usually lowers RMSE for skewed fare amounts while keeping the original model architecture unchanged. This small change lets the XGBoost regressor learn on a smoother signal and then convert predictions back to the original scale, moving the score nearer to the target.'
- What this solution (achieved 4.63524) has done: 'I fixed the NaN/invalid‑label issue by filtering the training targets before log‑transforming, added a haversine distance feature (which is cheap and improves model accuracy), and kept the rest of the pipeline unchanged. The script now fits without errors, produces predictions, and writes a correctly formatted `submission.csv` file.'
- What this solution (achieved 4.38506) has done: 'I add simple time‑based features (hour of day and weekday) to the preprocessing step and increase the number of trees slightly, which are low‑risk changes that usually lower RMSE for this dataset. These adjustments keep the original model and training flow unchanged while providing the regressor more relevant information, moving the score closer to the target.'
- What this solution (achieved 5.02442) has done: 'I tighten the data cleaning, add a few inexpensive time‑based and transformed distance features, and tune the XGBoost hyper‑parameters to learn more gradually – changes that keep the overall pipeline intact while should lower the RMSE toward the target.'
- What this solution (achieved 4.64315) has done: 'I enrich the feature set with cheap cyclic time encodings (sin/cos of hour and month) and an interaction term between distance and passenger count, tighten the target filtering to drop extreme fares, and introduce a small validation split with early‑stopping for XGBoost. These incremental tweaks keep the original model unchanged while expected to lower the RMSE toward the target.'
- What this solution (achieved 4.60523) has done: 'The changes focus on accelerating XGBoost training, which is the dominant runtime cost. By switching to the highly‑optimized histogram tree method and feeding NumPy arrays directly, we keep the exact same model configuration and data while vastly reducing overhead. Minor parameter additions (tree_method and eval_metric) do not alter the algorithmic logic or prediction semantics, preserving result accuracy.'
- What this solution (achieved 5.28107) has done: 'I revert the log‑transform of the target so the model is trained and validated on the original fare amounts, which aligns the training loss directly with the competition RMSE. This small change keeps the same preprocessing, feature set, and XGBoost configuration, but removes the exponential back‑conversion step and uses the raw fare values for splitting, fitting, and early‑stopping. The adjustment is expected to move the validation RMSE closer to the target (lower) while preserving the overall pipeline.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split




## === cell 1
train_cols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
df = pd.read_csv("../input/train.csv", usecols=train_cols)
test_set = pd.read_csv("../input/test.csv")  # keep all columns for submission




## === cell 2
def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Vectorised haversine distance in kilometres.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def preprocess(df_input, is_train=True):
    df = df_input.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["pickup_timestamp"] = df["pickup_datetime"].astype("int64") // 1_000_000_000
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_weekday"] = df["pickup_datetime"].dt.weekday
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_day"] = df["pickup_datetime"].dt.day

    df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["pickup_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["pickup_month"] / 12)

    df["distance"] = haversine_distance(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )
    df["distance_log"] = np.log1p(df["distance"])
    df["distance_passenger"] = df["distance"] * df["passenger_count"]

    df = df.drop(columns=["pickup_datetime"])

    keep_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
        "distance_log",
        "distance_passenger",
        "pickup_hour",
        "pickup_weekday",
        "pickup_month",
        "pickup_day",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "pickup_timestamp",  # new numeric time feature
    ]
    X = df[keep_cols]
    X = X.fillna(X.median())

    if is_train:
        y = df["fare_amount"]
        return X, y
    else:
        return X


X_train, y_train = preprocess(df, is_train=True)
test_set_key = test_set["key"].values
test_set_features = preprocess(test_set, is_train=False)




## === cell 3
valid_mask = (
    y_train.notnull()
    & (y_train >= 0)
    & (y_train <= 200)  # realistic fare range
    & (X_train["distance"] > 0)
    & (X_train["distance"] < 200)  # unrealistic long trips removed
    & (X_train["passenger_count"].between(1, 6))  # typical passenger range
)
X_train = X_train[valid_mask]
y_train = y_train[valid_mask]

X_train, y_train = (
    X_train.sample(frac=0.02, random_state=42),
    y_train.loc[X_train.sample(frac=1.0, random_state=42).index],
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3318930088.py in <cell line: 0>()
     20 )
     21 
---> 22 X_tr, X_val, y_tr, y_val = train_test_split(
     23     X_train, y_train, test_size=0.1, random_state=42
     24 )

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in train_test_split(test_size, train_size, random_state, shuffle, stratify, *arrays)
   2557         raise ValueError("At least one array required as input")
   2558 
-> 2559     arrays = indexable(*arrays)
   2560 
   2561     n_samples = _num_samples(arrays[0])

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in indexable(*iterables)
    441 
    442     result = [_make_indexable(X) for X in iterables]
--> 443     check_consistent_length(*result)
    444     return result
    445 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_consistent_length(*arrays)
    395     uniques = np.unique(lengths)
    396     if len(uniques) > 1:
--> 397         raise ValueError(
    398             "Found input variables with inconsistent numbers of samples: %r"
    399             % [int(l) for l in lengths]

ValueError: Found input variables with inconsistent numbers of samples: [1070664, 53533180]

## === cell 4
regressor = XGBRegressor(
    max_depth=10,
    learning_rate=0.04,
    n_estimators=800,  # reduced from 2000; early stopping will stop earlier if needed
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=113,
    min_child_weight=1,
    tree_method="hist",
    eval_metric="rmse",
)

regressor.fit(
    X_tr.values,
    y_tr,
    eval_set=[(X_val.values, y_val)],
    early_stopping_rounds=50,
    verbose=False,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3180766180.py in <cell line: 0>()
     15 
     16 regressor.fit(
---> 17     X_tr.values,
     18     y_tr,
     19     eval_set=[(X_val.values, y_val)],

NameError: name 'X_tr' is not defined

## === cell 5
y_pred_reg = regressor.predict(test_set_features.values)
y_pred_reg = np.clip(y_pred_reg, 0, 200)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/2716511178.py in <cell line: 0>()
----> 1 y_pred_reg = regressor.predict(test_set_features.values)
      2 y_pred_reg = np.clip(y_pred_reg, 0, 200)
      3 
      4 

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in predict(self, X, output_margin, validate_features, base_margin, iteration_range)
   1166             if self._can_use_inplace_predict():
   1167                 try:
-> 1168                     predts = self.get_booster().inplace_predict(
   1169                         data=X,
   1170                         iteration_range=iteration_range,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in get_booster(self)
    723             from sklearn.exceptions import NotFittedError
    724 
--> 725             raise NotFittedError("need to call fit or load_model beforehand")
    726         return self._Booster
    727 

NotFittedError: need to call fit or load_model beforehand

## === cell 6
submission = pd.DataFrame(
    {"key": test_set_key, "fare_amount": y_pred_reg}, columns=["key", "fare_amount"]
)
submission = submission.round({"fare_amount": 2})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Submission file written: submission.csv")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1914436642.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"key": test_set_key, "fare_amount": y_pred_reg}, columns=["key", "fare_amount"]
      3 )
      4 submission = submission.round({"fare_amount": 2})
      5 submission.to_csv("submission.csv", index=False)

NameError: name 'y_pred_reg' is not defined
