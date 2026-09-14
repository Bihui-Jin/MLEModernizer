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

4.38819

# 6. Current score

5.45066

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.88985) has done: 'Diagnosis: The crash happens because in xgboost==2.0.3 the trained `Booster` no longer exposes `best_ntree_limit` (it was used by older versions for early-stopping). With early stopping enabled, the correct attribute to use is `best_iteration`, and prediction should pass an `iteration_range` to limit trees to the best iteration.  
Patch summary: Update cell 26 to compute a safe best iteration from the booster (`best_iteration` when available), and call `predict(..., iteration_range=(0, best_iter+1))` instead of using the missing `best_ntree_limit`. This preserves the original “predict using best early-stopped model” semantics.  
Updated cells: Only cell 26 is changed.  
Compatibility notes for cell k+1: `xgbm_pred` remains a NumPy array aligned with `test_df` rows, so cell 27 continues to work unchanged.  
Assumptions: `xgb.train(..., early_stopping_rounds=20, ...)` sets `best_iteration` on the returned booster in this xgboost version; if not, we fall back to predicting with all iterations.'
- What this solution (achieved 5.89891) has done: 'To move RMSE down toward your target (lower is better) with minimal disruption, I keep your exact feature set (`distance`, `passenger_count`) and training flow but fix a few small issues that commonly hurt this baseline. Specifically: (1) make the train/validation split deterministic and slightly more reliable for early stopping by setting `random_state`, (2) use the correct modern XGBoost regression objective (`reg:squarederror`) while keeping RMSE eval, (3) add conservative regularization/defaults (`eta`, `max_depth`, `subsample`, `colsample_bytree`) to reduce overfitting on only 100k rows, and (4) clip negative predictions to 0 since fares cannot be negative, which typically improves RMSE. These are small parameter/post-processing adjustments; the core approach (same features, same xgb.train with early stopping) remains intact and it still write a valid submission CSV.'
- What this solution (achieved 5.88465) has done: 'Your current RMSE is worse than the target, so we should make a small, legitimate improvement while keeping your exact core approach (same two features, same XGBoost training flow with early stopping). The biggest low-risk gain here is to add very standard NYC Taxi data sanity filters (valid lat/lon bounds + removing extreme fares) because your current pipeline only filters distance/passengers, leaving many noisy/outlier rows that inflate RMSE. I also switch early-stopping monitoring to a small validation set (as you already do) but use both train+valid in `evals` to stabilize early stopping without changing the model type. Finally, we keep the same submission schema and ensure predictions are clipped to non-negative.'
- What this solution (achieved 5.88465) has done: 'We keep your exact pipeline (same two features, same XGBoost training with early stopping) and make only a small, legitimate adjustment that typically reduces RMSE for this competition: add a `base_score` equal to the mean training fare so boosting starts from a strong prior instead of 0. This does not change architecture or training flow, but usually improves both convergence and calibration for non-negative regression targets like fares. We also keep the modern `best_iteration` + `iteration_range` prediction logic and still clip negatives to 0, producing the same submission schema and filename. No extra data, no added features, and it remains comfortably within runtime.'
- What this solution (achieved 6.1685) has done: 'Your current RMSE (5.88465) is worse than the target (4.38819), so we should make a small, legitimate improvement while keeping your exact core approach (same two features and XGBoost training with early stopping). The biggest low-risk gain without changing feature extraction is to slightly tighten/standardize data cleaning for extreme outliers that remain even after your current filters (these outliers disproportionately hurt RMSE). I add two conservative filters that are commonly used in this competition: remove zero-distance trips (often bad data) and cap unrealistically high fares for short trips via a mild “fare per mile” sanity bound. This keeps the model and features identical, just improves training signal quality, and still produces the same required submission CSV.'
- What this solution (achieved 6.23157) has done: 'I keep your exact model, features (`distance`, `passenger_count`), and XGBoost training flow intact, and only make small changes aimed at reducing RMSE from 6.1685 toward your 4.38819 target. The main adjustment is a more standard NYC Taxi cleaning step that removes extreme long rides (your current `distance < 15` filter discards too many informative trips and typically hurts generalization), while still keeping conservative geographic/fare bounds and your existing “fare per mile” sanity filter. I also add a tiny amount of regularization (`min_child_weight`, `reg_lambda`) and set `tree_method="hist"` for stability/speed without changing semantics. Finally, I ensure the submission filename is Kaggle-friendly (no spaces) while keeping the required columns and row alignment unchanged.'
- What this solution (achieved 5.45068) has done: 'Your RMSE is worse than the target (lower is better), so we should make a small, legitimate improvement without changing your model or feature set. The biggest low-risk gain here is to align training and inference preprocessing: apply the same geographic bounds, passenger_count filter, and distance floor to the test set so the model isn’t asked to extrapolate on obviously invalid rows. For those filtered/invalid test rows, we fall back to the training mean fare (using the same base_score you already compute) to avoid extreme errors. This keeps the core logic intact (same features, same xgb.train + early stopping) and still writes a valid submission CSV.'
- What this solution (achieved 5.45066) has done: 'Your current RMSE (5.45068) is worse than the target (4.38819), so we should make a small, legitimate improvement while keeping your same two features and XGBoost training flow. The biggest low-risk gain is to align the training sample with the test distribution by increasing the training read size (still far below the full 55M) so the model sees more representative trips. I also add a very conservative “same preprocessing” step for the test set: compute distance the same way but cap extreme distances (only for inference) to reduce extrapolation on rare bad coordinates without changing your feature set. Finally, I keep your early-stopping/best-iteration prediction logic and submission schema unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")



## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=500000)



## === cell 2
train_df.shape



## === cell 3
test_df = pd.read_csv("../input/test.csv")



## === cell 4
test_df.shape



## === cell 5
train_df.head(5)



## === cell 6
train_df.isnull().sum()



## === cell 7
train_df.dropna(inplace=True)



## === cell 8
train_df.describe()



## === cell 9
train_df = train_df[train_df["fare_amount"] > 0]



## === cell 10
train_df.shape




## === cell 11
def distance(lat1, lon1, lat2, lon2):
    a = (
        0.5
        - np.cos((lat2 - lat1) * 0.017453292519943295) / 2
        + np.cos(lat1 * 0.017453292519943295)
        * np.cos(lat2 * 0.017453292519943295)
        * (1 - np.cos((lon2 - lon1) * 0.017453292519943295))
        / 2
    )
    res = 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
    return res




## === cell 12
train_df["distance"] = distance(
    train_df.pickup_latitude,
    train_df.pickup_longitude,
    train_df.dropoff_latitude,
    train_df.dropoff_longitude,
)



## === cell 13
test_df["distance"] = distance(
    test_df.pickup_latitude,
    test_df.pickup_longitude,
    test_df.dropoff_latitude,
    test_df.dropoff_longitude,
)



## === cell 14
train_df = train_df[
    (train_df["pickup_longitude"].between(-75, -72))
    & (train_df["dropoff_longitude"].between(-75, -72))
    & (train_df["pickup_latitude"].between(40, 42))
    & (train_df["dropoff_latitude"].between(40, 42))
]

train_df = train_df[train_df["fare_amount"] < 200]



## === cell 15
train_df = train_df[train_df["distance"] < 50]



## === cell 16
train_df.describe()



## === cell 17
train_df = train_df[
    (train_df["passenger_count"] != 0) & (train_df["passenger_count"] < 10)
]



## === cell 18
train_df = train_df[train_df["distance"] > 0.01].copy()
train_df["fare_per_mile"] = train_df["fare_amount"] / train_df["distance"]
train_df = train_df[train_df["fare_per_mile"].between(0.0, 80.0)].copy()
train_df.drop(columns=["fare_per_mile"], inplace=True)



## === cell 19
test_df["distance"] = test_df["distance"].clip(lower=0, upper=50)



## === cell 20
feat_cols_s = ["distance", "passenger_count"]

X = train_df[feat_cols_s]
y = train_df["fare_amount"]



## === cell 21
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)



## === cell 22
import xgboost as xgb




## === cell 23
def XGBoost(X_train, X_test, y_train, y_test, num_rounds=2000):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)

    base_score = float(np.mean(y_train))

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 2.0,
        "reg_lambda": 1.5,
        "tree_method": "hist",
        "seed": 42,
        "base_score": base_score,
    }

    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        early_stopping_rounds=50,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        verbose_eval=False,
    )




## === cell 24
xgbm = XGBoost(X_train, X_test, y_train, y_test)

test_valid_mask = (
    (test_df["pickup_longitude"].between(-75, -72))
    & (test_df["dropoff_longitude"].between(-75, -72))
    & (test_df["pickup_latitude"].between(40, 42))
    & (test_df["dropoff_latitude"].between(40, 42))
    & (test_df["passenger_count"] != 0)
    & (test_df["passenger_count"] < 10)
    & (test_df["distance"] > 0.01)
)

dtest_submit_valid = xgb.DMatrix(test_df.loc[test_valid_mask, feat_cols_s])

best_iter = getattr(xgbm, "best_iteration", None)
if best_iter is None:
    pred_valid = xgbm.predict(dtest_submit_valid)
else:
    pred_valid = xgbm.predict(dtest_submit_valid, iteration_range=(0, best_iter + 1))

pred_valid = np.clip(pred_valid, 0, None)

fallback = float(np.mean(y_train))
xgbm_pred = np.full(shape=(len(test_df),), fill_value=fallback, dtype=np.float32)
xgbm_pred[test_valid_mask.to_numpy()] = pred_valid.astype(np.float32)



## === cell 25
submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": xgbm_pred}, columns=["key", "fare_amount"]
)
out_path = "xgboost_regression.csv"
submission.to_csv(out_path, index=False)
print(submission.head())
print("Wrote submission:", out_path, "rows:", len(submission))
