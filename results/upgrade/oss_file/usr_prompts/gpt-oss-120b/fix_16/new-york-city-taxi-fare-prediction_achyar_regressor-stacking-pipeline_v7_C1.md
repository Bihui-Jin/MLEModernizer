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

3.9

# 3. Installed packages

catboost==1.2.8
geopandas==0.14.4
joblib==1.5.2
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
mlxtend==0.23.4
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

3.56643

# 6. Current score

4.57785

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.56414) has done: 'We eliminate the outer GridSearchCV wrapper, which was adding an unnecessary second layer of cross‑validation (doubling the training work) while not changing any model architecture or feature engineering. The pipeline (including the StackingCVRegressor that already performs its own CV) is fit directly using `TransformedTargetRegressor`. Corresponding prediction calls are updated to use this fitted model. This cuts the runtime roughly in half and keeps all original preprocessing, feature creation, and model stack unchanged.'
- What this solution (achieved 4.43054) has done: 'The fix raises the capacity of the two strongest learners in the stacking model – XGBoost and CatBoost – by increasing their number of trees/iterations. This adds predictive power while keeping the overall pipeline, feature engineering, and stacking logic unchanged, moving the RMSE closer to the target 3.56643.'
- What this solution (achieved 4.20916) has done: 'I increase the model capacity of the three strongest learners (XGBoost, LightGBM, CatBoost) by raising their number of trees/iterations, depth, and adding modest regularization parameters. This keeps the overall pipeline and feature engineering untouched while providing a realistic chance to lower the RMSE toward the target value.'
- What this solution (achieved 4.57785) has done: 'I import the missing libraries, define the required column lists and a small feature‑engineering transformer, build a simple preprocessing pipeline and train a LightGBM regressor on the sampled data. Then I load the test set, apply the same pipeline, generate predictions and write a correctly‑formatted `submission_v1a.csv` file.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor


def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Compute haversine distance in kilometers."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


class FeatureAdder(BaseEstimator, TransformerMixin):
    """Add distance and simple datetime features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["distance_km"] = haversine_vectorized(
            X["pickup_latitude"],
            X["pickup_longitude"],
            X["dropoff_latitude"],
            X["dropoff_longitude"],
        )
        X["pickup_hour"] = X["pickup_datetime"].dt.hour
        X["pickup_dayofweek"] = X["pickup_datetime"].dt.dayofweek
        X["pickup_month"] = X["pickup_datetime"].dt.month
        X = X.drop(columns=["pickup_datetime"])
        return X


train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
data = pd.read_csv(
    train_path, nrows=200_000, parse_dates=["pickup_datetime"]  # fast experiment
).drop(columns="key")
data = data.dropna().reset_index(drop=True)

target_col = "fare_amount"
ori_cols = [c for c in data.columns if c != target_col]

X_raw = data[ori_cols]
y = data[target_col].values

feature_adder = FeatureAdder()
numeric_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance_km",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
]

preprocess = ColumnTransformer(
    transformers=[("num", StandardScaler(), numeric_cols)], remainder="drop"
)

model = LGBMRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=-1, random_state=42, n_jobs=5
)

pipeline = Pipeline(
    steps=[("feat_add", feature_adder), ("preprocess", preprocess), ("model", model)]
)

pipeline.fit(X_raw, y)



## === cell 1
test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
test = pd.read_csv(test_path, parse_dates=["pickup_datetime"])

test_X_raw = test[ori_cols]  # same feature columns as training (excluding target)

test_pred = pipeline.predict(test_X_raw)

submission = pd.DataFrame({"key": test["key"], "fare_amount": test_pred})
submission_path = "submission_v1a.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
