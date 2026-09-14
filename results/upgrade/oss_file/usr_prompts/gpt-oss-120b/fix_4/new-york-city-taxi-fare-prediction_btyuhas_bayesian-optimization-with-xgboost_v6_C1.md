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

bayesian-optimization==3.1.0
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

3.24593

# 6. Current score

9.18855

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.53126) has done: 'The script had three critical failures: (1) the BayesianOptimization `maximize` call used an unsupported keyword, (2) the best‑parameter extraction accessed a wrong attribute, and (3) the subsequent code depended on the missing `params` variable. These are fixed by removing the `acq` argument, pulling the optimal parameters from `xgb_bo.max['params']`, adding the required XGBoost settings, and ensuring the workflow proceeds to train, predict, and write a proper `submission.csv` file.'
- What this solution (achieved 10.12095) has done: 'Improved the training step to let XGBoost stop early on the validation set, using a larger max boost round budget. This yields a model that better fits the data and reduces validation RMSE, moving the score closer to the target while keeping the original feature engineering and workflow unchanged.'
- What this solution (achieved 9.18855) has done: 'I increase the training sample size, add a proper haversine distance feature (which captures true travel distance better than Manhattan), and give XGBoost a larger boost‑round budget so it can converge further. These small, focused changes should lower the validation RMSE and move the score closer to the target while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



## === cell 1
df = pd.read_csv(
    "../input/train.csv",
    nrows=1_000_000,  # increased sample size for better learning
    usecols=[1, 2, 3, 4, 5, 6, 7],  # fare_amount … passenger_count
)



## === cell 2
df["pickup_datetime"] = df["pickup_datetime"].str.slice(0, 16)
df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"], utc=True, format="%Y-%m-%d %H:%M"
)



## === cell 3
df.dropna(how="any", axis="rows", inplace=True)

mask = df["pickup_longitude"].between(-75, -73)
mask &= df["dropoff_longitude"].between(-75, -73)
mask &= df["pickup_latitude"].between(40, 42)
mask &= df["dropoff_latitude"].between(40, 42)
mask &= df["passenger_count"].between(0, 8)
mask &= df["fare_amount"].between(0, 250)

df = df[mask]




## === cell 4
def dist(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    """Manhattan‑style distance (vectorised for pandas/ndarray)."""
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)




## === cell 5
def haversine(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    """Vectorised Haversine distance in kilometres."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(
        np.radians, [pickup_lat, pickup_long, dropoff_lat, dropoff_long]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c




## === cell 6
def transform(data):
    data["hour"] = data["pickup_datetime"].dt.hour
    data["day"] = data["pickup_datetime"].dt.day
    data["month"] = data["pickup_datetime"].dt.month
    data["year"] = data["pickup_datetime"].dt.year
    data = data.drop("pickup_datetime", axis=1)

    nyc = (-74.0063889, 40.7141667)
    jfk = (-73.7822222222, 40.6441666667)
    ewr = (-74.175, 40.69)
    lgr = (-73.87, 40.77)

    data["distance_to_center"] = dist(
        nyc[1], nyc[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["pickup_distance_to_jfk"] = dist(
        jfk[1], jfk[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["dropoff_distance_to_jfk"] = dist(
        jfk[1], jfk[0], data["dropoff_latitude"], data["dropoff_longitude"]
    )
    data["pickup_distance_to_ewr"] = dist(
        ewr[1], ewr[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["dropoff_distance_to_ewr"] = dist(
        ewr[1], ewr[0], data["dropoff_latitude"], data["dropoff_longitude"]
    )
    data["pickup_distance_to_lgr"] = dist(
        lgr[1], lgr[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["dropoff_distance_to_lgr"] = dist(
        lgr[1], lgr[0], data["dropoff_latitude"], data["dropoff_longitude"]
    )

    data["long_dist"] = data["pickup_longitude"] - data["dropoff_longitude"]
    data["lat_dist"] = data["pickup_latitude"] - data["dropoff_latitude"]

    data["dist"] = dist(
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["dropoff_latitude"],
        data["dropoff_longitude"],
    )
    data["haversine_dist"] = haversine(
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["dropoff_latitude"],
        data["dropoff_longitude"],
    )
    return data


df = transform(df)



## === cell 7
X_train, X_valid, y_train, y_valid = train_test_split(
    df.drop("fare_amount", axis=1), df["fare_amount"], test_size=0.25, random_state=42
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)




## === cell 8
def xgb_evaluate(max_depth, gamma, colsample_bytree):
    """Return negative RMSE for BayesianOptimization (we maximize)."""
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": int(max_depth),
        "subsample": 0.8,
        "eta": 0.1,
        "gamma": gamma,
        "colsample_bytree": colsample_bytree,
        "verbosity": 0,
    }
    cv_result = xgb.cv(
        params,
        dtrain,
        num_boost_round=50,
        nfold=3,
        early_stopping_rounds=10,
        seed=42,
        verbose_eval=False,
    )
    return -cv_result["test-rmse-mean"].iloc[-1]




## === cell 9
xgb_bo = BayesianOptimization(
    f=xgb_evaluate,
    pbounds={"max_depth": (3, 7), "gamma": (0, 1), "colsample_bytree": (0.3, 0.9)},
    random_state=42,
)
xgb_bo.maximize(init_points=3, n_iter=5)



## === cell 10
best_params = xgb_bo.max["params"]
best_params["max_depth"] = int(best_params["max_depth"])
best_params.update(
    {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "subsample": 0.8,
        "eta": 0.1,
        "verbosity": 0,
    }
)



## === cell 11
model2 = xgb.train(
    best_params,
    dtrain,
    num_boost_round=1000,  # allow more rounds; early stopping will pick optimum
    evals=[(dvalid, "valid")],
    early_stopping_rounds=20,
    verbose_eval=False,
)

y_pred_valid = model2.predict(dvalid)
print("Validation RMSE:", np.sqrt(mean_squared_error(y_valid, y_pred_valid)))



## === cell 12
fscores = pd.DataFrame(
    {
        "feature": list(model2.get_fscore().keys()),
        "importance": list(model2.get_fscore().values()),
    }
)
fscores.sort_values(by="importance", inplace=True)
fscores.plot.barh(x="feature", y="importance", legend=False)
plt.tight_layout()
plt.show()



## === cell 13
test = pd.read_csv("../input/test.csv").set_index("key")
test["pickup_datetime"] = test["pickup_datetime"].str.slice(0, 16)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], utc=True, format="%Y-%m-%d %H:%M"
)
test = transform(test)

dtest = xgb.DMatrix(test)
y_pred_test = model2.predict(dtest)



## === cell 14
submission = pd.DataFrame({"key": test.index, "fare_amount": y_pred_test})
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
