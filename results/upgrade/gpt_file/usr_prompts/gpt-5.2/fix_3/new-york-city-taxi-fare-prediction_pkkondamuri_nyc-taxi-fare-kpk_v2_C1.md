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

5.74679

# 6. Current score

1157.16471

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 739.77724) has done: 'Your RMSE is extremely high because the model is learning on absolute coordinate differences in degrees, which don’t correspond well to real travel distance; a minimal, legitimate fix is to compute a proper great-circle (Haversine) distance feature while keeping the same linear regression core. I keep your existing `lat_diff`/`lon_diff`/`passenger_count` features but add `haversine_km` (and reuse the same computation for train and test) so the model has a distance signal that matches fare. I also clamp negative predictions to 0 (fares can’t be negative), which typically reduces RMSE on this competition without changing the modeling approach. Finally, I make the train/test split deterministic to stabilize results and ensure the submission file is written correctly.'
- What this solution (achieved 1157.16471) has done: 'Your current RMSE is far above the target, so we should improve it with the smallest possible changes while keeping the same linear regression approach and existing feature set. The biggest remaining issue is that the training data still contains many invalid/outlier coordinates (and some unrealistic trips) that badly distort a linear model; we add minimal, standard NYC Taxi Fare filters (bounding box for NYC area + positive distance, optional “distance not crazy large”) without changing the model or features. We apply the exact same basic cleaning logic before feature creation so the learned coefficients reflect realistic trips and generalize to the test set. Finally, we keep your haversine feature and non-negative prediction clamp, and ensure the submission is written correctly as before.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))



## === cell 1
taxidata = pd.read_csv("../input/train.csv", nrows=20_000_000)



## === cell 2
import matplotlib.pyplot as plt
import seaborn as sns

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass



## === cell 3
taxidata.head()



## === cell 4
taxidata.info()



## === cell 5
taxidata.describe()



## === cell 6
print(taxidata.isnull().sum())



## === cell 7
taxidata = taxidata.dropna()



## === cell 8
len(taxidata)



## === cell 9
taxidata["lat_diff"] = (
    taxidata["dropoff_latitude"] - taxidata["pickup_latitude"]
).abs()
taxidata["lon_diff"] = (
    taxidata["dropoff_longitude"] - taxidata["pickup_longitude"]
).abs()



## === cell 10
plt.scatter(taxidata["lat_diff"][10000:20000], taxidata["lon_diff"][10000:20000])



## === cell 11
taxidata = taxidata[(taxidata["lat_diff"] < 5.0) & (taxidata["lon_diff"] < 5.0)]



## === cell 12
plt.hist(taxidata["passenger_count"][100000:200000], bins=20)



## === cell 13
sum(taxidata["passenger_count"] > 9)



## === cell 14
print(len(taxidata))
taxidata = taxidata[taxidata["passenger_count"] < 10]
print(len(taxidata))



## === cell 15
sum(taxidata["fare_amount"] > 3000)



## === cell 16
taxidata = taxidata[taxidata["fare_amount"] < 1000]




## === cell 17
def haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    R = 6371.0
    lat1 = np.radians(pickup_lat.astype(float))
    lon1 = np.radians(pickup_lon.astype(float))
    lat2 = np.radians(dropoff_lat.astype(float))
    lon2 = np.radians(dropoff_lon.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c




## === cell 18
nyc_bounds = {
    "min_lon": -74.3,
    "max_lon": -73.7,
    "min_lat": 40.5,
    "max_lat": 41.0,
}

taxidata = taxidata[
    (taxidata["pickup_longitude"].between(nyc_bounds["min_lon"], nyc_bounds["max_lon"]))
    & (
        taxidata["dropoff_longitude"].between(
            nyc_bounds["min_lon"], nyc_bounds["max_lon"]
        )
    )
    & (
        taxidata["pickup_latitude"].between(
            nyc_bounds["min_lat"], nyc_bounds["max_lat"]
        )
    )
    & (
        taxidata["dropoff_latitude"].between(
            nyc_bounds["min_lat"], nyc_bounds["max_lat"]
        )
    )
].copy()

taxidata["haversine_km"] = haversine_km(
    taxidata["pickup_latitude"],
    taxidata["pickup_longitude"],
    taxidata["dropoff_latitude"],
    taxidata["dropoff_longitude"],
)

taxidata = taxidata[
    (taxidata["haversine_km"] > 0.0) & (taxidata["haversine_km"] < 100.0)
]



## === cell 19
taxidata_X = taxidata[["lat_diff", "lon_diff", "passenger_count", "haversine_km"]]
taxidata_y = taxidata["fare_amount"]



## === cell 20
from sklearn.model_selection import train_test_split



## === cell 21
X_train, X_test, y_train, y_test = train_test_split(
    taxidata_X, taxidata_y, test_size=0.4, random_state=42
)



## === cell 22
from sklearn.linear_model import LinearRegression



## === cell 23
lm = LinearRegression()



## === cell 24
lm.fit(X_train, y_train)



## === cell 25
lm.intercept_



## === cell 26
lm.coef_



## === cell 27
test_pred = lm.predict(X_test)



## === cell 28
sns.histplot((y_test - test_pred), bins=50, kde=False)



## === cell 29
plt.scatter(y_test, test_pred, s=5, alpha=0.2)



## === cell 30
from sklearn import metrics



## === cell 31
np.sqrt(metrics.mean_squared_error(y_test, test_pred))



## === cell 32
taxifare_test = pd.read_csv("../input/test.csv")



## === cell 33
taxifare_test["lat_diff"] = (
    taxifare_test["dropoff_latitude"] - taxifare_test["pickup_latitude"]
).abs()
taxifare_test["lon_diff"] = (
    taxifare_test["dropoff_longitude"] - taxifare_test["pickup_longitude"]
).abs()



## === cell 34
taxifare_test["haversine_km"] = haversine_km(
    taxifare_test["pickup_latitude"],
    taxifare_test["pickup_longitude"],
    taxifare_test["dropoff_latitude"],
    taxifare_test["dropoff_longitude"],
)



## === cell 35
test_X = taxifare_test[["lat_diff", "lon_diff", "passenger_count", "haversine_km"]]



## === cell 36
predictions = lm.predict(test_X)



## === cell 37
predictions = np.maximum(predictions, 0.0)



## === cell 38
predictions.max()



## === cell 39
submission = pd.DataFrame(
    {"key": taxifare_test.key, "fare_amount": predictions},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)

print(os.listdir("."))
