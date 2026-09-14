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

5.58616

# 6. Current score

995.4734

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 995.16197) has done: 'The changes import the correct imputer (`SimpleImputer`), ensure pandas and numpy are available throughout, fix the broken import of `Imputer`, and reorganize the workflow so that data loading, feature creation, model training, evaluation, and submission generation run without errors. A small interaction feature (`distance_travel * passenger_count`) is added to help the linear model reach a lower RMSE, moving the score toward the target while preserving the original logic.'
- What this solution (achieved 995.4734) has done: 'I enable the linear model to learn an intercept (removing the manual bias column) and add a simple temporal feature (hour of pickup) which is known to affect taxi fares. These minimal changes keep the original workflow but give the model more expressive power, which should lower the validation RMSE and move the score toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

print("Input folder contents:", os.listdir("../input"))




## === cell 1
train_path = "../input/train.csv"
df = pd.read_csv(train_path, nrows=1_000_000)
df.head()




## === cell 2
alpha_ang = 0.506
epsilon = 1e-6  # avoid division by zero in trig calculations


def distance_travel(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs() * 50
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs() * 69
    df["displacement_vector"] = np.sqrt(
        df.abs_diff_latitude**2 + df.abs_diff_longitude**2
    )
    lat_term = df.abs_diff_latitude + epsilon
    df["actual_long"] = (
        df.displacement_vector
        * np.sin(np.arctan(df.abs_diff_longitude / lat_term) - alpha_ang)
    ).abs()
    df["actual_lat"] = (
        df.displacement_vector
        * np.cos(np.arctan(df.abs_diff_longitude / lat_term) - alpha_ang)
    ).abs()
    df["distance_travel"] = df.actual_long + df.actual_lat
    return df


df = df[(df.passenger_count > 0) & (df.fare_amount > 0)]
df = distance_travel(df)
df = df[(df.distance_travel > 0) & (df.distance_travel < 30) & (df.fare_amount < 100)]
df.head()




## === cell 3
l = len(df)
train_df = df[: int(0.7 * l)]
valid_df = df[int(0.7 * l) :]


def build_features(data):
    distance = data.distance_travel.values.reshape(-1, 1)
    passenger = data.passenger_count.values.reshape(-1, 1)
    interaction = (data.distance_travel * data.passenger_count).values.reshape(-1, 1)
    hour = pd.to_datetime(data.pickup_datetime).dt.hour.values.reshape(-1, 1)
    return np.hstack([distance, passenger, interaction, hour])


train_X = build_features(train_df)
valid_X = build_features(valid_df)

train_y = train_df.fare_amount.values
valid_y = valid_df.fare_amount.values

imputer = SimpleImputer(strategy="mean")
train_X = imputer.fit_transform(train_X)
valid_X = imputer.transform(valid_X)




## === cell 4
regr = linear_model.LinearRegression(fit_intercept=True, n_jobs=1)
regr.fit(train_X, train_y)
pred_valid = regr.predict(valid_X)
rmse = np.sqrt(mean_squared_error(valid_y, pred_valid))
print(f"Validation RMSE: {rmse:.4f}")
print("Model coefficients:", regr.coef_)
print("Model intercept:", regr.intercept_)




## === cell 5
test_path = "../input/test.csv"
tdf = pd.read_csv(test_path, nrows=1_000_000)
tdf = distance_travel(tdf)

test_X = build_features(tdf)
test_X = imputer.transform(test_X)
test_pred = regr.predict(test_X)




## === cell 6
submission = pd.DataFrame({"key": tdf["key"], "fare_amount": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
submission.head()
