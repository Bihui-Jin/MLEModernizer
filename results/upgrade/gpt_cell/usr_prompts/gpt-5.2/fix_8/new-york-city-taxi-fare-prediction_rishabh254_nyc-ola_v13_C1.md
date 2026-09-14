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

5.46206

# 6. Current score

15.00949

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 885.37292) has done: 'Diagnosis: Cell 4 crashes because newer pandas versions disallow selecting multiple columns from a GroupBy using a tuple (`['distance_miles', 'fare_amount']` written with comma syntax becomes a tuple). This worked in older pandas but now raises `ValueError: Cannot subset columns with a tuple... Use a list instead.` The fix is to change the GroupBy column selection to use a list of column names explicitly.

Patch summary: In cell 4, replace `data.groupby('passenger_count')['distance_miles', 'fare_amount'].mean()` with `data.groupby('passenger_count')[['distance_miles', 'fare_amount']].mean()`; no other logic is changed.

Updated cells: Cell 4 only.

Compatibility notes for cell k+1: The `data` DataFrame and all columns created in cell 4 (`distance_miles`, `fare_per_mile`, `inv_distance_miles`, `hour`, `year`, etc.) remain identical, so cell 5 (correlation/heatmap) work unchanged.

Assumptions: `passenger_count`, `distance_miles`, and `fare_amount` exist and are numeric enough for `mean()`; no changes are needed for the datetime parsing in earlier cells.'
- What this solution (achieved 884.74305) has done: 'The crash happens because `data.corr()` in pandas 2.2 tries to convert *all* columns to floats, but `data` contains non-numeric columns like `pickup_datetime` (and `datetime_object`), causing a string-to-float conversion error. The minimal fix is to compute correlations on numeric columns only, while keeping the rest of the heatmap logic unchanged. In pandas 2.x, the safest way is to pass `numeric_only=True` to `DataFrame.corr()`. This preserves the intended semantics (correlation heatmap over numeric features) and unblocks the notebook.'
- What this solution (achieved 884.45726) has done: 'Your score is extremely worse than the target (lower is better), and the main reason is that the model is trained with engineered features (distance/hour/year) but your train/val split is done **before** you compute `hour`/`year` for the split frames, so `train_df` likely lacks those columns when building `train_X` (or they’re misaligned/NaN), leading to a badly fit regression and huge RMSE. I make the smallest change that preserves your exact model and OLS training: compute `hour` and `year` using vectorized `pd.to_datetime` **once on the full dataset before splitting**, and do the same for test; this keeps the same features/linear model but fixes feature availability/consistency. I also remove the slow/fragile `datetime.strptime(... %Z)` list-comprehensions (which can silently parse wrong) and replace them with `pd.to_datetime(..., utc=True, errors='coerce')` to stabilize datetime parsing without changing semantics. Finally, I ensure submission predictions are finite and non-negative (fares can’t be negative) which typically reduces RMSE vs wild negatives without changing the modeling approach.'
- What this solution (achieved 980.62456) has done: 'Your RMSE is far above the target (lower is better), so the smallest meaningful improvement is to stop training OLS on raw, outlier-heavy data and instead apply the same basic geographic sanity filters to the training sample that the competition’s baseline solutions rely on. This keeps your exact core model (same engineered features and same `np.linalg.lstsq` OLS) but removes obviously wrong coordinates/fares that explode the fit and lead to huge errors. I also apply the same feature construction order consistently (create `distance_miles/hour/year` before splitting, as you intended) and ensure test-time feature columns match without changing the prediction formula. The submission format and path remain unchanged, and it still produces `submission.csv`.'
- What this solution (achieved 992.36693) has done: 'Your score is far worse than the target (lower RMSE is better), so the smallest change likely to move substantially toward the target is to keep your exact OLS model but make the engineered distance feature match the competition baseline more closely. I switch `distance_miles` from great-circle (haversine) to simple Manhattan distance in miles (still just a distance feature; same model/fit/predict pipeline), because taxi fares correlate better with street-grid distance in NYC and this usually reduces RMSE a lot for linear models. I also ensure the same distance definition is used consistently for train/validation/test, while keeping the same filters, split, training (`np.linalg.lstsq`), and submission format. No approximations, early stopping, or architecture changes are introduced—only the distance calculation used as an input feature is adjusted.'
- What this solution (achieved 696.68828) has done: 'Your RMSE is far above the target (lower is better), and the biggest “minimal-change” issue is that the model is fit on unscaled, outlier-heavy `distance_miles` while fares are much closer to linear in *log(1+distance)* for short trips and less sensitive for long trips. I keep your exact OLS training/prediction pipeline but add one extra engineered feature `log_distance_miles = log1p(distance_miles)` (and include it in the same linear input matrix) consistently for train/val/test, which typically improves linear fit without changing the approach. I also remove the premature `.round(2)` on predictions (rounding increases RMSE) while keeping the same non-negativity/finite guards. No changes to file paths or submission format; it still write `submission.csv`.'
- What this solution (achieved 15.00949) has done: 'Your current RMSE is far above the target (lower is better), so the smallest meaningful step is to keep your exact OLS model but fix a mismatch between what the model is trained on and how fares behave: the true relationship is closer to proportional to trip *distance*, not independently to `abs_diff_latitude` and `abs_diff_longitude`. Without changing the training approach, I add those two existing features into the linear input matrix (train/val/test consistently) so the model can correct distance linearization errors and reduce large residuals. I also clip predictions to a reasonable upper bound (based on your training filter `fare_amount <= 250`) to prevent occasional extreme predictions that massively inflate RMSE. All paths, reading logic, and the OLS fit/predict core remain the same, and the script still writes a valid `submission.csv`.'

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
data["datetime_object"] = pd.to_datetime(
    data["pickup_datetime"], utc=True, errors="coerce"
)
data["hour"] = data["datetime_object"].dt.hour.astype("float32")
data["year"] = data["datetime_object"].dt.year.astype("float32")




## === cell 4
def distance(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype="float64")
    lon1 = np.asarray(lon1, dtype="float64")
    lat2 = np.asarray(lat2, dtype="float64")
    lon2 = np.asarray(lon2, dtype="float64")

    dlat_miles = 69.0 * np.abs(lat2 - lat1)
    mean_lat_rad = np.deg2rad((lat1 + lat2) / 2.0)
    dlon_miles = 69.0 * np.cos(mean_lat_rad) * np.abs(lon2 - lon1)
    return dlat_miles + dlon_miles


data["distance_miles"] = distance(
    data.pickup_latitude,
    data.pickup_longitude,
    data.dropoff_latitude,
    data.dropoff_longitude,
)

data["log_distance_miles"] = np.log1p(data["distance_miles"].astype("float64"))

data.distance_miles.describe()

data.groupby("passenger_count")[["distance_miles", "fare_amount"]].mean()

print(
    "Average $USD/Mile : {:0.2f}".format(
        data.fare_amount.sum() / data.distance_miles.sum()
    )
)
data["fare_per_mile"] = data.fare_amount / data.distance_miles
data["inv_distance_miles"] = 1 / data.distance_miles



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

data = data[data.pickup_longitude.between(-75, -72)]
data = data[data.dropoff_longitude.between(-75, -72)]
data = data[data.pickup_latitude.between(40, 42)]
data = data[data.dropoff_latitude.between(40, 42)]

data = data[(data.abs_diff_longitude < 3.0) & (data.abs_diff_latitude < 3.0)]
data = data[data.fare_amount >= 0]
data = data[data.passenger_count <= 9]
data = data[(data.distance_miles > 0.0)]

data = data[data.fare_amount <= 250]

nyc = (-74.0063889, 40.7141667)
data["distance_to_center"] = distance(
    nyc[1], nyc[0], data.dropoff_latitude, data.dropoff_longitude
)
data = data[data.distance_to_center < 15.0]

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
    return np.column_stack(
        (
            df.distance_miles,
            df.log_distance_miles,
            df.abs_diff_longitude,
            df.abs_diff_latitude,
            df.passenger_count,
            df.hour,
            df.year,
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
w_OLS = np.matmul(
    np.matmul(np.linalg.inv(np.matmul(train_X.T, train_X)), train_X.T), train_y
)
print(w_OLS)



## === cell 13
test_df = pd.read_csv("../input/test.csv")

test_df["distance_miles"] = distance(
    test_df.pickup_latitude,
    test_df.pickup_longitude,
    test_df.dropoff_latitude,
    test_df.dropoff_longitude,
)
test_df["log_distance_miles"] = np.log1p(test_df["distance_miles"].astype("float64"))

test_df["datetime_object"] = pd.to_datetime(
    test_df["pickup_datetime"], utc=True, errors="coerce"
)
test_df["hour"] = test_df["datetime_object"].dt.hour.astype("float32")
test_df["year"] = test_df["datetime_object"].dt.year.astype("float32")

test_df.dtypes



## === cell 14
add_travel_vector_features(test_df)
test_X = get_input_matrix(test_df)

add_travel_vector_features(val_df)
val_X = get_input_matrix(val_df)

test_y_predictions = np.matmul(test_X, w)
val_y_predictions = np.matmul(val_X, w)

test_y_predictions = np.where(np.isfinite(test_y_predictions), test_y_predictions, 0.0)
val_y_predictions = np.where(np.isfinite(val_y_predictions), val_y_predictions, 0.0)

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
