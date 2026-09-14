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

3.12

# 3. Installed packages

geopandas==0.14.4
joblib==1.5.2
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

3.59144

# 6. Current score

5.71649

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.87588) has done: 'You don’t currently have a Kaggle score because your pipeline can silently produce an invalid submission: you filter rows out of `test_df`, which changes the required row count (9914) and drops keys. I keep your core feature engineering and XGBoost model intact, but remove all row-filtering on the test set (only clean NaNs and fill missing engineered features), so the submission always contains every test `key`. I also align the submission `key` order exactly to `sample_submission.csv` to avoid any potential ordering issues, and reindex predictions accordingly. These minimal fixes should turn your output into a valid submission and typically improve RMSE versus having missing rows/keys.'
- What this solution (achieved 9.95908) has done: 'Your current RMSE (10.87588) is far from the target (3.59144), so we need a modest-but-real improvement without changing your overall approach (same features + XGBoost regressor). The biggest score drag here is that you train on a random 5M rows without fixing the extreme outliers/noisy rows that NYC Taxi is notorious for (especially invalid coordinates and unrealistic ride distances that still pass your coarse NY bounding box). I keep your exact feature engineering and training flow, but add two minimal, standard “sanity filters” on the training set only: (1) tighten coordinate validity and (2) remove implausible ride distances; this typically drops RMSE substantially while preserving the same model and semantics. I also train the final XGBoost on the full cleaned dataset (not just the train split) before predicting test, which is a standard submission-time step and usually improves leaderboard RMSE with no change to core logic.'
- What this solution (achieved 6.91074) has done: 'You’re still far from the target RMSE (9.96 vs 3.59, lower is better), so the most “minimal but real” improvement is to fix label noise/outliers in the *training* set without changing your feature set or XGBoost setup. I keep the exact same feature engineering and model hyperparameters, but add a couple of standard NYC-Taxi sanity filters on training only (remove negative/zero fares, cap passenger_count, and remove obvious “teleport”/very-long rides with tiny fare by enforcing a loose fare-per-km band). I also add one missing strong baseline feature that doesn’t change the approach (same engineered-features pattern): absolute latitude/longitude deltas, which usually reduces RMSE materially with tree models. Finally, I keep your submission alignment to `sample_submission.csv` and keep test rows unfiltered to guarantee a valid submission.'
- What this solution (achieved 6.72842) has done: 'Your RMSE (6.91074) is still far from the target (3.59144, lower is better), so we need a real improvement while keeping your same feature set and XGBoost approach. The biggest remaining drag is training-set noise: your current filters still allow many mislabeled/odd rows (especially fares that don’t match distance) and also ignore obvious time-based signal. I keep your exact model/training flow, but add two minimal, standard NYC-taxi cleanups on the training set only (remove extreme fare-per-km outliers more tightly, and remove very-short “non-trip” rides), and add two very lightweight datetime features (minute-of-day and weekend) that match your existing datetime refactor style. I not filter test rows; submission remain aligned to `sample_submission.csv` and always have 9914 rows.'
- What this solution (achieved 6.9496) has done: 'Your current RMSE (6.72842) is still far above the target (3.59144), so we should make a small, legitimate improvement by reducing remaining training-label noise without changing your model/feature approach. I keep the exact same feature engineering and XGBRegressor setup, but tighten the training-only outlier filter to remove clearly mislabeled rides by using a slightly narrower (but still loose) fare-per-km band and also excluding extremely slow “non-trip” cases (very short distance with non-trivial fare). This typically improves RMSE materially on NYC Taxi while preserving identical semantics and leaving the test set unfiltered (so the submission keeps all 9914 keys). I also keep your sample_submission alignment and NaN-filling exactly as you already do to ensure a valid CSV.'
- What this solution (achieved 7.27858) has done: 'We need to move your RMSE down from 6.9496 toward 3.59144 (lower is better), so the smallest legitimate gain is to reduce training-label noise/outliers further without changing your core feature set or XGBoost approach. I keep the same feature engineering and model/training flow, but tighten the training-only cleanup in a targeted way: remove extreme “fare-per-km” outliers using robust quantile trimming and enforce a loose but realistic minimum fare floor for non-trivial trips; both are standard for this dataset and usually improve RMSE substantially. I also ensure the datetime parsing is consistent (UTC-naive) and keep the test set unfiltered while preserving your submission alignment to `sample_submission.csv` to guarantee a valid 9914-row submission. No architecture, loss, or loop changes—only training-data filtering and safe numeric handling that should move score toward the target.'
- What this solution (achieved 6.93581) has done: 'We need to move RMSE down from 7.27858 toward 3.59144 (lower is better), so we should remove a likely source of score regression: your recent quantile-based trimming is currently applied on `fare_per_km` *after* several hard filters and can easily over-prune valid but informative rides, worsening generalization. I keep your exact feature engineering and XGBRegressor setup, but replace the quantile trimming with a fixed, standard NYC-Taxi “fare-per-km” band computed safely (no inf/NaN) and add one minimal, common cleanup: drop rows with `ride_distance` extremely close to 0 while keeping the rest of your filters intact. This is a training-only change (test remains unfiltered), and the submission alignment to `sample_submission.csv` stays exactly as you already do to guarantee a valid 9914-row CSV. These minimal changes are targeted to reduce label-noise/outlier impact without changing the model architecture or training loop.'
- What this solution (achieved 6.79507) has done: 'You’re currently far above the target RMSE (6.94 vs 3.59, lower is better), so we should make a small, legitimate improvement by removing remaining label noise/outliers in the training set without changing your model, features, or training loop. The biggest safe gain here is to add a standard NYC Taxi cleanup: drop rows where pickup/dropoff are identical or nearly identical (GPS glitches) and remove obvious “wrong borough/airport teleport” cases using a generous bounding box plus a max-speed style constraint via distance vs time-of-day isn’t possible (no trip duration), so instead we use a tighter but still safe distance-to-NYC-center cap. These are training-only filters; the test set remains unfiltered to keep all 9914 keys and a valid submission. Everything else (feature engineering, XGBRegressor hyperparameters, fitting, and submission alignment) is preserved.'
- What this solution (achieved 5.62056) has done: 'We need to move RMSE down from 6.79507 toward 3.59144 (lower is better), and we’re still far outside the ±10% target band, so a small-but-real improvement is warranted without changing your model or feature set. The most likely remaining score drag is training label noise from rare but severe outliers that still pass the current rules; we add one additional *training-only* robust filter using log-space trimming on `fare_amount` and `ride_distance` to remove extreme tails while keeping most data. We also add a simple, standard “Manhattan distance” feature (L1 in lat/lon space) that matches your existing engineered-feature pattern and works well with tree models, without changing the learning approach. Everything else (XGBRegressor, parameters, training flow, test handling, and submission alignment) stays the same, and the script still writes a valid 9914-row `submission.csv`.'
- What this solution (achieved 5.69706) has done: 'Your current RMSE (5.62056) is still well above the target (3.59144), so we should make a small training-only improvement that reduces label noise without changing your model or feature set. The biggest remaining issue is that the NYC Taxi training set contains many “bad label” rows even after geographic filtering; tightening the fare-vs-distance consistency filter (using a more realistic but still generous $/km band and a small distance floor) typically yields a meaningful RMSE drop. I keep your exact feature engineering and XGBRegressor hyperparameters/training flow intact, only adjusting the training filters and making the fare_per_km computation more robust (avoid NaN/inf edge cases). Test handling and submission alignment to `sample_submission.csv` stay unchanged to guarantee a valid 9914-row `submission.csv`.'
- What this solution (achieved 5.61516) has done: 'Your current RMSE (5.69706) is still well above the target (3.59144), so we should make a small, training-only change that reduces label noise while keeping the same features and XGBRegressor training flow. The biggest low-risk gain at this stage is to (1) ensure `fare_per_km` is computed robustly with a slightly larger distance floor to avoid near-zero distance instability, and (2) tighten the fare-per-km band moderately (still generous) while keeping your existing geographic and log-tail trimming intact. These are minimal edits confined to the existing cleaning block; the model architecture, features list, and submission alignment remain unchanged. The test set stays unfiltered and we still write a valid 9914-row `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 5.71649) has done: 'We need to move RMSE down from 5.61516 toward 3.59144 (lower is better), so we focus on the largest remaining driver: training-label noise/outliers that still survive your current filters. I keep your feature engineering and XGBRegressor setup identical, but add one minimal, standard training-only cleanup: drop extreme outliers based on the model’s own residuals on a small held-out slice (this reduces mislabeled/garbage rows without changing the modeling approach). Then we refit the same XGBRegressor on the cleaned full training data and keep your existing test handling (no row filtering) and submission alignment to `sample_submission.csv` so the output stays valid (9914 rows). This should typically reduce RMSE meaningfully while keeping changes localized and within the same core logic.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=5_000_000
)
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
df.dtypes



## === cell 1
df.describe()



## === cell 2
df.head()




## === cell 3
def filter_column(df, column, range_min, range_max):
    return df[(df[column] >= range_min) & (df[column] <= range_max)]


ny_latitude_min, ny_latitude_max = 40.4772, 45.0153
ny_longitude_min, ny_longitude_max = -79.7624, -71.7517

df = df.dropna()  # remove nulls from training

df = filter_column(df, "pickup_longitude", ny_longitude_min, ny_longitude_max)
df = filter_column(df, "pickup_latitude", ny_latitude_min, ny_latitude_max)

df = filter_column(df, "dropoff_longitude", ny_longitude_min, ny_longitude_max)
df = filter_column(df, "dropoff_latitude", ny_latitude_min, ny_latitude_max)

df = filter_column(df, "passenger_count", 1, 6)

test_df = test_df.copy()



## === cell 4
df[df["fare_amount"] > 200].describe()



## === cell 5
df = filter_column(df, "fare_amount", 1, 200)




## === cell 6
def refactor_datetime(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True).dt.tz_convert(
        None
    )
    df["pickup_datetime"] = dt

    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["hour"] = df["pickup_datetime"].dt.hour

    df["minute"] = df["pickup_datetime"].dt.minute
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]
    df["is_weekend"] = (df["weekday"] >= 5).astype(np.int8)

    df.drop(columns=["pickup_datetime"], inplace=True)


refactor_datetime(df)
refactor_datetime(test_df)
df.head()




## === cell 7
def haversine_vec(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    dist = 2 * np.arcsin(
        np.sqrt(
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
    )
    km = 6367 * dist
    return km


ny_center = ("ny_center", (40.7128, -74.0060))
jfk_airport = ("jfk_airport", (40.6446, -73.7797))
lga_airport = ("lga_airport", (40.7733, -73.8718))
ewr_airport = ("ewr_airport", (40.6895, -74.1745))

locs = [ny_center, jfk_airport, lga_airport, ewr_airport]




## === cell 8
def insert_haversine_dists(df, locations):
    for name, (latc, lonc) in locations:
        df["pickup_dist_to_" + name] = haversine_vec(
            df["pickup_latitude"].values, df["pickup_longitude"].values, latc, lonc
        )
        df["dropoff_dist_to_" + name] = haversine_vec(
            df["dropoff_latitude"].values, df["dropoff_longitude"].values, latc, lonc
        )
    df["ride_distance"] = haversine_vec(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
    )

    df["abs_lon_diff"] = np.abs(
        df["dropoff_longitude"].values - df["pickup_longitude"].values
    )
    df["abs_lat_diff"] = np.abs(
        df["dropoff_latitude"].values - df["pickup_latitude"].values
    )

    df["manhattan_approx"] = df["abs_lon_diff"] + df["abs_lat_diff"]


insert_haversine_dists(df, locs)
insert_haversine_dists(test_df, locs)



## === cell 9
df.describe()



## === cell 10
df = df[df["ride_distance"] > 0]

df = df[
    (df["pickup_longitude"].between(-75, -72))
    & (df["dropoff_longitude"].between(-75, -72))
    & (df["pickup_latitude"].between(40, 42))
    & (df["dropoff_latitude"].between(40, 42))
]

df = df[df["ride_distance"] <= 100]

df = df[df["fare_amount"] > 0]
df = df[df["fare_amount"] <= 200]
df = df[df["passenger_count"].between(1, 6)]

ride_km = df["ride_distance"].astype(float).values
fare = df["fare_amount"].astype(float).values
min_km_for_ratio = 0.30
fare_per_km = fare / np.maximum(ride_km, min_km_for_ratio)
fare_per_km = pd.Series(fare_per_km, index=df.index).replace([np.inf, -np.inf], np.nan)
df = df[fare_per_km.notna()]

df = df[(fare_per_km >= 2.0) & (fare_per_km <= 20.0)]

df = df[df["ride_distance"] >= 0.10]
df = df[~((df["ride_distance"] < 0.2) & (df["fare_amount"] > 30.0))]

df = df[~((df["ride_distance"] >= 1.0) & (df["fare_amount"] < 3.0))]

same_loc = (df["abs_lon_diff"] < 1e-4) & (df["abs_lat_diff"] < 1e-4)
df = df[~same_loc]

df = df[
    (df["pickup_dist_to_ny_center"] <= 60.0) & (df["dropoff_dist_to_ny_center"] <= 60.0)
]

df = df[df["fare_amount"].notna() & df["ride_distance"].notna()]
log_fare = np.log1p(df["fare_amount"].astype(float))
log_dist = np.log1p(df["ride_distance"].astype(float))

fare_lo, fare_hi = log_fare.quantile([0.001, 0.999])
dist_lo, dist_hi = log_dist.quantile([0.001, 0.999])

df = df[(log_fare >= fare_lo) & (log_fare <= fare_hi)]
df = df[(log_dist >= dist_lo) & (log_dist <= dist_hi)]

df.describe()



## === cell 11
from sklearn.model_selection import train_test_split

train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)



## === cell 12
features = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "minute_of_day",
    "is_weekend",
    "ride_distance",
    "abs_lon_diff",
    "abs_lat_diff",
    "manhattan_approx",
]
features += ["pickup_dist_to_" + x[0] for x in locs]
features += ["dropoff_dist_to_" + x[0] for x in locs]
fare_amount = "fare_amount"

train_features = train_df[features]
train_fare_amount = train_df[fare_amount]

validation_features = validation_df[features]
validation_fare_amount = validation_df[fare_amount]
train_features.info()



## === cell 13
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()



## === cell 14
from sklearn.model_selection import cross_val_score


def estimate_model(model, df):
    X = df[features]
    y = df[fare_amount]
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-cv_scores)
    print("RMSE scores for each fold:", rmse_scores)
    print("Mean RMSE:", rmse_scores.mean())
    print("Standard Deviation of RMSE:", rmse_scores.std())




## === cell 15
estimate_model(linear_model, train_df)



## === cell 16
linear_model.fit(train_features, train_fare_amount)



## === cell 17
from sklearn.metrics import mean_squared_error

linear_predictions = linear_model.predict(validation_features)
mean_squared_error(validation_fare_amount, linear_predictions, squared=False)



## === cell 18
RUN_EXPENSIVE_CV = False

if RUN_EXPENSIVE_CV:
    from xgboost import XGBRegressor
    from sklearn.model_selection import KFold
    import matplotlib.pyplot as plt
    from joblib import Parallel, delayed
    import seaborn as sns

    learning_rates = [0.1, 0.15, 0.2]
    n_estimators = [80, 100, 150]

    sample_fraction = 0.1
    train_sample = df.sample(frac=sample_fraction, random_state=42)
    X = train_sample[features]
    y = train_sample[fare_amount]

    def cross_val_rmse(lr, ne, X, y):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_rmse = []

        for train_index, val_index in kf.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            model = XGBRegressor(
                objective="reg:squarederror",
                learning_rate=lr,
                n_estimators=ne,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

            predictions = model.predict(X_val)
            rmse = mean_squared_error(y_val, predictions, squared=False)
            fold_rmse.append(rmse)

        avg_rmse = np.mean(fold_rmse)
        return lr, ne, avg_rmse

    results = Parallel(n_jobs=-1)(
        delayed(cross_val_rmse)(lr, ne, X, y)
        for lr in learning_rates
        for ne in n_estimators
    )

    results_df = pd.DataFrame(
        results, columns=["learning_rate", "n_estimators", "rmse"]
    )
    results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=results_df, x="n_estimators", y="rmse", hue="learning_rate", marker="o"
    )
    plt.title("RMSE for Different Learning Rates and n_estimators")
    plt.xlabel("Number of Estimators")
    plt.ylabel("RMSE")
    plt.legend(title="Learning Rate")
    plt.show()



## === cell 19
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.15,
    n_estimators=150,
    max_depth=6,
    min_child_weight=10,
    n_jobs=-1,
    random_state=42,
)
xgb_model.fit(train_features, train_fare_amount)
xgb_predictions = xgb_model.predict(validation_features)
mean_squared_error(validation_fare_amount, xgb_predictions, squared=False)



## === cell 20
xgb_predictions = xgb_model.predict(train_features)
mean_squared_error(train_fare_amount, xgb_predictions, squared=False)



## === cell 21
from sklearn.model_selection import train_test_split

full_X = df[features]
full_y = df[fare_amount]

resid_train_idx, resid_val_idx = train_test_split(
    df.index, test_size=0.10, random_state=42
)

X_resid_train = full_X.loc[resid_train_idx]
y_resid_train = full_y.loc[resid_train_idx]
X_resid_val = full_X.loc[resid_val_idx]
y_resid_val = full_y.loc[resid_val_idx]

_tmp_model = XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.15,
    n_estimators=150,
    max_depth=6,
    min_child_weight=10,
    n_jobs=-1,
    random_state=42,
)
_tmp_model.fit(X_resid_train, y_resid_train)
val_pred = _tmp_model.predict(X_resid_val)
abs_resid = np.abs(y_resid_val.values - val_pred)

resid_cut = np.quantile(abs_resid, 0.99)
keep_val = abs_resid <= resid_cut
keep_val_idx = y_resid_val.index[keep_val]

clean_idx = resid_train_idx.union(keep_val_idx)
df_clean = df.loc[clean_idx].copy()

print("Original df shape:", df.shape)
print("Cleaned df shape:", df_clean.shape)

full_X_clean = df_clean[features]
full_y_clean = df_clean[fare_amount]
xgb_model.fit(full_X_clean, full_y_clean)

sample_sub = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)

test_feat = test_df.set_index("key")

missing_cols = [c for c in features if c not in test_feat.columns]
if missing_cols:
    raise ValueError(f"Missing required feature columns in test set: {missing_cols}")

test_feat = test_feat[features]

train_medians = full_X_clean.median(numeric_only=True)
test_feat = test_feat.apply(pd.to_numeric, errors="coerce")
test_feat = test_feat.fillna(train_medians)

test_feat = test_feat.reindex(sample_sub["key"])
test_feat = test_feat.fillna(train_medians)

pred = xgb_model.predict(test_feat)

pred = np.clip(pred, 0, None)

submission = pd.DataFrame({"key": sample_sub["key"], "fare_amount": pred})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
