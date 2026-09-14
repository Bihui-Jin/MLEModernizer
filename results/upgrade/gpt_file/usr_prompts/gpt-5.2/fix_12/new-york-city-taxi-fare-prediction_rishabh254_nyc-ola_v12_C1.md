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

5.51545

# 6. Current score

29.72353

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 786.85271) has done: 'I fix the pandas API breakages (tuple-based column selection in `groupby`, and `corr()` failing due to non-numeric columns) so the notebook runs on pandas 2.2. I also fix the datetime parsing (your format string doesn’t match the dataset) by using `pd.to_datetime`, which is both correct and much faster and ensures `hour`/`year` exist before splitting. Finally, I ensure the same feature engineering is applied consistently to train/val/test before building matrices, and that the script always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 28.87375) has done: 'Your RMSE is exploding because two features in your linear system (`fare_per_mile` and `inv_distance_miles`) create infinities/huge values when `distance_miles` is very small; those rows survive your `distance_miles > 0.0` filter and destabilize OLS, yielding extreme predictions. I make the smallest safe change: compute `fare_per_mile` and `inv_distance_miles` only after clipping `distance_miles` to a small epsilon, and tighten the filter to remove near-zero distances. I also clamp predictions to a reasonable non-negative range (still the same linear model; just prevents invalid negative fares) to reduce RMSE without changing the core approach. Finally, I keep your submission format identical and ensure it always writes `submission.csv`.'
- What this solution (achieved 28.91751) has done: 'Your current RMSE gap to the target is large, so the smallest legitimate improvement is to stabilize and de-bias the same OLS linear model without changing its core form. I (1) stop using the numerically unstable explicit matrix inverse (keep `lstsq` only), (2) add the same basic data cleaning commonly required for this competition (filter obvious out-of-bounds lat/lon and extreme fares) while keeping your existing feature set and training approach, and (3) ensure the exact same feature engineering is applied to train/val/test consistently before matrix building. These changes reduce the influence of bad rows and ill-conditioned fits, which should move RMSE substantially downward toward your target while preserving your model/loop/feature logic and producing the same submission format.'
- What this solution (achieved 28.95655) has done: 'Your RMSE is still very high mainly because the model is trained with `distance_miles` computed once, then later `val_df` recomputes `distance_miles` again (and `test_df` gets it only once) while other engineered columns are inconsistently present/absent; that mismatch plus remaining noisy/outlier rows makes the OLS coefficients poorly conditioned. I make the smallest change that preserves your linear least-squares core: compute *all* engineered features (distance + datetime parts + travel vectors) through one shared function and apply it consistently to train/val/test before building matrices. I also add one minimal NYC-taxi-specific cleanup that is standard for this competition and directly reduces RMSE without changing the model: filter out unrealistic speed outliers using pickup_datetime and distance (keeps the same features and OLS training). Finally, I keep your submission format identical and still write `submission.csv`.'
- What this solution (achieved 28.95655) has done: 'Your RMSE is far above the target, so we need a legitimate boost without changing your core OLS setup or feature set. The biggest remaining issue is that the fit is still being dominated by noisy/outlier rides; I add two minimal, standard NYC-taxi cleanups that directly reduce RMSE: (1) compute trip duration per-row (not from a global max) and filter implausible speeds, and (2) drop extreme `fare_per_mile` outliers that your own engineered feature exposes. I also ensure the model trains on a consistent, finite feature matrix by dropping any remaining NaNs/Infs after featurization and filtering, while keeping the same `lstsq` linear regression and submission format.'
- What this solution (achieved 28.95655) has done: 'Your failures come from the cleaning step filtering the 15M-row sample down to zero rows (so `.iloc[0]` and `train_test_split` crash), and then downstream variables never get defined. I keep your same OLS/feature logic, but add a minimal “safety fallback” that relaxes the strictest filters only if the cleaned dataframe becomes empty (or too small), so training always proceeds. I also guard the plotting/print cell so it doesn’t crash on empty data, and ensure `train_df/val_df` always exist before featurization and matrix building. Finally, I keep the submission format identical and always write `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 29.72352) has done: 'Your RMSE is far above the target, so we should improve (lower) it with the smallest legitimate changes that keep your same OLS/feature setup. The biggest current issue is that the model is being trained on raw longitude/latitude degrees with no NYC-specific distance scaling, so a simple linear model on `distance_miles` underfits badly; we can keep your exact linear least-squares approach but make `distance_miles` a more informative proxy by adding a standard NYC feature: distance to city center and (optionally) a “manhattan distance” proxy, then include them as extra linear columns. I also make the cleaning use `pickup_datetime` (not parsing timestamps from `key`) to compute trip duration and speed filtering, which removes label-noise/outliers without changing the model class. Finally, I keep the same submission format and ensure test-time NaN/Inf handling matches train/val so the matrices are consistent.'
- What this solution (achieved 29.72352) has done: 'Your current RMSE is still far above the target, so we should legitimately lower it with the smallest changes that keep your same OLS (np.linalg.lstsq) training and the same core feature set. The biggest avoidable error source is that you split *before* featurizing/cleaning, so `train_df`/`val_df` keep many rows that would have been removed by your own cleaning rules, and your model is fit on a different distribution than it’s evaluated on. I apply `featurize()` and the exact same `clean_data_*()` logic consistently to train and validation after splitting (without changing your model), and I also ensure the submission preserves the original test row order by predicting on a cleaned copy but writing keys/preds aligned back to the full test set (filling any dropped rows with a safe baseline). These are minimal semantic fixes that typically drop RMSE substantially while preserving the linear least-squares approach and producing a valid `submission.csv`.'
- What this solution (achieved 29.72353) has done: 'Your current RMSE (29.72) is far worse than the target (5.52), so we should legitimately *lower* it with very small, high-impact fixes while keeping your same linear least-squares model and training flow. The biggest avoidable error is that you currently include the `year` feature (mostly constant 2009–2015 in train) but test is all 2015, which causes a large systematic bias shift; I change that feature to `year - 2009` (same linear column, just centered) to stabilize extrapolation without changing the model class. I also fix a subtle leakage/bug: you clean `data` before splitting, then featurize+clean again post-split; instead, we keep the split but ensure the engineered datetime fields used in `get_input_matrix()` are always present/finite and consistently centered for train/val/test. Finally, I keep your submission alignment logic but remove prediction rounding (rounding hurts RMSE) while still clipping to a valid non-negative range.'
- What this solution (achieved 29.72353) has done: 'We need to lower RMSE from ~29.7 toward 5.52, so we keep your exact OLS (`np.linalg.lstsq`) and feature set, but remove a major source of instability: using 15M rows without any reweighting lets remaining outliers still dominate, and the current design matrix is poorly scaled (distance-related columns range far differently than hour/passenger/year), which makes the least-squares solution numerically ill-conditioned and hurts generalization. The smallest high-impact fix that preserves the same linear model is to standardize (z-score) the input columns using training statistics, then apply the same transform to val/test before `lstsq`/`matmul`; this does not change the model class, loss, or training approach, but typically improves fit stability a lot. Additionally, we ensure the exact same post-featurization cleaning is applied consistently before computing matrices, and we keep your submission alignment logic unchanged. These changes should legitimately reduce RMSE substantially while keeping the core logic intact and still producing `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

INPUT_DIR = "../input"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "/kaggle/input"

print(os.listdir(INPUT_DIR))



## === cell 1
data = pd.read_csv(f"{INPUT_DIR}/train.csv", nrows=15_000_000)




## === cell 2
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()




## === cell 3
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


def featurize(df):
    df = df.copy()

    add_travel_vector_features(df)

    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["datetime_object"] = dt.dt.tz_convert(None)  # tz-naive
    df["hour"] = df["datetime_object"].dt.hour
    df["year"] = df["datetime_object"].dt.year

    df["year_centered"] = df["year"] - 2009

    df["distance_miles"] = distance(
        df.pickup_latitude,
        df.pickup_longitude,
        df.dropoff_latitude,
        df.dropoff_longitude,
    )

    df["manhattan_miles"] = distance(
        df.pickup_latitude,
        df.pickup_longitude,
        df.pickup_latitude,
        df.dropoff_longitude,
    ) + distance(
        df.pickup_latitude,
        df.pickup_longitude,
        df.dropoff_latitude,
        df.pickup_longitude,
    )

    nyc_lat, nyc_lon = 40.7580, -73.9855
    df["pickup_to_center_miles"] = distance(
        df.pickup_latitude, df.pickup_longitude, nyc_lat, nyc_lon
    )
    df["dropoff_to_center_miles"] = distance(
        df.dropoff_latitude, df.dropoff_longitude, nyc_lat, nyc_lon
    )

    _eps = 1e-3  # miles (~1.6 meters)
    df["distance_miles_clipped"] = df["distance_miles"].clip(lower=_eps)
    if "fare_amount" in df.columns:
        df["fare_per_mile"] = df["fare_amount"] / df["distance_miles_clipped"]
    df["inv_distance_miles"] = 1.0 / df["distance_miles_clipped"]
    return df


data = featurize(data)

_ = data.groupby("passenger_count")[["distance_miles", "fare_amount"]].mean()

print(
    "Average $USD/Mile : {:0.2f}".format(
        data.fare_amount.sum() / data.distance_miles.sum()
    )
)



## === cell 4
import seaborn as sns
import matplotlib.pyplot as plt

numeric_data = data.select_dtypes(include=[np.number])
corrmat = numeric_data.corr()
f, ax = plt.subplots(figsize=(12, 9))

k = 14  # number of variables for heatmap
cols = corrmat.nlargest(k, "fare_amount")["fare_amount"].index
cm = np.corrcoef(numeric_data[cols].values.T)
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



## === cell 5
print("Old size: %d" % len(data))
data = data.dropna(how="any", axis="rows")
print("New size: %d" % len(data))



## === cell 6
print("Old size: %d" % len(data))


def clean_data_strict(df):
    df = df.copy()

    df = df[(df.abs_diff_longitude < 3.0) & (df.abs_diff_latitude < 3.0)]
    df = df[df.fare_amount > 0]
    df = df[df.passenger_count <= 9]
    df = df[(df.distance_miles > 1e-3)]

    df = df[(df.fare_amount <= 250.0)]
    df = df[(df.pickup_longitude >= -75) & (df.pickup_longitude <= -72)]
    df = df[(df.dropoff_longitude >= -75) & (df.dropoff_longitude <= -72)]
    df = df[(df.pickup_latitude >= 40) & (df.pickup_latitude <= 42)]
    df = df[(df.dropoff_latitude >= 40) & (df.dropoff_latitude <= 42)]

    df = df[(df.distance_miles <= 60.0)]

    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True).dt.tz_convert(
        None
    )
    df["datetime_object"] = dt
    df = df.dropna(subset=["datetime_object"])

    if "fare_per_mile" in df.columns:
        df = df[(df["fare_per_mile"] >= 0.5) & (df["fare_per_mile"] <= 40.0)]

    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    return df


def clean_data_fallback(df):
    df = df.copy()

    df = df[(df.abs_diff_longitude < 3.0) & (df.abs_diff_latitude < 3.0)]
    df = df[df.fare_amount > 0]
    df = df[df.passenger_count <= 9]
    df = df[(df.distance_miles > 1e-3)]

    df = df[(df.fare_amount <= 250.0)]
    df = df[(df.pickup_longitude >= -75) & (df.pickup_longitude <= -72)]
    df = df[(df.dropoff_longitude >= -75) & (df.dropoff_longitude <= -72)]
    df = df[(df.pickup_latitude >= 40) & (df.pickup_latitude <= 42)]
    df = df[(df.dropoff_latitude >= 40) & (df.dropoff_latitude <= 42)]
    df = df[(df.distance_miles <= 60.0)]

    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    return df


data_clean = clean_data_strict(data)
if len(data_clean) < 10_000:
    data_clean_fb = clean_data_fallback(data)
    if len(data_clean_fb) > len(data_clean):
        data_clean = data_clean_fb

data = data_clean
print("New size: %d" % len(data))



## === cell 7
if len(data) == 0:
    raise RuntimeError(
        "No training rows remain after cleaning. Increase nrows or relax cleaning."
    )

print(data["datetime_object"].iloc[0])
plot = data.iloc[:1000].plot.scatter("hour", "fare_amount")



## === cell 8
from sklearn.model_selection import train_test_split

y = data.fare_amount
X = data.drop("fare_amount", axis=1)

train_df, val_df, train_y, val_y = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_df.dtypes




## === cell 9
def get_input_matrix(df):
    return np.column_stack(
        (
            df.distance_miles,
            df.manhattan_miles,
            df.pickup_to_center_miles,
            df.dropoff_to_center_miles,
            df.passenger_count,
            df.hour,
            df.year_centered,
            np.ones(len(df)),
        )
    )


train_df = featurize(train_df.join(train_y.rename("fare_amount")))
val_df = featurize(val_df.join(val_y.rename("fare_amount")))

train_df_clean = clean_data_strict(train_df)
if len(train_df_clean) < 10_000:
    train_df_fb = clean_data_fallback(train_df)
    if len(train_df_fb) > len(train_df_clean):
        train_df_clean = train_df_fb

val_df_clean = clean_data_strict(val_df)
if len(val_df_clean) < 2_000:
    val_df_fb = clean_data_fallback(val_df)
    if len(val_df_fb) > len(val_df_clean):
        val_df_clean = val_df_fb

if len(train_df_clean) == 0 or len(val_df_clean) == 0:
    raise RuntimeError(
        "Train/val became empty after post-split cleaning; adjust nrows/filters."
    )

train_y = train_df_clean["fare_amount"].astype(float)
val_y = val_df_clean["fare_amount"].astype(float)

train_df = train_df_clean.drop(columns=["fare_amount"])
val_df = val_df_clean.drop(columns=["fare_amount"])

train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
train_y = train_y.loc[train_df.index]

val_mask = ~val_df.replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
val_df = val_df.loc[val_mask]
val_y = val_y.loc[val_mask]

train_X = get_input_matrix(train_df)
val_X = get_input_matrix(val_df)

print(train_X.shape)
print(train_y.shape)




## === cell 10
def standardize_matrices(train_X, val_X, test_X=None):
    feat_slice = slice(0, train_X.shape[1] - 1)
    mu = train_X[:, feat_slice].mean(axis=0)
    sigma = train_X[:, feat_slice].std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)

    def _apply(X):
        Xs = X.copy()
        Xs[:, feat_slice] = (Xs[:, feat_slice] - mu) / sigma
        return Xs

    out = (_apply(train_X), _apply(val_X), mu, sigma)
    if test_X is not None:
        out = out + (_apply(test_X),)
    return out


train_X_std, val_X_std, x_mu, x_sigma = standardize_matrices(train_X, val_X)



## === cell 11
(w, _, _, _) = np.linalg.lstsq(train_X_std, train_y, rcond=None)
print(w)



## === cell 12
w_OLS = w
print(w_OLS)



## === cell 13
test_df_full = pd.read_csv(f"{INPUT_DIR}/test.csv")
test_df_full_feat = featurize(test_df_full)

test_feat = test_df_full_feat.replace([np.inf, -np.inf], np.nan)
finite_mask = ~test_feat.isna().any(axis=1)

test_df = test_feat.loc[finite_mask].copy()
test_X = get_input_matrix(test_df)

test_X_std = test_X.copy()
test_X_std[:, :-1] = (test_X_std[:, :-1] - x_mu) / x_sigma

test_y_predictions_sub = np.matmul(test_X_std, w)

test_y_predictions_sub = np.clip(test_y_predictions_sub, 0.0, 500.0)

baseline = float(train_y.mean())
test_preds_full = np.full(shape=(len(test_df_full),), fill_value=baseline, dtype=float)
test_preds_full[finite_mask.values] = test_y_predictions_sub
test_preds_full = np.clip(test_preds_full, 0.0, 500.0)

val_y_predictions = np.matmul(val_X_std, w)
val_y_predictions = np.clip(val_y_predictions, 0.0, 500.0)

from sklearn.metrics import mean_squared_error

print(np.sqrt(mean_squared_error(val_y, val_y_predictions)))

submission = pd.DataFrame(
    {"key": test_df_full["key"], "fare_amount": test_preds_full},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)

print(os.listdir("."))
