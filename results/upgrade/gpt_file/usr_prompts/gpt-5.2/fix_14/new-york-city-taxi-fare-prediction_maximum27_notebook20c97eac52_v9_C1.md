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

3.12

# 3. Installed packages

cudf-polars-cu12==25.6.0
geopandas==0.14.4
lightgbm==4.6.0
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
polars==1.25.0
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

3.84929

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.80252) has done: 'Your current score (5.8506 RMSE) is worse than the target (3.84929), so we should improve accuracy with minimal, low-risk fixes. The biggest issue is that your train/test preprocessing creates mismatched one-hot columns (you dummify separately), and in cell 6 you also predict on a Polars DataFrame with the wrong feature set—both can silently degrade performance. I (1) make datetime parsing robust to the actual NYC Taxi format, (2) ensure train and test have identical feature columns after one-hot encoding, and (3) predict using the exact same pandas feature matrix columns used for training. These changes keep your model/feature idea intact while removing avoidable train/test schema bugs that commonly cause large RMSE.'
- What this solution (achieved 5.79027) has done: 'Your current RMSE (5.80252) is worse than the target (3.84929), so we should improve accuracy with very small, low-risk changes that keep your model and features the same. The biggest remaining accuracy drag is that you train with a random split while the underlying data is time-ordered; this tends to give a validation that looks OK but a worse Kaggle score due to distribution shift. I keep the same LightGBM regressor and the same engineered features, but (1) switch the split to be time-based using the parsed `pickup_datetime` before it gets dropped, and (2) add a tiny amount of label cleaning (remove non-positive/unrealistic fares) which is standard for this competition and doesn’t change the core approach. This should move RMSE down toward your target without altering the model architecture/loop or feature set.'
- What this solution (achieved 5.76594) has done: 'Your current RMSE (5.79027) is worse than the target (3.84929), so we should improve accuracy with minimal, low-risk fixes that keep your LightGBM model/training loop and engineered features intact. The largest remaining accuracy drag is the “distance” feature: it currently uses linear degree-to-km scaling on absolute deltas, which is much noisier than a simple Haversine distance while still staying within your existing “distance feature” core logic. I replace only the internal computation of `distance` with a Haversine formula (keeping the same column name), add a small, standard upper bound filter for unrealistic fares, and clip negative predictions to 0 to reduce RMSE from pathological outputs. All I/O paths and submission schema remain unchanged, and the script still writes a valid `.csv`.'
- What this solution (achieved 5.70068) has done: 'I fix the LightGBM training crash by ensuring no non-numeric columns (specifically the original `pickup_datetime` string) are passed into the model; this is the root cause of the current runtime error. I keep your feature pipeline and model/training setup the same, but explicitly drop `pickup_datetime` (and any other non-numeric columns if they sneak in) right before building LightGBM datasets. I also make the train/val/test column alignment robust after dropping these columns so inference uses exactly the same feature set as training. With training unblocked, the script run end-to-end and write a valid `submission_*.csv` with `key,fare_amount`.'
- What this solution (achieved 5.34899) has done: 'Your RMSE (5.70068) is still far from the target (3.84929), so we should improve accuracy with small, standard fixes that don’t change your model choice or overall training approach. The biggest remaining issue is that your features are built from raw lat/lon without removing obvious bad rows (e.g., zero-distance rides, extreme coordinates, and extreme fare/distance ratios), which adds heavy noise that LightGBM can’t fully learn away. I tighten the existing outlier filtering in your `drop_outliner` step using common NYC Taxi bounds (still the same “filter outliers” core logic), and I add one more simple, legitimate filter based on `distance` to remove implausible trips. These changes should move RMSE down toward your target while keeping the same feature engineering pipeline, LightGBM training call, and submission format.'
- What this solution (achieved 5.6684) has done: 'Your current RMSE (5.34899) is still worse than the target (3.84929), so we should improve accuracy with minimal, standard fixes that keep your LightGBM setup and engineered features intact. The biggest remaining gap is that the model is trying to learn the full fare without any explicit “NYC taxi fare structure” feature, even though you already compute `calculated_value` in `rule_base`. I keep the same feature pipeline and model/training call, but (1) add one additional feature `fare_estimate` derived from a simple, legitimate formula using `distance`, `passenger_count`, and a time-based surcharge, and (2) keep your current outlier filters while adding a very small ratio-based filter (fare per km) to remove impossible label noise. These are small changes that typically reduce RMSE substantially on this competition without changing the core approach or I/O.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import polars as pl
import numpy as np

TRAIN_ZIP = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv.zip"
TEST_CSV = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SAMPLE_N = 10_000_000
SEED = 0


def sample_train_from_zip(
    zip_path: str, n: int, seed: int = 0, batch_size: int = 1_000_000
) -> pl.DataFrame:
    """
    Fix: Polars batched CSV can error on occasional ragged lines in this dataset.
    Setting truncate_ragged_lines=True avoids 'found more fields than defined in Schema'.
    """
    rng = np.random.default_rng(seed)
    parts = []
    remaining = int(n)

    reader = pl.read_csv_batched(
        zip_path,
        batch_size=batch_size,
        truncate_ragged_lines=True,
        ignore_errors=True,
    )

    while remaining > 0:
        batches = reader.next_batches(1)  # returns list[DataFrame] or None
        if batches is None or len(batches) == 0:
            break
        dfb = batches[0]

        if dfb is None or dfb.height == 0:
            continue

        dfb = dfb.drop_nulls()
        if dfb.height == 0:
            continue

        keep_prob = min(1.0, max(0.0, remaining / max(1, dfb.height)))
        mask = rng.random(dfb.height) < keep_prob
        picked = dfb.filter(pl.Series(mask))

        if picked.height > remaining:
            idx = rng.choice(picked.height, size=remaining, replace=False)
            picked = picked[idx]

        if picked.height > 0:
            parts.append(picked)
            remaining -= picked.height

    if len(parts) == 0:
        raise RuntimeError(
            "Sampling produced 0 rows; check input path and file integrity."
        )

    out = pl.concat(parts, how="vertical_relaxed")
    if out.height < n:
        print(f"Warning: sampled {out.height} rows < requested {n}. Continuing.")
    return out


train_df = sample_train_from_zip(TRAIN_ZIP, SAMPLE_N, seed=SEED, batch_size=1_000_000)
test_df = pl.read_csv(TEST_CSV)

print("train_df shape:", train_df.shape)
print("test_df shape:", test_df.shape)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ComputeError                              Traceback (most recent call last)
/tmp/ipykernel_11/971350942.py in <cell line: 0>()
     62 
     63 
---> 64 train_df = sample_train_from_zip(TRAIN_ZIP, SAMPLE_N, seed=SEED, batch_size=1_000_000)
     65 test_df = pl.read_csv(TEST_CSV)
     66 

/tmp/ipykernel_11/971350942.py in sample_train_from_zip(zip_path, n, seed, batch_size)
     27 
     28     while remaining > 0:
---> 29         batches = reader.next_batches(1)  # returns list[DataFrame] or None
     30         if batches is None or len(batches) == 0:
     31             break

/usr/local/lib/python3.11/dist-packages/polars/io/csv/batched_reader.py in next_batches(self, n)
    133         list of DataFrames
    134         """
--> 135         if (batches := self._reader.next_batches(n)) is not None:
    136             if self.new_columns:
    137                 return [

ComputeError: could not parse `"��Ʌ�b��)*���?��4�C�Z���C���"�H� ��` as dtype `str` at column 'PK-    h�Ov$a ��������	 0 train.csvUT	 ���]���]ux |+  )    �' (column number 1)

The current offset in the file is 375318 bytes.

You might want to try:
- increasing `infer_schema_length` (e.g. `infer_schema_length=10000`),
- specifying correct dtype with the `schema_overrides` argument
- setting `ignore_errors` to `True`,
- adding `"��Ʌ�b��)*���?��4�C�Z���C���"�H� ��` to the `null_values` list.

Original error: ```invalid utf-8 sequence of 1 bytes from index 1```

## === cell 2
train = train_df
test = test_df



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1029062965.py in <cell line: 0>()
----> 1 train = train_df
      2 test = test_df
      3 

NameError: name 'train_df' is not defined

## === cell 3
import numpy as np
import polars as pl


def preprocess(df: pl.DataFrame) -> pl.DataFrame:
    cleaned = pl.col("pickup_datetime").str.replace(r"\sUTC$", "")
    parsed = cleaned.str.strptime(
        pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False
    )

    df = df.with_columns(parsed.alias("pickup_datetime_parsed"))

    df = df.with_columns(
        [
            pl.col("pickup_datetime_parsed").dt.year().alias("pickup_year"),
            pl.col("pickup_datetime_parsed").dt.month().alias("pickup_month"),
            pl.col("pickup_datetime_parsed").dt.day().alias("pickup_day"),
            pl.col("pickup_datetime_parsed").dt.hour().alias("pickup_hour"),
            pl.col("pickup_datetime_parsed").dt.minute().alias("pickup_minute"),
            pl.col("pickup_datetime_parsed").dt.second().alias("pickup_second"),
            pl.col("pickup_datetime_parsed").dt.weekday().alias("pickup_weekday"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("pickup_longitude") - pl.col("dropoff_longitude"))
            .abs()
            .alias("abs_longitude"),
            (pl.col("pickup_latitude") - pl.col("dropoff_latitude"))
            .abs()
            .alias("abs_latitude"),
        ]
    )

    return df


def distance(df: pl.DataFrame) -> pl.DataFrame:
    r_km = 6371.0
    df = df.with_columns(
        [
            (pl.col("pickup_latitude") * np.pi / 180.0).alias("_plat"),
            (pl.col("dropoff_latitude") * np.pi / 180.0).alias("_dlat"),
            (pl.col("pickup_longitude") * np.pi / 180.0).alias("_plon"),
            (pl.col("dropoff_longitude") * np.pi / 180.0).alias("_dlon"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("_dlat") - pl.col("_plat")).alias("_dphi"),
            (pl.col("_dlon") - pl.col("_plon")).alias("_dlambda"),
        ]
    )
    df = df.with_columns(
        [
            (
                (pl.col("_dphi") / 2.0).sin() ** 2
                + (
                    pl.col("_plat").cos()
                    * pl.col("_dlat").cos()
                    * (pl.col("_dlambda") / 2.0).sin() ** 2
                )
            ).alias("_a")
        ]
    )
    df = df.with_columns(
        [
            (2.0 * pl.col("_a").sqrt().arcsin()).alias("_c"),
            (pl.lit(r_km) * (2.0 * pl.col("_a").sqrt().arcsin())).alias("distance"),
        ]
    )
    df = df.drop(["_plat", "_dlat", "_plon", "_dlon", "_dphi", "_dlambda", "_a", "_c"])
    return df


def drop_encoding(df: pl.DataFrame) -> pl.DataFrame:
    drop_columns = ["key", "pickup_datetime", "pickup_minute", "pickup_second"]
    df = df.drop(drop_columns)
    return df


def cycling_encoding(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        [
            (pl.col("pickup_month") * 2 * np.pi / 12).sin().alias("pickup_month_sin"),
            (pl.col("pickup_month") * 2 * np.pi / 12).cos().alias("pickup_month_cos"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).sin().alias("pickup_hour_sin"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).cos().alias("pickup_hour_cos"),
        ]
    )
    return df


def one_hot_encoding(df: pl.DataFrame) -> pl.DataFrame:
    one_hot_cols = ["pickup_year", "pickup_day"]
    df = df.to_dummies(one_hot_cols)
    return df


def is_central(df: pl.DataFrame) -> pl.DataFrame:
    left_latitude = 40.737676
    right_latitude = 40.791716
    left_longitude = -74.015781
    right_longitude = -73.945521
    df = df.with_columns(
        (
            (pl.col("pickup_latitude") >= left_latitude)
            & (pl.col("pickup_latitude") <= right_latitude)
            & (pl.col("pickup_longitude") >= left_longitude)
            & (pl.col("pickup_longitude") <= right_longitude)
        )
        .cast(pl.Int8)
        .alias("is_pickup_central")
    )
    df = df.with_columns(
        (
            (pl.col("dropoff_latitude") >= left_latitude)
            & (pl.col("dropoff_latitude") <= right_latitude)
            & (pl.col("dropoff_longitude") >= left_longitude)
            & (pl.col("dropoff_longitude") <= right_longitude)
        )
        .cast(pl.Int8)
        .alias("is_dropoff_central")
    )
    return df


def diff_central(df: pl.DataFrame) -> pl.DataFrame:
    base_longitude = 85.393
    base_latitude = 111.034
    central_latitude = 40.764696
    central_longitude = -73.98065

    df = df.with_columns(
        [
            (pl.col("pickup_longitude") - central_longitude)
            .abs()
            .alias("central_pickup_abs_longitude"),
            (pl.col("pickup_latitude") - central_latitude)
            .abs()
            .alias("central_pickup_abs_latitude"),
        ]
    )
    df = df.with_columns(
        [
            (
                pl.col("central_pickup_abs_longitude") * base_longitude
                + pl.col("central_pickup_abs_latitude") * base_latitude
            ).alias("central_pickup_distance")
        ]
    )
    df = df.drop(["central_pickup_abs_longitude", "central_pickup_abs_latitude"])

    df = df.with_columns(
        [
            (pl.col("dropoff_longitude") - central_longitude)
            .abs()
            .alias("central_dropoff_abs_longitude"),
            (pl.col("dropoff_latitude") - central_latitude)
            .abs()
            .alias("central_dropoff_abs_latitude"),
        ]
    )
    df = df.with_columns(
        [
            (
                pl.col("central_dropoff_abs_longitude") * base_longitude
                + pl.col("central_dropoff_abs_latitude") * base_latitude
            ).alias("central_dropoff_distance")
        ]
    )
    df = df.drop(["central_dropoff_abs_longitude", "central_dropoff_abs_latitude"])

    return df


def is_short_distance(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col("distance") < 0.32).cast(pl.Int8).alias("is_short_distance")
    )
    return df


def drop_outliner(df: pl.DataFrame) -> pl.DataFrame:
    filtered = df

    filtered = filtered.filter(
        (pl.col("pickup_latitude") >= 40.5) & (pl.col("pickup_latitude") <= 41.0)
    )
    filtered = filtered.filter(
        (pl.col("pickup_longitude") >= -74.3) & (pl.col("pickup_longitude") <= -73.7)
    )
    filtered = filtered.filter(
        (pl.col("dropoff_latitude") >= 40.5) & (pl.col("dropoff_latitude") <= 41.0)
    )
    filtered = filtered.filter(
        (pl.col("dropoff_longitude") >= -74.3) & (pl.col("dropoff_longitude") <= -73.7)
    )

    filtered = filtered.filter(
        (pl.col("passenger_count") >= 1) & (pl.col("passenger_count") <= 6)
    )

    filtered = filtered.filter(
        (pl.col("fare_amount") > 0) & (pl.col("fare_amount") < 500)
    )

    filtered = filtered.filter(
        ~(
            (pl.col("pickup_latitude") == pl.col("dropoff_latitude"))
            & (pl.col("pickup_longitude") == pl.col("dropoff_longitude"))
        )
    )

    if "distance" in filtered.columns:
        filtered = filtered.filter(
            (pl.col("distance") > 0.05) & (pl.col("distance") < 80.0)
        )

    if "distance" in filtered.columns and "fare_amount" in filtered.columns:
        filtered = filtered.filter(
            (pl.col("fare_amount") / pl.col("distance") > 0.5)
            & (pl.col("fare_amount") / pl.col("distance") < 100.0)
        )

    return filtered


def rule_base(df: pl.DataFrame) -> pl.DataFrame:
    is_weekday = pl.col("pickup_weekday").is_in([0, 1, 2, 3, 4])
    is_weekend = pl.col("pickup_weekday").is_in([5, 6])

    df = df.with_columns(
        pl.when(is_weekday & pl.col("pickup_hour").is_in([16, 17, 18, 19, 20]))
        .then(9 * pl.col("distance") / 0.32)
        .when(
            is_weekday
            & pl.col("pickup_hour").is_in([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6])
        )
        .then(pl.col("distance") / 0.32)
        .when(is_weekday)
        .then(0.5 * pl.col("distance") / 0.32)
        .when(
            is_weekend
            & pl.col("pickup_hour").is_in([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6])
        )
        .then(60 * pl.col("distance") / 19.2)
        .otherwise(30 * pl.col("distance") / 19.2)
        .alias("calculated_value")
    )
    return df


def fare_estimate_feature(df: pl.DataFrame) -> pl.DataFrame:
    is_night = (
        pl.col("pickup_hour")
        .is_in([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6])
        .cast(pl.Float32)
    )
    passenger_adj = (pl.col("passenger_count").cast(pl.Float32) - 1.0).clip(0.0, 5.0)

    df = df.with_columns(
        (
            pl.lit(2.50)
            + pl.col("distance").cast(pl.Float32) * pl.lit(1.55)
            + is_night * pl.lit(0.50)
            + passenger_adj * pl.lit(0.15)
        ).alias("fare_estimate")
    )
    return df


test_key_all = test["key"].to_pandas()

test_raw = test.clone()

train = train.filter(
    (pl.col("pickup_latitude") >= 40.5)
    & (pl.col("pickup_latitude") <= 41.0)
    & (pl.col("pickup_longitude") >= -74.3)
    & (pl.col("pickup_longitude") <= -73.7)
    & (pl.col("dropoff_latitude") >= 40.5)
    & (pl.col("dropoff_latitude") <= 41.0)
    & (pl.col("dropoff_longitude") >= -74.3)
    & (pl.col("dropoff_longitude") <= -73.7)
    & (pl.col("passenger_count") >= 1)
    & (pl.col("passenger_count") <= 6)
)
test = test.filter(
    (pl.col("pickup_latitude") >= 40.5)
    & (pl.col("pickup_latitude") <= 41.0)
    & (pl.col("pickup_longitude") >= -74.3)
    & (pl.col("pickup_longitude") <= -73.7)
    & (pl.col("dropoff_latitude") >= 40.5)
    & (pl.col("dropoff_latitude") <= 41.0)
    & (pl.col("dropoff_longitude") >= -74.3)
    & (pl.col("dropoff_longitude") <= -73.7)
    & (pl.col("passenger_count") >= 1)
    & (pl.col("passenger_count") <= 6)
)

train = preprocess(train)
test = preprocess(test)

train = train.filter(pl.col("pickup_datetime_parsed").is_not_null())

train = distance(train)
train = cycling_encoding(train)
train = one_hot_encoding(train)
train = is_central(train)
train = diff_central(train)
train = drop_outliner(train)
train = is_short_distance(train)
train = rule_base(train)
train = fare_estimate_feature(train)

test = distance(test)
test = cycling_encoding(test)
test = one_hot_encoding(test)
test = is_central(test)
test = diff_central(test)
test = is_short_distance(test)
test = rule_base(test)
test = fare_estimate_feature(test)

test = test.filter(
    ~(
        (pl.col("pickup_latitude") == pl.col("dropoff_latitude"))
        & (pl.col("pickup_longitude") == pl.col("dropoff_longitude"))
    )
)
test = test.filter((pl.col("distance") > 0.05) & (pl.col("distance") < 80.0))

test_key = test["key"]

train = train.drop(["key", "pickup_datetime", "pickup_minute", "pickup_second"])
test = test.drop(["key", "pickup_datetime", "pickup_minute", "pickup_second"])

train_cols = set(train.columns)
test_cols = set(test.columns)

extra_in_test = sorted(list(test_cols - train_cols))
if len(extra_in_test) > 0:
    test = test.drop(extra_in_test)

missing_in_test = sorted(list(train_cols - test_cols - {"fare_amount"}))
if len(missing_in_test) > 0:
    test = test.with_columns(
        [pl.lit(0).cast(pl.Int8).alias(c) for c in missing_in_test]
    )

feature_cols = [c for c in train.columns if c != "fare_amount"]
test = test.select(feature_cols)

print("Processed train shape:", train.shape)
print("Processed test shape (filtered):", test.shape)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3356265769.py in <cell line: 0>()
    273 
    274 # Fix: preserve all original test keys for a valid submission even if we filter the test set later.
--> 275 test_key_all = test["key"].to_pandas()
    276 
    277 # Keep original test copy for later merge back (avoid NameError cascade and ensure alignment).

NameError: name 'test' is not defined

## === cell 4
import polars as pld

dtypes = train.dtypes
float64_columns = [
    col for col, dtype in zip(train.columns, dtypes) if dtype == pl.Float64
]
if len(float64_columns) > 0:
    train = train.with_columns(
        [pl.col(col).cast(pl.Float32) for col in float64_columns]
    )

dtypes = test.dtypes
float64_columns = [
    col for col, dtype in zip(test.columns, dtypes) if dtype == pl.Float64
]
if len(float64_columns) > 0:
    test = test.with_columns([pl.col(col).cast(pl.Float32) for col in float64_columns])



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1580942095.py in <cell line: 0>()
      2 
      3 # Keep memory manageable: cast Float64 -> Float32
----> 4 dtypes = train.dtypes
      5 float64_columns = [
      6     col for col, dtype in zip(train.columns, dtypes) if dtype == pl.Float64

NameError: name 'train' is not defined

## === cell 5
import warnings

warnings.simplefilter("ignore")

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

train = train.filter((pl.col("fare_amount") > 0) & (pl.col("fare_amount") <= 250))
if "distance" in train.columns:
    train = train.filter((pl.col("distance") > 0.05) & (pl.col("distance") < 80.0))

dt_order = None
if "pickup_datetime_parsed" in train.columns:
    dt_series = train["pickup_datetime_parsed"].to_pandas()
    dt_parsed = pd.to_datetime(dt_series, errors="coerce")
    if dt_parsed.notna().mean() > 0.95:
        dt_order = np.argsort(dt_parsed.values)

X = train.drop(["fare_amount"]).to_pandas()
y = train["fare_amount"].to_pandas()
X_test = test.to_pandas()

X_test = X_test.reindex(columns=X.columns, fill_value=0)

non_numeric_cols = [
    c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c].dtype)
]
if len(non_numeric_cols) > 0:
    X = X.drop(columns=non_numeric_cols)
    X_test = X_test.drop(columns=non_numeric_cols, errors="ignore")

if dt_order is not None:
    n = len(dt_order)
    cut = int(n * 0.8)
    tr_idx = dt_order[:cut]
    va_idx = dt_order[cut:]
    X_train = X.iloc[tr_idx].copy()
    y_train = y.iloc[tr_idx].copy()
    X_val = X.iloc[va_idx].copy()
    y_val = y.iloc[va_idx].copy()
else:
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "regression",
    "metric": "rmse",
    "device": "gpu",
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 0,
    "feature_fraction_seed": 0,
    "bagging_seed": 0,
}

bst = lgb.train(params, train_data, num_boost_round=100, valid_sets=[val_data])

y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mse = mean_squared_error(y_val, y_pred)
print(f"rmse:{rmse}")
model_name = "lgbm"



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/742128006.py in <cell line: 0>()
      9 
     10 # Keep existing label/distance cleaning (core logic unchanged)
---> 11 train = train.filter((pl.col("fare_amount") > 0) & (pl.col("fare_amount") <= 250))
     12 if "distance" in train.columns:
     13     train = train.filter((pl.col("distance") > 0.05) & (pl.col("distance") < 80.0))

NameError: name 'train' is not defined

## === cell 6
sub_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
sub_pred = np.maximum(sub_pred, 0.0)

test_key_pd_filtered = test_key.to_pandas()

filtered_submission = pd.DataFrame(
    {"key": test_key_pd_filtered, "fare_amount": sub_pred}
)

all_submission = pd.DataFrame({"key": test_key_all})
all_submission = all_submission.merge(filtered_submission, on="key", how="left")

fill_value = float(np.nanmedian(sub_pred)) if len(sub_pred) else 11.35
all_submission["fare_amount"] = (
    all_submission["fare_amount"].fillna(fill_value).astype(float)
)

exp_num = "rule_base"
submission_path = f"submission_{model_name}_{exp_num}.csv"
all_submission.to_csv(submission_path, index=False)

print(all_submission.head())
print("Saved:", submission_path)
print("Submission rows:", len(all_submission))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1364677790.py in <cell line: 0>()
      1 # Produce submission for all original test keys, filling any filtered-out rows with median prediction.
----> 2 sub_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
      3 sub_pred = np.maximum(sub_pred, 0.0)
      4 
      5 test_key_pd_filtered = test_key.to_pandas()

NameError: name 'bst' is not defined

## === cell 7
import matplotlib.pyplot as plt

feature_importance_df = pd.DataFrame()
feature_importance_df["feature"] = X_train.columns
feature_importance_df["importance"] = bst.feature_importance()
feature_importance_df = feature_importance_df.sort_values(
    by="importance", ascending=False
)
feature_importance_df = feature_importance_df.head(15)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["feature"], feature_importance_df["importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.show()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2522616328.py in <cell line: 0>()
      2 
      3 feature_importance_df = pd.DataFrame()
----> 4 feature_importance_df["feature"] = X_train.columns
      5 feature_importance_df["importance"] = bst.feature_importance()
      6 feature_importance_df = feature_importance_df.sort_values(

NameError: name 'X_train' is not defined
