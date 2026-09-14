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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

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

4.33573

# 6. Current score

6.53349

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.30369) has done: 'Diagnosis: The crash happens at import time in cell 1, before any of your code executes, due to an incompatibility between `google-api-python-client` (which imports `google.protobuf`) and the version of `protobuf` installed in the environment; this manifests as `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. Your notebook does not use the Google API client at all, but it is installed and its import side-effects can break the runtime when `protobuf` is mismatched. The minimal deterministic fix is to set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before any such imports occur, forcing the pure-Python protobuf implementation that is compatible here. This change is localized to cell 1 and does not alter your ML logic, data loading, or model semantics.

Patch summary: Add an environment variable override at the very top of cell 1 (before importing keras/other libs) to force the Python protobuf implementation, preventing the `MessageFactory.GetPrototype` crash.

Updated cells: cell 1 only.

Compatibility notes for cell k+1: No variables, imports, or interfaces used by cell 2 are changed; `np`, `pd`, and the path constants remain identical and available.

Assumptions: The error is triggered by the protobuf C++ implementation mismatch in this environment, and forcing the Python implementation is sufficient and safe; no code in later cells requires protobuf performance.'
- What this solution (achieved 15.26221) has done: 'The crash happens immediately on importing `keras` in cell 1 because this environment’s `keras==3.8.0` depends on a newer `protobuf` API than what’s available here, producing `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The minimal, deterministic fix is to use the already-installed `tf_keras==2.18.0` package instead of standalone Keras 3, without changing any model/training logic. This only requires changing the Keras-related imports in the failing cell; all constants and paths remain identical. No other cells need modification.'
- What this solution (achieved 15.25037) has done: 'The crash happens before any data is loaded, during `tf_keras` import, due to an incompatibility between the installed `protobuf` runtime and TensorFlow/Keras’ expected protobuf API (`MessageFactory.GetPrototype`). The existing environment variable override isn’t sufficient here, so the minimal fix is to apply a small, deterministic monkey-patch to `google.protobuf.message_factory.MessageFactory` to provide `GetPrototype` as an alias to `GetMessageClass` when missing. This keeps the rest of the notebook logic unchanged and allows the `tf_keras` imports to succeed. The patch is localized to cell 1 and does not modify training/evaluation semantics.'
- What this solution (achieved 15.21253) has done: 'The crash happens inside cell 1 during the protobuf compatibility shim: `google.protobuf.message_factory.MessageFactory` exists but does not expose `GetPrototype`, and the attempted conditional monkey‑patch didn’t apply correctly before something else tried to call `GetPrototype`. The safest minimal fix is to always provide a `GetPrototype` method on that class when it’s missing, mapping it to either `GetMessageClass` (newer protobuf) or `GetMessages` (older variants), and to do this patch immediately after importing protobuf. This keeps the rest of your imports/logic unchanged and prevents the AttributeError from occurring. No other cells need modification, and all variables/constants defined in cell 1 remain identical.'
- What this solution (achieved 15.25141) has done: 'Diagnosis: The crash happens inside `remove_datapoints_from_water()` when calling `plt.imread()` with an `https://` URL. Newer Matplotlib versions (as in this environment) no longer allow reading URLs directly, so it raises a `ValueError` telling you to open the URL yourself and pass a file-like object to Pillow. This breaks `clean()` in cell 9 because it depends on `remove_datapoints_from_water()`.

Patch summary: In cell 9 only, open the mask image URL via `urllib.request.urlopen`, load it with `PIL.Image.open`, and convert it to a NumPy array before applying the existing `[:,:,0] > 0.9` logic. This preserves the same mask-based filtering semantics while avoiding the Matplotlib URL limitation.

Updated cells: (cell 9)

Compatibility notes for cell k+1: `train_df` remains a cleaned pandas DataFrame with the same columns and filtering logic; `train_df.describe()` in cell 10 work unchanged.

Assumptions: Internet access to `https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png` is available (same requirement as the original code). Pillow (`PIL`) is available via Matplotlib’s dependency stack in this environment.'
- What this solution (achieved 15.21719) has done: 'The crash happens because `remove_datapoints_from_water()` in cell 9 downloads a NYC land/water mask from an external URL that currently returns HTTP 404, so `urlopen()` raises `HTTPError`. To keep the original cleaning logic intact while making the notebook runnable in offline/blocked-network environments, I add a small fallback inside the same function: try to download the mask, and if it fails, skip only the water-filter step by returning the input `df` unchanged. This preserves the interface and downstream expectations (same columns, same type) and only changes behavior when the external resource is unavailable. No other cells or model logic are modified.'
- What this solution (achieved 15.11612) has done: 'Diagnosis: The crash occurs because `tf_keras.optimizers` in tf_keras==2.18.0 does not expose a lowercase `adam` callable; instead it provides the `Adam` optimizer class. Calling `optimizers.adam(lr=...)` therefore raises `AttributeError`. To keep the same optimizer behavior, we should instantiate `optimizers.Adam` and pass the learning rate using the supported argument name (`learning_rate`).  

Patch summary: In cell 34, replace `optimizers.adam(lr=LEARNING_RATE)` with `optimizers.Adam(learning_rate=LEARNING_RATE)`. This is the minimal change needed to unblock model compilation and training while preserving the original training semantics.  

Updated cells / Compatibility notes for cell k+1 / Assumptions: The model object, `history`, and all variable names remain unchanged, so cell 35 continue to work as-is. Assumes tf_keras 2.18.0 follows the standard Keras optimizer API with `optimizers.Adam(learning_rate=...)`.'
- What this solution (achieved 5.73705) has done: 'Diagnosis: The crash happens during training when computing the custom `rmse` metric. In this environment `from keras import backend` resolves to Keras 3’s `keras.api.backend`, which no longer exposes legacy math ops like `sqrt/mean/square`, so `backend.sqrt(...)` raises `AttributeError`. The model itself is built with `tf_keras`, so the safest minimal fix is to use `tf_keras.backend` (or TensorFlow ops) for the metric implementation.

Patch summary: In cell 34 only, redefine `rmse` locally using `tf_keras.backend` and compile the model with this local `rmse`, avoiding reliance on the incompatible Keras 3 backend. No other logic (architecture/training/metrics list) is changed.

Updated cells: Only cell 34 is modified below.

Compatibility notes for cell k+1: `model` and `history` are still created exactly as before, so cell 35 can still visualize `model`. The metric name remains `rmse`, so any downstream references to `history.history['rmse']` keep working.

Assumptions: `tf_keras` is available (it is, per installed packages) and provides a legacy-compatible backend module with `sqrt/mean/square`.'
- What this solution (achieved 34.42011) has done: 'Diagnosis: The crash in cell 35 is caused by an outdated import path: `keras.utils.vis_utils` no longer exists in Keras 3 (and this environment uses `keras==3.8.0`). In Keras 3, `model_to_dot` is provided via `keras.utils.model_to_dot`. To keep the same behavior (rendering the model as SVG), we only need to update the import and leave the visualization call unchanged.

Patch summary: Modify cell 35 to import `model_to_dot` from `keras.utils` instead of `keras.utils.vis_utils`, preserving the exact SVG generation logic.

Updated cells: Only cell 35 is changed.

Compatibility notes for cell k+1: Cell 36 uses `history` and `plot_loss_accuracy_rmse(history)`; this patch does not touch those variables or training logic, so cell 36 remains fully compatible.

Assumptions: `keras.utils.model_to_dot` is available in the installed Keras 3.8.0 package (standard distribution). Graphviz may still be required at runtime for `.create(prog='dot', ...)`, but that is separate from the import error being fixed.'
- What this solution (achieved 159.37259) has done: 'Diagnosis: Cell 39 crashes because `keras.losses.mean_squared_error(y_true, y_pred)` requires `y_true` and `y_pred` to be broadcastable to the same shape, but the code slices them to different lengths (`test_labels[1000:]` vs `prediction[:1000]`). This produces incompatible shapes and triggers the `InvalidArgumentError` during subtraction inside the loss. The fix is to slice both arrays over the same index range and ensure they are compatible ranks (e.g., `(N,)` vs `(N,1)`).

Patch summary: Update cell 39 to compute MSE on matching slices (first 1000 samples) and reshape/flatten to make shapes compatible. Keep the same loss function call (`keras.losses.mean_squared_error`) and avoid changing any upstream modeling/training logic.

Updated cells: Only cell 39 is modified.

Compatibility notes for cell k+1: No variables used by later cells are changed; `prediction`, `predictionKaggle`, and `test_labels` remain intact. This patch only evaluates a metric and does not affect `output_submission(...)` in cell 41.

Assumptions: We assume the intent was to compare the same subset of ground-truth labels and predictions (likely the first 1000 examples) and that `prediction` is shaped `(N, 1)` while `test_labels` is `(N,)`, so flattening/reshaping is needed for broadcasting.'
- What this solution (achieved 8.75409) has done: 'Your current RMSE is extremely high mainly because the test split is taken from the raw (mostly uncleaned) data and is not feature-engineered consistently with the training data; the model is then evaluated on a different feature space (and in one place even on unscaled features). I make the smallest set of changes to ensure the same cleaning + feature engineering + column dropping is applied to `test_df` as to `train_df`, and ensure evaluation uses the scaled matrix (the same representation the model was trained on). This should move the score sharply downward toward your 4.33573 target without changing the model architecture, loss, or training loop. I also fix a couple of tiny logic bugs in the time-feature flags (currently always-true conditions) that harm signal quality but keep the overall feature set identical.'
- What this solution (achieved 6.53349) has done: 'Your score is worse than the target (RMSE 8.75 vs 4.34), so we should make small, legitimate changes that reduce error without changing the core model/training loop. The biggest remaining issue is inconsistent preprocessing: you clean the train/test split used for evaluation but you do not apply the same cleaning to the actual Kaggle `test.csv`, causing distribution shift at inference time; we apply a “test-safe” version of cleaning to `testKaggle` that preserves row count and key alignment (i.e., no row dropping) but clips invalid coordinates/passenger_count into the same feasible ranges. We also fix `add_time_features` parsing to robustly handle the competition’s datetime format (no timezone in most rows), preventing silent NaT/feature corruption. These are minimal changes that keep your architecture/loss/training intact while moving predictions closer to the training distribution and lowering RMSE toward the target.'

# 9. Code solution

## === cell 0
def clean(df):

    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != df["pickup_longitude"])
        & (df["dropoff_latitude"] != df["pickup_latitude"])
    ]
    print(" New size after removing same long lat: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    ]
    print(" New size after removing 0 long lat: %d" % len(df))

    MinMax = (-74.5, -72.8, 40.5, 41.8)
    df = df[
        (MinMax[0] <= df["pickup_longitude"]) & (df["pickup_longitude"] <= MinMax[1])
    ]
    df = df[
        (MinMax[0] <= df["dropoff_longitude"]) & (df["dropoff_longitude"] <= MinMax[1])
    ]
    df = df[(MinMax[2] <= df["pickup_latitude"]) & (df["pickup_latitude"] <= MinMax[3])]
    df = df[
        (MinMax[2] <= df["dropoff_latitude"]) & (df["dropoff_latitude"] <= MinMax[3])
    ]

    print(" New size after only NYC: %d" % len(df))
    df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]

    print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)  # Statue of Liberty

    df = df[
        (nyc_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != nyc_coord[0])
    ]
    df = df[
        (nyc_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != nyc_coord[0])
    ]

    print(" New size after NY airport: %d" % len(df))

    df = df[
        (fk_coord[1] != df["pickup_longitude"]) & (df["pickup_latitude"] != fk_coord[0])
    ]
    df = df[
        (fk_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != fk_coord[0])
    ]

    print(" New size after jfk airport: %d" % len(df))

    df = df[
        (ewr_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != ewr_coord[0])
    ]
    df = df[
        (ewr_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != ewr_coord[0])
    ]

    print(" New size after ewr airport: %d" % len(df))
    df = df[
        (lga_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != lga_coord[0])
    ]
    df = df[
        (lga_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != lga_coord[0])
    ]

    print(" New size after lgr airport: %d" % len(df))

    df = df[
        (sol_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != sol_coord[0])
    ]
    df = df[
        (sol_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != sol_coord[0])
    ]

    print(" New size after sol removed: %d" % len(df))

    print("Old size: %d" % len(df))
    df = remove_datapoints_from_water(df)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))

    return df


def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    nyc_mask = (
        plt.imread("https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png")[
            :, :, 0
        ]
        > 0.9
    )

    pickup_x, pickup_y = lonlat_to_xy(
        df.pickup_longitude,
        df.pickup_latitude,
        nyc_mask.shape[1],
        nyc_mask.shape[0],
        BB,
    )
    dropoff_x, dropoff_y = lonlat_to_xy(
        df.dropoff_longitude,
        df.dropoff_latitude,
        nyc_mask.shape[1],
        nyc_mask.shape[0],
        BB,
    )
    idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]

    return df[idx]


def late_night(row):
    if (row["hour"] <= 3) or (row["hour"] >= 22):
        return 1
    else:
        return 0


def night(row):
    if ((row["hour"] > 20) or (row["hour"] < 6)) and (row["weekday"] < 5):
        return 1
    else:
        return 0


def rush_hour(row):
    if ((row["hour"] <= 20) and (row["hour"] >= 16)) and (row["weekday"] < 5):
        return 1
    else:
        return 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=False, infer_datetime_format=True
    )
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.weekday
    df["pickup_datetime"] = dt.astype(str)
    df["night"] = df.apply(lambda x: night(x), axis=1)
    df["late_night"] = df.apply(lambda x: late_night(x), axis=1)
    df["rush_hour"] = df.apply(lambda x: rush_hour(x), axis=1)

    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    df["latdiff"] = lat1 - lat2
    df["londiff"] = lon1 - lon2

    return df


def add_distances_features(df):

    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["distance"] = np.sqrt(
        np.abs(df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
        + np.abs(df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
    )

    return df


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


def distanceP(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column]
    df[[id_column, prediction_column]].to_csv((file_name), index=False)
    print("Output complete")


def plot_loss_accuracy_rmse(history):

    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(history.history["rmse"])
    plt.plot(history.history["val_rmse"])
    plt.title("Model rmse")
    plt.ylabel("rmse")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()


def clean_test_safe(df):
    df = df.copy()
    for c in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]:
        if c in df.columns:
            df[c] = df[c].astype("float32")
            df[c] = df[c].fillna(df[c].median())
    if "passenger_count" in df.columns:
        df["passenger_count"] = df["passenger_count"].fillna(1).astype("float32")

    MinMax = (-74.5, -72.8, 40.5, 41.8)
    df["pickup_longitude"] = df["pickup_longitude"].clip(MinMax[0], MinMax[1])
    df["dropoff_longitude"] = df["dropoff_longitude"].clip(MinMax[0], MinMax[1])
    df["pickup_latitude"] = df["pickup_latitude"].clip(MinMax[2], MinMax[3])
    df["dropoff_latitude"] = df["dropoff_latitude"].clip(MinMax[2], MinMax[3])

    df["passenger_count"] = df["passenger_count"].clip(1, 6).round().astype("uint8")

    return df




## === cell 1
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory"):
        _mf_cls = _message_factory.MessageFactory
        if not hasattr(_mf_cls, "GetPrototype"):

            def _GetPrototype(self, descriptor):
                if hasattr(self, "GetMessageClass"):
                    return self.GetMessageClass(descriptor)
                if hasattr(self, "GetMessages"):
                    msg_map = self.GetMessages([descriptor.file])
                    return msg_map.get(descriptor.full_name)
                raise AttributeError(
                    "protobuf MessageFactory has no compatible method to provide GetPrototype"
                )

            _mf_cls.GetPrototype = _GetPrototype
except Exception:
    pass

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tf_keras.callbacks import EarlyStopping
from tf_keras import optimizers
from tf_keras import regularizers

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 1000
EPOCHS = 50
LEARNING_RATE = 0.001
DATASET_SIZE = 80000



## === cell 2
datatypes = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

trainKaggle = pd.read_csv(
    TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes, usecols=[1, 2, 3, 4, 5, 6, 7]
)
testKaggle = pd.read_csv(TEST_PATH)



## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)



## === cell 4
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))



## === cell 5
train_df.describe()



## === cell 6
test_df.describe()




## === cell 7
def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"
    try:
        from urllib.request import urlopen
        from PIL import Image

        nyc_mask_img = np.array(Image.open(urlopen(url)))
        nyc_mask = nyc_mask_img[:, :, 0] > 0.9
    except Exception as e:
        print(
            f"Warning: could not load NYC water mask from {url} ({type(e).__name__}: {e}). "
            f"Skipping water filter."
        )
        return df

    pickup_x, pickup_y = lonlat_to_xy(
        df.pickup_longitude,
        df.pickup_latitude,
        nyc_mask.shape[1],
        nyc_mask.shape[0],
        BB,
    )
    dropoff_x, dropoff_y = lonlat_to_xy(
        df.dropoff_longitude,
        df.dropoff_latitude,
        nyc_mask.shape[1],
        nyc_mask.shape[0],
        BB,
    )
    idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]

    return df[idx]




## === cell 8
print("train_df clean")
train_df = clean(train_df)

print("test_df clean")
test_df = clean(test_df)

print("testKaggle clean_test_safe (no row dropping)")
testKaggle = clean_test_safe(testKaggle)



## === cell 9
train_df.describe()



## === cell 10
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 11
train_df.describe()



## === cell 12
print("train_df add_coordinate_features")
add_coordinate_features(train_df)
print("test_df add_coordinate_features Disabled!")
add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
add_coordinate_features(testKaggle)



## === cell 13
train_df.describe()



## === cell 14
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## === cell 15
train_df.describe()



## === cell 16
dropped_columns = [
    "passenger_count",
    "pickup_datetime",
]  #'pickup_latitude','pickup_longitude', 'dropoff_longitude', 'dropoff_latitude']

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")



## === cell 17
train_df.shape



## === cell 18
train_df.describe()



## === cell 19
test_df.describe()



## === cell 20
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## === cell 21
train_df.describe()



## === cell 22
validation_df.describe()



## === cell 23
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 24
test_labels



## === cell 25
train_df.describe()



## === cell 26
validation_df.describe()



## === cell 27
test_df.describe()



## === cell 28
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)



## === cell 29
test_scaled



## === cell 30
from keras import backend


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))




## === cell 31
from tf_keras import backend as tfk_backend


def rmse(y_true, y_pred):
    return tfk_backend.sqrt(
        tfk_backend.mean(tfk_backend.square(y_pred - y_true), axis=-1)
    )


model = Sequential()
model.add(
    Dense(
        256,
        activation="relu",
        input_dim=train_df_scaled.shape[1],
        activity_regularizer=regularizers.l1(0.01),
    )
)
model.add(BatchNormalization())
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(32, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(8, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(1))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    loss="mean_squared_error", optimizer=adam, metrics=["mae", "accuracy", rmse, "mse"]
)

print("Dataset size: %s" % DATASET_SIZE)
print("Epochs: %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size: %s" % BATCH_SIZE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % train_df.columns)
model.summary()

history = model.fit(
    x=train_df_scaled,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(validation_df_scaled, validation_labels),
    shuffle=True,
)



## === cell 32
from IPython.display import SVG

from keras.utils import model_to_dot

SVG(model_to_dot(model).create(prog="dot", format="svg"))



## === cell 33
plot_loss_accuracy_rmse(history)



## === cell 34
score = model.evaluate(test_scaled, test_labels, verbose=1)
print(score)
print("Test loss:", score[0])
print("Test accuracy:", score[1])



## === cell 35
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)



## === cell 36
import keras.losses  # import mean_squared_error
import numpy as np

y_true = np.asarray(test_labels[:1000]).reshape(-1, 1)
y_pred = np.asarray(prediction[:1000])

keras.losses.mean_squared_error(y_true, y_pred)



## === cell 37
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)



## === cell 38
if len(prediction) > 10000 and len(test_labels) > 10000:
    print(prediction[10000])
    print(test_labels[10000])
else:
    print("Not enough rows to print index 10000 after cleaning.")
