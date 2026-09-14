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

5.00327

# 6. Current score

112.98568

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.23538) has done: 'Diagnosis: The crash happens during `import keras...` in cell 1, before any data is loaded. With Keras 3.x in this environment, importing the standalone `keras` package can trigger a protobuf incompatibility leading to `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The project also has `tf_keras==2.18.0` installed, which provides the same Keras API but avoids this protobuf path and is compatible with the rest of the notebook. The minimal fix is to switch the Keras imports in cell 1 from `keras` to `tf_keras`, keeping the same classes and names used later.

Patch summary: Replace all `from keras...` imports in cell 1 with `from tf_keras...` equivalents, leaving all constants and variables unchanged so downstream cells continue to work.

Updated cells: cell 1 only.

Compatibility notes for cell k+1: Cell 2 only depends on `TRAIN_PATH`, `TEST_PATH`, and `DATASET_SIZE` defined in cell 1; these remain unchanged. No interfaces or variable names used by later cells are modified—only the import source is changed.

Assumptions: `tf_keras` is installed and importable (it is listed as `tf_keras==2.18.0`), and the rest of the notebook uses standard Keras APIs that are compatible with `tf_keras`.'
- What this solution (achieved 15.25374) has done: 'Diagnosis: The crash happens during the `tf_keras` import chain in cell 1, where an incompatible `protobuf` runtime is pulled in and triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This is a known compatibility issue between newer protobuf versions and some TensorFlow/Keras builds. Since cell 1 only needs Keras symbols, the safest minimal fix is to switch the imports to the already-installed `keras` package (v3.8.0) to avoid importing the problematic TensorFlow/protobuf stack.

Patch summary: Modify only the Keras-related imports in cell 1 to use `keras` instead of `tf_keras`, keeping the same layer/model/callback/optimizer/regularizer names and leaving all constants and subsequent variable names unchanged.

Updated cells: Only cell 1 is changed.

Compatibility notes for cell k+1: Cell 2 only depends on `pd`, `TRAIN_PATH`, `TEST_PATH`, and `DATASET_SIZE`, which remain unchanged. The model symbols imported in cell 1 keep the same identifiers (`Sequential`, `Dense`, etc.), so later cells that build/train the model still find these names.

Assumptions: The environment can import `keras==3.8.0` successfully (it is installed) and later cells do not rely on TensorFlow-specific behaviors unique to `tf_keras` beyond the standard Keras API.'
- What this solution (achieved 15.28275) has done: 'Diagnosis: The crash happens in cell 1 during `import keras...` because the installed `keras==3.8.0` triggers a protobuf incompatibility in this Python 3.7 environment, resulting in `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This is a known symptom of mismatched `protobuf`/runtime expectations and is avoided here by using the already-installed `tf_keras==2.18.0`, which is compatible with the environment and provides the same high-level Keras APIs used in the notebook. The fix is to switch the Keras imports in cell 1 from `keras` to `tf_keras` without changing any model/training logic.

Patch summary: In cell 1 only, replace `keras.*` imports with `tf_keras.*` equivalents (models/layers/callbacks/optimizers/regularizers) to bypass the incompatible standalone Keras 3 import path that triggers the protobuf error. No other logic, constants, paths, or semantics are modified.

Updated cells:'
- What this solution (achieved 15.14918) has done: 'Diagnosis: The crash happens in cell 1 during imports, before any data is read. With this environment (Python 3.7, `google-api-python-client==2.177.0`), importing `googleapiclient` can indirectly trigger an incompatible `protobuf`/`google` stack where `MessageFactory.GetPrototype` is missing, raising `AttributeError` even though the notebook itself does not use Google APIs. The simplest deterministic fix is to remove/disable that unused dependency by ensuring the `google` client stack is not imported/initialized from this notebook’s runtime, while keeping the ML/data logic unchanged.

Patch summary: In cell 1, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` *before* any other imports to force the pure-Python protobuf implementation and avoid the failing `GetPrototype` access. This is a minimal, localized environment workaround that does not change model/training semantics.

Updated cells: Only cell 1 is modified.

Compatibility notes for cell k+1: All variables defined in cell 1 (`TRAIN_PATH`, `TEST_PATH`, `SUBMISSION_NAME`, hyperparameters, and keras/sklearn imports) remain unchanged and available for cell 2+.

Assumptions: The failure is triggered by a protobuf C++ implementation incompatibility in the current runtime; forcing the Python implementation is sufficient and does not materially affect downstream computations (since protobuf is not used by this notebook’s logic).'
- What this solution (achieved 15.0908) has done: 'Diagnosis: The crash in cell 1 happens during importing `tf_keras` because the installed `protobuf` runtime is incompatible with the expected API (`MessageFactory.GetPrototype`), and forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is not sufficient to avoid that in this environment. This is a known protobuf/TensorFlow-Keras interop issue that is resolved by ensuring the pure-Python implementation is used *before* protobuf is first imported and by disabling the C++ fast implementation explicitly. Since we cannot change packages, the minimal fix is to set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` and `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION` before any imports that may trigger protobuf (including `tf_keras`). No model/training logic is changed.

Patch summary: Update only cell 1 to set the protobuf environment variables (including implementation version) at the very top, before importing any libraries, so `tf_keras` imports do not crash with the missing `GetPrototype` attribute.

Updated cells / Compatibility notes for cell k+1 / Assumptions: Cell 2 expects `TRAIN_PATH`, `TEST_PATH`, and other constants from cell 1; these remain unchanged. Assumption: This environment supports protobuf’s pure-Python implementation when `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` is set, which avoids the failing code path used by the C++ implementation.'
- What this solution (achieved 15.18197) has done: 'Diagnosis: The crash happens during `tf_keras` import in cell 1 due to an incompatibility between the installed protobuf runtime and TensorFlow/Keras, producing `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The environment variables set in the cell do not guarantee a compatible protobuf implementation/version for `tf_keras` in this environment. The minimal unblock is to force the pure-Python protobuf implementation and use a protobuf runtime version that still provides `MessageFactory.GetPrototype` (protobuf 3.20.x), then perform the imports.

Patch summary: In cell 1 only, add a small pre-import compatibility shim that (1) forces the python protobuf backend, and (2) pins protobuf to 3.20.* at runtime (via pip) if a newer incompatible protobuf is present, then proceeds with the original imports unchanged. No model/training logic is modified.

Updated cells: cell 1 only (buggy cell).

Compatibility notes for cell k+1: All symbols defined in cell 1 (`np`, `pd`, `plt`, `Sequential`, layer classes, constants like `TRAIN_PATH`, etc.) remain defined with the same names and types, so cell 2 can run unchanged.

Assumptions: Network access for `pip install` is available in this environment (typical Kaggle notebook behavior). If protobuf is already compatible, the pip step is a no-op and imports proceed normally.'
- What this solution (achieved 15.4347) has done: 'Diagnosis: The crash happens inside `remove_datapoints_from_water()` when calling `plt.imread()` with an `https://` URL; recent Matplotlib/Pillow versions do not support reading URLs directly and raise a `ValueError`. The rest of the cleaning logic expects `nyc_mask` to be a boolean 2D array derived from that image, so we must load the same PNG via `urllib.request.urlopen` and `PIL.Image.open`, then convert it to a NumPy array. This keeps the cleaning/evaluation semantics identical while making the image read compatible with the installed stack.

Patch summary: Update only cell 9 to monkey-patch `remove_datapoints_from_water` so it loads the NYC mask image from the URL using `urllib` + `PIL`, then computes the same boolean mask and filtering as before. No other logic is changed.

Updated cells: cell 9 only.

Compatibility notes for cell k+1: `train_df` and `test_df` remain pandas DataFrames with the same columns as before; only the internal URL image loading method changes, so cell 10 (`train_df.describe()`) works unchanged.

Assumptions: Outbound HTTPS access is available in the environment (as implied by the original code using the URL). Pillow is available via Matplotlib’s dependency stack; if not, the import would fail early (but in this environment it should exist).'
- What this solution (achieved 15.40744) has done: 'Diagnosis: Cell 9 crashes because `remove_datapoints_from_water()` downloads a NYC land mask image from `https://aiblog.nl/...png`, but that URL now returns HTTP 404 in this environment. Since `clean()` (defined earlier) calls `remove_datapoints_from_water()`, the whole cleaning step fails before any model code can run. The minimal safe fix is to make `remove_datapoints_from_water()` robust to download failure by falling back to a no-op (return the original `df`) when the mask cannot be retrieved/parsed; this preserves the pipeline and keeps the same interface.

Patch summary: In cell 9 only, wrap the remote mask download/decoding in a `try/except` and return `df` unchanged on any exception (HTTPError/URLError/etc.). This avoids the crash while keeping the rest of the cleaning logic and downstream variables (`train_df`, `test_df`) intact.

Updated cells: (cell 9 only)

Compatibility notes for cell k+1: Cell 10 expects `train_df` to exist and be a cleaned DataFrame; with this patch `train_df`/`test_df` are still produced (possibly without the water-mask filtering if the download fails), so `train_df.describe()` continues to work unchanged.

Assumptions: If the mask URL is unavailable (as indicated by the 404), skipping only the water-mask filter is acceptable to unblock execution; all other cleaning filters remain applied exactly as before.'
- What this solution (achieved 15.31986) has done: 'Diagnosis: Cell 36 crashes because `tf_keras.optimizers` (TF-Keras 2.18) does not expose a lowercase factory `optimizers.adam`; the correct API is the `Adam` class (or `optimizers.legacy.Adam`). Using the nonexistent attribute raises `AttributeError` before training starts.  
Patch summary: Replace `optimizers.adam(lr=...)` with a version-compatible optimizer construction while keeping the same learning rate and training semantics. Prefer `optimizers.Adam(learning_rate=...)` and fall back to `optimizers.legacy.Adam(...)` if needed for compatibility.  
Updated cells: Only cell 36 is changed, and the model architecture, compile arguments, and fit call remain the same aside from the optimizer instantiation.  
Compatibility notes for cell k+1: `model` and `history` are still created exactly as before, so visualization in cell 37 continues to work unchanged.  
Assumptions: `tf_keras.optimizers.Adam` exists in this environment; if not, `tf_keras.optimizers.legacy.Adam` exists (common in TF-Keras).'
- What this solution (achieved 266.68573) has done: 'Diagnosis: The crash happens during training when Keras tries to call the custom `rmse` metric. In this environment, `from keras import backend` resolves to Keras 3’s `keras.api.backend`, which no longer exposes legacy math ops like `sqrt/mean/square`, causing `AttributeError: module 'keras.api.backend' has no attribute 'sqrt'`. The rest of the model uses `tf_keras`, so the metric must also use `tf_keras` (or TensorFlow ops) to be compatible with the training loop.

Patch summary: In cell 36, replace the `keras.backend`-based RMSE implementation by importing `backend` from `tf_keras` (same stack as the model) and compute RMSE with those backend ops. Keep the function name/signature (`rmse(y_true, y_pred)`) unchanged so `model.compile(..., metrics=[..., rmse, ...])` continues to work without changing training semantics.

Updated cells: Only cell 36 is modified.

Compatibility notes for cell k+1: Cell 37 still references `model` and `model_to_dot(model)` exactly as before; no interface/variable names change.

Assumptions: `tf_keras` is available (it is installed and already used for the model), and its backend provides `sqrt/mean/square` compatible with the training graph.'
- What this solution (achieved 325.8975) has done: 'Diagnosis: Cell 37 crashes because Keras 3 removed/moved `keras.utils.vis_utils`, so `from keras.utils.vis_utils import model_to_dot` raises `ModuleNotFoundError`. The visualization is optional and should not affect training/inference semantics. We need a version-compatible import path and a safe fallback if Graphviz-related utilities are unavailable.

Patch summary: Update cell 37 to try importing `model_to_dot` from the Keras 3 supported location (`keras.utils`) and fall back to `tf_keras.utils` if needed; if neither is available, skip the SVG rendering gracefully without stopping the notebook. This keeps the `model` variable and training results unchanged and preserves compatibility with the next cell.

Updated cells: Only cell 37 is modified.

Compatibility notes for cell k+1: Cell 38 only uses `history` and plotting; this patch does not modify `history` or `model` and not interfere with loss plotting.

Assumptions: Rendering the model diagram is non-essential; if the environment lacks the necessary visualization dependencies (e.g., Graphviz/pydot), we should not crash the run.'
- What this solution (achieved 414.34596) has done: 'Diagnosis: Cell 46 crashes because it references a variable named `prediction` that is never defined anywhere earlier; the actual predictions computed in cell 43 are stored in `test_predictions` (and Kaggle predictions in `predictionKaggle` in cell 44). This mismatch triggers a `NameError` on the first line of the cell. The fix is to use the already-defined `test_predictions` variable in this diagnostic cell while keeping the same argmax-based inspection logic.

Patch summary: Update cell 46 to replace all uses of the undefined `prediction` with `test_predictions`, which is created in cell 43 and matches the intended “test set predictions” semantics. No other logic or outputs are changed.

Updated cells: (cell 46 only)

Compatibility notes for cell k+1: Cell 47 still references `prediction` and may also fail; this patch does not modify cell 47 per the constraint to only change the failing cell. The updated cell 46 does not alter any variables used later.

Assumptions: The intended variable for the argmax inspection in cell 46 is the test-set prediction array produced in cell 43 (`test_predictions`), not the Kaggle submission predictions (`predictionKaggle`).'
- What this solution (achieved 211.42485) has done: 'Diagnosis: Cell 47 crashes with `NameError: name 'prediction' is not defined` because no variable named `prediction` exists in the notebook state. Earlier cells create `test_predictions` (predictions for `test_scaled`) and `predictionKaggle` (predictions for Kaggle test set), but cell 47 (and cell 48) incorrectly reference `prediction`.  
Patch summary: In cell 47, define `prediction` to point to the already-computed `test_predictions` so that the argmin inspection and downstream plotting in cell 48 work without changing any model/training logic or evaluation semantics.  
Updated cells: Only cell 47 is modified.  
Compatibility notes for cell k+1: Cell 48 expects `prediction` to be defined and aligned with `test_labels`; mapping `prediction = test_predictions` preserves shape compatibility and intended “measured vs predicted” plotting for the local test split.  
Assumptions: The intent of cells 47–48 is to inspect/plot predictions against `test_labels` (the local split), not the Kaggle submission predictions.'
- What this solution (achieved 112.98568) has done: 'The crash happens because `train_df` still contains at least one non-numeric column (most likely the `key` string ID), so `MinMaxScaler` can’t convert it to floats. The intended model features are numeric, and `key` is only used later for submission, so dropping it for scaling preserves the core training/inference semantics. In cell 29, drop `key` from all feature DataFrames (and align `testKaggle_clean`) right before scaling, without changing any other upstream logic. This keeps the existing variables (`train_df_scaled`, `validation_df_scaled`, `test_scaled`, `testKaggle_scaled`, `scaler`) available for cell 30 and later cells.'

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

    print(" New size after NYC lang lot: %d" % len(df))

    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["dropoff_longitude"] != 0)]
    df = df[(df["dropoff_latitude"] != 0)]

    print(" New size after lang lot > 0: %d" % len(df))

    df = df[((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > 0.001)]
    df = df[((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > 0.001)]

    print(" New size after lang - lot > 0.001: %d" % len(df))

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
    if (row["hour"] <= 3) or (row["hour"] >= 0):
        return 1
    else:
        return 0


def night(row):
    if ((row["hour"] > 20) and (row["hour"] > 0)) and (row["weekday"] < 5):
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
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S %Z"
    )
    df["year"] = df["pickup_datetime"].apply(lambda x: x.year)
    df["month"] = df["pickup_datetime"].apply(lambda x: x.month)
    df["day"] = df["pickup_datetime"].apply(lambda x: x.day)
    df["hour"] = df["pickup_datetime"].apply(lambda x: x.hour)
    df["weekday"] = df["pickup_datetime"].apply(lambda x: x.weekday())
    df["pickup_datetime"] = df["pickup_datetime"].apply(lambda x: str(x))
    df["night"] = df.apply(lambda x: night(x), axis=1)
    df["late_night"] = df.apply(lambda x: late_night(x), axis=1)
    df["rush_hour"] = df.apply(lambda x: rush_hour(x), axis=1)

    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    df["latdiff"] = (lat1 - lat2).abs()
    df["londiff"] = (lon1 - lon2).abs()

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




## === cell 1
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_version
    from packaging.version import Version

    if Version(_pb_version) >= Version("4.0.0"):
        import sys
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.*"]
        )
        import importlib

        importlib.invalidate_caches()
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

BATCH_SIZE = 256
EPOCHS = 100
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
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    dtype=datatypes,
    usecols=[
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)
testKaggle = pd.read_csv(
    TEST_PATH, dtype={k: v for k, v in datatypes.items() if k != "fare_amount"}
)


## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000]


## === cell 4
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))


## === cell 5
train_df.describe()


## === cell 6
test_df.describe()


## === cell 7
import urllib.request
from PIL import Image


def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"
    try:
        with urllib.request.urlopen(url) as resp:
            img = Image.open(resp)
            nyc_mask = np.array(img)[:, :, 0] > 0.9
    except Exception:
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


print("train_df clean")
train_df = clean(train_df)
test_df = clean(test_df)


## === cell 8
train_df.describe()


## === cell 9
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)


## === cell 10
train_df.describe()


## === cell 11
print("train_df add_coordinate_features")
add_coordinate_features(train_df)
print("test_df add_coordinate_features Disabled!")
add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
add_coordinate_features(testKaggle)


## === cell 12
train_df.describe()


## === cell 13
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)


print("Done with Adding features")


## === cell 14
train_df.describe()


## === cell 15
plot = train_df.iloc[:2000].plot.scatter("latdiff", "londiff")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "weekday")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "night")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "late_night")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "rush_hour")


## === cell 16
dropped_columns = [
    "passenger_count",
    "pickup_datetime",
]  #'pickup_latitude','pickup_longitude', 'dropoff_longitude', 'dropoff_latitude']

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)

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
testKaggle_clean = testKaggle.drop(dropped_columns, axis=1).copy()
testKaggle_clean = testKaggle_clean[train_df.columns]


## === cell 29
scaler = preprocessing.MinMaxScaler()

if "key" in train_df.columns:
    train_df = train_df.drop(["key"], axis=1)
if "key" in validation_df.columns:
    validation_df = validation_df.drop(["key"], axis=1)
if "key" in test_df.columns:
    test_df = test_df.drop(["key"], axis=1)
if "key" in testKaggle_clean.columns:
    testKaggle_clean = testKaggle_clean.drop(["key"], axis=1)

testKaggle_clean = testKaggle_clean[train_df.columns]

train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)


## === cell 30
test_scaled


## === cell 31
from keras import backend


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))




## === cell 32
from tf_keras import backend


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


model = Sequential()
model.add(
    Dense(
        256,
        activation="linear",
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

try:
    adam = optimizers.Adam(learning_rate=LEARNING_RATE)
except AttributeError:
    adam = optimizers.legacy.Adam(learning_rate=LEARNING_RATE)

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


## === cell 33
from IPython.display import SVG

try:
    from keras.utils import model_to_dot  # Keras 3+
except Exception:
    try:
        from tf_keras.utils import model_to_dot  # fallback for tf_keras environments
    except Exception:
        model_to_dot = None

if model_to_dot is not None:
    try:
        SVG(model_to_dot(model).create(prog="dot", format="svg"))
    except Exception as e:
        print(f"Model visualization skipped (graphviz/pydot may be missing): {e}")
else:
    print(
        "Model visualization skipped: model_to_dot is not available in this environment."
    )


## === cell 34
plot_loss_accuracy_rmse(history)


## === cell 35
score = model.evaluate(train_df_scaled, train_labels, verbose=1)
print(score)
print("train mean_squared_error:", score[0])
print("train mae:", score[1])
print("train accuracy:", score[2])
print("train rmse:", score[3])
print("train mse:", score[4])


## === cell 36
score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
print(score)
print("Validation mean_squared_error:", score[0])
print("Validation mae:", score[1])
print("Validation accuracy:", score[2])
print("Validation rmse:", score[3])
print("Validation mse:", score[4])


## === cell 37
score = model.evaluate(test_scaled, test_labels, verbose=1)
print(score)
print("Test mean_squared_error:", score[0])
print("Test mae:", score[1])
print("Test accuracy:", score[2])
print("Test rmse:", score[3])
print("Test mse:", score[4])


## === cell 38
validation_predictions = model.predict(validation_df_scaled).flatten()

plt.scatter(validation_labels, validation_predictions)
plt.xlabel("True Values [10$]")
plt.ylabel("Predictions [10$]")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])


## === cell 39
test_predictions = model.predict(test_scaled).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values [10$]")
plt.ylabel("Predictions [10$]")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])


## === cell 40
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)




## === cell 41
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())




## === cell 42
print(np.argmax(test_predictions))
print(test_predictions[np.argmax(test_predictions)])
print(test_labels[np.argmax(test_predictions)])
test_df.iloc[np.argmax(test_predictions)]


## === cell 43
prediction = test_predictions

print(np.argmin(prediction))
print(prediction[np.argmin(prediction)])
print(test_labels[np.argmin(prediction)])
test_df.iloc[np.argmin(prediction)]


## === cell 44
fig, ax = plt.subplots()
ax.scatter(test_labels, prediction)
ax.plot(
    [test_labels.min(), test_labels.max()],
    [test_labels.min(), test_labels.max()],
    "k--",
    lw=4,
)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.show()


## === cell 45
predictionKaggle = np.clip(predictionKaggle, 0, None)

output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
