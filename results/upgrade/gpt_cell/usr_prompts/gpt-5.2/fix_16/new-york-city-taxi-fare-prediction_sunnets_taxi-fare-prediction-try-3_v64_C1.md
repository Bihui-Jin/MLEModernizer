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

4.60368

# 6. Current score

91.93103

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.29432) has done: 'Diagnosis: The crash happens during `import keras` in cell 1, not from your code logic. With `keras==3.8.0` on Python 3.7, Keras pulls in protobuf-related code paths that are incompatible with the protobuf version in the environment, producing `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This is an import-time dependency mismatch, so execution never reaches the data loading in cell 2. The minimal fix is to avoid importing standalone `keras` and instead use the already-installed compatible `tf_keras==2.18.0` package, keeping the same APIs (Sequential, layers, callbacks, optimizers, regularizers).

Patch summary: In cell 1 only, replace all `keras.*` imports with `tf_keras.*` equivalents so imports succeed under Python 3.7 while preserving the same model-building and training semantics. Do not change any hyperparameters, paths, or downstream variable names.

Updated cells:'
- What this solution (achieved 15.20452) has done: 'Diagnosis: The crash happens during imports in cell 1, before any model/data code runs. With the current package set, importing `google-api-python-client` pulls in a newer `protobuf` stack where `MessageFactory.GetPrototype` was removed, triggering `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` at import time. This notebook doesn’t use any Google API modules, so the safest fix is to force protobuf to use its pure-Python implementation before any such transitive import occurs. This avoids the incompatible C++/runtime path that triggers the missing attribute, without changing the notebook’s ML logic.

Patch summary: In cell 1 only, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version `2`) before other imports. Keep all existing imports and constants unchanged so downstream cells behave identically.

Updated cells: Only cell 1 is modified.

Compatibility notes for cell k+1: Cell 2 relies on `np`, `pd`, and constants like `TRAIN_PATH`/`TEST_PATH`/`DATASET_SIZE` defined in cell 1; these remain unchanged and available.

Assumptions: The error is caused by a transitive protobuf incompatibility from the environment (not from the user’s code), and switching protobuf to the pure-Python implementation is sufficient and does not materially affect numerical/model behavior.'
- What this solution (achieved 15.20917) has done: 'Diagnosis: The crash in cell 1 (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) is a known incompatibility between newer `protobuf` (v5+) and libraries that still expect the older `MessageFactory.GetPrototype` API during import-time initialization (often triggered by TensorFlow/Keras-related imports). In this notebook, the error is raised while importing `tf_keras`/Keras components, before any model code runs. The existing env vars forcing the pure-Python protobuf implementation are not sufficient to restore the missing method. The minimal fix is to monkey-patch `google.protobuf.message_factory.MessageFactory` to provide a `GetPrototype` alias (and `GetMessageClass`) when missing, before importing `tf_keras`.

Patch summary: In cell 1 only, insert a small protobuf compatibility patch right after setting the `PROTOCOL_BUFFERS_*` env vars and before importing `tf_keras`. This restores the expected method so the imports succeed, without changing any modeling/training logic or downstream variables.

Updated cells: Only cell 1 is modified.

Compatibility notes for cell k+1: All variables/constants imported/defined in cell 1 (`np`, `pd`, `TRAIN_PATH`, `TEST_PATH`, etc.) remain unchanged, so cell 2 run identically.

Assumptions: The environment has a `protobuf` version where `MessageFactory.GetPrototype` was removed (protobuf>=5), and adding a backward-compatible alias is sufficient to unblock `tf_keras` imports.'
- What this solution (achieved 15.31738) has done: 'Diagnosis: Cell 1 crashes while trying to monkey‑patch `google.protobuf.message_factory.MessageFactory` by accessing `MessageFactory.GetPrototype`. In this environment, `message_factory.MessageFactory` is not the expected class (or is a module-level factory instance), so the attribute lookup itself raises `AttributeError` before the guarded patching logic can proceed. This protobuf patch is not required for the rest of the notebook (which uses `numpy/pandas/tf_keras`), so the safest minimal fix is to remove the fragile attribute access and replace it with a no-op guarded block that never touches missing attributes.

Patch summary: In cell 1 only, replace the protobuf monkey‑patch section with a fully guarded/optional compatibility block that catches all exceptions and does not access `MessageFactory.GetPrototype` directly. All imports/constants and downstream variables remain unchanged.

Updated cells: Only cell 1 is changed.

Compatibility notes for cell k+1: Cell 2 relies on `np`, `pd`, and the path/constants defined in cell 1; these remain identical, so `pd.read_csv(TRAIN_PATH, ...)` work as before.

Assumptions: The protobuf monkey‑patch is not required for subsequent cells in this notebook; it was defensive boilerplate. If protobuf is needed elsewhere, the guarded block still attempts a safe patch only when the expected attributes exist.'
- What this solution (achieved 15.24448) has done: 'Diagnosis: The crash happens inside the protobuf compatibility shim in cell 1: it tries to access/patch `MessageFactory.GetPrototype`, but in the installed `google.protobuf` version `MessageFactory` is an instance without that attribute (and also may not expose `GetMessageClass` the way the shim expects). This makes the shim itself raise `AttributeError` before the rest of the notebook can run. The fix is to make the shim robust by importing `MessageFactory` from the correct module (`google.protobuf.message_factory`) and only applying patches if the attributes exist on the class, otherwise safely skipping.

Patch summary: Modify only the protobuf shim portion in cell 1 to avoid accessing missing attributes and to patch class methods only when present. Keep all later imports/variables unchanged so downstream cells continue to work identically.

Updated cells: Only cell 1 is changed.

Compatibility notes for cell k+1: All names imported in cell 1 (`np`, `pd`, `plt`, sklearn, `tf_keras`, and constants like `TRAIN_PATH`) remain defined exactly as before, so cell 2 can run unchanged.

Assumptions: The intent of the shim is only to prevent protobuf-related incompatibilities during imports; if the expected protobuf API is absent, skipping the patch is safe and preferable to crashing.'
- What this solution (achieved 15.32238) has done: 'Diagnosis: The crash happens in cell 1 while trying to monkey‑patch `google.protobuf.message_factory.MessageFactory` to provide `GetPrototype`. In this environment, `MessageFactory` is being treated as an *instance* (or an object without those methods), so the patch attempts to access/modify attributes on the wrong object type and triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. We only need to make this compatibility shim safe by patching the actual class when possible and otherwise skipping without raising.

Patch summary: Update the protobuf shim in cell 1 to (a) import `MessageFactory` safely, (b) detect whether it is a class vs instance, (c) patch the class (or the instance’s class) only when the counterpart method exists, and (d) catch/ignore any unexpected protobuf layouts. This keeps the intent (avoid protobuf incompatibilities) but prevents the AttributeError so the rest of the notebook can run unchanged.

Updated cells: Cell 1 only (minimal localized change).

Compatibility notes for cell k+1: All imports and constants defined in cell 1 remain the same (`np`, `pd`, `TRAIN_PATH`, `TEST_PATH`, etc.), so cell 2 continues to work without any interface/variable changes.

Assumptions: The protobuf shim is not required for core model logic, only to avoid certain protobuf/TF/Keras import issues; therefore safely skipping the patch when not applicable is acceptable and deterministic.'
- What this solution (achieved 15.22562) has done: 'Diagnosis: The crash happens in cell 1 during the protobuf compatibility shim: it tries to add `GetPrototype` onto a `MessageFactory` instance/class via `google.protobuf.message_factory`. In this environment, `MessageFactory` is an object where that attribute assignment path does not work as written, causing `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` at import time. The simplest safe fix is to avoid touching `MessageFactory` entirely and instead patch the symbol that TensorFlow/Keras commonly expects: `google.protobuf.message_factory.GetPrototype` (module-level) when only `GetMessageClass` exists. This keeps the intent (protobuf API-compat) but avoids the failing attribute access.

Patch summary: Update the protobuf shim in cell 1 to conditionally define `message_factory.GetPrototype` at the module level using `GetMessageClass` when missing, and remove the fragile instance/class attribute manipulation that triggers the error. No other imports, constants, or downstream interfaces are changed.

Updated cells: cell 1 only.

Compatibility notes for cell k+1: All variables and imports defined in cell 1 (`np`, `pd`, `plt`, `TRAIN_PATH`, etc.) remain unchanged, so cell 2 continues to run identically.

Assumptions: The protobuf incompatibility is what triggered the import-time crash and TensorFlow/Keras only requires the presence of `google.protobuf.message_factory.GetPrototype` (or equivalent) rather than instance-level mutation of `MessageFactory`.'
- What this solution (achieved 15.22831) has done: 'Diagnosis: The crash happens in cell 1 while importing packages, before any data/model code runs. Recent `protobuf` versions removed/changed `MessageFactory.GetPrototype`, and `tf_keras` (and/or one of its transitive deps like TensorBoard) still expects it to exist, leading to `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The existing shim patches only the module-level `google.protobuf.message_factory` function, but the failing code is calling the *instance method* on a `MessageFactory` object. We need to patch the `MessageFactory` class itself to provide `GetPrototype` by delegating to `GetMessageClass` when available.

Patch summary: In cell 1 only, extend the protobuf compatibility shim to monkey-patch `google.protobuf.message_factory.MessageFactory.GetPrototype` (class method) when missing, mapping it to `GetMessageClass`. This resolves the import-time crash without changing any training/data logic or downstream interfaces.

Updated cells: (cell 1 only)

Compatibility notes for cell k+1: All variables/constants/imports defined in cell 1 remain the same (`np`, `pd`, model/layers/callbacks, paths, hyperparameters). The change only affects protobuf internals during import and does not alter any interfaces used by cell 2.

Assumptions: `google.protobuf.message_factory.MessageFactory` exists in this environment and provides `GetMessageClass` when `GetPrototype` is missing (typical for newer protobuf releases).'
- What this solution (achieved 15.32002) has done: 'Diagnosis: The crash happens during the protobuf compatibility monkey-patch in cell 1: it assumes `google.protobuf.message_factory.MessageFactory` exists and can be modified, but in your protobuf version `message_factory.MessageFactory` is already an *instance* (or otherwise not the expected class), so `_MF.GetPrototype = ...` triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The fix is to avoid touching this object unless it’s actually a class/type that can be patched, and otherwise safely skip the patch. This keeps the intent (compatibility with older code expecting `GetPrototype`) without breaking on newer protobuf internals.

Patch summary: In cell 1 only, adjust the protobuf patch block to (a) retrieve the `MessageFactory` attribute if present, (b) patch only when it’s a class/type (not an instance), and (c) otherwise no-op. All imports/constants and downstream variables remain unchanged.

Updated cells: cell 1 only.

Compatibility notes for cell k+1: Cell 2 relies on `TRAIN_PATH`, `TEST_PATH`, `SUBMISSION_NAME`, and the imported packages (`numpy`, `pandas`, etc.); these remain defined exactly as before. The patch does not change any model/training semantics—only prevents the early protobuf patch crash so execution can proceed to cell 2.

Assumptions: `google.protobuf` is installed and importable; the environment’s protobuf implementation may not expose a patchable `MessageFactory` class, so skipping the patch is acceptable if TensorFlow/Keras imports succeed.'
- What this solution (achieved 15.23587) has done: 'The crash happens while importing TensorFlow/Keras because the protobuf version in this environment no longer provides `MessageFactory.GetPrototype`, and the current monkey-patch tries to call `self.GetMessageClass(...)` on a `MessageFactory` instance that doesn’t have it. The safest minimal fix is to patch `MessageFactory.GetPrototype` to use module-level `message_factory.GetMessageClass(descriptor)` when available, and otherwise fall back to a no-op (so imports proceed) rather than raising `AttributeError`. This keeps the intent of the compatibility shim but avoids calling missing instance methods. No other logic (paths, model code, training settings) is changed.'
- What this solution (achieved 15.06479) has done: 'Diagnosis: The crash occurs inside `remove_datapoints_from_water()` when calling `plt.imread()` with an HTTP URL. Newer Matplotlib versions no longer allow passing a URL string directly to `imread`, raising a `ValueError` and halting `clean()`. The fix is to open the URL as a file-like object and decode it via Pillow (PIL), then convert to a NumPy array—exactly as the error message suggests—so the mask creation logic remains unchanged.

Patch summary: In cell 9, replace `plt.imread(url)` with `np.array(PIL.Image.open(urllib.request.urlopen(url)))` (and keep the `[:,:,0] > 0.9` logic intact). Add the minimal imports (`urllib.request` and `PIL.Image`) locally inside the function to avoid touching earlier cells.

Updated cells: cell 9 only.

Compatibility notes for cell k+1: `clean()` still returns the same filtered DataFrame type/structure, so downstream cells (including cell 10’s `train_df.describe()`) remain compatible.

Assumptions: Pillow is available via Matplotlib’s dependencies in this environment (common on Kaggle/Colab); if not, this would require installing it, but we cannot do that here.'
- What this solution (achieved 15.33668) has done: 'Diagnosis: Cell 9 fails inside `clean()` when it calls `remove_datapoints_from_water()`, which downloads a NYC land mask from `https://aiblog.nl/...png`. That URL now returns HTTP 404, so `urllib.request.urlopen()` raises `HTTPError` and the whole cleaning step crashes. Since the mask is an optional external resource and Kaggle environments can’t rely on it being available, the safest fix is to make `remove_datapoints_from_water()` resilient: if the download fails, return the input `df` unchanged. This preserves the notebook’s core logic and keeps `clean()` callable without changing other cells.

Patch summary: In cell 9 only, wrap the mask download/load in a `try/except` and add bounds checking for computed pixel indices; on any failure, fall back to returning `df` unchanged so execution continues deterministically.

Updated cells: Only cell 9 is modified below.

Compatibility notes for cell k+1: `train_df` and `test_df` are still produced as DataFrames (same variable names), so `train_df.describe()` in cell 10 works unchanged.

Assumptions: Network access and/or the external mask URL may be unavailable or changed; in that case skipping the water-mask filter is acceptable to unblock execution while keeping the rest of the pipeline intact.'
- What this solution (achieved 15.29492) has done: 'Diagnosis: The crash in cell 36 comes from calling `optimizers.adam(...)`, but in `tf_keras` (and modern Keras/TensorFlow APIs) the optimizer class is `optimizers.Adam` (capital “A”), not a lowercase factory function. As a result, `tf_keras.optimizers` has no attribute `adam`, raising `AttributeError`.  
Patch summary: In cell 36 only, replace `optimizers.adam(lr=...)` with `optimizers.Adam(learning_rate=...)` (keeping the same learning rate value) so model compilation and training proceed.  
Updated cells: Only cell 36 is modified below.  
Compatibility notes for cell k+1: The variables `model` and `history` are still created with the same meanings, so cell 37 (model visualization) remains compatible.  
Assumptions: `tf_keras.optimizers.Adam` is available in the provided `tf_keras==2.18.0` and accepts `learning_rate` (standard API).'
- What this solution (achieved 374.1933) has done: 'The crash happens because the custom `rmse` metric imports `backend` from the standalone `keras` (Keras 3), but the model is built/trained with `tf_keras`; in this environment `keras.backend` no longer exposes `sqrt/mean/square` as expected, causing `AttributeError`. The minimal fix is to implement `rmse` using the `tf_keras.backend` module so it matches the training stack. This keeps the same metric definition and preserves model/training semantics. No other logic is changed.'
- What this solution (achieved 91.93103) has done: 'The crash is caused by an outdated import path: in Keras 3, `keras.utils.vis_utils` no longer exists, so importing `model_to_dot` from there raises `ModuleNotFoundError`. To keep the same behavior (rendering the model graph via Graphviz) without changing any training/evaluation logic, update cell 37 to import `model_to_dot` from its current Keras 3 location, with a safe fallback to `tf_keras.utils` if needed. The rest of the cell remains the same so downstream variables and output semantics are unchanged. This patch is fully localized to the failing cell and only addresses the import error.'

# 9. Code solution

## === cell 0
def clean(df):
    
    print(' Old size: %d' % len(df))
    df = df.dropna(how = 'any', axis = 'rows')
    print(' New size after dropna: %d' % len(df))  
    
    df = df[(df['dropoff_longitude'] != df['pickup_longitude']) & (df['dropoff_latitude'] != df['pickup_latitude'])]
    print(' New size after removing same long lat: %d' % len(df))                 
    
    df = df[(df['dropoff_longitude'] != 0) & (df['pickup_longitude'] != 0) & (df['dropoff_latitude'] != 0) & (df['pickup_latitude'] != 0)] 
    print(' New size after removing 0 long lat: %d' % len(df))                          
    MinMax = (-74.5, -72.8, 40.5, 41.8)

    df = df[(MinMax[0] <= df['pickup_longitude']) & (df['pickup_longitude'] <= MinMax[1])]
    df = df[(MinMax[0] <= df['dropoff_longitude']) & (df['dropoff_longitude'] <= MinMax[1])]
    df = df[(MinMax[2] <= df['pickup_latitude']) & (df['pickup_latitude'] <= MinMax[3])]
    df = df[(MinMax[2] <= df['dropoff_latitude']) & (df['dropoff_latitude'] <= MinMax[3])]
       

    print(' New size after NYC lang lot: %d' % len(df))         
    
    df = df[(df['pickup_latitude'] != 0)]
    df = df[(df['pickup_latitude'] != 0)]
    df = df[(df['dropoff_longitude']!= 0)]
    df = df[(df['dropoff_latitude'] != 0)]
    
    print(' New size after lang lot > 0: %d' % len(df))         

    df = df[((df['pickup_latitude'] - df['dropoff_latitude']).abs() > 0.001)]
    df = df[((df['pickup_longitude'] - df['dropoff_longitude']).abs() > 0.001)]
    
    print(' New size after lang - lot > 0.001: %d' % len(df))         
    
    print(' New size after only NYC: %d' % len(df)) 
    df = df[(0 < df['fare_amount']) & (df['fare_amount'] <= 50)]
    
    print(' New size after removing outliers: %d' % len(df)) 
    
    df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]
    print(' New size after removing 6=>passenger_count > 0 : %d' % len(df)) 
    
    
    
    
    nyc_coord = (40.7141667,-74.0063889) 
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892,-74.0445) # Statue of Liberty
    
             

    df = df[(nyc_coord[1] != df['pickup_longitude']) & (df['pickup_latitude'] != nyc_coord[0])]
    df = df[(nyc_coord[1] != df['dropoff_longitude']) & (df['dropoff_latitude'] != nyc_coord[0])]
    
    print(' New size after NY airport: %d' % len(df))
    
    df = df[(fk_coord[1] != df['pickup_longitude']) & (df['pickup_latitude'] != fk_coord[0])]
    df = df[(fk_coord[1] != df['dropoff_longitude']) & (df['dropoff_latitude'] != fk_coord[0])]
    
    print(' New size after jfk airport: %d' % len(df))
    
    df = df[(ewr_coord[1] != df['pickup_longitude']) & (df['pickup_latitude'] != ewr_coord[0])]
    df = df[(ewr_coord[1] != df['dropoff_longitude']) & (df['dropoff_latitude'] != ewr_coord[0])]
    
    print(' New size after ewr airport: %d' % len(df))
    df = df[(lga_coord[1] != df['pickup_longitude']) & (df['pickup_latitude'] != lga_coord[0])]
    df = df[(lga_coord[1] != df['dropoff_longitude']) & (df['dropoff_latitude'] != lga_coord[0])]
    

    print(' New size after lgr airport: %d' % len(df))
             
    df = df[(sol_coord[1] != df['pickup_longitude']) & (df['pickup_latitude'] != sol_coord[0])]
    df = df[(sol_coord[1] != df['dropoff_longitude']) & (df['dropoff_latitude'] != sol_coord[0])]
    

    print(' New size after sol removed: %d' % len(df))             
    
            
    print('Old size: %d' % len(df))
    df = remove_datapoints_from_water(df)
    print('New size: %d' % len(df))
    
        
    print(' New size: %d' % len(df))
    
    return df

def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx*(longitude - BB[0])/(BB[1]-BB[0])).astype('int'), \
               (dy - dy*(latitude - BB[2])/(BB[3]-BB[2])).astype('int')

    BB = (-74.5, -72.8, 40.5, 41.8)
    
    nyc_mask = plt.imread('https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png')[:,:,0] > 0.9
    
    pickup_x, pickup_y = lonlat_to_xy(df.pickup_longitude, df.pickup_latitude, 
                                      nyc_mask.shape[1], nyc_mask.shape[0], BB)
    dropoff_x, dropoff_y = lonlat_to_xy(df.dropoff_longitude, df.dropoff_latitude, 
                                      nyc_mask.shape[1], nyc_mask.shape[0], BB)    
    idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]
    
    return df[idx]
    
def late_night (row):
     if (row['hour'] <= 3) or (row['hour'] >= 0):
        return 1
     else:
        return 0


def night (row):
    if ((row['hour'] > 20) and (row['hour'] > 0)) and (row['weekday'] < 5):
        return 1
    else:
        return 0
    
def rush_hour (row):
    if ((row['hour'] <= 20) and (row['hour'] >= 16)) and (row['weekday'] < 5):
        return 1
    else:
        return 0   
    

       
    
    
def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    df['pickup_datetime'] =  pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S %Z')
    df['year'] = df['pickup_datetime'].apply(lambda x: x.year)
    df['month'] = df['pickup_datetime'].apply(lambda x: x.month)
    df['day'] = df['pickup_datetime'].apply(lambda x: x.day)
    df['hour'] = df['pickup_datetime'].apply(lambda x: x.hour)
    df['weekday'] = df['pickup_datetime'].apply(lambda x: x.weekday())
    df['pickup_datetime'] =  df['pickup_datetime'].apply(lambda x: str(x))
    df['night'] = df.apply (lambda x: night(x), axis=1)
    df['late_night'] = df.apply (lambda x: late_night(x), axis=1)
    df['rush_hour'] = df.apply (lambda x: rush_hour(x), axis=1)
    
    return df


def add_coordinate_features(df):
    lat1 = df['pickup_latitude']
    lat2 = df['dropoff_latitude']
    lon1 = df['pickup_longitude']
    lon2 = df['dropoff_longitude']
    
    df['latdiff'] = (lat1 - lat2).abs()
    df['londiff'] = (lon1 - lon2).abs()
    
    
    return df


def add_distances_features(df):
    
    lat1 = df['pickup_latitude']
    lat2 = df['dropoff_latitude']
    lon1 = df['pickup_longitude']
    lon2 = df['dropoff_longitude']
    
    df['manhattan'] = manhattan(lat1, lon1, lat2, lon2)
    df['distance'] = np.sqrt(np.abs(df['pickup_longitude']-df['dropoff_longitude'])**2 + np.abs(df['pickup_latitude']-df['dropoff_latitude'])**2)
    
  
    return df

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...
def distanceP(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...
    
def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column]
    df[[id_column, prediction_column]].to_csv((file_name), index=False)
    print('Output complete')
    
    
def plot_loss_accuracy_rmse(history):
    
    plt.figure(figsize=(20,10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    
    plt.figure(figsize=(20,10))
    plt.plot(history.history['rmse'])
    plt.plot(history.history['val_rmse'])
    plt.title('Model rmse')
    plt.ylabel('rmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    


## === cell 1
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf import message_factory as _message_factory  # noqa: F401

    _MF = getattr(_message_factory, "MessageFactory", None)

    if _MF is not None and isinstance(_MF, type):
        if not hasattr(_MF, "GetPrototype"):
            if hasattr(_message_factory, "GetMessageClass"):

                def _GetPrototype(self, descriptor):
                    return _message_factory.GetMessageClass(descriptor)

                _MF.GetPrototype = _GetPrototype  # type: ignore[attr-defined]
            else:
                def _GetPrototype(self, descriptor):
                    raise AttributeError(
                        "GetPrototype is not available in this protobuf version"
                    )

                _MF.GetPrototype = _GetPrototype  # type: ignore[attr-defined]

    if not hasattr(_message_factory, "GetPrototype") and hasattr(
        _message_factory, "GetMessageClass"
    ):
        _message_factory.GetPrototype = _message_factory.GetMessageClass
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
datatypes = {'key': 'str', 
              'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

trainKaggle = pd.read_csv(TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes, usecols=[1,2,3,4,5,6,7])
testKaggle = pd.read_csv(TEST_PATH)    


## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000]


## === cell 5
print('testKaggle Size %d' % len(testKaggle))
print('train_df Size %d' % len(train_df))
print('test_df Size %d' % len(test_df))


## === cell 6
train_df.describe()


## === cell 7
test_df.describe()


## === cell 9


def remove_datapoints_from_water(df):
    import urllib.request
    from PIL import Image

    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"
    try:
        nyc_mask_img = np.array(Image.open(urllib.request.urlopen(url)))
        nyc_mask = nyc_mask_img[:, :, 0] > 0.9
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

    in_bounds = (
        (pickup_x >= 0)
        & (pickup_x < nyc_mask.shape[1])
        & (pickup_y >= 0)
        & (pickup_y < nyc_mask.shape[0])
        & (dropoff_x >= 0)
        & (dropoff_x < nyc_mask.shape[1])
        & (dropoff_y >= 0)
        & (dropoff_y < nyc_mask.shape[0])
    )

    idx = np.zeros(len(df), dtype=bool)
    if in_bounds.any():
        idx[in_bounds.values if hasattr(in_bounds, "values") else in_bounds] = (
            nyc_mask[pickup_y[in_bounds], pickup_x[in_bounds]]
            & nyc_mask[dropoff_y[in_bounds], dropoff_x[in_bounds]]
        )
    return df[idx]


print("train_df clean")
train_df = clean(train_df)
test_df = clean(test_df)


## === cell 10
train_df.describe()


## === cell 11
print('train_df add_time_features')
train_df = add_time_features(train_df)
print('test_df add_time_features')
test_df = add_time_features(test_df)
print('testKaggle add_time_features')
testKaggle = add_time_features(testKaggle)


   


## === cell 12
train_df.describe()


## === cell 13
print('train_df add_coordinate_features')
add_coordinate_features(train_df)
print('test_df add_coordinate_features Disabled!')
add_coordinate_features(test_df)
print('testKaggle add_coordinate_features')
add_coordinate_features(testKaggle)


## === cell 14
train_df.describe()


## === cell 15
print('train_df add_distances_features')
train_df = add_distances_features(train_df)
print('test_df add_distances_features')
test_df = add_distances_features(test_df)
print('testKaggle add_distances_features')
testKaggle = add_distances_features(testKaggle)


print('Done with Adding features')


## === cell 16
train_df.describe()


## === cell 17
plot = train_df.iloc[:2000].plot.scatter('latdiff', 'londiff')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'passenger_count')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'year')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'month')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'day')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'hour')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'weekday')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'night')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'late_night')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'rush_hour')


## === cell 19
dropped_columns = ['passenger_count', 'pickup_datetime'] #'pickup_latitude','pickup_longitude', 'dropoff_longitude', 'dropoff_latitude']

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ['key'], axis=1)

print('Done with dropped_columns')


## === cell 20
train_df.shape


## === cell 21
train_df.describe()


## === cell 22
test_df.describe()


## === cell 23
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)


## === cell 24
train_df.describe()


## === cell 25
validation_df.describe()


## === cell 26
train_labels = train_df['fare_amount'].values
validation_labels = validation_df['fare_amount'].values
test_labels = test_df['fare_amount'].values

train_df = train_df.drop(['fare_amount'], axis=1)
validation_df = validation_df.drop(['fare_amount'], axis=1)
test_df = test_df.drop(['fare_amount'], axis=1)

print('Done with Labels')


## === cell 27
test_labels


## === cell 30
train_df.describe()


## === cell 31
validation_df.describe()


## === cell 32
test_df.describe()


## === cell 33
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)


## === cell 34
test_scaled


## === cell 35
from keras import backend
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


## === cell 36
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


## === cell 37
from IPython.display import SVG

try:
    from keras.utils import model_to_dot
except Exception:
    from tf_keras.utils import model_to_dot

SVG(model_to_dot(model).create(prog="dot", format="svg"))


## === cell 38
plot_loss_accuracy_rmse(history)


## === cell 39
score = model.evaluate(train_df_scaled, train_labels, verbose=1)
print(score)
print('train mean_squared_error:', score[0])
print('train mae:', score[1])
print('train accuracy:', score[2])
print('train rmse:', score[3])
print('train mse:', score[4])


## === cell 40
score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
print(score)
print('Validation mean_squared_error:', score[0])
print('Validation mae:', score[1])
print('Validation accuracy:', score[2])
print('Validation rmse:', score[3])
print('Validation mse:', score[4])


## === cell 41
score = model.evaluate(test_scaled, test_labels, verbose=1)
print(score)
print('Test mean_squared_error:', score[0])
print('Test mae:', score[1])
print('Test accuracy:', score[2])
print('Test rmse:', score[3])
print('Test mse:', score[4])


## === cell 42
validation_predictions = model.predict(validation_df_scaled).flatten()

plt.scatter(validation_labels, validation_predictions)
plt.xlabel('True Values [10$]')
plt.ylabel('Predictions [10$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])


## === cell 43
test_predictions = model.predict(test_scaled).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [10$]')
plt.ylabel('Predictions [10$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])


## === cell 44
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)


## === cell 45
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


## === cell 46
print(np.argmax(test_predictions))
print(test_predictions[np.argmax(test_predictions)])
print(test_labels[np.argmax(test_predictions)])
test_df.iloc[np.argmax(test_predictions)]


## === cell 47
print(np.argmin(test_predictions))
print(test_predictions[np.argmin(test_predictions)])
print(test_labels[np.argmin(test_predictions)])
test_df.iloc[np.argmin(test_predictions)]


## === cell 48
fig, ax = plt.subplots()
ax.scatter(test_labels, test_predictions)
ax.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


## === cell 50

output_submission(testKaggle, predictionKaggle, 'key', 'fare_amount', SUBMISSION_NAME)
