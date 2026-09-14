# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Given readings from several seismic sensors around a volcano, estimate how long it will be until the next eruption.

## Metric
Mean absolute error (MAE) between the predicted loss and the actual loss.

## Submission Format
For every id in the test set, you should predict the time until the next eruption. The file should contain a header and have the following format:

```
segment_id,time_to_eruption
1,1
2,2
3,3
etc.
```

## Data
### Dataset Description

#### Files
**train.csv** Metadata for the train files.

- `segment_id`: ID code for the data segment. Matches the name of the associated data file.
- `time_to_eruption`: The target value, the time until the next eruption.

**[train|test]/*.csv**: the data files. Each file contains ten minutes of logs from ten different sensors arrayed around a volcano. The readings have been normalized within each segment, in part to ensure that the readings fall within the range of int16 values. If you are using the Pandas library you may find that you still need to load the data as float32 due to the presence of some nulls.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (70 lines)
            sample_submission.csv (445 lines)
            sample_submission.csv.zip (2.8 kB)
            test.zip (514.3 MB)
            train.csv (3988 lines)
            train.csv.zip (39.1 kB)
            train.zip (4.6 GB)
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
            test/
                1003520023.csv (60002 lines)
                1004346803.csv (60002 lines)
                ... and 442 other files
                test/
            train/
                1000015382.csv (60002 lines)
                1000554676.csv (60002 lines)
                ... and 3985 other files
                train/
        input/
            description.md (70 lines)
            sample_submission.csv (445 lines)
            sample_submission.csv.zip (2.8 kB)
            test.zip (514.3 MB)
            train.csv (3988 lines)
            train.csv.zip (39.1 kB)
            train.zip (4.6 GB)
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
            test/
                1003520023.csv (60002 lines)
                1004346803.csv (60002 lines)
                ... and 442 other files
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
            train/
                1000015382.csv (60002 lines)
                1000554676.csv (60002 lines)
                ... and 3985 other files
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
        working/
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
```

-> data/predict-volcanic-eruptions-ingv-oe/sample_submission.csv has 444 rows and 2 columns.
The columns are: segment_id, time_to_eruption

-> data/predict-volcanic-eruptions-ingv-oe/test/1003520023.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1004346803.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1007996426.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1009749143.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1016956864.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1024522044.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1028325789.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> (stopped after 10 files for performance)

# 5. Target score

6646327.0

# 6. Current score

4878669.0

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4878669.0) has done: 'I remove the non-code prose and stray Markdown backticks that are currently being executed as Python, which causes the SyntaxErrors and prevents a submission from being written. I also fix the TensorFlow import crash caused by an incompatible protobuf implementation by forcing the pure-Python protobuf backend before importing TensorFlow (a standard Kaggle fix for this error). Finally, I correct the feature-dimension mismatch (your feature engineering creates 120 features, not 12×10=120 based on `n_f=12`) by computing the feature dimension from the quantiles list so the arrays allocate correctly and the pipeline runs end-to-end to write `submission.csv`.'
- What this solution (achieved 4878669.0) has done: 'The immediate blocker is the TensorFlow import crash (`MessageFactory.GetPrototype`) caused by an incompatibility between TF 2.18 and protobuf 6.x in this environment; I fix it by forcing the pure-Python protobuf runtime **and** disabling the C++ implementation before importing TensorFlow. I also add a safe fallback so if TensorFlow still cannot import, the pipeline continues with XGBoost-only training and still writes a valid `submission.csv`. These changes are execution/stability fixes and should be largely score-neutral (your current submission is already better than the target MAE, so we avoid score-improving model changes). The rest of the core logic (feature engineering, model definitions, training approach) is preserved.'
- What this solution (achieved 4878669.0) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from running by forcing the pure-Python protobuf backend early and (if available) using TensorFlow’s `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` setting before importing TF. I also make the TensorFlow branch safe by only defining/using the NN model when TF actually imported, so the script always reaches submission writing. Because your current MAE (4,878,669) is already better than the target (6,646,327) for a lower-is-better metric, I avoid any score-improving modeling changes and keep the training/inference logic intact aside from the stability fix. The pipeline run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 4878669.0) has done: 'The run is currently blocked by a TensorFlow↔protobuf incompatibility (`MessageFactory.GetPrototype`) that happens during `import tensorflow`, so the script never reaches feature building or submission writing. I fix this by forcing the pure-Python protobuf backend **before any protobuf/TensorFlow import** and by preventing TensorFlow from loading its own bundled protobuf by importing `google.protobuf` first. If TensorFlow still fails, the code safely fall back to XGBoost-only (same modeling logic as you already intended) and still write a valid `submission.csv`. These changes are execution/stability-focused and should be largely score-neutral (your current MAE is already better than the target for a lower-is-better metric).'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import google.protobuf  # noqa: F401
except Exception as _e:
    print("WARNING: protobuf pre-import failed:", repr(_e))

import numpy as np
import pandas as pd

TF_AVAILABLE = True
try:
    import tensorflow as tf

    import tensorflow.keras.layers as ly
    import tensorflow.keras.models as ml
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    ly = None
    ml = None
    EarlyStopping = None
    ReduceLROnPlateau = None
    Adam = None
    print(
        "WARNING: TensorFlow import failed; will fall back to XGBoost-only. Reason:",
        repr(e),
    )

from sklearn.model_selection import train_test_split
import xgboost as xgb

np.random.seed(3)
if TF_AVAILABLE:
    tf.random.set_seed(3)

DATA_ROOT = "/kaggle/input/predict-volcanic-eruptions-ingv-oe"
TRAIN_META_PATH = f"{DATA_ROOT}/train.csv"
TRAIN_DIR = f"{DATA_ROOT}/train"
TEST_DIR = f"{DATA_ROOT}/test"
SAMPLE_SUB_PATH = f"{DATA_ROOT}/sample_submission.csv"

print("TF available:", TF_AVAILABLE)
if TF_AVAILABLE:
    print("TensorFlow:", tf.__version__)
print("XGBoost:", xgb.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
n_sensors = 10
n_basic_stats = 7  # abs(mean), std, mean, var, min, max, median
n_features = n_sensors * (n_basic_stats + len(quantiles))  # 10 * (7 + 5) = 120

train_df = pd.read_csv(TRAIN_META_PATH)

total_data = np.empty((train_df.shape[0], n_features), dtype=np.float32)
time_ = np.empty((train_df.shape[0], 1), dtype=np.float32)

for i_, seg_ in enumerate(train_df["segment_id"].values):
    the_df = pd.read_csv(f"{TRAIN_DIR}/{seg_}.csv").fillna(0).astype(np.float32)

    feats = np.concatenate(
        (
            the_df.abs().mean().to_numpy(),
            the_df.std().to_numpy(),
            the_df.mean().to_numpy(),
            the_df.var().to_numpy(),
            the_df.min().to_numpy(),
            the_df.max().to_numpy(),
            the_df.median().to_numpy(),
            the_df.quantile(quantiles).to_numpy().reshape(1, -1)[0],
        )
    ).astype(np.float32)

    if feats.shape[0] != n_features:
        raise ValueError(
            f"Feature length mismatch: got {feats.shape[0]}, expected {n_features}"
        )

    total_data[i_, :] = feats
    time_[i_, 0] = train_df.loc[i_, "time_to_eruption"]

print("Train features shape:", total_data.shape, "Target shape:", time_.shape)



## === cell 2
sample_submission_df = pd.read_csv(SAMPLE_SUB_PATH)

total_data_test_ = np.empty(
    (sample_submission_df.shape[0], n_features), dtype=np.float32
)

for i_, seg_ in enumerate(sample_submission_df["segment_id"].values):
    the_df = pd.read_csv(f"{TEST_DIR}/{seg_}.csv").fillna(0).astype(np.float32)

    feats = np.concatenate(
        (
            the_df.abs().mean().to_numpy(),
            the_df.std().to_numpy(),
            the_df.mean().to_numpy(),
            the_df.var().to_numpy(),
            the_df.min().to_numpy(),
            the_df.max().to_numpy(),
            the_df.median().to_numpy(),
            the_df.quantile(quantiles).to_numpy().reshape(1, -1)[0],
        )
    ).astype(np.float32)

    if feats.shape[0] != n_features:
        raise ValueError(
            f"Test feature length mismatch: got {feats.shape[0]}, expected {n_features}"
        )

    total_data_test_[i_, :] = feats

print("Test features shape:", total_data_test_.shape)



## === cell 3
try:
    del the_df
except NameError:
    pass




## === cell 4
def create_my_model(input_dim: int):
    model = ml.Sequential()
    model.add(ly.Input(shape=(input_dim,)))
    model.add(ly.BatchNormalization())
    model.add(ly.Dense(1000, activation="relu"))
    model.add(ly.BatchNormalization())
    model.add(ly.Dropout(0.7))
    model.add(ly.Dense(1, activation="relu"))

    model.compile(
        optimizer=Adam(learning_rate=1.0, clipvalue=900.0), loss="mean_absolute_error"
    )
    return model




## === cell 5
X_train1, X_val, y_train1, y_val = train_test_split(
    total_data, time_, test_size=0.1, random_state=3
)

val_mae = np.inf
model = None

if TF_AVAILABLE:
    cb_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, min_lr=1e-7, patience=2, verbose=1, mode="min"
    )
    cb_early = EarlyStopping(
        monitor="val_loss", mode="min", restore_best_weights=True, patience=5, verbose=1
    )

    model = create_my_model(total_data.shape[1])
    history = model.fit(
        X_train1,
        y_train1,
        batch_size=8,
        epochs=600,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=[cb_lr, cb_early],
    )

    val_mae = model.evaluate(X_val, y_val, verbose=0)
    print("NN val MAE:", float(val_mae))
else:
    print("Skipping NN training because TensorFlow is unavailable.")



## === cell 6
xgb_params = dict(
    n_estimators=100000,
    max_depth=8,
    learning_rate=0.05,
    alpha=0.1,
    subsample=0.6,
    objective="reg:absoluteerror",  # aligns with MAE
    random_state=3,
)

try:
    model1 = xgb.XGBRegressor(tree_method="gpu_hist", **xgb_params)
    model1.fit(
        X_train1,
        y_train1.ravel(),
        eval_set=[(X_val, y_val.ravel())],
        eval_metric="mae",
        verbose=False,
        early_stopping_rounds=5,
    )
    print("XGB trained with GPU.")
except Exception as e:
    print("GPU training unavailable, falling back to CPU. Reason:", repr(e))
    model1 = xgb.XGBRegressor(tree_method="hist", **xgb_params)
    model1.fit(
        X_train1,
        y_train1.ravel(),
        eval_set=[(X_val, y_val.ravel())],
        eval_metric="mae",
        verbose=False,
        early_stopping_rounds=5,
    )
    print("XGB trained with CPU.")

pred_val = model1.predict(X_val)
xgb_val_mae = np.mean(np.abs(pred_val - y_val.ravel()))
print("XGB val MAE:", float(xgb_val_mae))



## === cell 7
use_xgb = (not TF_AVAILABLE) or (xgb_val_mae <= float(val_mae))
print("Submitting:", "XGB" if use_xgb else "NN")

if use_xgb:
    test_pred = model1.predict(total_data_test_).reshape(-1).astype(np.float32)
else:
    test_pred = (
        model.predict(total_data_test_, verbose=0).reshape(-1).astype(np.float32)
    )

submission = pd.read_csv(SAMPLE_SUB_PATH)
submission["time_to_eruption"] = test_pred

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote", submission_path, "shape:", submission.shape)
print(submission.head())
print(
    "File exists:",
    os.path.exists(submission_path),
    "size:",
    os.path.getsize(submission_path),
)
