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
Given time series of breaths, predict the airway pressure in the respiratory circuit during the breath, given the time series of control inputs.

The best submissions will take lung attributes compliance and resistance into account.

## Metric
Mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored.

## Submission Format
For each `id` in the test set, you must predict a value for the `pressure` variable. The file should contain a header and have the following format:

```
id,pressure
1,20
2,23
3,24
etc.
```

## Dataset
The ventilator data used in this competition was produced using a modified [open-source ventilator](https://pvp.readthedocs.io/) connected to an [artificial bellows test lung](https://www.ingmarmed.com/product/quicklung/) via a respiratory circuit. The diagram below illustrates the setup, with the two control inputs highlighted in green and the state variable (airway pressure) to predict in blue. The first control input is a continuous variable from 0 to 100 representing the percentage the inspiratory solenoid valve is open to let air into the lung (i.e., 0 is completely closed and no air is let in and 100 is completely open). The second control input is a binary variable representing whether the exploratory valve is open (1) or closed (0) to let air out.

![Ventilator diagram](https://raw.githubusercontent.com/google/deluca-lung/main/assets/2020-10-02%20Ventilator%20diagram.svg)

Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - globally-unique time step identifier across an entire file
- `breath_id` - globally-unique time step for breaths
- `R` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change `R` by changing the diameter of the straw, with higher `R` being harder to blow.
- `C` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change `C` by changing the thickness of the balloon’s latex, with higher `C` having thinner latex and easier to blow.
- `time_step` - the actual time stamp.
- `u_in` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
- `u_out` - the control input for the exploratory solenoid valve. Either 0 or 1.
- `pressure` - the airway pressure measured in the respiratory circuit, measured in cmH2O.

# 2. Python version

3.9

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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        input/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        working/
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
```

-> data/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/ventilator-pressure-prediction/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> input/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> (stopped after 10 files for performance)

# 5. Target score

0.1953

# 6. Current score

0.33999

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.33999) has done: 'Main bottlenecks are (1) repeatedly materializing large NumPy slices for the selected fold, (2) feeding huge arrays through Keras without an input pipeline (extra copies / Python overhead), and (3) doing an unnecessary full `model.evaluate` pass before prediction. The changes below keep the exact same data, features, model, loss, epochs, and fold selection, but switch training/prediction to `tf.data` pipelines (zero-copy where possible, prefetch), remove the extra evaluation pass, and compute the fold-5 indices directly (no loop). I also enable XLA JIT (graph-level compile) and keep determinism settings unchanged to preserve accuracy semantics.'
- What this solution (achieved 0.33999) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` in cell 0 because TensorFlow is pulling in an incompatible Protobuf runtime. In this environment `protobuf==6.33.0` is installed, and TensorFlow 2.18 expects a Protobuf 5.x API; the mismatch triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during import. The two `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION*` environment variables set in the cell do not fix this incompatibility and still allow the broken import path. The minimal unblock is to force Protobuf to use the pure-Python implementation before importing TensorFlow, which avoids the incompatible C++/upb path that is causing this specific error.

Patch summary: In cell 0, change `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` from `"upb"` to `"python"` and remove/stop setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION`, keeping everything else (seeds, determinism flags, model/training logic in later cells) unchanged. This is a localized environment configuration fix to make TensorFlow import reliably with the installed Protobuf.

Updated cells: Cell 0 only (buggy cell).

Compatibility notes for cell k+1: No variables, imports, or interfaces used by cell 1 are changed; `tf`, `layers`, `KFold`, `RobustScaler`, and `rb` are still defined exactly as before after TensorFlow successfully imports.

Assumptions: The environment allows selecting the Protobuf Python implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` at runtime, and using the pure-Python protobuf is acceptable for this notebook’s execution (it only changes protobuf backend, not the model logic).'
- What this solution (achieved 0.33999) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` because the installed `protobuf==6.33.0` is incompatible with TensorFlow 2.18 in this environment, leading to `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` inside protobuf internals. The attempted workaround `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is set too late (it must be set before any protobuf/TensorFlow-related imports) and still may not help with protobuf 6.x. The minimal deterministic fix is to pin protobuf to a TensorFlow-compatible 4.x version at runtime (via pip) before importing TensorFlow, then restart the protobuf module import path in-process by importing TF after the pin. This keeps the rest of the notebook logic unchanged and unblocks execution.

Patch summary: In cell 0 only, add a small pre-import guard that checks the installed protobuf major version and, if it is >=5, installs `protobuf==4.25.3` (a widely compatible TF-supported version) before importing TensorFlow. Keep all existing environment variables and seeding logic intact, and do not change any downstream variables or interfaces.

Updated cells: (cell 0 only)

Compatibility notes for cell k+1: No variables, paths, or interfaces used by cell 1 are changed; it run exactly the same once TensorFlow imports successfully.

Assumptions: Internet/package install is available in this environment (standard for Kaggle-like runtimes); installing protobuf 4.25.3 is sufficient for TensorFlow 2.18 compatibility here.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import numpy as np
import pandas as pd

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    _pb_major = int(str(_pb_ver).split(".", 1)[0])
except Exception:
    _pb_major = None

if _pb_major is not None and _pb_major >= 5:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

rb = RobustScaler()


## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub = "../input/ventilator-pressure-prediction/sample_submission.csv"




## === cell 2
def dropCols(df, cols):
    return df.drop(columns=cols)




## === cell 3
train_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
train_data = pd.read_csv(train_path, dtype=train_dtypes)

g = train_data.groupby("breath_id", sort=False)
train_data["diff_u_in1"] = g["u_in"].diff().fillna(0).astype("float32")
train_data["u_in_cumsum"] = g["u_in"].cumsum().astype("float32")



## === cell 4
cols_2_drop = ["id", "breath_id"]



## === cell 5
train_df = dropCols(train_data, cols_2_drop)
Y = train_df.pop("pressure")



## === cell 6
rb.fit(train_df)
train_arr = rb.transform(train_df).astype(np.float32, copy=False)

del train_df, train_data, g



## === cell 7
train_arr = train_arr.reshape(-1, 80, train_arr.shape[-1])
Y_arr = Y.to_numpy(dtype=np.float32).reshape(-1, 80, 1)
del Y



## === cell 8
train_arr.shape, Y_arr.shape




## === cell 9
def build_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Bidirectional(
            layers.LSTM(300, return_sequences=True),
            input_shape=[80, train_arr.shape[-1]],
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                150,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(80, return_sequences=True, kernel_initializer="random_normal")
        )
    )
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1))

    opt = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=opt, loss=tf.keras.losses.MeanAbsoluteError(), metrics=["mae"]
    )
    return model




## === cell 10
def scheduler(epoch, lr):
    if epoch > 200 and epoch % 10 == 0:
        return lr * tf.math.exp(-0.01)
    else:
        return lr


callback1 = tf.keras.callbacks.LearningRateScheduler(scheduler)



## === cell 11
EPOCH = 100
BATCH_SIZE = 512

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
except Exception:
    tpu = None
    tpu_strategy = tf.distribute.get_strategy()

with tpu_strategy.scope():
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(train_arr, Y_arr))
    fold5_train_idx, fold5_valid_idx = splits[4]

    idx_train = tf.convert_to_tensor(fold5_train_idx, dtype=tf.int32)
    idx_valid = tf.convert_to_tensor(fold5_valid_idx, dtype=tf.int32)
    X_tf = tf.convert_to_tensor(train_arr)
    y_tf = tf.convert_to_tensor(Y_arr)

    def _gather_xy(i):
        return tf.gather(X_tf, i), tf.gather(y_tf, i)

    ds_train = (
        tf.data.Dataset.from_tensor_slices(idx_train)
        .shuffle(
            buffer_size=min(int(idx_train.shape[0]), 8192),
            seed=42,
            reshuffle_each_iteration=True,
        )
        .map(_gather_xy, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    ds_valid = (
        tf.data.Dataset.from_tensor_slices(idx_valid)
        .map(_gather_xy, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    fold_ckpt = "./AdamPressurePreModel5.h5"
    if os.path.exists(fold_ckpt):
        print(f"Found existing {fold_ckpt}; skipping training to save time.")
    else:
        pretrained_path = "./AdamPressurePreModel3.h5"
        if os.path.exists(pretrained_path):
            model = tf.keras.models.load_model(pretrained_path)
        else:
            model = build_model()

        callback0 = tf.keras.callbacks.ModelCheckpoint(
            fold_ckpt, monitor="val_loss", save_best_only=True
        )

        his = model.fit(
            ds_train,
            validation_data=ds_valid,
            epochs=EPOCH,
            shuffle=False,  # handled by ds_train.shuffle; keeps effective behavior (shuffled training).
            callbacks=[callback0, callback1],
            verbose=2,
        )
        print("\n\n")



## === cell 12
models_paths = ["./AdamPressurePreModel5.h5"]
models = [tf.keras.models.load_model(model_path) for model_path in models_paths]



## === cell 13
test_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}
test_data = pd.read_csv(test_path, dtype=test_dtypes)

gt = test_data.groupby("breath_id", sort=False)
test_data["diff_u_in1"] = gt["u_in"].diff().fillna(0).astype("float32")
test_data["u_in_cumsum"] = gt["u_in"].cumsum().astype("float32")

test_data = dropCols(test_data, cols_2_drop)
test_arr = rb.transform(test_data).astype(np.float32, copy=False)
test_arr = test_arr.reshape(-1, 80, test_arr.shape[-1])

del test_data, gt



## === cell 14
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_arr)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

p = []
for model in models:
    p.append(model.predict(test_ds, verbose=0).reshape(-1, 1))



## === cell 15
p = sum(p) / 1



## === cell 16
p.shape



## === cell 17
p



## === cell 18
submission_file = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
submission_file["pressure"] = p.reshape(
    -1,
)
submission_file.to_csv("submission.csv", index=False)
