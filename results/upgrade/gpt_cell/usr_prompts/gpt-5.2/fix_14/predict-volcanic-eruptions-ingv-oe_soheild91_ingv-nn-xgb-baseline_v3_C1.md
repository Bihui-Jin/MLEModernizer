# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

4250539.0

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4250539.0) has done: 'Diagnosis: The crash happens in cell 0 during `import tensorflow_addons as tfa`. With TensorFlow 2.18.0 and protobuf 6.33.0, `tensorflow_addons` is not compatible and triggers the protobuf `MessageFactory.GetPrototype` AttributeError at import time, preventing the notebook from running at all.

Patch summary: Remove the hard dependency on `tensorflow_addons` by making its import optional. This keeps the rest of the code unchanged and allows execution to proceed in environments where TFA is incompatible/unavailable.

Updated cells: Only cell 0 is modified; all other cells remain untouched.

Compatibility notes for cell k+1: The variable name `tfa` is still defined (set to `None` when import fails), so any later references won’t raise `NameError`. If later code requires `tfa`, it would still fail at usage time, but this patch strictly fixes the immediate import-time crash that blocks execution.

Assumptions: `tensorflow_addons` is not strictly required for the provided pipeline to run, or later cells either do not use it or can tolerate `tfa` being `None`.'
- What this solution (achieved 4250539.0) has done: 'Diagnosis: The crash happens in cell 0 during import time, before any data loading. It is caused by an incompatibility between the installed `protobuf==6.33.0` and `tensorflow_addons` (tfa), which triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` while importing tfa. Because tfa is optional here and the notebook already guards it with `try/except`, the simplest deterministic fix is to avoid importing `tensorflow_addons` entirely so the environment mismatch cannot crash the run.

Patch summary: In cell 0, replace the `try: import tensorflow_addons as tfa ...` block with a safe `tfa = None` assignment and a comment explaining why. This keeps the rest of the notebook logic unchanged while preventing the protobuf-related import crash.

Updated cells: Only cell 0 is modified.

Compatibility notes for cell k+1: `tfa` still exists and is set to `None`, matching the original intent of the fallback path when tfa is unavailable; later cells referencing `tfa` behave the same as they would when the import fails.

Assumptions: The notebook does not strictly require tensorflow_addons for core execution; if it does, it already contains a fallback path when `tfa is None` (as implied by the original guarded import).'
- What this solution (achieved 4250539.0) has done: 'The crash happens immediately when importing TensorFlow in cell 0; with `protobuf==6.33.0` this is a known incompatibility that raises `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The smallest deterministic fix is to force TensorFlow to use the pure-Python protobuf implementation before TensorFlow is imported. This avoids the incompatible C++ implementation path that triggers the error. No model/training logic is changed; only the environment variable is set early enough to take effect.'
- What this solution (achieved 4250539.0) has done: 'The crash happens in cell 0 during TensorFlow import because protobuf 6.x is incompatible with the TensorFlow build in this environment, leading to an internal AttributeError (`MessageFactory.GetPrototype`). The safest minimal fix is to force TensorFlow to use the pure-Python protobuf implementation and to avoid importing TensorFlow at module import time (which triggers the failure). We keep the rest of the logic intact by deferring TensorFlow/Keras imports until later (when/if used), without changing any modeling code. This unblocks execution so the subsequent (non-TF) feature extraction in cell 1 can run.'
- What this solution (achieved 4250539.0) has done: 'Cell 5 crashes because `ReduceLROnPlateau` and `EarlyStopping` (and also `ml/ly/tfa`) were left as `None` in cell 0 and never imported/assigned before being called. The minimal fix is to import TensorFlow/Keras and TensorFlow Addons inside cell 5 and bind `EarlyStopping` and `ReduceLROnPlateau` to the correct callback classes before they’re instantiated. This keeps the existing model/training logic unchanged and only restores the missing dependencies required for this cell to run. No other cells are modified, and the `model` object created here remains compatible with cell 6.'
- What this solution (achieved 4250539.0) has done: 'The crash happens as soon as TensorFlow Addons is imported/used: with your environment (protobuf==6.33.0), `tensorflow_addons` triggers a protobuf `MessageFactory.GetPrototype` AttributeError due to an incompatibility. The smallest safe fix is to remove the dependency on `tensorflow_addons` in this cell by providing minimal local replacements for `tfa.layers.WeightNormalization` and `tfa.optimizers.AdamW` using built-in Keras equivalents. This keeps the same model architecture and training loop intact, only swapping the failing import-dependent components for compatible implementations. No other cells are changed, and `create_my_model()` continues to work because it still references `tfa.*` names that exist after this patch.'
- What this solution (achieved 4250539.0) has done: 'Diagnosis: The crash occurs when importing/initializing TensorFlow Addons (or something that pulls in protobuf-generated code) under `protobuf==6.33.0`, which triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` due to an incompatibility between TensorFlow/TFA and protobuf 6.x. The notebook already tries to force the pure-Python protobuf implementation, but protobuf 6 has removed APIs that TFA (or its dependencies) still use. The minimal unblock is to avoid importing TensorFlow Addons entirely and instead provide a tiny local shim that preserves the exact interfaces used (`tfa.layers.WeightNormalization` and `tfa.optimizers.AdamW`) without touching the model/training loop semantics beyond removing the failing dependency.

Patch summary: In cell 5, remove/avoid the `import tensorflow_addons as tfa` attempt (which triggers the protobuf error) and unconditionally define a small `tfa` namespace that provides `WeightNormalization` as an identity wrapper and `AdamW` mapped to `tf.keras.optimizers.AdamW`. This keeps `create_my_model()` working unchanged and allows training to run.

Updated cells: Only cell 5 is modified.

Compatibility notes for cell k+1: `model` is still created and trained, and `sample_submission_df`/`total_data_test_` usage in cell 6 remains compatible. `tfa` still exists with the attributes referenced by `create_my_model()`.

Assumptions: TensorFlow 2.18’s built-in `tf.keras.optimizers.AdamW` is available (it is in TF 2.18), and the only required TFA features are the two referenced symbols (WeightNormalization and AdamW). The identity WeightNormalization wrapper is acceptable as a compatibility shim to unblock execution when TFA cannot be imported in this environment.'
- What this solution (achieved 4250539.0) has done: 'The crash happens when importing TensorFlow because it is incompatible with the installed `protobuf==6.33.0`, producing the `MessageFactory.GetPrototype` AttributeError during TF initialization. The minimal fix is to force TensorFlow to use its pure-Python protobuf implementation before importing TensorFlow in the failing cell, ensuring a compatible API surface. Since `os.environ[...]` was set in cell 0 but cell 5 runs the actual `import tensorflow as tf`, we replicate the required environment-variable setup at the top of cell 5 so it takes effect before the import. No model/training logic is changed, and all variables used by cell 6 (`model`, `total_data_test_`, `sample_submission_df`) remain available.'
- What this solution (achieved 4250539.0) has done: 'The crash happens when importing TensorFlow due to an incompatibility between TensorFlow 2.18 and the installed protobuf 6.33.0; this triggers the `MessageFactory.GetPrototype` AttributeError deep inside protobuf/TensorFlow initialization. The minimal fix is to force TensorFlow to use the pure-Python protobuf implementation *before* importing TensorFlow, and to disable the C++ protobuf implementation explicitly. This keeps the model/training logic unchanged and only adjusts environment variables needed for deterministic import behavior. No other cells are modified, and variables used by cell 6 (`model`, `sample_submission_df`, `total_data_test_`) remain defined as before.'
- What this solution (achieved 4250539.0) has done: 'The crash happens during `import tensorflow as tf` because TensorFlow 2.18 is incompatible with the installed `protobuf==6.33.0`, triggering the `MessageFactory.GetPrototype` AttributeError. The safest minimal fix in the failing cell is to force TensorFlow to use the pure-Python protobuf implementation and pin protobuf to a TF-compatible major version before importing TensorFlow. This keeps your model code, architecture, training loop, and outputs unchanged while unblocking execution. The rest of the cell remains the same so `model` is still created and trained for cell 6.'
- What this solution (achieved 4250539.0) has done: 'The crash happens because recent Keras versions require `Input(shape=...)` to receive a tuple, but the notebook passes an integer (`total_data.shape[1]`), which raises `ValueError: Cannot convert '120' to a shape.` Since `create_my_model()` is defined in an earlier cell, the smallest fix within the failing cell is to monkey-patch `ly.Input` locally to wrap any integer input into a 1D tuple before the model is created. This keeps the model architecture/training semantics identical and avoids touching other cells. No other logic is changed, and `model` remains defined for cell 6.'
- What this solution (achieved 4250539.0) has done: 'The crash happens because in TensorFlow/Keras 2.18 the `AdamW` optimizer no longer accepts the legacy keyword `lr`; it requires `learning_rate`. Since `create_my_model()` (defined in an earlier cell) is called inside this failing cell, we can fix the issue by wrapping `tf.keras.optimizers.AdamW` with a small compatibility function that converts `lr` to `learning_rate` before instantiation. This keeps the model architecture, training loop, and loss exactly the same while making the optimizer construction compatible with the installed Keras version. The patch is localized to cell 5 and preserves the `tfa.optimizers.AdamW` interface used by `create_my_model()`.'
- What this solution (achieved 4250539.0) has done: 'Diagnosis: The crash happens when constructing `tf.keras.optimizers.AdamW` via `_AdamW_compat`: in TF/Keras 2.18 the `learning_rate` argument must be a Python `float` (or schedule/callable), but the code passes an `int` (`lr=1`), which gets forwarded as `learning_rate=1` and triggers a `ValueError`. This is an API type strictness issue, not a modeling logic issue.  
Patch summary: Update `_AdamW_compat` in cell 5 to coerce an integer `learning_rate` (or `lr`) into `float` before creating the optimizer, keeping all other behavior identical. This is the minimal change needed to unblock model compilation/training and keeps the same effective learning rate value.  
Updated cells: Only cell 5 is modified as required.  
Compatibility notes for cell k+1: `model` remains a compiled and trained Keras model object, so `model.predict(total_data_test_)` in the next cell continues to work without interface changes.  
Assumptions: The intent of `lr=1` is to use learning rate value `1.0` (same magnitude), and casting to float is acceptable and semantically identical.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd

tf = None
bk = None
ly = None
ml = None
EarlyStopping = None
ReduceLROnPlateau = None

tfa = None

from sklearn.model_selection import train_test_split
import xgboost


## === cell 1
%%time
train_df=pd.read_csv('/kaggle/input/predict-volcanic-eruptions-ingv-oe/train.csv')
n_f=12
total_data=np.empty((train_df.shape[0],n_f*10))
time_=np.empty((train_df.shape[0],1))
for i_,seg_ in enumerate(train_df['segment_id']):
    the_df=pd.read_csv(f'/kaggle/input/predict-volcanic-eruptions-ingv-oe/train/{seg_}.csv').fillna(0)
    total_data[i_,:]=np.concatenate((the_df.abs().mean().to_numpy(),
                                    the_df.std().to_numpy(),
                                    the_df.mean().to_numpy(),
                                    the_df.var().to_numpy(),
                                    the_df.min().to_numpy(),
                                    the_df.max().to_numpy(),
                                    the_df.median().to_numpy(),
                                    the_df.quantile([0.1,0.25,0.5,0.75,0.9]).to_numpy().reshape(1,-1)[0]))
    time_[i_,0]=train_df.loc[i_,'time_to_eruption']


## === cell 2
%%time
sample_submission_df=pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')
n_f=12
total_data_test_=np.empty((sample_submission_df.shape[0],n_f*10))
for i_,seg_ in enumerate(sample_submission_df['segment_id']):
    the_df=pd.read_csv(f'/kaggle/input/predict-volcanic-eruptions-ingv-oe/test/{seg_}.csv').fillna(0)
    total_data_test_[i_,:]=np.concatenate((the_df.abs().mean().to_numpy(),
                                    the_df.std().to_numpy(),
                                    the_df.mean().to_numpy(),
                                    the_df.var().to_numpy(),
                                    the_df.min().to_numpy(),
                                    the_df.max().to_numpy(),
                                    the_df.median().to_numpy(),
                                    the_df.quantile([0.1,0.25,0.5,0.75,0.9]).to_numpy().reshape(1,-1)[0]))


## === cell 3
del the_df


## === cell 4
def create_my_model():
    model = ml.Sequential()
    model.add(ly.Input(total_data.shape[1]))
    model.add(ly.BatchNormalization())
    model.add(tfa.layers.WeightNormalization(ly.Dense(1000,activation='relu')))
    model.add(ly.BatchNormalization())
    model.add(ly.Dropout(0.7))
    model.add(tfa.layers.WeightNormalization(ly.Dense(1,activation='relu')))


    model.compile(optimizer=tfa.optimizers.AdamW(lr = 1, weight_decay = 1e-5, clipvalue = 900),loss='mean_absolute_error')
    return model


## === cell 5
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ["PROTOCOL_BUFFERS_PYTHON_DISABLE_CXX_IMPLEMENTATION"] = "1"

import pkgutil
import subprocess
import sys

if pkgutil.find_loader("google.protobuf") is not None:
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as _pb_ver

        if int(_pb_ver.split(".", 1)[0]) >= 5:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
            )
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
else:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

import tensorflow as tf
from tensorflow.keras import backend as bk
from tensorflow.keras import layers as ly
from tensorflow.keras import models as ml
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import types


class _IdentityWeightNormalization(ly.Wrapper):
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)

    def call(self, inputs, training=None):
        return self.layer(inputs, training=training)


def _AdamW_compat(*args, **kwargs):
    if "lr" in kwargs and "learning_rate" not in kwargs:
        kwargs["learning_rate"] = kwargs.pop("lr")
    if "learning_rate" in kwargs and isinstance(kwargs["learning_rate"], int):
        kwargs["learning_rate"] = float(kwargs["learning_rate"])
    return tf.keras.optimizers.AdamW(*args, **kwargs)


tfa = types.SimpleNamespace(
    layers=types.SimpleNamespace(WeightNormalization=_IdentityWeightNormalization),
    optimizers=types.SimpleNamespace(AdamW=_AdamW_compat),
)

_keras_input = ly.Input


def _input_compat(shape=None, *args, **kwargs):
    if isinstance(shape, int):
        shape = (shape,)
    return _keras_input(shape=shape, *args, **kwargs)


ly.Input = _input_compat

cb_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, min_lr=1e-7, patience=2, verbose=1, mode="min"
)
cb_early = EarlyStopping(
    monitor="val_loss", mode="min", restore_best_weights=True, patience=5, verbose=1
)
model = create_my_model()
X_train1, X_val, y_train1, y_val = train_test_split(
    total_data, time_, test_size=0.1, random_state=3
)
model.fit(
    X_train1,
    y_train1,
    batch_size=8,
    epochs=600,
    verbose=1,
    validation_data=(X_val, y_val),
    callbacks=[cb_lr, cb_early],
)


## === cell 6
sample_submission_df1=pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')
sample_submission_df['time_to_eruption']=model.predict(total_data_test_)
sample_submission_df.to_csv('nn_res.csv',index=False)


## === cell 7
model1 = xgboost.XGBRegressor(n_estimators=100000,tree_method='gpu_hist',max_depth=8,learning_rate=0.05,alpha=0.1,SUBSAMPLE=0.6)
X_train1, X_val, y_train1, y_val = train_test_split(total_data, time_, test_size=0.1, random_state=3)
eval_set = [(X_val, y_val)]
model1.fit(X_train1, y_train1,early_stopping_rounds=5,eval_metric='mae', eval_set=eval_set, verbose=True)


## === cell 8
sample_submission_df1=pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')
sample_submission_df1['time_to_eruption']=model1.predict(total_data_test_)[:,None]
sample_submission_df1.to_csv('xgb_res.csv',index=False)
