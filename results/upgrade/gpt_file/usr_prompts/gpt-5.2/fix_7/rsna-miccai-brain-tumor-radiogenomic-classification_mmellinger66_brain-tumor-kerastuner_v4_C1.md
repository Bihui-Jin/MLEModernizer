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
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
keras-tuner==1.4.7
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
pydicom==3.0.1
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

0.4221

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.48471) has done: 'I fix the immediate runtime/import crash by avoiding the known protobuf/TensorFlow incompatibility in this environment and by importing TensorFlow only after setting safe environment flags. Then I fix DICOM loading by switching from the removed `pydicom.read_file` to `pydicom.dcmread`, plus add a small safety fallback for corrupted/missing slices so data extraction completes. Next, I correct the Keras API usage (`layers.Rescaling` instead of the removed `keras.layers.experimental.preprocessing.Rescaling`) and fix the callback monitor typo so training and tuning can run. Finally, I fix the prediction logic to output valid probabilities (not `argmax` class labels) and ensure the submission file is aligned to `sample_submission.csv` order and saved as `submission.csv` with correct columns.'
- What this solution (achieved 0.47647) has done: 'I fix the TensorFlow import crash by removing the incompatible protobuf “cpp” override and forcing the pure-Python protobuf implementation before importing TensorFlow/Keras. Then I restore missing symbols (`SEED`, `to_categorical`, etc.) by ensuring the failed import cell no longer aborts, which resolves the downstream `NameError`s. Finally, I keep your model/tuning/training logic unchanged but make the tuner import robust (`tensorflow.keras`-backed Keras) so KerasTuner can run under TF 2.18, and ensure a valid `submission.csv` is always written with correct column names and ID formatting.'
- What this solution (achieved 0.54353) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from running by making the protobuf implementation selection compatible with TF 2.18 (without relying on the broken python-only path in this environment). Then I keep your data loading, model, tuning, training, and prediction logic the same, only adding small robustness guards so training/inference won’t fail if some arrays are empty. Finally, because your current score (0.47647) is higher than the target (0.4221) and higher-is-better, I make a minimal, metric-consistent calibration nudge (light probability shrinking toward 0.5) to move the score downward toward the target band while still producing valid probabilities and a correct `submission.csv`.'
- What this solution (achieved 0.47647) has done: 'I fix the crash happening before any training by forcing a protobuf configuration that is compatible with TensorFlow 2.18 in this Kaggle image, and I also make TensorFlow import occur only after those env vars are set. Everything else (data loading, model/tuner, training, and prediction) be kept the same, including your existing probability “shrink toward 0.5” calibration (which already moves the score downward toward the 0.4221 target band). I also add a tiny fallback from `tqdm.notebook` to regular `tqdm` so the script runs in both notebook and script contexts without errors. The code still write a valid `submission.csv` with the required columns and ID formatting.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault(
    "TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
)

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import warnings

warnings.filterwarnings("ignore")

import gc, random, math, json, sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L

from sklearn.model_selection import train_test_split, KFold

SEED = 34
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

print("TF version:", tf.__version__)
print("TF_USE_LEGACY_KERAS:", os.environ.get("TF_USE_LEGACY_KERAS"))
print("GPU devices:", tf.config.list_physical_devices("GPU"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
sample_sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")

print(train.shape, test.shape, sample_sub.shape)
print("test seq_length unique:", sorted(test["seq_length"].unique().tolist()))



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1276076435.py in <cell line: 0>()
----> 1 train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
      2 test = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
      3 sample_sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")
      4 
      5 print(train.shape, test.shape, sample_sub.shape)

/usr/local/lib/python3.11/dist-packages/pandas/io/json/_json.py in read_json(path_or_buf, orient, typ, dtype, convert_axes, convert_dates, keep_default_dates, precise_float, date_unit, encoding, encoding_errors, lines, chunksize, compression, nrows, storage_options, dtype_backend, engine)
    789         convert_axes = True
    790 
--> 791     json_reader = JsonReader(
    792         path_or_buf,
    793         orient=orient,

/usr/local/lib/python3.11/dist-packages/pandas/io/json/_json.py in __init__(self, filepath_or_buffer, orient, typ, dtype, convert_axes, convert_dates, keep_default_dates, precise_float, date_unit, encoding, lines, chunksize, compression, nrows, storage_options, encoding_errors, dtype_backend, engine)
    902             self.data = filepath_or_buffer
    903         elif self.engine == "ujson":
--> 904             data = self._get_data_from_filepath(filepath_or_buffer)
    905             self.data = self._preprocess_data(data)
    906 

/usr/local/lib/python3.11/dist-packages/pandas/io/json/_json.py in _get_data_from_filepath(self, filepath_or_buffer)
    958             and not file_exists(filepath_or_buffer)
    959         ):
--> 960             raise FileNotFoundError(f"File {filepath_or_buffer} does not exist")
    961         else:
    962             warnings.warn(

FileNotFoundError: File /kaggle/input/stanford-covid-vaccine/train.json does not exist

## === cell 2
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]



## === cell 3
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
print("Vocab size:", len(token2int))




## === cell 4
def preprocess_inputs(df, cols=("sequence", "structure", "predicted_loop_type")):
    """
    Returns int32 array of shape (n_samples, seq_len, 3)
    """
    arr = (
        df.loc[:, list(cols)]
        .applymap(lambda seq: [token2int[x] for x in seq])
        .values.tolist()
    )
    x = np.array(arr, dtype=np.int32)  # (n, 3, seq_len)
    x = np.transpose(x, (0, 2, 1))  # (n, seq_len, 3)
    return x




## === cell 5
train_filt = train[train.signal_to_noise > 1].copy()

train_inputs = preprocess_inputs(train_filt)
train_labels = np.array(
    train_filt[target_cols].values.tolist(), dtype=np.float32
).transpose(
    (0, 2, 1)
)  # (n, 68, 5)

print("train_inputs:", train_inputs.shape, train_inputs.dtype)
print("train_labels:", train_labels.shape, train_labels.dtype)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3094398645.py in <cell line: 0>()
----> 1 train_filt = train[train.signal_to_noise > 1].copy()
      2 
      3 train_inputs = preprocess_inputs(train_filt)
      4 train_labels = np.array(
      5     train_filt[target_cols].values.tolist(), dtype=np.float32

NameError: name 'train' is not defined

## === cell 6
def gru_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def build_model(
    gru=False, seq_len=107, pred_len=68, dropout=0.4, embed_dim=75, hidden_dim=96
):

    inputs = tf.keras.layers.Input(shape=(seq_len, 3), dtype=tf.int32)

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        inputs
    )

    reshaped = tf.keras.layers.Reshape((seq_len, 3 * embed_dim))(embed)

    reshaped = tf.keras.layers.SpatialDropout1D(0.2)(reshaped)

    if gru:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
    else:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)

    truncated = hidden[:, :pred_len]
    out = tf.keras.layers.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    adam = tf.optimizers.Adam()
    model.compile(optimizer=adam, loss="mse")

    return model




## === cell 7
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    train_inputs, train_labels, test_size=0.1, random_state=SEED
)

print("train split:", train_inputs.shape, train_labels.shape)
print("val split:", val_inputs.shape, val_labels.shape)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2949584448.py in <cell line: 0>()
      1 train_inputs, val_inputs, train_labels, val_labels = train_test_split(
----> 2     train_inputs, train_labels, test_size=0.1, random_state=SEED
      3 )
      4 
      5 print("train split:", train_inputs.shape, train_labels.shape)

NameError: name 'train_inputs' is not defined

## === cell 8
if len(tf.config.list_physical_devices("GPU")) > 0:
    print("Training on GPU")
else:
    print("Training on CPU")

lr_callback = tf.keras.callbacks.ReduceLROnPlateau()



## === cell 9
gru = build_model(gru=True, seq_len=107, pred_len=68)
sv_gru = tf.keras.callbacks.ModelCheckpoint(
    "model_gru.weights.h5", save_weights_only=True, save_best_only=False
)

history_gru = gru.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=72,
    callbacks=[lr_callback, sv_gru],
    verbose=2,
)

print(
    f"Min training loss={min(history_gru.history['loss'])}, min validation loss={min(history_gru.history['val_loss'])}"
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/433153080.py in <cell line: 0>()
      5 
      6 history_gru = gru.fit(
----> 7     train_inputs,
      8     train_labels,
      9     validation_data=(val_inputs, val_labels),

NameError: name 'train_inputs' is not defined

## === cell 10
lstm = build_model(gru=False, seq_len=107, pred_len=68)
sv_lstm = tf.keras.callbacks.ModelCheckpoint(
    "model_lstm.weights.h5", save_weights_only=True, save_best_only=False
)

history_lstm = lstm.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=72,
    callbacks=[lr_callback, sv_lstm],
    verbose=2,
)

print(
    f"Min training loss={min(history_lstm.history['loss'])}, min validation loss={min(history_lstm.history['val_loss'])}"
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1953731830.py in <cell line: 0>()
      5 
      6 history_lstm = lstm.fit(
----> 7     train_inputs,
      8     train_labels,
      9     validation_data=(val_inputs, val_labels),

NameError: name 'train_inputs' is not defined

## === cell 11
fig, ax = plt.subplots(1, 2, figsize=(20, 6))

ax[0].plot(history_gru.history["loss"])
ax[0].plot(history_gru.history["val_loss"])
ax[0].set_title("GRU")
ax[0].legend(["train", "validation"], loc="upper right")
ax[0].set_ylabel("Loss")
ax[0].set_xlabel("Epoch")

ax[1].plot(history_lstm.history["loss"])
ax[1].plot(history_lstm.history["val_loss"])
ax[1].set_title("LSTM")
ax[1].legend(["train", "validation"], loc="upper right")
ax[1].set_ylabel("Loss")
ax[1].set_xlabel("Epoch")

plt.show()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4045418568.py in <cell line: 0>()
      1 fig, ax = plt.subplots(1, 2, figsize=(20, 6))
      2 
----> 3 ax[0].plot(history_gru.history["loss"])
      4 ax[0].plot(history_gru.history["val_loss"])
      5 ax[0].set_title("GRU")

NameError: name 'history_gru' is not defined

## === cell 12
test_df = test.copy()
assert (
    test_df["seq_length"].nunique() == 1 and int(test_df["seq_length"].iloc[0]) == 107
), "Unexpected test seq_length; update inference handling."

test_inputs = preprocess_inputs(test_df)
print("test_inputs:", test_inputs.shape)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3646382736.py in <cell line: 0>()
----> 1 test_df = test.copy()
      2 assert (
      3     test_df["seq_length"].nunique() == 1 and int(test_df["seq_length"].iloc[0]) == 107
      4 ), "Unexpected test seq_length; update inference handling."
      5 

NameError: name 'test' is not defined

## === cell 13
gru.load_weights("model_gru.weights.h5")
lstm.load_weights("model_lstm.weights.h5")

gru_preds_68 = gru.predict(test_inputs, batch_size=64, verbose=1)  # (n_test, 68, 5)
lstm_preds_68 = lstm.predict(test_inputs, batch_size=64, verbose=1)  # (n_test, 68, 5)

print("gru_preds_68:", gru_preds_68.shape, "lstm_preds_68:", lstm_preds_68.shape)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1505594205.py in <cell line: 0>()
----> 1 gru.load_weights("model_gru.weights.h5")
      2 lstm.load_weights("model_lstm.weights.h5")
      3 
      4 gru_preds_68 = gru.predict(test_inputs, batch_size=64, verbose=1)  # (n_test, 68, 5)
      5 lstm_preds_68 = lstm.predict(test_inputs, batch_size=64, verbose=1)  # (n_test, 68, 5)

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'model_gru.weights.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 14
def pad_to_107(preds_68, seq_len=107):
    n, L, c = preds_68.shape
    assert L == 68 and c == 5
    out = np.zeros((n, seq_len, c), dtype=np.float32)
    out[:, :68, :] = preds_68.astype(np.float32)
    return out


gru_preds = pad_to_107(gru_preds_68, seq_len=107)
lstm_preds = pad_to_107(lstm_preds_68, seq_len=107)

print("padded gru_preds:", gru_preds.shape, "padded lstm_preds:", lstm_preds.shape)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1092801449.py in <cell line: 0>()
      7 
      8 
----> 9 gru_preds = pad_to_107(gru_preds_68, seq_len=107)
     10 lstm_preds = pad_to_107(lstm_preds_68, seq_len=107)
     11 

NameError: name 'gru_preds_68' is not defined

## === cell 15
preds_gru = []
for i, uid in enumerate(test_df.id.values):
    single_pred = gru_preds[i]  # (107, 5)
    single_df = pd.DataFrame(single_pred, columns=target_cols)
    single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
    preds_gru.append(single_df)

preds_gru_df = pd.concat(preds_gru, axis=0, ignore_index=True)
print(preds_gru_df.head(), preds_gru_df.shape)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1764526261.py in <cell line: 0>()
      1 preds_gru = []
----> 2 for i, uid in enumerate(test_df.id.values):
      3     single_pred = gru_preds[i]  # (107, 5)
      4     single_df = pd.DataFrame(single_pred, columns=target_cols)
      5     single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]

NameError: name 'test_df' is not defined

## === cell 16
preds_lstm = []
for i, uid in enumerate(test_df.id.values):
    single_pred = lstm_preds[i]  # (107, 5)
    single_df = pd.DataFrame(single_pred, columns=target_cols)
    single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
    preds_lstm.append(single_df)

preds_lstm_df = pd.concat(preds_lstm, axis=0, ignore_index=True)
print(preds_lstm_df.head(), preds_lstm_df.shape)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1924283768.py in <cell line: 0>()
      1 preds_lstm = []
----> 2 for i, uid in enumerate(test_df.id.values):
      3     single_pred = lstm_preds[i]  # (107, 5)
      4     single_df = pd.DataFrame(single_pred, columns=target_cols)
      5     single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]

NameError: name 'test_df' is not defined

## === cell 17
blend_preds_df = pd.DataFrame()
blend_preds_df["id_seqpos"] = preds_gru_df["id_seqpos"]
blend_preds_df["reactivity"] = (
    0.4 * preds_gru_df["reactivity"] + 0.6 * preds_lstm_df["reactivity"]
)
blend_preds_df["deg_Mg_pH10"] = (
    0.4 * preds_gru_df["deg_Mg_pH10"] + 0.6 * preds_lstm_df["deg_Mg_pH10"]
)
blend_preds_df["deg_pH10"] = (
    0.4 * preds_gru_df["deg_pH10"] + 0.6 * preds_lstm_df["deg_pH10"]
)
blend_preds_df["deg_Mg_50C"] = (
    0.4 * preds_gru_df["deg_Mg_50C"] + 0.6 * preds_lstm_df["deg_Mg_50C"]
)
blend_preds_df["deg_50C"] = (
    0.4 * preds_gru_df["deg_50C"] + 0.6 * preds_lstm_df["deg_50C"]
)

print(blend_preds_df.head(), blend_preds_df.shape)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/231359592.py in <cell line: 0>()
      1 blend_preds_df = pd.DataFrame()
----> 2 blend_preds_df["id_seqpos"] = preds_gru_df["id_seqpos"]
      3 blend_preds_df["reactivity"] = (
      4     0.4 * preds_gru_df["reactivity"] + 0.6 * preds_lstm_df["reactivity"]
      5 )

NameError: name 'preds_gru_df' is not defined

## === cell 18
submission = sample_sub[["id_seqpos"]].merge(blend_preds_df, on="id_seqpos", how="left")

missing = int(submission[target_cols].isna().any(axis=1).sum())
print("Missing rows after merge:", missing)

submission[target_cols] = submission[target_cols].fillna(0.0)
submission = submission[["id_seqpos"] + target_cols]

assert (
    submission.shape[0] == sample_sub.shape[0]
), "Row count mismatch vs sample_submission"
assert (
    submission.columns.tolist() == ["id_seqpos"] + target_cols
), "Submission columns mismatch"
assert submission["id_seqpos"].is_unique, "id_seqpos must be unique"
assert set(submission["id_seqpos"]) == set(
    sample_sub["id_seqpos"]
), "id_seqpos set mismatch vs sample_submission"

print(submission.head(), submission.shape)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3651547001.py in <cell line: 0>()
----> 1 submission = sample_sub[["id_seqpos"]].merge(blend_preds_df, on="id_seqpos", how="left")
      2 
      3 missing = int(submission[target_cols].isna().any(axis=1).sum())
      4 print("Missing rows after merge:", missing)
      5 

NameError: name 'sample_sub' is not defined

## === cell 19
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print("Saved columns:", submission.columns.tolist())
print("Saved rows:", len(submission))

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/865138129.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 print("Submission saved to submission.csv")
      3 print("Saved columns:", submission.columns.tolist())
      4 print("Saved rows:", len(submission))

NameError: name 'submission' is not defined
