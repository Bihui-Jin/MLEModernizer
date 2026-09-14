# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

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
tqdm==4.67.1

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        input/
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        working/
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
```

-> data/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/stanford-covid-vaccine/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> data/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> input/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")

import pandas as pd, numpy as np
import math, json, gc, random, os, sys
from matplotlib import pyplot as plt
from tqdm import tqdm

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

try:
    import google.protobuf  # noqa: F401
    from packaging import version as _pkg_version  # type: ignore
    import google.protobuf as _gp

    if _pkg_version.parse(_gp.__version__) >= _pkg_version.parse("5.0.0"):
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        import importlib

        importlib.invalidate_caches()
        for _m in list(sys.modules):
            if _m.startswith("google.protobuf"):
                sys.modules.pop(_m, None)
except Exception:
    pass

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L

try:
    import tensorflow_addons as tfa  # type: ignore
except Exception:
    tfa = None

from sklearn.model_selection import train_test_split, KFold

SEED = 34
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## === cell 1
train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
sample_sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")



## === cell 2
print(train.shape)
if not train.isnull().values.any():
    print("No missing values")
train.head()



## === cell 3
print(test.shape)
if not test.isnull().values.any():
    print("No missing values")
test.head()



## === cell 4
print(sample_sub.shape)
if not sample_sub.isnull().values.any():
    print("No missing values")
sample_sub.head()



## === cell 5
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]



## === cell 6
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}




## === cell 7
def preprocess_inputs(df, cols=["sequence", "structure", "predicted_loop_type"]):
    return np.transpose(
        np.array(
            df[cols].applymap(lambda seq: [token2int[x] for x in seq]).values.tolist()
        ),
        (0, 2, 1),
    )




## === cell 8
train_inputs = preprocess_inputs(train)
train_labels = np.array(train[target_cols].values.tolist()).transpose((0, 2, 1))




## === cell 9
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
    gru=False, seq_len=107, pred_len=68, dropout=0.5, embed_dim=75, hidden_dim=128
):
    inputs = tf.keras.layers.Input(shape=(seq_len, 3))

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        inputs
    )
    reshaped = tf.keras.layers.Reshape(
        (embed.shape[1], embed.shape[2] * embed.shape[3])
    )(embed)

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
    model.compile(optimizer=tf.optimizers.Adam(), loss="mse")
    return model




## === cell 10
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    train_inputs, train_labels, test_size=0.1, random_state=SEED
)



## === cell 11
if tf.config.list_physical_devices("GPU") is not None:
    print("Training on GPU")



## === cell 12
lr_callback = tf.keras.callbacks.ReduceLROnPlateau()



## === cell 13
gru = build_model(gru=True)

sv_gru = tf.keras.callbacks.ModelCheckpoint(
    "model_gru.weights.h5", save_weights_only=True, save_best_only=False
)

history_gru = gru.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=75,
    callbacks=[lr_callback, sv_gru],
    verbose=2,
)

print(
    f"Min training loss={min(history_gru.history['loss'])}, min validation loss={min(history_gru.history['val_loss'])}"
)


## === cell 14
lstm = build_model(gru=False)

sv_lstm = tf.keras.callbacks.ModelCheckpoint(
    "model_lstm.weights.h5", save_weights_only=True, save_best_only=False
)

history_lstm = lstm.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=75,
    callbacks=[lr_callback, sv_lstm],
    verbose=2,
)

print(
    f"Min training loss={min(history_lstm.history['loss'])}, min validation loss={min(history_lstm.history['val_loss'])}"
)


## === cell 15
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].plot(history_gru.history["loss"])
ax[0].plot(history_gru.history["val_loss"])

ax[1].plot(history_lstm.history["loss"])
ax[1].plot(history_lstm.history["val_loss"])

ax[0].set_title("GRU")
ax[1].set_title("LSTM")

ax[0].legend(["train", "validation"], loc="upper right")
ax[1].legend(["train", "validation"], loc="upper right")

ax[0].set_ylabel("Loss")
ax[0].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax[1].set_xlabel("Epoch")



## === cell 16
seq_lens = sorted(test["seq_length"].dropna().astype(int).unique().tolist())
if len(seq_lens) == 0:
    raise ValueError("No seq_length values found in test data.")
test_len = int(seq_lens[-1])

test_df = test.copy()
test_df["sequence"] = (
    test_df["sequence"].astype(str).map(lambda s, __l=test_len: s[:__l])
)
test_df["structure"] = (
    test_df["structure"].astype(str).map(lambda s, __l=test_len: s[:__l])
)
test_df["predicted_loop_type"] = (
    test_df["predicted_loop_type"].astype(str).map(lambda s, __l=test_len: s[:__l])
)

test_inputs = preprocess_inputs(test_df)
test_len = int(test_inputs.shape[1])
print("Test inputs shape:", test_inputs.shape)



## === cell 17
pred_len = int(test_df["seq_scored"].iloc[0]) if "seq_scored" in test_df.columns else 68
pred_len = int(pred_len)

gru_pred_model = build_model(gru=True, seq_len=test_len, pred_len=pred_len)
lstm_pred_model = build_model(gru=False, seq_len=test_len, pred_len=pred_len)

gru_pred_model.load_weights("model_gru.h5")
lstm_pred_model.load_weights("model_lstm.h5")

gru_test_preds = gru_pred_model.predict(test_inputs, batch_size=64, verbose=0)
lstm_test_preds = lstm_pred_model.predict(test_inputs, batch_size=64, verbose=0)

blend_test_preds = 0.5 * gru_test_preds + 0.5 * lstm_test_preds  # (n, pred_len, 5)



## --- ERROR in cell 17, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_10/1645800972.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      7[0m [0mlstm_pred_model[0m [0;34m=[0m [0mbuild_model[0m[0;34m([0m[0mgru[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m [0mseq_len[0m[0;34m=[0m[0mtest_len[0m[0;34m,[0m [0mpred_len[0m[0;34m=[0m[0mpred_len[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m [0;34m[0m[0m
[0;32m----> 9[0;31m [0mgru_pred_model[0m[0;34m.[0m[0mload_weights[0m[0;34m([0m[0;34m"model_gru.h5"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     10[0m [0mlstm_pred_model[0m[0;34m.[0m[0mload_weights[0m[0;34m([0m[0;34m"model_lstm.h5"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m    120[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m             [0;31m# `keras.config.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 122[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    123[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py[0m in [0;36m__init__[0;34m(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)[0m
[1;32m    562[0m                                  [0mfs_persist[0m[0;34m=[0m[0mfs_persist[0m[0;34m,[0m [0mfs_threshold[0m[0;34m=[0m[0mfs_threshold[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    563[0m                                  fs_page_size=fs_page_size)
[0;32m--> 564[0;31m                 [0mfid[0m [0;34m=[0m [0mmake_fid[0m[0;34m([0m[0mname[0m[0;34m,[0m [0mmode[0m[0;34m,[0m [0muserblock_size[0m[0;34m,[0m [0mfapl[0m[0;34m,[0m [0mfcpl[0m[0;34m,[0m [0mswmr[0m[0;34m=[0m[0mswmr[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    565[0m [0;34m[0m[0m
[1;32m    566[0m             [0;32mif[0m [0misinstance[0m[0;34m([0m[0mlibver[0m[0;34m,[0m [0mtuple[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py[0m in [0;36mmake_fid[0;34m(name, mode, userblock_size, fapl, fcpl, swmr)[0m
[1;32m    236[0m         [0;32mif[0m [0mswmr[0m [0;32mand[0m [0mswmr_support[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    237[0m             [0mflags[0m [0;34m|=[0m [0mh5f[0m[0;34m.[0m[0mACC_SWMR_READ[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 238[0;31m         [0mfid[0m [0;34m=[0m [0mh5f[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0mname[0m[0;34m,[0m [0mflags[0m[0;34m,[0m [0mfapl[0m[0;34m=[0m[0mfapl[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    239[0m     [0;32melif[0m [0mmode[0m [0;34m==[0m [0;34m'r+'[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    240[0m         [0mfid[0m [0;34m=[0m [0mh5f[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0mname[0m[0;34m,[0m [0mh5f[0m[0;34m.[0m[0mACC_RDWR[0m[0;34m,[0m [0mfapl[0m[0;34m=[0m[0mfapl[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32mh5py/_objects.pyx[0m in [0;36mh5py._objects.with_phil.wrapper[0;34m()[0m

[0;32mh5py/_objects.pyx[0m in [0;36mh5py._objects.with_phil.wrapper[0;34m()[0m

[0;32mh5py/h5f.pyx[0m in [0;36mh5py.h5f.open[0;34m()[0m

[0;31mFileNotFoundError[0m: [Errno 2] Unable to synchronously open file (unable to open file: name = 'model_gru.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 18
n = blend_test_preds.shape[0]
if test_len < pred_len:
    raise ValueError(f"test_len ({test_len}) < pred_len ({pred_len}) unexpectedly.")

full_preds = np.zeros((n, test_len, 5), dtype=np.float32)
full_preds[:, :pred_len, :] = blend_test_preds.astype(np.float32)

if test_len > pred_len:
    last_vals = blend_test_preds[:, -1:, :].astype(np.float32)  # (n,1,5)
    full_preds[:, pred_len:, :] = np.repeat(
        last_vals, repeats=(test_len - pred_len), axis=1
    )
