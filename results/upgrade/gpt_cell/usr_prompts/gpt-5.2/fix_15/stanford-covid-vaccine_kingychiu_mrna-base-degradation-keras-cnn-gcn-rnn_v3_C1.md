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

3.9

# 2. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0
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
import sys
import subprocess

try:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )
except Exception:
    pass

import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm

from sklearn.model_selection import train_test_split

try:
    from keras.utils.vis_utils import plot_model
except Exception:
    plot_model = None

import tensorflow.keras.layers as L
import keras.backend as K
import tensorflow as tf


## === cell 1
!pip install spektral -q


## === cell 2
try:
    from spektral.layers import GraphConv  # older Spektral
except ImportError:
    from spektral.layers import GCNConv as GraphConv  # newer Spektral


## === cell 3
train_json_path = "/kaggle/input/stanford-covid-vaccine/train.json"
test_json_path = "/kaggle/input/stanford-covid-vaccine/test.json"
sample_sub_path = "/kaggle/input/stanford-covid-vaccine/sample_submission.csv"

output_path = "./"
bpps_path = "/kaggle/input/stanford-covid-vaccine/bpps"

train_df = pd.read_json(train_json_path, lines=True)
test_df = pd.read_json(test_json_path, lines=True)

public_df = test_df.query("seq_length == 107").copy()
private_df = test_df.query("seq_length == 130").copy()


## === cell 4
train_df.shape


## === cell 5
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
train_df[pred_cols].head()


## === cell 6
train_y = np.array(train_df[pred_cols].values.tolist()).transpose((0, 2, 1))
train_y.shape


## === cell 7
train_df[["id", "sequence", "structure", "predicted_loop_type"]].head()


## === cell 8
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
sequence_token2int = {x: i for i, x in enumerate("AGUC")}
structure_token2int = {
    ".": 0,
    "(": 1,
    ")": 2,
}
loop_token2int = {x: i for i, x in enumerate("SMIBHEX")}
token2int_map = {
    "sequence": sequence_token2int,
    "structure": structure_token2int,
    "predicted_loop_type": loop_token2int,
}
sequence_columns = ["sequence", "structure", "predicted_loop_type"]


def to_seq(df):
    if df is None or len(df) == 0:
        return np.zeros((0, 0, len(sequence_columns)), dtype=np.int32)

    cols_tok = [
        np.array(
            df[col].apply(lambda s: [token2int[x] for x in s]).values.tolist(),
            dtype=np.int32,
        )
        for col in sequence_columns
    ]  # each is (n, L)
    arr = np.stack(cols_tok, axis=1)  # (n, 3, L)
    return np.transpose(arr, (0, 2, 1))  # (n, L, 3)


train = to_seq(train_df)
public = to_seq(public_df)
private = to_seq(private_df)

train.shape, public.shape, private.shape


## === cell 9
def to_one_hot(df):
    if df is None or len(df) == 0:
        return np.zeros((0, 0, 14), dtype=np.float32)

    cols_idx = []
    for col in sequence_columns:
        idx = np.stack(
            df[col]
            .apply(lambda seq: [token2int_map[col][x] for x in seq])
            .values.tolist(),
            axis=0,
        ).astype(
            np.int32
        )  # (n, L)
        cols_idx.append(idx)

    temp = np.stack(cols_idx, axis=2)  # (n, L, 3)

    ohe_1 = tf.keras.utils.to_categorical(temp[:, :, 0], 4)
    ohe_2 = tf.keras.utils.to_categorical(temp[:, :, 1], 3)
    ohe_3 = tf.keras.utils.to_categorical(temp[:, :, 2], 7)
    return np.concatenate([ohe_1, ohe_2, ohe_3], axis=2)


train_ohe = to_one_hot(train_df)
public_ohe = to_one_hot(public_df)
private_ohe = to_one_hot(private_df)

train_ohe.shape, public_ohe.shape, private_ohe.shape


## === cell 10
def get_adjacency_matrix(inps):
    As = []
    for row in range(0, inps.shape[0]):
        A = np.zeros((inps.shape[1], inps.shape[1]))
        stack = []
        opened_so_far = []

        for seqpos in range(0, inps.shape[1]):
            if inps[row, seqpos, 1] == 0:
                stack.append(seqpos)
                opened_so_far.append(seqpos)
            elif inps[row, seqpos, 1] == 1:
                openpos = stack.pop()
                A[openpos, seqpos] = 1
                A[seqpos, openpos] = 1
        As.append(A)
    return np.array(As)

train_adj = get_adjacency_matrix(train)
public_adj = get_adjacency_matrix(public)
private_adj = get_adjacency_matrix(private)

train_adj.shape, public_adj.shape, private_adj.shape


## === cell 11
train_adj.mean(), public_adj.mean(), private_adj.mean()


## === cell 12
def get_bpps(mRNA_ids, length_map=None):
    candidate_dirs = [
        bpps_path,
        "/kaggle/input/stanford-covid-vaccine/stanford-covid-vaccine/bpps",
        "/kaggle/data/stanford-covid-vaccine/bpps",
        "/kaggle/data/stanford-covid-vaccine/stanford-covid-vaccine/bpps",
    ]

    if length_map is None:
        length_map = {}

    bpps = []
    for mRNA_id in tqdm(mRNA_ids):
        arr = None
        for d in candidate_dirs:
            fp = os.path.join(d, f"{mRNA_id}.npy")
            if os.path.exists(fp):
                arr = np.load(fp)
                break

        if arr is None:
            L = int(length_map.get(mRNA_id, 0))
            arr = np.zeros((L, L), dtype=np.float32)

        bpps.append(arr)

    return np.array(bpps)


train_len_map = dict(zip(train_df["id"].values, train_df["seq_length"].values))
public_len_map = dict(zip(public_df["id"].values, public_df["seq_length"].values))
private_len_map = dict(zip(private_df["id"].values, private_df["seq_length"].values))

train_bpps = get_bpps(train_df.id.values, length_map=train_len_map)
public_bpps = get_bpps(public_df.id.values, length_map=public_len_map)
private_bpps = get_bpps(private_df.id.values, length_map=private_len_map)

train_bpps.shape, public_bpps.shape, private_bpps.shape


## === cell 13
train_bpps.mean(), public_bpps.mean(), private_bpps.mean() 


## === cell 14
def _bpps_row_stats(bpps_arr):
    if bpps_arr is None:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    if isinstance(bpps_arr, np.ndarray):
        if bpps_arr.ndim == 3:
            if bpps_arr.shape[0] == 0:
                return (
                    np.zeros((0, 0), dtype=np.float32),
                    np.zeros((0, 0), dtype=np.float32),
                )
            return bpps_arr.mean(axis=2), bpps_arr.max(axis=2)
        if bpps_arr.ndim == 0:
            return np.zeros((0, 0), dtype=np.float32), np.zeros(
                (0, 0), dtype=np.float32
            )

    if hasattr(bpps_arr, "__len__") and len(bpps_arr) == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    means, maxs = [], []
    for mat in bpps_arr:
        mat = np.asarray(mat)
        if mat.ndim == 2 and mat.size:
            means.append(mat.mean(axis=1))
            maxs.append(mat.max(axis=1))
        else:
            means.append(np.zeros((0,), dtype=np.float32))
            maxs.append(np.zeros((0,), dtype=np.float32))
    return np.stack(means, axis=0), np.stack(maxs, axis=0)


train_bpps_stats = list(_bpps_row_stats(train_bpps))
public_bpps_stats = list(_bpps_row_stats(public_bpps))
private_bpps_stats = list(_bpps_row_stats(private_bpps))


## === cell 15
train_bpps_stats = np.concatenate([stats[:,:,None] for stats in train_bpps_stats], axis=2)
public_bpps_stats = np.concatenate([stats[:,:,None] for stats in public_bpps_stats], axis=2)
private_bpps_stats = np.concatenate([stats[:,:,None] for stats in private_bpps_stats], axis=2)

train_bpps_stats.shape, public_bpps_stats.shape, private_bpps_stats.shape


## === cell 16
scored_seq_length = 68


def rmse(y_actual, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)
    return K.sqrt(mse)


def mcrmse(y_actual, y_pred):
    score = 0
    for i in range(y_actual.shape[2]):
        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / y_actual.shape[2]
    return score


def build_model(input_seq_len=107, output_seq_len=scored_seq_length):
    class _GraphConvNoMask(GraphConv):
        def call(self, inputs, mask=None, **kwargs):
            if mask is not None:
                try:
                    if isinstance(mask, (list, tuple)) and all(m is None for m in mask):
                        mask = None
                    elif mask is None:
                        mask = None
                except TypeError:
                    mask = None
            return super().call(inputs, mask=mask, **kwargs)

    def _bi_gru_block(x, hidden_dim, dropout):
        gru = L.Bidirectional(
            L.GRU(
                hidden_dim,
                dropout=dropout,
                return_sequences=True,
            ),
        )(x)
        return gru

    def _conv_block(x, adj_m, bpp_m, conv_filters, graph_channels):
        conv = L.Conv1D(
            conv_filters,
            5,
            padding="same",
            activation="tanh",
        )(x)

        gcn_1 = _GraphConvNoMask(
            graph_channels,
            activation="tanh",
        )([conv, adj_m])

        gcn_2 = _GraphConvNoMask(
            graph_channels,
            activation="tanh",
        )([conv, bpp_m])

        conv = L.Concatenate()([conv, gcn_1, gcn_2])
        conv = L.Activation("relu")(conv)
        conv = L.SpatialDropout1D(0.1)(conv)

        return conv

    one_hot_encoding_inputs = L.Input(shape=(input_seq_len, 14), name="onehot")
    adj_matrix_inputs = L.Input((input_seq_len, input_seq_len), name="adjmatrix")
    base_pair_proba_inputs = L.Input((input_seq_len, input_seq_len), name="pairproba")
    base_pair_proba_stats_inputs = L.Input(
        shape=(input_seq_len, 2), name="pairprobastats"
    )

    merged_inputs = L.Concatenate()(
        [one_hot_encoding_inputs, base_pair_proba_stats_inputs]
    )

    hidden = _conv_block(
        merged_inputs, adj_matrix_inputs, base_pair_proba_inputs, 512, 80
    )
    hidden = _bi_gru_block(hidden, 256, 0.5)
    hidden = _conv_block(hidden, adj_matrix_inputs, base_pair_proba_inputs, 512, 80)
    hidden = _bi_gru_block(hidden, 256, 0.5)

    out = hidden[:, :output_seq_len]
    out = L.Dense(5, activation="linear")(out)

    model = tf.keras.Model(
        inputs=[
            one_hot_encoding_inputs,
            adj_matrix_inputs,
            base_pair_proba_inputs,
            base_pair_proba_stats_inputs,
        ],
        outputs=out,
    )

    return model


model = build_model()

if callable(plot_model):
    plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)


## === cell 17
split_results  = train_test_split(
    train_ohe,
    train_adj,
    train_bpps,
    train_bpps_stats,
    train_y,
    train_df.signal_to_noise,
    train_df.SN_filter,
    test_size=0.1,
    random_state=42,
)

[a.shape for a in split_results]


## === cell 18
trn_ohe, val_ohe, trn_adj, val_adj, trn_bpps, val_bpps, trn_bpps_stats, val_bpps_stats, trn_y, val_y, trn_snr, val_snr, trn_snf, val_snf = split_results


## === cell 19
model = build_model()
model.compile(tf.keras.optimizers.Adam(), loss=mcrmse)


## === cell 20
trn_inputs = [trn_ohe, trn_adj, trn_bpps, trn_bpps_stats]
val_inputs = [val_ohe, val_adj, val_bpps, val_bpps_stats]


## === cell 21
val_mask = np.where((val_snf==1))
val_inputs = [val_input[val_mask] for val_input in val_inputs]
val_y = val_y[val_mask]


## === cell 22
sample_weight = np.log(trn_snr+1.11)/2


## === cell 23
def rmse(y_actual, y_pred):
    mse = tf.reduce_mean(tf.math.squared_difference(y_actual, y_pred), axis=-1)
    return K.sqrt(mse)


def mcrmse(y_actual, y_pred):
    score = 0
    for i in range(y_actual.shape[2]):
        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / y_actual.shape[2]
    return score


model.compile(tf.keras.optimizers.Adam(), loss=mcrmse)

history = model.fit(
    trn_inputs,
    trn_y,
    validation_data=(val_inputs, val_y),
    batch_size=64,
    epochs=300,
    sample_weight=sample_weight,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(verbose=1, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(
            f"model.h5", save_best_only=True, verbose=0, monitor="val_loss"
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=20,
            monitor="val_loss",
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        ),
    ],
    verbose=2,
)
print(f"Min validation loss history={min(history.history['val_loss'])}")


## --- ERROR in cell 23, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2013795755.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     15[0m [0mmodel[0m[0;34m.[0m[0mcompile[0m[0;34m([0m[0mtf[0m[0;34m.[0m[0mkeras[0m[0;34m.[0m[0moptimizers[0m[0;34m.[0m[0mAdam[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0mloss[0m[0;34m=[0m[0mmcrmse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m [0;34m[0m[0m
[0;32m---> 17[0;31m history = model.fit(
[0m[1;32m     18[0m     [0mtrn_inputs[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     19[0m     [0mtrn_y[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m    120[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m             [0;31m# `keras.config.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 122[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    123[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2013795755.py[0m in [0;36mmcrmse[0;34m(y_actual, y_pred)[0m
[1;32m      9[0m     [0mscore[0m [0;34m=[0m [0;36m0[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m     [0;32mfor[0m [0mi[0m [0;32min[0m [0mrange[0m[0;34m([0m[0my_actual[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m2[0m[0;34m][0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 11[0;31m         [0mscore[0m [0;34m+=[0m [0mrmse[0m[0;34m([0m[0my_actual[0m[0;34m[[0m[0;34m:[0m[0;34m,[0m [0;34m:[0m[0;34m,[0m [0mi[0m[0;34m][0m[0;34m,[0m [0my_pred[0m[0;34m[[0m[0;34m:[0m[0;34m,[0m [0;34m:[0m[0;34m,[0m [0mi[0m[0;34m][0m[0;34m)[0m [0;34m/[0m [0my_actual[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m2[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     12[0m     [0;32mreturn[0m [0mscore[0m[0;34m[0m[0;34m[0m[0m
[1;32m     13[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2013795755.py[0m in [0;36mrmse[0;34m(y_actual, y_pred)[0m
[1;32m      3[0m [0;32mdef[0m [0mrmse[0m[0;34m([0m[0my_actual[0m[0;34m,[0m [0my_pred[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m     [0mmse[0m [0;34m=[0m [0mtf[0m[0;34m.[0m[0mreduce_mean[0m[0;34m([0m[0mtf[0m[0;34m.[0m[0mmath[0m[0;34m.[0m[0msquared_difference[0m[0;34m([0m[0my_actual[0m[0;34m,[0m [0my_pred[0m[0;34m)[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;34m-[0m[0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m     [0;32mreturn[0m [0mK[0m[0;34m.[0m[0msqrt[0m[0;34m([0m[0mmse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m

[0;31mAttributeError[0m: module 'keras.backend' has no attribute 'sqrt'

## === cell 24
model.load_weights(f'model.h5')
