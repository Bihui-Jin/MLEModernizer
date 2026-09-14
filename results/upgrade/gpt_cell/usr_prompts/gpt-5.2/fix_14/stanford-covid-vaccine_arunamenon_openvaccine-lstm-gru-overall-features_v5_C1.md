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
protobuf==6.33.0
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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    if int(_pb_ver.split(".", 1)[0]) >= 6:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        import importlib
        import google.protobuf as _gp

        importlib.reload(_gp)
except Exception:
    pass

import json
import tensorflow as tf
from matplotlib import pyplot as plt


## === cell 2
os.chdir('/kaggle/')
os.getcwd()


## === cell 3
train_data = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines = True)
test_data = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines = True)
submission_format = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv', encoding = 'utf-8-sig')


## === cell 4
train_data.head()


## === cell 5
train_data.shape


## === cell 6
train_data.groupby(['SN_filter']).size()


## === cell 7
test_data.head()


## === cell 8
test_data.shape


## === cell 9
submission_format.head()


## === cell 10
print(train_data.shape)
print(test_data.shape)
print(submission_format.shape)


## === cell 11
print('Training data:\n',train_data['seq_scored'].value_counts())
print('Test data:\n',test_data['seq_scored'].value_counts())
len(train_data['reactivity'].iloc[0])


## === cell 12
len(train_data['sequence'].iloc[0])


## === cell 13
flag = False
for i in range(0,len(train_data)):
    if(([x<0 for x in train_data['reactivity_error'].iloc[0]].count(True) > 0) |
       ([x<0 for x in train_data['deg_error_Mg_pH10'].iloc[0]].count(True) > 0) |
       ([x<0 for x in train_data['deg_error_pH10'].iloc[0]].count(True) > 0) |
       ([x<0 for x in train_data['deg_error_Mg_50C'].iloc[0]].count(True) > 0) |
       ([x<0 for x in train_data['deg_error_50C'].iloc[0]].count(True) > 0)):
        flag = True
print(flag)


## === cell 14
train_data.columns


## === cell 15
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


## === cell 16
token2int


## === cell 17
def _resolve_bpps_dir():
    candidates = [
        "/kaggle/input/stanford-covid-vaccine/bpps",
        "/kaggle/data/stanford-covid-vaccine/bpps",
        "/kaggle/working/stanford-covid-vaccine/bpps",
        "/kaggle/input/bpps",
        "/kaggle/data/bpps",
        "/kaggle/working/bpps",
    ]
    for d in candidates:
        if os.path.isdir(d):
            return d
    return None


_BPPS_DIR = _resolve_bpps_dir()


def read_bpps_sum(df):
    bpps_arr = []
    for mol_id, L in zip(df.id.to_list(), df.seq_length.to_list()):
        if _BPPS_DIR is not None:
            p = os.path.join(_BPPS_DIR, f"{mol_id}.npy")
            if os.path.exists(p):
                bpps_arr.append(np.load(p).sum(axis=1))
                continue
        bpps_arr.append(np.zeros(int(L), dtype=np.float32))
    return bpps_arr


def read_bpps_max(df):
    bpps_arr = []
    for mol_id, L in zip(df.id.to_list(), df.seq_length.to_list()):
        if _BPPS_DIR is not None:
            p = os.path.join(_BPPS_DIR, f"{mol_id}.npy")
            if os.path.exists(p):
                bpps_arr.append(np.load(p).max(axis=1))
                continue
        bpps_arr.append(np.zeros(int(L), dtype=np.float32))
    return bpps_arr


def read_bpps_nb(df):
    bpps_nb_mean = 0.077522
    bpps_nb_std = 0.08914
    bpps_arr = []
    for mol_id, L in zip(df.id.to_list(), df.seq_length.to_list()):
        if _BPPS_DIR is not None:
            p = os.path.join(_BPPS_DIR, f"{mol_id}.npy")
            if os.path.exists(p):
                bpps = np.load(p)
                bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
                bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
                bpps_arr.append(bpps_nb)
                continue
        bpps_arr.append(np.zeros(int(L), dtype=np.float32))
    return bpps_arr


os.chdir("/kaggle/working/")
train_data["bpps_sum"] = read_bpps_sum(train_data)
test_data["bpps_sum"] = read_bpps_sum(test_data)
train_data["bpps_max"] = read_bpps_max(train_data)
test_data["bpps_max"] = read_bpps_max(test_data)
train_data["bpps_nb"] = read_bpps_nb(train_data)
test_data["bpps_nb"] = read_bpps_nb(test_data)

train_data.head()


## === cell 18
def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    base_fea = np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )
    
    bpps_sum_fea = np.array(df['bpps_sum'].to_list())[:,:,np.newaxis]
    bpps_max_fea = np.array(df['bpps_max'].to_list())[:,:,np.newaxis]
    bpps_nb_fea = np.array(df['bpps_nb'].to_list())[:,:,np.newaxis]
    
    return np.concatenate([base_fea,bpps_sum_fea,bpps_max_fea,bpps_nb_fea], 2)


## === cell 19
train_inputs = preprocess_inputs(train_data.loc[train_data['signal_to_noise'] > 1])
train_labels = np.array(train_data.loc[train_data['signal_to_noise'] > 1][target_cols].values.tolist()).transpose((0, 2, 1))


## === cell 20
train_data.loc[[0]]


## === cell 21
preprocess_inputs(train_data.loc[[0]])


## === cell 22
test_data.head()


## === cell 23
from keras.losses import mean_squared_error

def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(mean_squared_error(y_true, y_pred))

def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)

def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
                                tf.keras.layers.LSTM(hidden_dim,
                                dropout=dropout,
                                return_sequences=True,
                                kernel_initializer = 'orthogonal'))

def build_model(seq_len = 107, embed_dim = 100, hidden_dim = 256, dropout = 0.2, pred_len = 68):
    
    inputs = tf.keras.layers.Input(shape=(seq_len, 6))
    categorical_feats = inputs[:, :, :3]
    numerical_feats = inputs[:, :, 3:]

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(categorical_feats)
    
    reshaped = tf.reshape(embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
    
    reshaped = tf.keras.layers.concatenate([reshaped, numerical_feats], axis=2)
    
    LSTM_layer = lstm_layer(hidden_dim, dropout)(reshaped)

    truncated = LSTM_layer[:, :pred_len]

    out = tf.keras.layers.Dense(5, activation='linear')(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    adam = tf.optimizers.Adam()

    model.compile(optimizer = adam, loss = MCRMSE)
    
    return model


## === cell 24
EPOCHS = 60
BATCH_SIZE = 32


def build_model(seq_len=107, embed_dim=100, hidden_dim=256, dropout=0.2, pred_len=68):
    inputs = tf.keras.layers.Input(shape=(seq_len, 6))
    categorical_feats = inputs[:, :, :3]
    numerical_feats = inputs[:, :, 3:]

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        categorical_feats
    )

    reshaped = tf.keras.layers.Reshape((seq_len, 3 * embed_dim))(embed)

    reshaped = tf.keras.layers.concatenate([reshaped, numerical_feats], axis=2)

    LSTM_layer = lstm_layer(hidden_dim, dropout)(reshaped)
    truncated = LSTM_layer[:, :pred_len]
    out = tf.keras.layers.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    adam = tf.optimizers.Adam()
    model.compile(optimizer=adam, loss=MCRMSE)
    return model


model_on_train_data = build_model()
model_on_train_data.summary()
model_callback = tf.keras.callbacks.ModelCheckpoint(f"LSTM model.h5")

history = model_on_train_data.fit(
    train_inputs,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
    callbacks=[model_callback],
)


## === cell 25
print(f" LSTM mean fold validation loss: {min(history.history['loss'])}")

fig, ax = plt.subplots(1, 1, figsize = (20, 10))

ax.plot(history.history['loss'])

ax.set_title('LSTM Model')

ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')


## === cell 26
def preprocess_inputs(df, cols=["sequence", "structure", "predicted_loop_type"]):
    if "seq_length" in df.columns and len(df) > 0:
        L = int(df["seq_length"].max())
    else:
        L = len(str(df[cols[0]].iloc[0])) if len(df) > 0 else 0

    pad_token = "X"
    pad_val = token2int.get(pad_token, 0)

    def _encode_str(s):
        s = str(s)
        if len(s) < L:
            s = s + (pad_token * (L - len(s)))
        elif len(s) > L:
            s = s[:L]
        return [token2int.get(ch, pad_val) for ch in s]

    def _pad_num(x):
        arr = np.asarray(x, dtype=np.float32)
        if arr.shape[0] < L:
            arr = np.pad(
                arr, (0, L - arr.shape[0]), mode="constant", constant_values=0.0
            )
        elif arr.shape[0] > L:
            arr = arr[:L]
        return arr

    cat_feats = np.stack(
        [np.stack(df[c].map(_encode_str).to_list(), axis=0) for c in cols],
        axis=-1,
    ).astype(np.int32)

    bpps_sum_fea = np.stack(df["bpps_sum"].map(_pad_num).to_list(), axis=0)[
        :, :, np.newaxis
    ]
    bpps_max_fea = np.stack(df["bpps_max"].map(_pad_num).to_list(), axis=0)[
        :, :, np.newaxis
    ]
    bpps_nb_fea = np.stack(df["bpps_nb"].map(_pad_num).to_list(), axis=0)[
        :, :, np.newaxis
    ]

    return np.concatenate([cat_feats, bpps_sum_fea, bpps_max_fea, bpps_nb_fea], axis=2)


## === cell 27
public_df = test_data.loc[test_data["seq_length"] == 107].reset_index(drop=True)
public_inputs = preprocess_inputs(public_df)

model_on_test_data_public = build_model(seq_len=107, pred_len=107)
model_on_test_data_public.load_weights(f"LSTM model.h5")
pred_test_data_public = model_on_test_data_public.predict(public_inputs)


## === cell 28
private_df = test_data.loc[test_data["seq_length"] == 130].reset_index(drop=True)
private_inputs = preprocess_inputs(private_df)

model_on_test_data_private = build_model(seq_len=130, pred_len=130)
model_on_test_data_private.load_weights(f"LSTM model.h5")
pred_test_data_private = model_on_test_data_private.predict(private_inputs)


## --- ERROR in cell 28, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/63717159.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0;31m# so prediction can run and cell 29 can access private_df and pred_test_data_private.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0mprivate_df[0m [0;34m=[0m [0mtest_data[0m[0;34m.[0m[0mloc[0m[0;34m[[0m[0mtest_data[0m[0;34m[[0m[0;34m"seq_length"[0m[0;34m][0m [0;34m==[0m [0;36m130[0m[0;34m][0m[0;34m.[0m[0mreset_index[0m[0;34m([0m[0mdrop[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m [0mprivate_inputs[0m [0;34m=[0m [0mpreprocess_inputs[0m[0;34m([0m[0mprivate_df[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0;34m[0m[0m
[1;32m      6[0m [0mmodel_on_test_data_private[0m [0;34m=[0m [0mbuild_model[0m[0;34m([0m[0mseq_len[0m[0;34m=[0m[0;36m130[0m[0;34m,[0m [0mpred_len[0m[0;34m=[0m[0;36m130[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1669131572.py[0m in [0;36mpreprocess_inputs[0;34m(df, cols)[0m
[1;32m     30[0m     [0;31m# categorical: (N, L, 3)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     31[0m     cat_feats = np.stack(
[0;32m---> 32[0;31m         [0;34m[[0m[0mnp[0m[0;34m.[0m[0mstack[0m[0;34m([0m[0mdf[0m[0;34m[[0m[0mc[0m[0;34m][0m[0;34m.[0m[0mmap[0m[0;34m([0m[0m_encode_str[0m[0;34m)[0m[0;34m.[0m[0mto_list[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;36m0[0m[0;34m)[0m [0;32mfor[0m [0mc[0m [0;32min[0m [0mcols[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     33[0m         [0maxis[0m[0;34m=[0m[0;34m-[0m[0;36m1[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     34[0m     ).astype(np.int32)

[0;32m/tmp/ipykernel_11/1669131572.py[0m in [0;36m<listcomp>[0;34m(.0)[0m
[1;32m     30[0m     [0;31m# categorical: (N, L, 3)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     31[0m     cat_feats = np.stack(
[0;32m---> 32[0;31m         [0;34m[[0m[0mnp[0m[0;34m.[0m[0mstack[0m[0;34m([0m[0mdf[0m[0;34m[[0m[0mc[0m[0;34m][0m[0;34m.[0m[0mmap[0m[0;34m([0m[0m_encode_str[0m[0;34m)[0m[0;34m.[0m[0mto_list[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;36m0[0m[0;34m)[0m [0;32mfor[0m [0mc[0m [0;32min[0m [0mcols[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     33[0m         [0maxis[0m[0;34m=[0m[0;34m-[0m[0;36m1[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     34[0m     ).astype(np.int32)

[0;32m/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py[0m in [0;36mstack[0;34m(arrays, axis, out, dtype, casting)[0m
[1;32m    443[0m     [0marrays[0m [0;34m=[0m [0;34m[[0m[0masanyarray[0m[0;34m([0m[0marr[0m[0;34m)[0m [0;32mfor[0m [0marr[0m [0;32min[0m [0marrays[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    444[0m     [0;32mif[0m [0;32mnot[0m [0marrays[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 445[0;31m         [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m'need at least one array to stack'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    446[0m [0;34m[0m[0m
[1;32m    447[0m     [0mshapes[0m [0;34m=[0m [0;34m{[0m[0marr[0m[0;34m.[0m[0mshape[0m [0;32mfor[0m [0marr[0m [0;32min[0m [0marrays[0m[0;34m}[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: need at least one array to stack

## === cell 29
def format_predictions(public_preds, private_preds):
    preds = []
    
    for df, preds_ in [(public_df, pred_test_data_public), (private_df, pred_test_data_private)]:
        for i, uid in enumerate(df.id):
            single_pred = preds_[i]

            single_df = pd.DataFrame(single_pred, columns=target_cols)
            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

            preds.append(single_df)
    return pd.concat(preds).reset_index(drop = True)
