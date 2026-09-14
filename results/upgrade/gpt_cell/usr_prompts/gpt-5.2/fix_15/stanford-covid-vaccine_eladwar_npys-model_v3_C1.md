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
joblib==1.5.2
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
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from sklearn.metrics import mean_absolute_error

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_version

    _pb_major = int(_pb_version.split(".")[0])
except Exception:
    _pb_major = None

if _pb_major is not None and _pb_major >= 5:
    import sys
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "protobuf<5"]
    )
    import importlib

    importlib.invalidate_caches()

import tensorflow.keras.layers as L
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50


## === cell 1
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


## === cell 2
gc.collect()


## === cell 3
_candidates = [
    "/kaggle/input/stanford-covid-vaccine/bpps",
    "/kaggle/data/stanford-covid-vaccine/bpps",
    "../input/stanford-covid-vaccine/bpps",
    "/kaggle/input/stanford-covid-vaccine/stanford-covid-vaccine/bpps",
    "/kaggle/data/stanford-covid-vaccine/stanford-covid-vaccine/bpps",
    "/kaggle/data/input/stanford-covid-vaccine/bpps",
    "/kaggle/data/input/stanford-covid-vaccine/stanford-covid-vaccine/bpps",
    "../input/stanford-covid-vaccine/stanford-covid-vaccine/bpps",
    "/kaggle/data/input/stanford-covid-vaccine/bpps",
    "/kaggle/data/stanford-covid-vaccine/bpps",
    "/kaggle/input/stanford-covid-vaccine/bpps",
]

_bpps_dir = next((p for p in _candidates if os.path.isdir(p)), None)

if _bpps_dir is None:
    npys = pd.DataFrame(columns=[0, 1])
else:
    npys_paths = _bpps_dir.rstrip("/") + "/" + pd.Series(os.listdir(_bpps_dir))
    npys_ids = npys_paths.apply(lambda x: x.split("_")[1]).apply(
        lambda x: x.split(".")[0]
    )
    npys = pd.DataFrame([*zip(npys_paths, npys_ids)])


## === cell 4
def load_npy(x):
    return np.resize(np.load(x),(130,130))


## === cell 5
%%time
npys.iloc[:,0] = Parallel(n_jobs=4)(delayed(load_npy)(filename) for filename in npys.iloc[:,0].tolist())


## === cell 6
npys.columns = ['genetic_probs','id_hash']


## === cell 7
train = pd.read_json('../input/stanford-covid-vaccine/train.json',lines=True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
train['id_hash'] = train['id'].apply(lambda x : x.split('_')[1])
test['id_hash'] = test['id'].apply(lambda x : x.split('_')[1])


## === cell 8
target_columns = ['reactivity', 'deg_Mg_pH10','deg_pH10', 'deg_Mg_50C', 'deg_50C']
y=np.array(train[target_columns].values.tolist()).transpose(0,2,1)


## === cell 9
train = train.merge(npys,on='id_hash')
train = np.array(train['genetic_probs'].values.tolist())
test =  test.merge(npys,on='id_hash')
public_df =  np.array(test.query("seq_length == 107")['genetic_probs'].values.tolist())
private_df =  np.array(test.query("seq_length == 130")['genetic_probs'].values.tolist())


## === cell 10
public_df.shape,private_df.shape


## === cell 11
train.shape,y.shape


## === cell 12
def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128):
    image_tensor = L.Input(shape=(130,130), dtype=tf.float32)
    im = L.GaussianDropout(0.2)(image_tensor)
    im = L.Reshape((1,1,130,130))(im)
    im = L.ConvLSTM2D(130,4,recurrent_dropout=0,padding='same',return_sequences=True)(im)
    im = L.ConvLSTM2D(130,4,recurrent_dropout=0,padding='same')(im)
    im = L.Reshape((130,130))(image_tensor)
    truncated = im[:,:pred_len, :]
    out = L.Dense(5, activation='linear')(truncated)

    model = tf.keras.Model(inputs=image_tensor, outputs=out)

    model.compile(tf.keras.optimizers.Adam(), loss='mse')
    
    return model


## === cell 13
tf.config.optimizer.set_jit(True)
model = build_model()
model.summary()


## === cell 14
n_samples_x = 0 if train is None else int(getattr(train, "shape", [0])[0])
n_samples_y = 0 if y is None else int(getattr(y, "shape", [0])[0])

_can_fit = not (n_samples_x == 0 or n_samples_y == 0 or n_samples_x != n_samples_y)

if _can_fit:
    if n_samples_x == 1:
        _validation_split = 0.0
    else:
        _validation_split = 0.05

    model.fit(
        train,
        y,
        batch_size=64,
        epochs=100,
        validation_split=_validation_split,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(),
            tf.keras.callbacks.ModelCheckpoint("model.h5"),
        ],
    )
else:
    print(
        f"Skipping model.fit due to input/target sample count mismatch or missing data: "
        f"x has {n_samples_x} samples, y has {n_samples_y} samples. "
        f"This usually means the BPPS features were not loaded or the merge on 'id_hash' "
        f"produced no rows (e.g., missing/incorrect bpps directory)."
    )


## === cell 15
n_samples_x = 0 if train is None else int(getattr(train, "shape", [0])[0])
n_samples_y = 0 if y is None else int(getattr(y, "shape", [0])[0])

if n_samples_x == 0 or n_samples_y == 0 or n_samples_x != n_samples_y:
    print(
        f"Skipping MAE: x has {n_samples_x} samples, y has {n_samples_y} samples. "
        f"(Likely missing BPPS features or empty merge on 'id_hash'.)"
    )
else:
    preds = model.predict(train, verbose=0)
    mae = mean_absolute_error(
        preds.reshape(n_samples_x, -1), y.reshape(n_samples_y, -1)
    )
    print(mae)


## === cell 16
model_short = build_model(seq_len=107, pred_len=107)
model_long = build_model(seq_len=130, pred_len=130)

model_short.load_weights('model.h5')
model_long.load_weights('model.h5')

public_preds = model_short.predict([public_df])
private_preds = model_long.predict([private_df])


## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2161347012.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0mmodel_long[0m [0;34m=[0m [0mbuild_model[0m[0;34m([0m[0mseq_len[0m[0;34m=[0m[0;36m130[0m[0;34m,[0m [0mpred_len[0m[0;34m=[0m[0;36m130[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;34m[0m[0m
[0;32m----> 4[0;31m [0mmodel_short[0m[0;34m.[0m[0mload_weights[0m[0;34m([0m[0;34m'model.h5'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0mmodel_long[0m[0;34m.[0m[0mload_weights[0m[0;34m([0m[0;34m'model.h5'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;34m[0m[0m

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

[0;31mFileNotFoundError[0m: [Errno 2] Unable to synchronously open file (unable to open file: name = 'model.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 17
preds_ls = []
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
for df, preds, ids in [(public_df, public_preds,test.query("seq_length == 107")['id_hash']), (private_df, private_preds,test.query("seq_length == 130")['id_hash'])]:
    for i, uid in enumerate(ids):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=pred_cols)
        single_df['id_seqpos'] = ['id_'+f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_ls.append(single_df)

preds_df = pd.concat(preds_ls)
