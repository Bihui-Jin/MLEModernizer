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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

from google.protobuf import message_factory as _message_factory  # noqa: E402
from google.protobuf import descriptor_pool as _descriptor_pool  # noqa: E402

if not hasattr(_message_factory.MessageFactory, "GetPrototype"):

    def _get_prototype(self, descriptor):
        if hasattr(_message_factory, "GetMessageClass"):
            return _message_factory.GetMessageClass(descriptor)
        pool = getattr(self, "pool", None) or _descriptor_pool.Default()
        return pool.GetMessageClass(descriptor.full_name)

    _message_factory.MessageFactory.GetPrototype = _get_prototype

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from sklearn.metrics import mean_absolute_error

import tensorflow.keras.layers as L
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50


## === cell 1
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


## === cell 2
gc.collect()


## === cell 3
_candidate_roots = [
    "/kaggle/input/stanford-covid-vaccine",
    "/kaggle/data/stanford-covid-vaccine",
    "/kaggle/input/stanford-covid-vaccine/stanford-covid-vaccine",
    "/kaggle/data/stanford-covid-vaccine/stanford-covid-vaccine",
    "/kaggle/input",
    "/kaggle/data",
]

_bpps_dir = None

for _root in _candidate_roots:
    _cand = os.path.join(_root, "bpps")
    if os.path.isdir(_cand):
        _bpps_dir = _cand
        break

if _bpps_dir is None:
    _search_roots = [
        r
        for r in _candidate_roots
        if os.path.isdir(r)
        and ("stanford-covid-vaccine" in r or r in ("/kaggle/input", "/kaggle/data"))
    ]
    for _root in _search_roots:
        for _dirpath, _dirnames, _filenames in os.walk(_root):
            if "bpps" in _dirnames:
                _bpps_dir = os.path.join(_dirpath, "bpps")
                break
        if _bpps_dir is not None:
            break

if _bpps_dir is None:
    npys_paths = pd.Series([], dtype="object")
    npys_ids = pd.Series([], dtype="object")
    npys = pd.DataFrame([], columns=[0, 1])
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

    im = L.Activation('linear')(image_tensor)
    im = L.Activation('linear')(im)
    im = L.Activation('linear')(im)
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
n_samples = 0
try:
    n_samples = int(getattr(train, "shape", [0])[0])
except Exception:
    n_samples = 0

if n_samples == 0:
    print(
        "Skipping model.fit: training data contains 0 samples after preprocessing/merge."
    )
else:
    if n_samples < 2:
        print(
            f"Skipping model.fit: training data contains {n_samples} sample(s), "
            "not sufficient for validation_split=0.05."
        )
    else:
        with tf.device("/gpu"):
            model.fit(
                train,
                y,
                batch_size=64,
                epochs=100,
                validation_split=0.05,
                callbacks=[
                    tf.keras.callbacks.ReduceLROnPlateau(),
                    tf.keras.callbacks.ModelCheckpoint("model.h5"),
                ],
            )


## === cell 15
n_train = 0
try:
    n_train = int(getattr(train, "shape", [0])[0])
except Exception:
    n_train = 0

if n_train == 0:
    print("Skipping MAE: train contains 0 samples after preprocessing/merge.")
else:
    preds = model.predict(train)
    if preds.shape != y.shape:
        if preds.size == y.size:
            mae = mean_absolute_error(preds.reshape(-1), y.reshape(-1))
            print(mae)
        else:
            raise ValueError(
                f"Prediction/target size mismatch: preds.shape={preds.shape}, y.shape={y.shape}"
            )
    else:
        mae = mean_absolute_error(preds.reshape(-1), y.reshape(-1))
        print(mae)


## === cell 16
model_short = build_model(seq_len=107, pred_len=107)
model_long = build_model(seq_len=130, pred_len=130)

if os.path.exists("model.h5"):
    model_short.load_weights("model.h5")
    model_long.load_weights("model.h5")
else:
    print(
        "Warning: 'model.h5' not found; skipping load_weights and using current model weights."
    )

public_n = int(getattr(public_df, "shape", (0,))[0]) if public_df is not None else 0
private_n = int(getattr(private_df, "shape", (0,))[0]) if private_df is not None else 0

public_preds = (
    model_short.predict(public_df)
    if public_n > 0
    else np.zeros((0, 107, 5), dtype=np.float32)
)
private_preds = (
    model_long.predict(private_df)
    if private_n > 0
    else np.zeros((0, 130, 5), dtype=np.float32)
)


## === cell 17
preds_ls = []
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
for df, preds, ids in [
    (public_df, public_preds, test.query("seq_length == 107")["id_hash"]),
    (private_df, private_preds, test.query("seq_length == 130")["id_hash"]),
]:
    for i, uid in enumerate(ids):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=pred_cols)
        single_df["id_seqpos"] = [
            "id_" + f"{uid}_{x}" for x in range(single_df.shape[0])
        ]

        preds_ls.append(single_df)

if len(preds_ls) == 0:
    preds_df = pd.DataFrame(columns=pred_cols + ["id_seqpos"])
else:
    preds_df = pd.concat(preds_ls)


## === cell 18
preds_df


## === cell 19
sample_df


## --- ERROR in cell 19, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/690972938.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0msample_df[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mNameError[0m: name 'sample_df' is not defined

## === cell 20
sample_df = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')
submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])
submission.to_csv('submission.csv', index=False)
