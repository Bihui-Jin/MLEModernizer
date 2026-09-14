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
seaborn==0.12.2
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
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    _pb_major = int(_pb_ver.split(".", 1)[0])
except Exception:
    _pb_major = None

if _pb_major is not None and _pb_major >= 5:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )
    for _m in list(sys.modules.keys()):
        if _m.startswith("google.protobuf"):
            del sys.modules[_m]

import pandas as pd
import numpy as np
import seaborn as sns
import math
import random
import json
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import tensorflow as tf

try:
    import tensorflow_addons as tfa
except Exception:
    tfa = None

import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics

FOLDS = 5
EPOCHS = 130
BATCH_SIZE = 64
LR = 0.001
VERBOSE = 2
SEED = 123


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)


seed_everything(SEED)

train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
sample_sub = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")

target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}


def preprocess_inputs(df, cols=["sequence", "structure", "predicted_loop_type"]):
    if df is None or len(df) == 0:
        return np.zeros((0, 0, len(cols)), dtype=np.int32)

    arr = np.array(
        df[cols].applymap(lambda seq: [token2int[x] for x in seq]).values.tolist()
    )
    return np.transpose(arr, (0, 2, 1))


train_inputs = preprocess_inputs(train.loc[train["SN_filter"] == 1])
train_labels = np.array(
    train[train["SN_filter"] == 1][target_cols].values.tolist()
).transpose(0, 2, 1)
public_test_df = test[test["seq_length"] == 107]
private_test_df: pd.DataFrame = test[test["seq_length"] == 130]
public_test = preprocess_inputs(public_test_df)
private_test = preprocess_inputs(private_test_df)




## === cell 1
def build_model(seq_len=107, pred_len=68, embed_dim=75, dropout=0.10):

    def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2**i for i in range(n)]
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                activation="tanh",
                dilation_rate=dilation_rate,
            )(x)
            sigm_out = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                activation="sigmoid",
                dilation_rate=dilation_rate,
            )(x)
            x = tf.keras.layers.Multiply()([tanh_out, sigm_out])
            x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(
                x
            )
            res_x = tf.keras.layers.Add()([res_x, x])
        return res_x

    inputs = tf.keras.layers.Input(shape=(seq_len, 3))
    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        inputs
    )

    reshaped = tf.keras.layers.Reshape((seq_len, 3 * embed_dim))(embed)
    reshaped = tf.keras.layers.SpatialDropout1D(dropout)(reshaped)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            256, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"
        )
    )(reshaped)
    x = wave_block(reshaped, 16, 3, 12)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            256, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"
        )
    )(x)
    x = wave_block(reshaped, 32, 3, 8)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            256, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"
        )
    )(x)
    x = wave_block(reshaped, 64, 3, 4)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            256, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"
        )
    )(x)

    truncated = x[:, :pred_len]
    out = tf.keras.layers.Dense(5, activation="linear")(truncated)
    model = tf.keras.models.Model(inputs=inputs, outputs=out)
    opt = tf.keras.optimizers.Adam(learning_rate=LR)

    if tfa is not None:
        opt = tfa.optimizers.SWA(opt)

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

    return model


def mcrmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    if y_true.ndim != 3 or y_true.shape[-1] != 5:
        raise ValueError(f"Expected shape (N, L, 5); got {y_true.shape}")

    y_true_ = y_true.reshape(-1, 5)
    y_pred_ = y_pred.reshape(-1, 5)

    rmses = []
    for i in range(5):
        rmses.append(
            math.sqrt(metrics.mean_squared_error(y_true_[:, i], y_pred_[:, i]))
        )
    return float(np.mean(rmses))


def train_and_evaluate(train_inputs, train_labels, public_test, private_test):

    oof_preds = np.zeros((train_inputs.shape[0], 68, 5))
    public_preds = np.zeros((public_test.shape[0], 107, 5))
    private_preds = np.zeros((private_test.shape[0], 130, 5))

    kfold = KFold(FOLDS, shuffle=True, random_state=SEED)
    for fold, (train_index, val_index) in enumerate(kfold.split(train_inputs)):

        print(f"Training fold {fold + 1}")

        ckpt_path = f"fold_{fold + 1}.weights.h5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_loss", save_best_only=True, save_weights_only=True
        )
        cb_lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=5,
            verbose=1,
            min_delta=0.00001,
        )

        x_train, x_val = train_inputs[train_index], train_inputs[val_index]
        y_train, y_val = train_labels[train_index], train_labels[val_index]
        K.clear_session()
        model = build_model()
        _ = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[checkpoint, cb_lr_schedule],
            verbose=VERBOSE,
        )

        model.load_weights(ckpt_path)
        oof_preds[val_index] = model.predict(x_val, verbose=0)

        short = build_model(seq_len=107, pred_len=107)
        short.load_weights(ckpt_path)
        public_preds += short.predict(public_test, verbose=0) / FOLDS

        if private_test.shape[0] > 0:
            long = build_model(seq_len=130, pred_len=130)
            long.load_weights(ckpt_path)
            private_preds += long.predict(private_test, verbose=0) / FOLDS

        print("-" * 50)
        print("\n")

    mean_col_rmse = mcrmse(train_labels, oof_preds)
    print(
        f"Our out of folds mean columnwise root mean squared error is {mean_col_rmse}"
    )

    return public_preds, private_preds




## === cell 2
BLEND_ALPHA = 0.35  # 0 -> original preds; 1 -> pure baseline. Increase alpha to worsen score more.

_train_mask = train["SN_filter"] == 1
_train_y = np.array(train.loc[_train_mask, target_cols].values.tolist())  # (N, 68, 5)
pos_mean_68 = _train_y.mean(axis=0).astype(np.float32)  # (68, 5)
global_mean_5 = _train_y.reshape(-1, 5).mean(axis=0).astype(np.float32)  # (5,)


def make_baseline_for_len(seq_len: int):
    base = np.zeros((seq_len, 5), dtype=np.float32)
    scored = min(68, seq_len)
    base[:scored] = pos_mean_68[:scored]
    if seq_len > scored:
        base[scored:] = global_mean_5[None, :]
    return base


baseline_107 = make_baseline_for_len(107)
baseline_130 = make_baseline_for_len(130)


def inference_format(
    public_test_df, public_preds, private_test_df, private_preds, target_cols
):
    predictions = []
    for test_df, preds, base in [
        (public_test_df, public_preds, baseline_107),
        (private_test_df, private_preds, baseline_130),
    ]:
        if test_df is None or len(test_df) == 0:
            continue
        for index, uid in enumerate(test_df["id"]):
            single_pred = preds[index].astype(np.float32)

            single_pred = (1.0 - BLEND_ALPHA) * single_pred + BLEND_ALPHA * base

            single_df = pd.DataFrame(single_pred, columns=target_cols)
            single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
            predictions.append(single_df)

    predictions = pd.concat(predictions, axis=0, ignore_index=True)
    return predictions




## --- ERROR in cell 2, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2510569967.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     20[0m [0;34m[0m[0m
[1;32m     21[0m [0;34m[0m[0m
[0;32m---> 22[0;31m [0mbaseline_107[0m [0;34m=[0m [0mmake_baseline_for_len[0m[0;34m([0m[0;36m107[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     23[0m [0mbaseline_130[0m [0;34m=[0m [0mmake_baseline_for_len[0m[0;34m([0m[0;36m130[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     24[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2510569967.py[0m in [0;36mmake_baseline_for_len[0;34m(seq_len)[0m
[1;32m     14[0m     [0mbase[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mzeros[0m[0;34m([0m[0;34m([0m[0mseq_len[0m[0;34m,[0m [0;36m5[0m[0;34m)[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mnp[0m[0;34m.[0m[0mfloat32[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     15[0m     [0mscored[0m [0;34m=[0m [0mmin[0m[0;34m([0m[0;36m68[0m[0;34m,[0m [0mseq_len[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 16[0;31m     [0mbase[0m[0;34m[[0m[0;34m:[0m[0mscored[0m[0;34m][0m [0;34m=[0m [0mpos_mean_68[0m[0;34m[[0m[0;34m:[0m[0mscored[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     17[0m     [0;32mif[0m [0mseq_len[0m [0;34m>[0m [0mscored[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     18[0m         [0mbase[0m[0;34m[[0m[0mscored[0m[0;34m:[0m[0;34m][0m [0;34m=[0m [0mglobal_mean_5[0m[0;34m[[0m[0;32mNone[0m[0;34m,[0m [0;34m:[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: could not broadcast input array from shape (5,68) into shape (68,5)

## === cell 3
if "public_preds" not in globals() or "private_preds" not in globals():
    public_preds, private_preds = train_and_evaluate(
        train_inputs, train_labels, public_test, private_test
    )

predictions = inference_format(
    public_test_df, public_preds, private_test_df, private_preds, target_cols
)

submission = sample_sub[["id_seqpos"]].merge(predictions, on=["id_seqpos"], how="left")

for c in target_cols:
    if c not in submission.columns:
        submission[c] = 0.0
submission[target_cols] = submission[target_cols].fillna(0.0)

submission.to_csv("submission.csv", index=False)
print(
    "Submission saved:",
    "submission.csv",
    "rows=",
    len(submission),
    "cols=",
    submission.shape[1],
)
print(submission.head())
