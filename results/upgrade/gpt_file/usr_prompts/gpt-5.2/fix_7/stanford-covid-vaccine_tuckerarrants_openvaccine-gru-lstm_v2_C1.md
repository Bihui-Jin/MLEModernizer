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
Predict likely degradation rates at each base of an RNA molecule.

## Metric
Mean columnwise root mean squared error:

$\textrm{MCRMSE} = \frac{1}{N_{t}}\sum_{j=1}^{N_{t}}\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{ij} - \hat{y}_{ij})^2}$

where $N_{t}$ is the number of scored ground truth target columns, and $y$ and $\hat{y}$ are the actual and predicted values, respectively.

There are multiple ground truth values provided in the training data. While the submission format requires all 5 to be predicted, only the following are scored: reactivity, deg_Mg_pH10, and deg_Mg_50C.

## Submission Formats
For each sample `id` in the test set, you must predict targets for *each* sequence position (`seqpos`), one per row. If the length of the `sequence` of an `id` is, e.g., 107, then you should make 107 predictions. Positions greater than the `seq_scored` value of a sample are not scored, but still need a value in the solution file.

```csv
id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C
id_d190610e8_0,0.1,0.3,0.2,0.5,0.4
id_d190610e8_1,0.3,0.2,0.5,0.4,0.2
id_d190610e8_2,0.5,0.4,0.2,0.1,0.2
etc.
```

## Dataset 
- **train.json** - the training data
- **test.json** - the test set, without any columns associated with the ground truth.
- **sample_submission.csv** - a sample submission file in the correct format

#### Columns
- `id` - An arbitrary identifier for each sample.
- `seq_scored` - (68 in Train and Public Test, 68 in Private Test) Integer value denoting the number of positions used in scoring with predicted values. This should match the length of `reactivity`, `deg_*` and `*_error_*` columns.
- `seq_length` - (107 in Train and Public Test, 107 in Private Test) Integer values, denotes the length of `sequence`.
- `sequence` - (1x107 string in Train and Public Test, 107 in Private Test) Describes the RNA sequence, a combination of `A`, `G`, `U`, and `C` for each sample. Should be 107 characters long, and the first 68 bases should correspond to the 68 positions specified in `seq_scored` (note: indexed starting at 0).
- `structure` - (1x107 string in Train and Public Test, 107 in Private Test) An array of `(`, `)`, and `.` characters that describe whether a base is estimated to be paired or unpaired. Paired bases are denoted by opening and closing parentheses e.g. (....) means that base 0 is paired to base 5, and bases 1-4 are unpaired.
- `reactivity` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likely secondary structure of the RNA sample.
- `deg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high pH (pH 10).
- `deg_Mg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium in high pH (pH 10).
- `deg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high temperature (50 degrees Celsius).
- `deg_Mg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium at high temperature (50 degrees Celsius).
- `*_error_*` - An array of floating point numbers, should have the same length as the corresponding `reactivity` or `deg_*` columns, calculated errors in experimental values obtained in `reactivity` and `deg_*` columns.
- `predicted_loop_type` - (1x107 string) Describes the structural context (also referred to as 'loop type')of each character in `sequence`. Loop types assigned by bpRNA from Vienna RNAfold 2 structure. From the bpRNA_documentation: S: paired "Stem" M: Multiloop I: Internal loop B: Bulge H: Hairpin loop E: dangling End X: eXternal loop
    - `S/N filter` Indicates if the sample passed filters described below in `Additional Notes`.

#### Additional Notes
At the beginning of the competition, Stanford scientists have data on 2400 RNA sequences of length 107. For technical reasons, measurements cannot be carried out on the final bases of these RNA sequences, so we have experimental data (ground truth) in 5 conditions for the first 68 bases.

We have split out 240 of these 2400 sequences for a public test set to allow for continuous evaluation through the competition, on the public leaderboard. These sequences, in `test.json`, have been additionally filtered based on three criteria detailed below to ensure that this subset is not dominated by any large cluster of RNA molecules with poor data, which might bias the public leaderboard. The remaining 2160 sequences for which we have data are in `train.json`.

For our final and most important scoring (the Private Leaderbooard), Stanford scientists are carrying out measurements on 240 new RNAs. For these data, we expect to have measurements for the first 68 bases, again missing the ends of the RNA. These sequences constitute the 240 sequences in `test.json`.

For those interested in how the sequences in `test.json` were filtered, here were the steps to ensure a diverse and high quality test set for public leaderboard scoring:

1. Minimum value across all 5 conditions must be greater than -0.5.
2. Mean signal/noise across all 5 conditions must be greater than 1.0. [Signal/noise is defined as mean( measurement value over 68 nts )/mean( statistical error in measurement value over 68 nts)]
3. To help ensure sequence diversity, the resulting sequences were clustered into clusters with less than 50% sequence similarity, and the 240 test set sequences were chosen from clusters with 3 or fewer members. That is, any sequence in the test set should be sequence similar to at most 2 other sequences.

Note that these filters have not been applied to the 2160 RNAs in the public training data `train.json` -- some of those measurements have negative values or poor signal-to-noise, or some RNA sequences have near-identical sequences in that set. But we are providing all those data in case competitors can squeeze out more signal.

# 2. Python version

3.8

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
tqdm==4.67.1

# 4. Data file paths

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

# 5. Target score

0.40564

# 6. Current score

0.28915

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.28892) has done: 'I (1) remove `tensorflow_addons` usage because it’s incompatible with the provided TF/protobuf stack and is causing the very first import-time crash, (2) fix the missing `train_test_split`/`KFold` import usage and make the train/val split deterministic, and (3) replace the invalid `tf.reshape` on a KerasTensor with an equivalent Keras layer (`Reshape`) so model building works in TF 2.18. I also fix test preprocessing to handle the (now only) 107-length test set correctly (no empty “private” split), and ensure predictions are expanded to all 107 positions per id and merged back into `sample_submission.csv` to produce a valid `submission.csv` with the correct columns and row order.'
- What this solution (achieved 0.28563) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by pinning protobuf to the TensorFlow-compatible “python” implementation before TensorFlow loads, which is the root cause of the first-cell failure in this environment. I also fix a small logic bug where `~train.isnull().values.any()` is not the intended boolean check and can behave incorrectly; this is score-neutral but improves correctness. Additionally, I make the inference models match the training architecture exactly and only change the `pred_len` via slicing (rather than rebuilding with different output length), which preserves core logic but avoids potential weight-shape mismatches and improves stability. No changes are made to the model layers, training loop, loss, or feature extraction beyond these compatibility/stability fixes.'
- What this solution (achieved 0.28603) has done: 'You’re currently crashing at TensorFlow import with `MessageFactory.GetPrototype`, which is a protobuf-runtime incompatibility; setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` isn’t sufficient in this environment unless we also force the pure-Python protobuf module to be used before TensorFlow loads. I patch cell 1 to (a) force `google.protobuf` to use the python implementation via environment + module reload, and (b) fall back cleanly if it’s already loaded, so the notebook runs end-to-end. I also keep the model/training logic unchanged and only add a small safety check to ensure the submission merge doesn’t leave missing `id_seqpos` predictions unnoticed (score-neutral, correctness-focused). No score-tuning changes are made since your score (0.28563) is already better than the target (0.40564) for a lower-is-better metric.'
- What this solution (achieved 0.28873) has done: 'I fix the TensorFlow import crash by forcing protobuf’s pure-Python implementation *before* TensorFlow is imported, and (critically) ensuring the C++ `_message` backend is not already loaded in `sys.modules` (the root cause of `MessageFactory.GetPrototype` in this environment). I keep the model, training loop, blending, and preprocessing semantics the same, only making the minimal environment/module-load adjustments needed for runtime stability. I also keep submission creation identical, with a small safety assertion that the produced submission matches the sample row count and required columns. Since your current score (0.28603) is already better than the target (0.40564) for a lower-is-better metric, I won’t introduce any score-improving changes.'
- What this solution (achieved 0.2888) has done: 'I fix the TensorFlow import crash caused by the protobuf C++ backend being loaded before TensorFlow (triggering `MessageFactory.GetPrototype` errors) by forcing the pure-Python protobuf implementation *and* preventing the C++/upb backends from importing via environment flags set before any `google.protobuf`/TF import. This is a runtime-only compatibility change and does not alter model/training logic or score behavior. I also keep the rest of the pipeline intact so it trains, predicts, and writes a valid `submission.csv` with the required columns/row order. No score-tuning changes be made because your current score (0.28873, lower-is-better) is already better than the target (0.40564), so we should avoid changing performance.'
- What this solution (achieved 0.28915) has done: 'The crash happens before any training because TensorFlow 2.18 in this environment is incompatible with the currently-installed protobuf runtime (the `MessageFactory.GetPrototype` attribute error). I fix this by forcing TensorFlow to use the pure-Python protobuf runtime via environment flags *and* by ensuring `google.protobuf` and its C++/upb backends are not imported prior to TensorFlow (then verifying the import succeeds). These changes are runtime/compatibility-only and do not alter your model, training loop, preprocessing, or submission logic, so they should be score-neutral (and you’re already better than the target on a lower-is-better metric). The rest of the pipeline remain the same and still write a valid `submission.csv` with the correct columns and row order.'

# 9. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")

import os, gc, random, math, json, sys, importlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_DISABLE_CEXT", "1")

purge_prefixes = (
    "google.protobuf.pyext",
    "google._upb",
    "google.protobuf.internal",
    "google.protobuf",
)
for name in list(sys.modules.keys()):
    if name.startswith(purge_prefixes):
        del sys.modules[name]

try:
    import google.protobuf  # noqa: F401

    importlib.reload(google.protobuf)
except Exception:
    pass

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L

from sklearn.model_selection import train_test_split, KFold

SEED = 34
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
def preprocess_inputs(df, cols=("sequence", "structure", "predicted_loop_type")):
    if df.shape[0] == 0:
        return np.zeros((0, 0, len(cols)), dtype=np.int32)
    arr = (
        df.loc[:, list(cols)]
        .applymap(lambda seq: [token2int[x] for x in seq])
        .values.tolist()
    )
    x = np.array(arr, dtype=np.int32)  # (n, 3, seq_len)
    return np.transpose(x, (0, 2, 1))  # (n, seq_len, 3)




## === cell 8
train_inputs = preprocess_inputs(train)
train_labels = np.array(train[target_cols].values.tolist(), dtype=np.float32).transpose(
    (0, 2, 1)
)  # (n, 68, 5)

print("train_inputs:", train_inputs.shape, train_inputs.dtype)
print("train_labels:", train_labels.shape, train_labels.dtype)




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
    model.compile(optimizer=tf.optimizers.Adam(), loss="mse")
    return model




## === cell 10
train_inputs_tr, val_inputs, train_labels_tr, val_labels = train_test_split(
    train_inputs, train_labels, test_size=0.1, random_state=SEED
)

print("train split:", train_inputs_tr.shape, train_labels_tr.shape)
print("val split:", val_inputs.shape, val_labels.shape)



## === cell 11
if len(tf.config.list_physical_devices("GPU")) > 0:
    print("Training on GPU")
else:
    print("Training on CPU")



## === cell 12
lr_callback = tf.keras.callbacks.ReduceLROnPlateau()



## === cell 13
gru = build_model(gru=True)
sv_gru = tf.keras.callbacks.ModelCheckpoint(
    "model_gru.weights.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
)

history_gru = gru.fit(
    train_inputs_tr,
    train_labels_tr,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=70,
    callbacks=[lr_callback, sv_gru],
    verbose=2,
)

print(
    f"Min training loss={min(history_gru.history['loss'])}, min validation loss={min(history_gru.history['val_loss'])}"
)



## === cell 14
lstm = build_model(gru=False)
sv_lstm = tf.keras.callbacks.ModelCheckpoint(
    "model_lstm.weights.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
)

history_lstm = lstm.fit(
    train_inputs_tr,
    train_labels_tr,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=70,
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
public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()

public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df)

print("public_df:", public_df.shape, "public_inputs:", public_inputs.shape)
print("private_df:", private_df.shape, "private_inputs:", private_inputs.shape)



## === cell 17
gru_infer = build_model(gru=True, seq_len=107, pred_len=68)
lstm_infer = build_model(gru=False, seq_len=107, pred_len=68)

gru_infer.load_weights("model_gru.weights.h5")
lstm_infer.load_weights("model_lstm.weights.h5")

gru_public_68 = gru_infer.predict(public_inputs, batch_size=64, verbose=0)  # (n,68,5)
lstm_public_68 = lstm_infer.predict(public_inputs, batch_size=64, verbose=0)


def extend_to_full_len(pred68, full_len=107):
    if pred68.shape[0] == 0:
        return np.zeros((0, full_len, 5), dtype=np.float32)
    n = pred68.shape[0]
    out = np.zeros((n, full_len, pred68.shape[-1]), dtype=np.float32)
    out[:, : pred68.shape[1], :] = pred68
    out[:, pred68.shape[1] :, :] = pred68[:, pred68.shape[1] - 1 : pred68.shape[1], :]
    return out


gru_public_preds = extend_to_full_len(gru_public_68, full_len=107)
lstm_public_preds = extend_to_full_len(lstm_public_68, full_len=107)

if private_df.shape[0] > 0:
    gru_infer_130 = build_model(gru=True, seq_len=130, pred_len=68)
    lstm_infer_130 = build_model(gru=False, seq_len=130, pred_len=68)
    gru_infer_130.load_weights("model_gru.weights.h5")
    lstm_infer_130.load_weights("model_lstm.weights.h5")
    gru_private_68 = gru_infer_130.predict(private_inputs, batch_size=64, verbose=0)
    lstm_private_68 = lstm_infer_130.predict(private_inputs, batch_size=64, verbose=0)
    gru_private_preds = extend_to_full_len(gru_private_68, full_len=130)
    lstm_private_preds = extend_to_full_len(lstm_private_68, full_len=130)
else:
    gru_private_preds = np.zeros((0, 0, 5), dtype=np.float32)
    lstm_private_preds = np.zeros((0, 0, 5), dtype=np.float32)

print(
    "gru_public_preds:",
    gru_public_preds.shape,
    "lstm_public_preds:",
    lstm_public_preds.shape,
)



## === cell 18
preds_gru = []
for df_part, preds in [(public_df, gru_public_preds), (private_df, gru_private_preds)]:
    for i, uid in enumerate(df_part.id.values):
        single_pred = preds[i]  # (seq_len, 5)
        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
        preds_gru.append(single_df)

preds_gru_df = pd.concat(preds_gru, ignore_index=True)
preds_gru_df.head()



## === cell 19
preds_lstm = []
for df_part, preds in [
    (public_df, lstm_public_preds),
    (private_df, lstm_private_preds),
]:
    for i, uid in enumerate(df_part.id.values):
        single_pred = preds[i]
        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
        preds_lstm.append(single_df)

preds_lstm_df = pd.concat(preds_lstm, ignore_index=True)
preds_lstm_df.head()



## === cell 20
blend_preds_df = pd.DataFrame()
blend_preds_df["id_seqpos"] = preds_gru_df["id_seqpos"].values
blend_preds_df["reactivity"] = (
    0.7 * preds_gru_df["reactivity"].values + 0.3 * preds_lstm_df["reactivity"].values
)
blend_preds_df["deg_Mg_pH10"] = (
    0.7 * preds_gru_df["deg_Mg_pH10"].values + 0.3 * preds_lstm_df["deg_Mg_pH10"].values
)
blend_preds_df["deg_pH10"] = (
    0.7 * preds_gru_df["deg_pH10"].values + 0.3 * preds_lstm_df["deg_pH10"].values
)
blend_preds_df["deg_Mg_50C"] = (
    0.7 * preds_gru_df["deg_Mg_50C"].values + 0.3 * preds_lstm_df["deg_Mg_50C"].values
)
blend_preds_df["deg_50C"] = (
    0.7 * preds_gru_df["deg_50C"].values + 0.3 * preds_lstm_df["deg_50C"].values
)

blend_preds_df.head()



## === cell 21
submission = sample_sub[["id_seqpos"]].merge(blend_preds_df, on="id_seqpos", how="left")

missing = submission[target_cols].isna().any(axis=1).sum()
print("Missing prediction rows after merge:", int(missing))

for c in target_cols:
    submission[c] = submission[c].fillna(0.0).astype(np.float32)

assert (
    submission.shape[0] == sample_sub.shape[0]
), "Row count mismatch vs sample_submission"
assert (
    list(submission.columns) == ["id_seqpos"] + target_cols
), "Submission columns mismatch"

print("submission:", submission.shape)
submission.head()



## === cell 22
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
