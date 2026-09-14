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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
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

0.40747

# 6. Current score

0.35996

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.28852) has done: 'I remove the `plotly` import that is triggering the protobuf `MessageFactory` error and keep visualization optional so the notebook can run in Kaggle’s environment. I fix the Keras Functional model construction error by replacing the raw `tf.reshape` with a Keras `Reshape` layer (same tensor math, valid in TF/Keras 2.18). I also fix inference for the test set by not assuming a private set of length 130 (this dataset’s test is all length 107), and by producing predictions for all 107 positions by padding the last 39 positions with the model’s last available prediction (score-neutral since those positions are not scored). Finally, I ensure the generated `submission.csv` matches the sample submission’s row order and required columns.'
- What this solution (achieved 0.29362) has done: 'I fix the runtime crash caused by an incompatible protobuf/TF interaction by forcing the pure-Python protobuf implementation before importing TensorFlow, which resolves the `MessageFactory` `GetPrototype` error in Kaggle. I also make the ModelCheckpoint monitor valid by explicitly providing validation data (via a deterministic split) rather than relying on `validation_split`, which can interact poorly with callbacks expecting a stable validation stream. To move the score toward your higher-error target (0.40747) while keeping core modeling/training unchanged, I only adjust the final prediction calibration by blending predictions slightly toward 0 (a score-controlled, post-processing-only change that preserves model semantics and produces a valid submission). The pipeline still run end-to-end and write `submission.csv` in the required format and row order.'
- What this solution (achieved 0.29758) has done: 'We fix the crash in the very first cell caused by TensorFlow 2.18 importing `google.protobuf` APIs that are incompatible with the forced pure-Python protobuf implementation; the minimal fix is to stop overriding protobuf implementation/version and let TensorFlow use the default C++ protobuf. Then we keep the existing model/training/inference logic intact, only adding a safe fallback import guard and ensuring seeds are set after TensorFlow import. Finally, we keep the existing post-processing (SHRINK) and submission construction unchanged so the score behavior stays close to your current calibrated level while producing a valid `submission.csv`.'
- What this solution (achieved 0.36099) has done: 'I fix the TensorFlow/protobuf import crash by forcing the compatible pure-Python protobuf implementation *before* importing TensorFlow (TF 2.18 on Kaggle can otherwise hit `MessageFactory.GetPrototype` errors). I keep the model, training loop, and inference logic the same, but make the environment setup deterministic and safe in this Kaggle runtime. To move your score toward the higher (worse) target (0.40747) from the current 0.29758 (lower is better), I make the smallest possible calibration-only change by increasing the existing `SHRINK` factor so predictions are shrunk closer to 0 (this should mildly worsen MCRMSE without changing the core model). The script still run end-to-end and write a valid `submission.csv` matching the sample submission row order and columns.'
- What this solution (achieved 0.36181) has done: 'I fix the runtime crash in the first cell caused by forcing an incompatible pure-Python protobuf implementation with TensorFlow 2.18, by removing that override (this is the direct cause of the `MessageFactory.GetPrototype` error). I keep the model, training loop, inference, padding-to-107, and submission construction unchanged so behavior stays close to your current 0.36099 score. Since your score is already within ±10% of the target 0.40747 (lower is better), I not change the `SHRINK` calibration or anything that would intentionally move the score further. The result run end-to-end in Kaggle and write a valid `submission.csv` with the required columns and row order.'
- What this solution (achieved 0.3641) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from even importing TF by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this is the known safe ordering for TF 2.18 in Kaggle). I keep the model, training loop, inference, padding, and submission formatting identical to preserve core logic and score behavior. I also keep the existing `SHRINK` calibration unchanged since your current score (0.36181) is already within the ±10% tolerance band around the target (0.40747, lower is better). Finally, I ensure the script always writes a valid `submission.csv` with the exact required columns and row order.'
- What this solution (achieved 0.35996) has done: 'I fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf environment override, which is what triggers the `MessageFactory.GetPrototype` error in this TF 2.18 Kaggle environment. I keep the model, training loop, inference, padding-to-107, and submission formatting identical to preserve core logic and semantics. Because your current score (0.3641, lower-is-better) is already within ±10% of the target (0.40747), I won’t change `SHRINK` or other calibration knobs that would intentionally move the score. I also add a small, score-neutral guard to ensure the output CSV has the correct columns/order and no missing rows.'

# 9. Code solution

## === cell 0
import os
import random
import json
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf
import tensorflow.keras.layers as L

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF:", tf.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]



## === cell 2
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}


def preprocess_inputs(df, cols=("sequence", "structure", "predicted_loop_type")):
    """
    Returns int tokens shaped (n_samples, seq_len, 3).
    """
    arr = (
        df.loc[:, list(cols)]
        .applymap(lambda seq: [token2int[x] for x in seq])
        .values.tolist()
    )
    arr = np.array(arr, dtype=np.int32)  # (n_samples, 3, seq_len)
    arr = np.transpose(arr, (0, 2, 1))  # (n_samples, seq_len, 3)
    return arr




## === cell 3
def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True))


def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128):
    """
    Core architecture preserved.
    """
    inputs = L.Input(shape=(seq_len, 3), dtype=tf.int32)

    embed = L.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        inputs
    )  # (B, seq_len, 3, embed_dim)

    reshaped = L.Reshape((seq_len, 3 * embed_dim))(embed)

    hidden = gru_layer(hidden_dim, dropout)(reshaped)
    hidden = gru_layer(hidden_dim, dropout)(hidden)
    hidden = gru_layer(hidden_dim, dropout)(hidden)

    truncated = hidden[:, :pred_len]  # (B, pred_len, 2*hidden_dim)

    out = L.Dense(5, activation="linear")(truncated)  # (B, pred_len, 5)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(tf.keras.optimizers.Adam(), loss="mse")
    return model




## === cell 4
train_path = "/kaggle/input/stanford-covid-vaccine/train.json"
test_path = "/kaggle/input/stanford-covid-vaccine/test.json"
sample_path = "/kaggle/input/stanford-covid-vaccine/sample_submission.csv"

train = pd.read_json(train_path, lines=True)
test = pd.read_json(test_path, lines=True)
sample_df = pd.read_csv(sample_path)

print(train.shape, test.shape, sample_df.shape)
print("Test seq_length unique:", sorted(test["seq_length"].unique().tolist()))
print("Train seq_scored unique:", sorted(train["seq_scored"].unique().tolist()))



## === cell 5
train_inputs = preprocess_inputs(train)
train_labels = np.array(train[pred_cols].values.tolist(), dtype=np.float32).transpose(
    (0, 2, 1)
)  # (n, 68, 5)

print("train_inputs:", train_inputs.shape, train_inputs.dtype)
print("train_labels:", train_labels.shape, train_labels.dtype)



## === cell 6
model = build_model(seq_len=107, pred_len=68)
model.summary()



## === cell 7
n = train_inputs.shape[0]
idx = np.arange(n)
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.05
n_val = int(round(n * val_frac))
val_idx = idx[:n_val]
tr_idx = idx[n_val:]

x_tr, y_tr = train_inputs[tr_idx], train_labels[tr_idx]
x_val, y_val = train_inputs[val_idx], train_labels[val_idx]

history = model.fit(
    x_tr,
    y_tr,
    batch_size=64,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(),
        tf.keras.callbacks.ModelCheckpoint(
            "model.weights.h5",
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
        ),
    ],
    validation_data=(x_val, y_val),
    verbose=2,
)



## === cell 8
print("Final training loss:", history.history["loss"][-1])
print("Final val loss:", history.history["val_loss"][-1])



## === cell 9
test_df = test.copy()
test_inputs = preprocess_inputs(test_df)

if os.path.exists("model.weights.h5"):
    model.load_weights("model.weights.h5")

test_preds_68 = model.predict(test_inputs, batch_size=64, verbose=0)  # (n_test, 68, 5)
print("test_preds_68:", test_preds_68.shape)

pad_len = 107 - test_preds_68.shape[1]
if pad_len > 0:
    last = test_preds_68[:, -1:, :]  # (n_test, 1, 5)
    pad = np.repeat(last, repeats=pad_len, axis=1)  # (n_test, pad_len, 5)
    test_preds_107 = np.concatenate([test_preds_68, pad], axis=1)
else:
    test_preds_107 = test_preds_68[:, :107, :]

print("test_preds_107:", test_preds_107.shape)

SHRINK = 0.60
test_preds_107 = test_preds_107 * SHRINK



## === cell 10
preds_ls = []
for i, uid in enumerate(test_df["id"].values):
    single_pred = test_preds_107[i]  # (107, 5)
    single_df = pd.DataFrame(single_pred, columns=pred_cols)
    single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
    preds_ls.append(single_df)

preds_df = pd.concat(preds_ls, axis=0, ignore_index=True)
print(preds_df.shape)
print(preds_df.head())



## === cell 11
submission = sample_df[["id_seqpos"]].merge(preds_df, on="id_seqpos", how="left")

for c in pred_cols:
    submission[c] = submission[c].astype(np.float32)
submission[pred_cols] = submission[pred_cols].fillna(0.0)

submission = submission[["id_seqpos"] + pred_cols]

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path, "shape:", submission.shape)
print("Missing after merge:", submission[pred_cols].isna().sum().sum())
print(submission.head())
