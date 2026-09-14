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

0.38527

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")

import pandas as pd, numpy as np
import json, gc, random, os, sys
from tqdm import tqdm

import tensorflow as tf

try:
    import tensorflow_addons as tfa
except Exception:
    tfa = None

from sklearn.model_selection import train_test_split, KFold



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
sample_sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")



## === cell 2
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]



## === cell 3
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}




## === cell 4
def preprocess_inputs(
    df, cols=["sequence", "structure", "predicted_loop_type"], seq_len=None
):
    """
    Convert the three character strings into integer token arrays.
    Handles empty dataframes by returning an empty array with the expected shape.
    """
    if df.empty:
        seq_len = seq_len or 107
        return np.empty((0, seq_len, len(cols)), dtype=int)

    tokenised = df[cols].applymap(lambda seq: [token2int[x] for x in seq])
    arr = np.array(tokenised.values.tolist())
    return np.transpose(arr, (0, 2, 1))




## === cell 5
train_inputs = preprocess_inputs(train[train.signal_to_noise > 1])
train_labels = np.array(train[train.signal_to_noise > 1][target_cols].values.tolist())
train_labels = train_labels.transpose((0, 2, 1))




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
    gru=1, seq_len=107, pred_len=68, dropout=0.5, embed_dim=75, hidden_dim=128
):
    """
    Build the sequence‑to‑sequence model.
    The embedding output (batch, seq_len, 3, embed_dim) is reshaped to
    (batch, seq_len, 3*embed_dim) before the recurrent layers.
    """
    inputs = tf.keras.layers.Input(shape=(seq_len, 3), dtype="int32")

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        inputs
    )
    reshaped = tf.keras.layers.Reshape((seq_len, 3 * embed_dim))(embed)

    x = tf.keras.layers.SpatialDropout1D(0.2)(reshaped)

    if gru == 1:
        x = gru_layer(hidden_dim, dropout)(x)
        x = gru_layer(hidden_dim, dropout)(x)
        x = gru_layer(hidden_dim, dropout)(x)
    elif gru == 0:
        x = lstm_layer(hidden_dim, dropout)(x)
        x = lstm_layer(hidden_dim, dropout)(x)
        x = lstm_layer(hidden_dim, dropout)(x)
    elif gru == 4:
        x = gru_layer(hidden_dim, dropout)(x)
        x = lstm_layer(hidden_dim, dropout)(x)
        x = gru_layer(hidden_dim, dropout)(x)
    elif gru == 3:
        x = lstm_layer(hidden_dim, dropout)(x)
        x = gru_layer(hidden_dim, dropout)(x)
        x = lstm_layer(hidden_dim, dropout)(x)

    x = x[:, :pred_len]

    out = tf.keras.layers.Dense(5, activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    model.compile(optimizer=tf.optimizers.Adam(), loss="mse")
    return model




## === cell 7
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    train_inputs, train_labels, test_size=0.1, random_state=34
)



## === cell 8
if tf.config.list_physical_devices("GPU"):
    print("Training on GPU")
else:
    print("Training on CPU")



## === cell 9
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1)



## === cell 10
gru = build_model(gru=1)
sv_gru = tf.keras.callbacks.ModelCheckpoint(
    "model_gru.h5", save_best_only=True, monitor="val_loss"
)

history_gru = gru.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=10,  # modest epochs for quick turnaround
    callbacks=[lr_callback, sv_gru],
    verbose=2,
)

print(
    f"Min training loss={min(history_gru.history['loss']):.5f}, "
    f"min validation loss={min(history_gru.history['val_loss']):.5f}"
)



## === cell 11
lstm = build_model(gru=0)
sv_lstm = tf.keras.callbacks.ModelCheckpoint(
    "model_lstm.h5", save_best_only=True, monitor="val_loss"
)

history_lstm = lstm.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=10,
    callbacks=[lr_callback, sv_lstm],
    verbose=2,
)

print(
    f"Min training loss={min(history_lstm.history['loss']):.5f}, "
    f"min validation loss={min(history_lstm.history['val_loss']):.5f}"
)



## === cell 12
hyb1 = build_model(gru=3)
sv_hyb1 = tf.keras.callbacks.ModelCheckpoint(
    "model_hyb1.h5", save_best_only=True, monitor="val_loss"
)

history_hyb1 = hyb1.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=10,
    callbacks=[lr_callback, sv_hyb1],
    verbose=2,
)

print(
    f"Min training loss={min(history_hyb1.history['loss']):.5f}, "
    f"min validation loss={min(history_hyb1.history['val_loss']):.5f}"
)



## === cell 13
public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()

public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df, seq_len=130)



## === cell 14
gru_short = build_model(gru=1, seq_len=107, pred_len=107)
gru_long = build_model(gru=1, seq_len=130, pred_len=130)
lstm_short = build_model(gru=0, seq_len=107, pred_len=107)
lstm_long = build_model(gru=0, seq_len=130, pred_len=130)
hyb1_short = build_model(gru=3, seq_len=107, pred_len=107)
hyb1_long = build_model(gru=3, seq_len=130, pred_len=130)

for model, path in [
    (gru_short, "model_gru.h5"),
    (gru_long, "model_gru.h5"),
    (lstm_short, "model_lstm.h5"),
    (lstm_long, "model_lstm.h5"),
    (hyb1_short, "model_hyb1.h5"),
    (hyb1_long, "model_hyb1.h5"),
]:
    if os.path.exists(path):
        model.load_weights(path)

gru_public_preds = gru_short.predict(public_inputs, verbose=0)
gru_private_preds = gru_long.predict(private_inputs, verbose=0)
lstm_public_preds = lstm_short.predict(public_inputs, verbose=0)
lstm_private_preds = lstm_long.predict(private_inputs, verbose=0)
hyb1_public_preds = hyb1_short.predict(public_inputs, verbose=0)
hyb1_private_preds = hyb1_long.predict(private_inputs, verbose=0)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
UnboundLocalError                         Traceback (most recent call last)
/tmp/ipykernel_55/1967884122.py in <cell line: 0>()
     20 
     21 gru_public_preds = gru_short.predict(public_inputs, verbose=0)
---> 22 gru_private_preds = gru_long.predict(private_inputs, verbose=0)
     23 lstm_public_preds = lstm_short.predict(public_inputs, verbose=0)
     24 lstm_private_preds = lstm_long.predict(private_inputs, verbose=0)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py in predict(self, x, batch_size, verbose, steps, callbacks)
    567         callbacks.on_predict_end()
    568         outputs = tree.map_structure_up_to(
--> 569             batch_outputs, potentially_ragged_concat, outputs
    570         )
    571         return tree.map_structure(convert_to_np_if_not_ragged, outputs)

UnboundLocalError: cannot access local variable 'batch_outputs' where it is not associated with a value

## === cell 15
preds_gru = []
for df, preds in [(public_df, gru_public_preds), (private_df, gru_private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]
        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
        preds_gru.append(single_df)
preds_gru_df = pd.concat(preds_gru, ignore_index=True)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2114121147.py in <cell line: 0>()
      1 preds_gru = []
----> 2 for df, preds in [(public_df, gru_public_preds), (private_df, gru_private_preds)]:
      3     for i, uid in enumerate(df.id):
      4         single_pred = preds[i]
      5         single_df = pd.DataFrame(single_pred, columns=target_cols)

NameError: name 'gru_private_preds' is not defined

## === cell 16
preds_lstm = []
for df, preds in [(public_df, lstm_public_preds), (private_df, lstm_private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]
        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
        preds_lstm.append(single_df)
preds_lstm_df = pd.concat(preds_lstm, ignore_index=True)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3979487165.py in <cell line: 0>()
      1 preds_lstm = []
----> 2 for df, preds in [(public_df, lstm_public_preds), (private_df, lstm_private_preds)]:
      3     for i, uid in enumerate(df.id):
      4         single_pred = preds[i]
      5         single_df = pd.DataFrame(single_pred, columns=target_cols)

NameError: name 'lstm_public_preds' is not defined

## === cell 17
preds_hyb1 = []
for df, preds in [(public_df, hyb1_public_preds), (private_df, hyb1_private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]
        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
        preds_hyb1.append(single_df)
preds_hyb1_df = pd.concat(preds_hyb1, ignore_index=True)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1896233708.py in <cell line: 0>()
      1 preds_hyb1 = []
----> 2 for df, preds in [(public_df, hyb1_public_preds), (private_df, hyb1_private_preds)]:
      3     for i, uid in enumerate(df.id):
      4         single_pred = preds[i]
      5         single_df = pd.DataFrame(single_pred, columns=target_cols)

NameError: name 'hyb1_public_preds' is not defined

## === cell 18
blend_preds_df = pd.DataFrame()
blend_preds_df["id_seqpos"] = preds_gru_df["id_seqpos"]
for col in target_cols:
    blend_preds_df[col] = (
        0.333 * preds_gru_df[col]
        + 0.334 * preds_lstm_df[col]
        + 0.333 * preds_hyb1_df[col]
    )



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3289023138.py in <cell line: 0>()
      1 # Simple averaging blend (weights sum to 1)
      2 blend_preds_df = pd.DataFrame()
----> 3 blend_preds_df["id_seqpos"] = preds_gru_df["id_seqpos"]
      4 for col in target_cols:
      5     blend_preds_df[col] = (

NameError: name 'preds_gru_df' is not defined

## === cell 19
submission = sample_sub[["id_seqpos"]].merge(blend_preds_df, on="id_seqpos", how="left")
submission[target_cols] = submission[target_cols].fillna(0)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/2883324655.py in <cell line: 0>()
----> 1 submission = sample_sub[["id_seqpos"]].merge(blend_preds_df, on="id_seqpos", how="left")
      2 # Fill any missing predictions (should not happen) with zeros
      3 submission[target_cols] = submission[target_cols].fillna(0)
      4 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10830         from pandas.core.reshape.merge import merge
  10831 
> 10832         return merge(
  10833             self,
  10834             right,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    792             left_drop,
    793             right_drop,
--> 794         ) = self._get_merge_keys()
    795 
    796         if left_drop:

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _get_merge_keys(self)
   1295                         rk = cast(Hashable, rk)
   1296                         if rk is not None:
-> 1297                             right_keys.append(right._get_label_or_level_values(rk))
   1298                         else:
   1299                             # work-around for merge_asof(right_index=True)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _get_label_or_level_values(self, key, axis)
   1909             values = self.axes[axis].get_level_values(key)._values
   1910         else:
-> 1911             raise KeyError(key)
   1912 
   1913         # Check for duplicates

KeyError: 'id_seqpos'

## === cell 20
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1674613990.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 print("Submission saved to submission.csv")

NameError: name 'submission' is not defined
