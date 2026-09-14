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

0.40932

# 6. Current score

0.27381

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.27253) has done: 'I fix the TensorFlow/Keras runtime errors by replacing raw `tf.transpose` on a KerasTensor with a Keras `Permute` layer and by correcting the custom TF loss to compute the competition MCRMSE on the 3 scored targets with proper tensor shapes. I also address the `Tokenizer` crash by switching to the legacy Keras tokenizer import that avoids the protobuf `MessageFactory` issue in this environment. To ensure the script always produces a valid submission, I remove the unsupported sklearn wrapper cell, handle the fact that this dataset has no 130-length private set (so stacking doesn’t fail), and build the submission by aligning predictions to `sample_submission.csv`’s `id_seqpos` ordering. Core model architecture/training loop remain the same; changes are strictly to make it run and to make the loss consistent with the evaluation metric.'
- What this solution (achieved 0.27381) has done: 'I fix the runtime crash caused by the protobuf/Keras tokenizer import mismatch by switching to the stable legacy tokenizer import path and ensuring it’s imported before any other Keras submodules are pulled in. I also remove the currently-unused EarlyStopping callback to keep training behavior consistent with your stated “no relaxed convergence criteria” requirement (this is score-neutral in intent but avoids unintended early termination). Finally, I make the data path resolution more robust (without changing I/O locations) and add a small safety check to guarantee the submission rows exactly match `sample_submission.csv` ordering and length so a valid `.csv` is always produced.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam

np.random.seed(0)
tf.random.set_seed(0)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]




## === cell 2
def read_json(filename):
    """
    reads in train/test json data as pandas DataFrame
    """
    with open(filename, "r") as f:
        df = pd.read_json(path_or_buf=f, orient="records", lines=True)
    return df




## === cell 3
CANDIDATE_DIRS = [
    "/kaggle/input/stanford-covid-vaccine",
    "/kaggle/data/stanford-covid-vaccine",
    "../input/stanford-covid-vaccine",
    "../data/stanford-covid-vaccine",
]
DATA_DIR = None
for d in CANDIDATE_DIRS:
    if os.path.exists(d):
        DATA_DIR = d
        break
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find stanford-covid-vaccine dataset directory in expected locations."
    )

train_path = os.path.join(DATA_DIR, "train.json")
test_path = os.path.join(DATA_DIR, "test.json")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_df = read_json(train_path)
test_df = read_json(test_path)

print(train_df["id"].nunique())
print(train_df.columns)




## === cell 4
def unpack_df_lists(df, col_names):
    """
    turn list-like elements of dataframe into tabular data
    """
    if isinstance(col_names, str):
        col_names = [col_names]

    all_series = [df[c] for c in col_names]
    unpacked = [ser.explode() for ser in all_series]
    data = pd.concat(unpacked, axis=1)

    original = df.drop(col_names, axis=1)
    data = original.join(data)
    return data




## === cell 5
tokenize_cols = ["sequence", "structure", "predicted_loop_type"]


def tokenize_df(df, tokenizer, cols=tokenize_cols):
    """
    tokenizer is a keras Tokenizer already fitted on text
    """
    data = df.copy()
    for c in cols:
        data[c] = tokenizer.texts_to_sequences(data[c])
    return data




## === cell 6
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(filters=None, lower=False, char_level=True)
tokenizer.fit_on_texts("().ACGUBEHIMSX")

temp = train_df[train_df["SN_filter"] == 1].copy()
temp = tokenize_df(temp, tokenizer)
train_df = temp

print(train_df.shape)
train_df.head()




## === cell 7
def score(raw_values=False, use_tf=False, **kwargs):
    """
    Competition metric (MCRMSE) over scored columns:
    reactivity, deg_Mg_pH10, deg_Mg_50C

    y_true/y_pred expected shape: (batch, 5, seq_scored) in this notebook.
    """
    col_dict = {"reactivity": 0, "deg_Mg_pH10": 1, "deg_Mg_50C": 3}
    scored_idx = list(col_dict.values())

    if not use_tf:

        def loss_np(y_true, y_pred):
            from sklearn.metrics import mean_squared_error

            y_true = np.array(y_true)[:, scored_idx]
            y_pred = np.array(y_pred)[:, scored_idx]
            multi = "raw_values" if raw_values else "uniform_average"
            return mean_squared_error(y_true, y_pred, squared=False, multioutput=multi)

        return loss_np

    def loss_tf(y_true, y_pred):
        y_true = tf.gather(y_true, scored_idx, axis=1)  # (B, 3, L)
        y_pred = tf.gather(y_pred, scored_idx, axis=1)  # (B, 3, L)

        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=2)  # (B, 3)
        rmse = tf.sqrt(mse + 1e-8)  # (B, 3)
        return tf.reduce_mean(rmse, axis=1)  # (B,)

    return loss_tf




## === cell 8
train_only_cols = [
    "reactivity_error",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_error_Mg_50C",
    "deg_error_50C",
]
signal_cols = ["signal_to_noise", "SN_filter"]
drop_cols = ["seq_length", "seq_scored", "index", "id"]
train_drop_cols = drop_cols + target_cols + train_only_cols + signal_cols

X_train = (
    train_df.drop(train_drop_cols, axis=1)
    .apply(lambda row: [e for e in row], axis=1)
    .apply(lambda e: np.array(e))
)
X_train = np.stack(X_train.values, axis=0).astype(np.int32)  # (n,3,107) tokens

y_train = (
    train_df[target_cols]
    .apply(lambda row: [e for e in row], axis=1)
    .apply(lambda e: np.array(e))
)
y_train = np.stack(y_train.values, axis=0).astype(np.float32)  # (n,5,68)

print("X_train:", X_train.shape, X_train.dtype)
print("y_train:", y_train.shape, y_train.dtype)




## === cell 9
def make_model():
    """
    Creates a tensorflow keras sequence model (core architecture preserved).
    """
    EMBEDDING_PARAMS = {
        "input_dim": len(tokenizer.word_index) + 1,
        "output_dim": 100,
    }

    shape = (3, None)  # 3 sequences of unknown length

    seq_inputs = tf.keras.Input(shape=shape)  # (B, 3, seq_length)
    embed = layers.Embedding(**EMBEDDING_PARAMS)(
        seq_inputs
    )  # (B, 3, seq_length, out_dim)

    def rnn_layer(num_neurons):
        return layers.Bidirectional(
            layers.LSTM(num_neurons, return_sequences=True, dropout=0.2)
        )

    rnn_layers = []
    for i in range(embed.shape[1]):  # loop thru sequences
        r = rnn_layer(30)(embed[:, i])
        r = rnn_layer(30)(r)
        rnn_layers.append(r)

    x = layers.Concatenate()(rnn_layers)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dense(5, activation="linear")(x)  # (B, seq_length, 5)

    x = layers.Permute((2, 1))(x)  # (B, 5, seq_length)

    x = layers.Lambda(lambda t: t[:, :, :-39])(x)  # (B, 5, 68)

    model = tf.keras.Model(inputs=seq_inputs, outputs=x)

    optimizer = Adam(learning_rate=0.01)
    loss = score(use_tf=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=["mse"])
    return model




## === cell 10
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split

TF_FITPARAMS = {"epochs": 150, "batch_size": 100, "validation_batch_size": 50}
fp = TF_FITPARAMS


def schedule_func(epoch, lr):
    if epoch < 50:
        return lr
    else:
        return lr * np.exp(-0.13)


callbacks = [
    LearningRateScheduler(schedule_func),
]

model = make_model()
model.summary()
print(X_train.shape, y_train.shape)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=0
)

history = model.fit(
    X_tr, y_tr, validation_data=(X_val, y_val), callbacks=callbacks, **fp
)



## === cell 11
test_df = read_json(test_path)

test_public = test_df[test_df["seq_length"] == 107].copy()
test_public = tokenize_df(test_public, tokenizer)

test_private = test_df[test_df["seq_length"] == 130].copy()
if len(test_private) > 0:
    test_private = tokenize_df(test_private, tokenizer)

print("test_public:", test_public.shape)
print("test_private:", test_private.shape)

X_test_public = (
    test_public.drop(drop_cols, axis=1)
    .apply(lambda row: [e for e in row], axis=1)
    .apply(lambda e: np.array(e))
)
X_test_public = np.stack(X_test_public.values, axis=0).astype(np.int32)
print("X_test_public:", X_test_public.shape)

if len(test_private) > 0:
    X_test_private = (
        test_private.drop(drop_cols, axis=1)
        .apply(lambda row: [e for e in row], axis=1)
        .apply(lambda e: np.array(e))
    )
    X_test_private = np.stack(X_test_private.values, axis=0).astype(np.int32)
    print("X_test_private:", X_test_private.shape)



## === cell 12
test_pred_public = model.predict(X_test_public, batch_size=64)  # (n,5,68)
print("test_pred_public:", test_pred_public.shape)

if len(test_private) > 0:
    test_pred_private = model.predict(X_test_private, batch_size=64)
    print("test_pred_private:", test_pred_private.shape)
else:
    test_pred_private = None




## === cell 13
def preds_to_long_df(test_ids, preds_5x68, seq_length=107, seq_scored=68):
    """
    Convert per-id predictions shaped (n, 5, 68) into long dataframe with rows for each seqpos 0..seq_length-1.
    For seqpos >= seq_scored, fill 0.0.
    """
    rows = []
    for i, rid in enumerate(test_ids):
        p = preds_5x68[i]  # (5,68)
        for pos in range(seq_length):
            if pos < seq_scored:
                vals = p[:, pos]
            else:
                vals = np.zeros((5,), dtype=np.float32)
            rows.append((f"{rid}_{pos}", *vals.tolist()))
    return pd.DataFrame(rows, columns=["id_seqpos"] + target_cols)


sub_public_pred = preds_to_long_df(
    test_public["id"].values, test_pred_public, seq_length=107, seq_scored=68
)

if test_pred_private is not None and len(test_private) > 0:
    seq_scored_private = int(test_private["seq_scored"].iloc[0])
    sub_private_pred = preds_to_long_df(
        test_private["id"].values,
        test_pred_private,
        seq_length=130,
        seq_scored=seq_scored_private,
    )
    pred_all = pd.concat([sub_public_pred, sub_private_pred], ignore_index=True)
else:
    pred_all = sub_public_pred

print("Pred long df:", pred_all.shape)
pred_all.head()



## === cell 14
sample_sub = pd.read_csv(sample_sub_path)

sub_df = sample_sub[["id_seqpos"]].merge(pred_all, on="id_seqpos", how="left")

for c in target_cols:
    if c not in sub_df.columns:
        sub_df[c] = 0.0
sub_df[target_cols] = sub_df[target_cols].astype(np.float32).fillna(0.0)

sub_df = sub_df[["id_seqpos"] + target_cols]

assert len(sub_df) == len(sample_sub), (len(sub_df), len(sample_sub))
assert (sub_df["id_seqpos"].values == sample_sub["id_seqpos"].values).all()

print(sub_df.shape)
sub_df.head()



## === cell 15
out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(sub_df))
