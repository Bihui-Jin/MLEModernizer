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

0.38739

# 6. Current score

0.24573

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.25996) has done: 'I remove the `tensorflow_addons` dependency that is crashing under this Kaggle image (protobuf incompatibility), and keep your model/training logic the same by using plain Adam optimizers only. I also fix the Keras Functional graph errors by replacing raw `tf.reshape` on a `KerasTensor` with a proper Keras `Reshape` layer (same tensor shape, same semantics). Next, I make preprocessing robust for any `seq_length` by padding/truncating to the requested length so public/private splits can be handled without “axes don’t match” errors. Finally, I ensure prediction-to-submission alignment by building `id_seqpos` rows for every test id and merging onto `sample_submission.csv`, then writing a valid `submission.csv`.'
- What this solution (achieved 0.24416) has done: 'I fix the import-time crash by patching the `google.protobuf` MessageFactory API change before TensorFlow is imported, which resolves the `'MessageFactory' object has no attribute 'GetPrototype'` error in this Kaggle image. Then I fix the blending KeyError by ensuring the merge applies suffixes to both GRU and LSTM prediction columns (the current merge can leave GRU columns unsuffixed, so `reactivity_gru` doesn’t exist). Finally, I keep your model/training/inference logic unchanged and only make the submission-merge more robust so it always writes a valid `submission.csv` with the correct columns and row count.'
- What this solution (achieved 0.24604) has done: 'The crash happens before TensorFlow imports because the protobuf shim only patches `MessageFactory.GetPrototype`, but the actual error in this environment is raised from an instance missing that attribute during import. I make the protobuf patch more robust by also patching the concrete C++/python message factory class that TensorFlow uses when present, and I do it before importing TensorFlow so the runtime error is removed. The rest of the pipeline (data loading, preprocessing, model definitions/training, blending, and submission building) is kept the same to avoid changing your achieved score unnecessarily. The script run end-to-end and always write a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.24573) has done: 'I fix the protobuf shim so it patches the *actual* `google.protobuf.message_factory.MessageFactory` class used at runtime (and does it before importing TensorFlow), which is why you currently still get `'MessageFactory' object has no attribute 'GetPrototype'`. I also make the null-check logic correct (the current `~train.isnull().values.any()` is a bitwise-not bug) but keep it score-neutral. Everything else (data filtering, preprocessing, model definitions, training schedules, blending, and submission construction) remain unchanged to preserve your current leaderboard behavior while restoring end-to-end execution and a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")

import os, gc, random, math, json, sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

try:
    import google.protobuf.message_factory as _mf

    if hasattr(_mf, "MessageFactory"):
        _MessageFactoryCls = _mf.MessageFactory
        if (not hasattr(_MessageFactoryCls, "GetPrototype")) and hasattr(
            _MessageFactoryCls, "GetMessageClass"
        ):

            def _GetPrototype(self, descriptor):
                return self.GetMessageClass(descriptor)

            _MessageFactoryCls.GetPrototype = _GetPrototype
except Exception:
    pass

try:
    from google.protobuf.pyext import _message as _pb_message  # type: ignore

    if hasattr(_pb_message, "MessageFactory"):
        _CppMessageFactory = _pb_message.MessageFactory
        if (not hasattr(_CppMessageFactory, "GetPrototype")) and hasattr(
            _CppMessageFactory, "GetMessageClass"
        ):

            def _GetPrototype_cpp(self, descriptor):
                return self.GetMessageClass(descriptor)

            _CppMessageFactory.GetPrototype = _GetPrototype_cpp
except Exception:
    pass

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
from tensorflow import keras
from sklearn.model_selection import train_test_split, KFold

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

print("set up complete!")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
sample_sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")

print("Data Load Complete")



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
token2int["U"]



## === cell 8
cols = ["sequence", "structure", "predicted_loop_type"]
train[cols].applymap(lambda seq: [token2int[x] for x in seq]).head(1)




## === cell 9
def preprocess_inputs(
    df, cols=("sequence", "structure", "predicted_loop_type"), seq_len=107
):
    """
    Fix: make preprocessing robust for variable seq_length by padding/truncating to seq_len.
    Output shape: (n_samples, seq_len, 3)
    """
    arr = (
        df.loc[:, list(cols)]
        .applymap(lambda s: [token2int[x] for x in s])
        .values.tolist()
    )
    x = np.array(
        arr, dtype=object
    )  # (n_samples, 3), each element is a list[int] length L
    n = len(df)
    out = np.zeros((n, seq_len, 3), dtype=np.int32)

    for i in range(n):
        for j in range(3):
            seq = x[i, j]
            Lseq = len(seq)
            if Lseq >= seq_len:
                out[i, :, j] = np.array(seq[:seq_len], dtype=np.int32)
            else:
                out[i, :Lseq, j] = np.array(seq, dtype=np.int32)
    return out




## === cell 10
train_filtered = train[train.signal_to_noise > 1].copy()
train_inputs = preprocess_inputs(train_filtered, seq_len=107)
train_y = (
    np.array(train_filtered[target_cols].values.tolist())
    .transpose((0, 2, 1))
    .astype(np.float32)
)

print(train_inputs.shape)
print(train_y.shape)



## === cell 11
inputs_demo = tf.keras.layers.Input(shape=(107, 3))
embed_demo = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=75)(
    inputs_demo
)
reshaped_demo = tf.keras.layers.Reshape((107, 3 * 75))(embed_demo)
print("Demo shapes:", embed_demo.shape, reshaped_demo.shape)



## === cell 12
embed_demo.shape




## === cell 13
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

    adam = tf.optimizers.Adam()
    model.compile(optimizer=adam, loss="mse")

    return model


print("Model structure defined")




## === cell 14
def lstm_model(seq_len=107, output_dim=75, dropout=0.5, pred_len=68):

    inputs = tf.keras.layers.Input(shape=(seq_len, 3))
    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=output_dim)(
        inputs
    )

    reshaped = tf.keras.layers.Reshape((seq_len, 3 * output_dim))(embed)

    hidden = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128, dropout=dropout, kernel_initializer="orthogonal", return_sequences=True
        )
    )(reshaped)
    hidden = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128, dropout=dropout, kernel_initializer="orthogonal", return_sequences=True
        )
    )(hidden)
    hidden = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128, dropout=dropout, kernel_initializer="orthogonal", return_sequences=True
        )
    )(hidden)

    truncated = hidden[:, :pred_len]
    output = tf.keras.layers.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    adam = tf.optimizers.Adam(learning_rate=0.01)
    model.compile(loss="mse", optimizer=adam)
    return model




## === cell 15
train_data, val_data, train_labels, val_labels = train_test_split(
    train_inputs, train_y, test_size=0.2, random_state=4
)

print(train_data.shape, val_data.shape, train_labels.shape, val_labels.shape)



## === cell 16
lr_callback = tf.keras.callbacks.ReduceLROnPlateau()



## === cell 17
smpl_lstm = lstm_model(seq_len=107, output_dim=75, dropout=0.5, pred_len=68)
sv_smpl_lstm = tf.keras.callbacks.ModelCheckpoint(
    "model_smpl_lstm.weights.h5", save_weights_only=True
)



## === cell 18
smpl_lstm.summary()



## === cell 19
history_smpl_lstm = smpl_lstm.fit(
    train_data,
    train_labels,
    validation_data=(val_data, val_labels),
    batch_size=64,
    epochs=80,
    callbacks=[lr_callback, sv_smpl_lstm],
    verbose=2,
)

print(
    f"Min training loss={min(history_smpl_lstm.history['loss'])}, min validation loss={min(history_smpl_lstm.history['val_loss'])}"
)



## === cell 20
gru = build_model(gru=True, seq_len=107, pred_len=68)
sv_gru = tf.keras.callbacks.ModelCheckpoint(
    "model_gru.weights.h5", save_weights_only=True
)



## === cell 21
gru.summary()



## === cell 22
history_gru = gru.fit(
    train_data,
    train_labels,
    validation_data=(val_data, val_labels),
    batch_size=64,
    epochs=70,
    callbacks=[lr_callback, sv_gru],
    verbose=2,
)

print(
    f"Min training loss={min(history_gru.history['loss'])}, min validation loss={min(history_gru.history['val_loss'])}"
)



## === cell 23
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].plot(history_gru.history["loss"])
ax[0].plot(history_gru.history["val_loss"])

ax[1].plot(history_smpl_lstm.history["loss"])
ax[1].plot(history_smpl_lstm.history["val_loss"])

ax[0].set_title("GRU")
ax[1].set_title("SMPL_LSTM")

ax[0].legend(["train", "validation"], loc="upper right")
ax[1].legend(["train", "validation"], loc="upper right")

ax[0].set_ylabel("Loss")
ax[0].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax[1].set_xlabel("Epoch")

plt.show()



## === cell 24
public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()  # may be empty

public_inputs = (
    preprocess_inputs(public_df, seq_len=107)
    if len(public_df)
    else np.zeros((0, 107, 3), dtype=np.int32)
)
private_inputs = (
    preprocess_inputs(private_df, seq_len=130)
    if len(private_df)
    else np.zeros((0, 130, 3), dtype=np.int32)
)

print("Test data prepared!", public_inputs.shape, private_inputs.shape)



## === cell 25
gru_short = build_model(gru=True, seq_len=107, pred_len=107)
lstm_short = lstm_model(seq_len=107, output_dim=75, dropout=0.5, pred_len=107)

gru_short.load_weights("model_gru.weights.h5")
lstm_short.load_weights("model_smpl_lstm.weights.h5")

if len(private_df):
    gru_long = build_model(gru=True, seq_len=130, pred_len=130)
    lstm_long = lstm_model(seq_len=130, output_dim=75, dropout=0.5, pred_len=130)
    gru_long.load_weights("model_gru.weights.h5")
    lstm_long.load_weights("model_smpl_lstm.weights.h5")

print("models built and trained")



## === cell 26
gru_public_preds = (
    gru_short.predict(public_inputs, verbose=0)
    if len(public_df)
    else np.zeros((0, 107, 5), dtype=np.float32)
)
lstm_public_preds = (
    lstm_short.predict(public_inputs, verbose=0)
    if len(public_df)
    else np.zeros((0, 107, 5), dtype=np.float32)
)

if len(private_df):
    gru_private_preds = gru_long.predict(private_inputs, verbose=0)
    lstm_private_preds = lstm_long.predict(private_inputs, verbose=0)
else:
    gru_private_preds = np.zeros((0, 130, 5), dtype=np.float32)
    lstm_private_preds = np.zeros((0, 130, 5), dtype=np.float32)

print(
    "Predictions complete:",
    gru_public_preds.shape,
    lstm_public_preds.shape,
    gru_private_preds.shape,
    lstm_private_preds.shape,
)



## === cell 27
preds_gru = []

for df_part, preds in [(public_df, gru_public_preds), (private_df, gru_private_preds)]:
    for i, uid in enumerate(df_part.id.values):
        single_pred = preds[i]  # (seq_len, 5)
        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
        preds_gru.append(single_df)

preds_gru_df = (
    pd.concat(preds_gru, ignore_index=True)
    if len(preds_gru)
    else pd.DataFrame(columns=target_cols + ["id_seqpos"])
)
preds_gru_df.head()



## === cell 28
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

preds_lstm_df = (
    pd.concat(preds_lstm, ignore_index=True)
    if len(preds_lstm)
    else pd.DataFrame(columns=target_cols + ["id_seqpos"])
)
preds_lstm_df.head()



## === cell 29
lstm_weight = 0.5
gru_weight = 0.5

preds_gru_sfx = preds_gru_df.rename(columns={c: f"{c}_gru" for c in target_cols})
preds_lstm_sfx = preds_lstm_df.rename(columns={c: f"{c}_lstm" for c in target_cols})

blend_preds_df = preds_gru_sfx.merge(preds_lstm_sfx, on="id_seqpos", how="inner")

for col in target_cols:
    blend_preds_df[col] = gru_weight * blend_preds_df[f"{col}_gru"].astype(
        np.float32
    ) + lstm_weight * blend_preds_df[f"{col}_lstm"].astype(np.float32)

blend_preds_df = blend_preds_df[["id_seqpos"] + target_cols]
blend_preds_df.head()



## === cell 30
submission = sample_sub[["id_seqpos"]].merge(blend_preds_df, on="id_seqpos", how="left")

submission[target_cols] = submission[target_cols].fillna(0.0)

submission = submission[["id_seqpos"] + target_cols]

submission.head(), len(submission)



## === cell 31
submission.to_csv("submission.csv", index=False)
print(
    "Submission saved:",
    os.path.abspath("submission.csv"),
    "rows:",
    len(submission),
    "cols:",
    submission.shape[1],
)
