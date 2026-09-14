# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.7082

# 6. Current score

0.28421

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28077) has done: 'I fix the environment-breaking import (tensorflow_addons) that triggers the protobuf `MessageFactory.GetPrototype` error, since it’s not used anywhere in your pipeline. Next, I restore the missing `train_test_split`/`KFold` import and replace the invalid `tf.reshape` on a KerasTensor with a proper Keras layer (Flatten/Reshape), which preserves the exact intended tensor shape for the RNN stack. I also fix the test preprocessing so it works for any `seq_length` (the competition test is 107), removing the public/private split that currently creates empty arrays and causes the transpose error. Finally, I ensure prediction-to-submission row alignment matches `sample_submission.csv` exactly and that a valid `submission.csv` is always written.'
- What this solution (achieved 0.2832) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf implementation by forcing TensorFlow to use the pure-Python protobuf runtime before importing it (this is a runtime-only change and score-neutral). I also add a small safety fallback so the code can still run even if that env var isn’t honored early enough, and I keep all model/training/prediction logic identical. Finally, I ensure the submission dataframe is strictly aligned to `sample_submission.csv` ordering and that all required columns are present and numeric, then write `submission.csv`.'
- What this solution (achieved 0.27982) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by ensuring the pure-Python protobuf runtime is selected *before* TensorFlow (and protobuf) are imported, and by providing a safe monkey-patch fallback if the environment still exposes a protobuf build missing that method. This is runtime-only and should be score-neutral, but it unblocks the notebook so it can train and generate predictions end-to-end. I also correct the null-check logic (`~train.isnull().values.any()` is wrong due to bitwise negation) to avoid misleading diagnostics, without affecting modeling. Finally, I keep the model/training/prediction logic unchanged and ensure `submission.csv` is written with correct columns and ordering.'
- What this solution (achieved 0.28446) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any protobuf/TensorFlow import* and removing the brittle monkey-patch that triggers the `GetPrototype` AttributeError in this environment. Then I keep your exact model/training/prediction logic the same, only adding a small safety clamp for tokenization so unexpected characters don’t crash preprocessing. Finally, I ensure the submission rows are perfectly aligned to `sample_submission.csv` and always written to `submission.csv` with the required columns and numeric dtypes (score-neutral but prevents invalid submissions).'
- What this solution (achieved 0.28635) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf runtime as early as possible and (only if needed) providing a safe fallback that restores `MessageFactory.GetPrototype` without touching model/training logic. Then I keep your architecture and training exactly the same, but correct the inference mismatch where you build a 107-length model and load weights trained for 68 positions (this was silently breaking weight loading/usage and hurting score). Finally, I ensure predictions are generated for exactly the `id_seqpos` rows in `sample_submission.csv` by slicing to each sample’s `seq_length` and aligning strictly to the sample submission ordering, producing a valid `submission.csv`.'
- What this solution (achieved 0.28497) has done: 'I fix the runtime crash occurring before TensorFlow import by removing the brittle protobuf monkey-patch that is triggering the `MessageFactory.GetPrototype` AttributeError in this environment, while still forcing the pure-Python protobuf runtime early (score-neutral). Then I keep the model/training/inference logic the same, only adding small safety guards to ensure deterministic, correct input shapes and that the submission rows align exactly to `sample_submission.csv` with no missing ids. Finally, I ensure the script always writes a valid `submission.csv` with the required columns and numeric dtypes.'
- What this solution (achieved 0.32568) has done: 'I fix the TensorFlow import crash (`MessageFactory` missing `GetPrototype`) by forcing the pure-Python protobuf runtime and then adding a small, safe compatibility shim that defines `GetPrototype` via `GetMessageClass` only if it’s missing, before importing TensorFlow. This is a runtime-only change and won’t alter your model/training logic, but it unblocks execution so training and submission generation can complete. Because your current score is far worse than the target, I also fix the shape mismatch bug where `train_labels` are length 68 but the model is being trained against 107 timesteps; this currently makes training invalid/incorrect and prevents meaningful learning. Finally, I keep your architecture/loops intact and ensure the submission remains aligned to `sample_submission.csv` and is written as `submission.csv`.'
- What this solution (achieved 0.28421) has done: 'Your current pipeline is likely scoring poorly because it trains/predicts 107 positions and pads labels by repeating the last measured value, while Kaggle only scores the first 68; this introduces a large amount of noisy, unscored supervision that can distort learning on the scored region. I make the smallest change that aligns training/inference with the evaluation: train the model to predict only `seq_scored` (=68) by setting `pred_len` to 68 and removing the padded-label training. I keep the same model architecture (Embedding → Reshape → 3x BiRNN → Dense) and the same training loops/epochs/optimizers, only adjusting the label tensor shape and the slicing length. I also keep submission generation identical, but fill positions 68..106 by repeating position 67 predictions (instead of relying on a model trained on padded targets), which is score-neutral for the metric and preserves valid submission format.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import warnings

warnings.filterwarnings("ignore")

import gc, random, math, json, sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

try:
    from google.protobuf.message_factory import MessageFactory

    if not hasattr(MessageFactory, "GetPrototype"):

        def _getprototype(self, descriptor):
            from google.protobuf.message_factory import GetMessageClass

            return GetMessageClass(descriptor)

        MessageFactory.GetPrototype = _getprototype
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

print("TF version:", tf.__version__)



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
print("Vocab size:", len(token2int), "tokens:", list(token2int.keys()))




## === cell 7
def preprocess_inputs(df, cols=["sequence", "structure", "predicted_loop_type"]):
    fallback = token2int["."]

    def _encode(seq):
        return [token2int.get(x, fallback) for x in seq]

    arr = np.array(df[cols].applymap(_encode).values.tolist())
    return np.transpose(arr, (0, 2, 1)).astype(np.int32)




## === cell 8
train_inputs = preprocess_inputs(train)

SEQ_LEN = int(train["seq_length"].iloc[0])  # expected 107

PRED_LEN = int(train["seq_scored"].iloc[0])  # expected 68

train_labels = (
    np.array(train[target_cols].values.tolist()).transpose((0, 2, 1)).astype(np.float32)
)  # (n,68,5)

print("train_inputs:", train_inputs.shape, train_inputs.dtype)
print("train_labels:", train_labels.shape, train_labels.dtype)
print("SEQ_LEN:", SEQ_LEN, "PRED_LEN:", PRED_LEN)




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
    gru=False,
    lstm=False,
    seq_len=107,
    pred_len=68,
    dropout=0.5,
    embed_dim=100,
    hidden_dim=128,
):
    inputs = tf.keras.layers.Input(shape=(seq_len, 3), dtype=tf.int32)

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        inputs
    )
    reshaped = tf.keras.layers.Reshape((seq_len, 3 * embed_dim))(embed)

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
    model.compile(tf.keras.optimizers.Adam(), loss="mse")
    return model




## === cell 10
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    train_inputs, train_labels, test_size=0.1, random_state=SEED
)
print("Split shapes:")
print(" train_inputs:", train_inputs.shape, "val_inputs:", val_inputs.shape)
print(" train_labels:", train_labels.shape, "val_labels:", val_labels.shape)



## === cell 11
if tf.config.list_physical_devices("GPU"):
    print("Training on GPU")
else:
    print("Training on CPU")



## === cell 12
gru = build_model(gru=True, seq_len=SEQ_LEN, pred_len=PRED_LEN)
sv_gru = tf.keras.callbacks.ModelCheckpoint(
    "model_gru.weights.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
)

history_gru = gru.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=70,
    callbacks=[tf.keras.callbacks.ReduceLROnPlateau(), sv_gru],
    verbose=0,
)

print(
    f"Min training loss={min(history_gru.history['loss'])}, min validation loss={min(history_gru.history['val_loss'])}"
)



## === cell 13
lstm = build_model(lstm=True, seq_len=SEQ_LEN, pred_len=PRED_LEN)
sv_lstm = tf.keras.callbacks.ModelCheckpoint(
    "model_lstm.weights.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
)

history_lstm = lstm.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=70,
    callbacks=[tf.keras.callbacks.ReduceLROnPlateau(), sv_lstm],
    verbose=0,
)

print(
    f"Min training loss={min(history_lstm.history['loss'])}, min validation loss={min(history_lstm.history['val_loss'])}"
)



## === cell 14
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].plot(history_gru.history["loss"])
ax[0].plot(history_gru.history["val_loss"])

ax[1].plot(history_lstm.history["loss"])
ax[1].plot(history_lstm.history["val_loss"])

ax[0].set_title("GRU")
ax[1].set_title("LSTM")

ax[0].set_ylabel("Loss")
ax[0].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax[1].set_xlabel("Epoch")



## === cell 15
test_df = test.copy()
test_inputs = preprocess_inputs(test_df)

print("test_inputs:", test_inputs.shape)



## === cell 16
gru_full = build_model(gru=True, seq_len=SEQ_LEN, pred_len=PRED_LEN)
lstm_full = build_model(lstm=True, seq_len=SEQ_LEN, pred_len=PRED_LEN)

gru_full.load_weights("model_gru.weights.h5")
lstm_full.load_weights("model_lstm.weights.h5")

gru_preds_scored = gru_full.predict(test_inputs, batch_size=64, verbose=0)  # (n,68,5)
lstm_preds_scored = lstm_full.predict(test_inputs, batch_size=64, verbose=0)  # (n,68,5)

print(
    "gru_preds_scored:",
    gru_preds_scored.shape,
    "lstm_preds_scored:",
    lstm_preds_scored.shape,
)



## === cell 17
preds_gru = []
preds_lstm = []

for i, (uid, seq_len) in enumerate(zip(test_df.id.values, test_df.seq_length.values)):
    scored = int(test_df.seq_scored.values[i])  # expected 68
    full_len = int(seq_len)  # expected 107

    single_gru_scored = gru_preds_scored[i, :scored]  # (68,5)
    single_lstm_scored = lstm_preds_scored[i, :scored]  # (68,5)

    if full_len > scored:
        pad_len = full_len - scored
        gru_pad = np.repeat(single_gru_scored[-1:, :], pad_len, axis=0)
        lstm_pad = np.repeat(single_lstm_scored[-1:, :], pad_len, axis=0)
        single_gru_full = np.concatenate(
            [single_gru_scored, gru_pad], axis=0
        )  # (107,5)
        single_lstm_full = np.concatenate(
            [single_lstm_scored, lstm_pad], axis=0
        )  # (107,5)
    else:
        single_gru_full = single_gru_scored[:full_len]
        single_lstm_full = single_lstm_scored[:full_len]

    df_gru = pd.DataFrame(single_gru_full, columns=target_cols)
    df_gru["id_seqpos"] = [f"{uid}_{x}" for x in range(df_gru.shape[0])]
    preds_gru.append(df_gru)

    df_lstm = pd.DataFrame(single_lstm_full, columns=target_cols)
    df_lstm["id_seqpos"] = [f"{uid}_{x}" for x in range(df_lstm.shape[0])]
    preds_lstm.append(df_lstm)

preds_gru_df = pd.concat(preds_gru, axis=0, ignore_index=True)
preds_lstm_df = pd.concat(preds_lstm, axis=0, ignore_index=True)

print(preds_gru_df.shape, preds_lstm_df.shape)
preds_gru_df.head()



## === cell 18
blend_preds_df = pd.DataFrame()
blend_preds_df["id_seqpos"] = preds_gru_df["id_seqpos"]
blend_preds_df["reactivity"] = (
    preds_gru_df["reactivity"] + preds_lstm_df["reactivity"]
) / 2
blend_preds_df["deg_Mg_pH10"] = (
    preds_gru_df["deg_Mg_pH10"] + preds_lstm_df["deg_Mg_pH10"]
) / 2
blend_preds_df["deg_pH10"] = (preds_gru_df["deg_pH10"] + preds_lstm_df["deg_pH10"]) / 2
blend_preds_df["deg_Mg_50C"] = (
    preds_gru_df["deg_Mg_50C"] + preds_lstm_df["deg_Mg_50C"]
) / 2
blend_preds_df["deg_50C"] = (preds_gru_df["deg_50C"] + preds_lstm_df["deg_50C"]) / 2

blend_preds_df.head()



## === cell 19
submission = sample_sub[["id_seqpos"]].merge(blend_preds_df, on="id_seqpos", how="left")

for c in target_cols:
    submission[c] = submission[c].fillna(0.0).astype(np.float32)

submission = submission[["id_seqpos"] + target_cols]

print("submission shape:", submission.shape)
submission.head()



## === cell 20
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print("Columns:", submission.columns.tolist())
print("Any NA:", submission.isna().any().to_dict())
print(
    "First/last id_seqpos:", submission.id_seqpos.iloc[0], submission.id_seqpos.iloc[-1]
)
