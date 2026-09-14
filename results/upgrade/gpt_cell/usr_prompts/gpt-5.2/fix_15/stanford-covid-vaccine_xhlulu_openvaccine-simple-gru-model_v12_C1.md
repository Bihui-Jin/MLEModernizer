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

0.39012

# 6. Current score

0.24983

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.25024) has done: 'Your code doesn’t yield a score mainly because it may fail before producing a valid `submission.csv` (the protobuf downgrade/re-exec is risky on Kaggle, and there’s no safeguard to ensure predictions align 1:1 with `sample_submission.csv`). I remove the protobuf self-downgrade (TensorFlow 2.18 works with your installed protobuf 6.x), add a strict alignment step that generates predictions in exactly the same `id_seqpos` order as the sample submission, and keep your model/training logic unchanged. I also ensure `pred_cols` ordering matches the submission columns and that all required rows are filled deterministically. These minimal fixes should produce a valid submission and typically improve score vs. silent misalignment/zero-filled rows, moving toward your target.'
- What this solution (achieved 0.24923) has done: 'The crash happens during TensorFlow import because the installed `protobuf==6.33.0` is incompatible with `tensorflow==2.18.0` in this environment, triggering `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The minimal fix is to force protobuf to use the pure-Python implementation before importing TensorFlow, which avoids the incompatible C++/upb path. This change is localized to cell 0 and preserves all modeling/training logic. No other cells need modification and TensorFlow import cleanly.'
- What this solution (achieved 0.25094) has done: 'Diagnosis: The crash happens while importing TensorFlow because the notebook forces `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="cpp"`, but the installed `protobuf==6.33.0` in this environment does not provide the expected C++ extension module (`google.protobuf.pyext._message`). That makes TensorFlow’s generated `*_pb2.py` imports fail during initialization. This is an environment compatibility issue, not a model/code logic issue.

Patch summary: In cell 0, stop forcing the Protobuf C++ implementation and instead force the pure-Python implementation (`"python"`), which is available with protobuf 6.x and avoids the missing `_message` import. This change is minimal, deterministic, and only affects the import-time runtime configuration, leaving all subsequent TensorFlow/Keras logic unchanged.

Updated cells: Only cell 0 is modified.

Compatibility notes for cell k+1: All imports (`tf`, `L`, `train_test_split`, etc.) remain available under the same names, so cell 1 (`tf.random.set_seed(...)`) work unchanged.

Assumptions: The environment does not have a working Protobuf C++ extension compatible with protobuf 6.x, and switching to the Python Protobuf runtime is acceptable for this notebook (performance impact is negligible for this dataset/training setup).'
- What this solution (achieved 0.25223) has done: 'The crash happens before any model/data code runs: importing TensorFlow triggers a protobuf compatibility issue (`MessageFactory.GetPrototype` missing). This comes from having `protobuf==6.33.0`, which is incompatible with the TensorFlow 2.18 stack in this environment. The minimal fix is to force TensorFlow to use the Python protobuf implementation before importing it, and to apply a small compatibility shim that provides `GetPrototype` via `GetMessageClass` when protobuf v6 removes it. This keeps the rest of the notebook unchanged and unblocks execution so cell 1 (and later) can run normally.'
- What this solution (achieved 0.24983) has done: 'Diagnosis: The crash happens during the protobuf compatibility shim in cell 0. In protobuf 6.x, `google.protobuf.message_factory.MessageFactory` no longer exposes `GetPrototype` or `GetMessageClass` the way older versions did, so the current conditional patch doesn’t run and some downstream import path still tries to call `GetPrototype`, triggering `AttributeError`. The fix is to always provide a `GetPrototype` method on `MessageFactory` when it’s missing, using the appropriate fallback available in the installed protobuf version (either `GetMessageClass` on the instance/class, or the module-level `message_factory.GetMessageClass`).  

Patch summary: Modify only cell 0 to make the protobuf shim robust across protobuf versions by defining `MessageFactory.GetPrototype` whenever it is absent, using whichever `GetMessageClass` implementation exists. This keeps the rest of the notebook unchanged and unblocks execution.

Updated cells: cell 0 only.

Compatibility notes for cell k+1: No variables or interfaces used by cell 1 are changed; imports and seeding work the same.

Assumptions: The environment is using protobuf 6.33.0 as listed, and the intent of the shim is purely to restore the legacy `GetPrototype` API expected by some dependency.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import google.protobuf  # noqa: F401
from google.protobuf.message_factory import MessageFactory
import google.protobuf.message_factory as _pb_message_factory

if not hasattr(MessageFactory, "GetPrototype"):

    def _get_prototype(self, descriptor):
        if hasattr(self, "GetMessageClass"):
            return self.GetMessageClass(descriptor)
        if hasattr(MessageFactory, "GetMessageClass"):
            return MessageFactory.GetMessageClass(descriptor)
        if hasattr(_pb_message_factory, "GetMessageClass"):
            return _pb_message_factory.GetMessageClass(descriptor)
        raise AttributeError(
            "No GetMessageClass available to emulate MessageFactory.GetPrototype"
        )

    MessageFactory.GetPrototype = _get_prototype

import json  # noqa: F401
import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
import tensorflow.keras.layers as L
from sklearn.model_selection import train_test_split


## === cell 1
tf.random.set_seed(2020)
np.random.seed(2020)



## === cell 2
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]



## === cell 3
y_true = tf.random.normal((32, 68, 3))
y_pred = tf.random.normal((32, 68, 3))




## === cell 4
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(
        tf.square(y_true - y_pred), axis=1
    )  # (batch, n_targets)
    per_sample = tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)  # (batch,)
    return tf.reduce_mean(per_sample)  # scalar




## === cell 5
def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(
        L.GRU(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )




## === cell 6
def build_model(
    embed_size,
    seq_len=107,
    pred_len=68,
    dropout=0.5,
    sp_dropout=0.2,
    embed_dim=75,
    hidden_dim=128,
    n_layers=2,
):
    inputs = L.Input(shape=(seq_len, 3), dtype=tf.int32)

    x0 = L.Embedding(input_dim=embed_size, output_dim=embed_dim)(inputs[..., 0])
    x1 = L.Embedding(input_dim=embed_size, output_dim=embed_dim)(inputs[..., 1])
    x2 = L.Embedding(input_dim=embed_size, output_dim=embed_dim)(inputs[..., 2])
    hidden = L.Concatenate(axis=-1)([x0, x1, x2])  # (batch, seq_len, 3*embed_dim)

    hidden = L.SpatialDropout1D(sp_dropout)(hidden)

    for _ in range(n_layers):
        hidden = gru_layer(hidden_dim, dropout)(hidden)

    truncated = hidden[:, :pred_len]
    out = L.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(tf.optimizers.Adam(), loss=MCRMSE)
    return model




## === cell 7
def pandas_list_to_array(df):
    """
    Input: dataframe of shape (x, y), containing list of length l
    Return: np.array of shape (x, l, y)
    """
    return np.transpose(np.array(df.values.tolist()), (0, 2, 1))




## === cell 8
def preprocess_inputs(
    df, token2int, cols=["sequence", "structure", "predicted_loop_type"]
):
    arr = pandas_list_to_array(
        df[cols].applymap(lambda seq: [token2int[x] for x in seq])
    )
    return arr.astype(np.int32)




## === cell 9
data_dir = "/kaggle/input/stanford-covid-vaccine/"
train = pd.read_json(os.path.join(data_dir, "train.json"), lines=True)
test = pd.read_json(os.path.join(data_dir, "test.json"), lines=True)
sample_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))



## === cell 10
train = train.query("SN_filter == 1").copy()



## === cell 11
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}

train_inputs = preprocess_inputs(train, token2int)
train_labels = pandas_list_to_array(train[pred_cols]).astype(np.float32)



## === cell 12
x_train, x_val, y_train, y_val = train_test_split(
    train_inputs, train_labels, test_size=0.2, random_state=34
)



## === cell 13
test_df = test.query("seq_length == 107").copy()
test_inputs = preprocess_inputs(test_df, token2int)



## === cell 14
model = build_model(embed_size=len(token2int))
model.summary()



## === cell 15
CKPT_PATH = "model.weights.h5"

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    batch_size=64,
    epochs=100,
    verbose=2,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=5),
        tf.keras.callbacks.ModelCheckpoint(
            CKPT_PATH,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
        ),
    ],
)



## === cell 16
fig = px.line(
    history.history,
    y=["loss", "val_loss"],
    labels={"index": "epoch", "value": "MCRMSE"},
    title="Training History",
)
fig.show()



## === cell 17
model_infer = build_model(seq_len=107, pred_len=68, embed_size=len(token2int))
model_infer.load_weights(CKPT_PATH)

test_preds_68 = model_infer.predict(
    test_inputs, batch_size=64, verbose=0
)  # (n_test, 68, 5)

n_test = test_preds_68.shape[0]
pad_len = 107 - test_preds_68.shape[1]
last_step = test_preds_68[:, -1:, :]  # (n_test, 1, 5)
test_preds_107 = np.concatenate(
    [test_preds_68, np.repeat(last_step, pad_len, axis=1)], axis=1
).astype(
    np.float32
)  # (n_test, 107, 5)



## === cell 18
preds_ls = []
for i, uid in enumerate(test_df.id.values):
    single_pred = test_preds_107[i]  # (107, 5)
    single_df = pd.DataFrame(single_pred, columns=pred_cols)
    single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
    preds_ls.append(single_df)

preds_df = pd.concat(preds_ls, ignore_index=True)
for c in pred_cols:
    preds_df[c] = preds_df[c].astype(np.float32)

submission = sample_df[["id_seqpos"]].merge(
    preds_df, on="id_seqpos", how="left", validate="one_to_one"
)
for c in pred_cols:
    submission[c] = submission[c].fillna(0.0).astype(np.float32)

submission = submission[["id_seqpos"] + pred_cols]
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("Null counts per column:\n", submission.isna().sum())
