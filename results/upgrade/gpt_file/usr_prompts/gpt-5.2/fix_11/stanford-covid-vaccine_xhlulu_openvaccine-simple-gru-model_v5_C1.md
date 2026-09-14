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

0.41269

# 6. Current score

0.29201

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.29218) has done: 'I fix the initial import crash by removing the unused `plotly` import that triggers a protobuf incompatibility in this environment. Then I fix the model-building error by replacing the raw `tf.reshape` on a KerasTensor with a proper Keras layer (`Reshape`), while keeping the same embedding→flatten-channels→stacked BiGRU→Dense logic. Next I remove the incorrect 130-length “private” branch (this dataset only has length 107 here) and make prediction always output 107 positions by padding the last 39 positions with the model’s last available (68th) prediction, which is score-neutral for the metric since only the first 68 are scored. Finally, I ensure the submission aligns exactly with `sample_submission.csv` and is written as `submission.csv`.'
- What this solution (achieved 0.29036) has done: 'I fix the protobuf/TensorFlow import crash that prevents the notebook from running by setting a safe protobuf implementation before TensorFlow is imported, without changing any modeling logic. I also make the submission-generation more robust by ensuring predictions are fully aligned to the sample submission order and by asserting there are no missing ids after the merge (so you don’t silently submit NaNs). These changes are score-neutral (they don’t alter the model/training/predictions) but ensure the pipeline runs end-to-end and always writes a valid `submission.csv`. Since your current score (0.29218, lower is better) is already better than the target (0.41269), I not make any score-improving changes.'
- What this solution (achieved 0.28739) has done: 'I fix the TensorFlow import crash by switching the protobuf implementation from the unavailable C++ backend (`cpp`) to the pure-Python backend before importing TensorFlow. This unblock execution so `tf` and `L` are defined, which also resolves the downstream `NameError` failures in model building/training/prediction cells. I keep the model/training logic identical, and only add a small safety fallback to re-import `tf`/`L` if a cell is run out of order. Finally, I ensure the pipeline always writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.28991) has done: 'I fix the TensorFlow import crash by setting the protobuf implementation to pure-Python *before* any TensorFlow import attempt, removing the invalid default `"cpp"` that triggers the `_message` ImportError in this environment. This unblock all downstream cells (model definition, training, prediction, submission creation) without changing the model architecture, training loop, or prediction semantics. I also keep deterministic seeding and add a small safety check to ensure the submission is fully aligned to `sample_submission.csv` and always written as `submission.csv`. These are correctness/stability fixes; since you don’t currently get a valid run/submission, score tuning is not applicable yet.'
- What this solution (achieved 0.29824) has done: 'You’re hitting a TensorFlow/protobuf incompatibility at import time (`MessageFactory.GetPrototype` missing), so the pipeline never reaches training or submission writing. I fix this by pinning a safe protobuf Python implementation and version *before* importing TensorFlow, and by forcing TensorFlow to use the Python protobuf backend (a common Kaggle workaround for protobuf 6.x). These changes are execution/stability-only and won’t alter the model/training logic, so the score should remain in the same band (already better than the target). I also keep the submission alignment checks intact to guarantee a valid `submission.csv`.'
- What this solution (achieved 0.29478) has done: 'The crash happens before any training because TensorFlow 2.18 in this environment is importing code paths that expect an older protobuf API (it calls `MessageFactory.GetPrototype`, which was removed in protobuf 5/6). The minimal, stable fix is to force TensorFlow to use the pure-Python protobuf implementation and to pre-import `google.protobuf` before importing TensorFlow, so it doesn’t try to use the incompatible fast backend. This is an execution-only fix and does not change your model/training/prediction logic, so it should keep your score in the same (already better-than-target) band while making the notebook run end-to-end and always write a valid `submission.csv`. I also keep all paths and the submission alignment logic unchanged.'
- What this solution (achieved 0.29311) has done: 'The only blocking issue is the TensorFlow import crash caused by a protobuf API mismatch (`MessageFactory.GetPrototype` missing) in this Kaggle environment. I apply a minimal, execution-only compatibility shim that restores `GetPrototype` by aliasing it to `GetMessageClass` (where available) before importing TensorFlow, while keeping your model/training/prediction logic unchanged. Since your current score (0.29478, lower is better) is already better than the target (0.41269), I won’t make any score-improving changes—just ensure the notebook runs end-to-end and writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.29201) has done: 'I fix the TensorFlow/protobuf import crash by applying the `GetPrototype` shim at the correct level (the C++ `MessageFactory` instance that TensorFlow uses), rather than on the Python wrapper class where it doesn’t take effect. This is an execution-only fix: it doesn’t change your model, training, or prediction logic, so it should keep your score in the same band (already better than the target, lower-is-better). I also keep the environment variables set before importing TensorFlow, and add a small defensive check to ensure the shim is applied even if protobuf internals differ slightly. The rest of the pipeline stays the same and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import google.protobuf  # noqa: F401

try:
    from google.protobuf import message_factory as _py_message_factory
    from google.protobuf.pyext import _message as _cpp_message
except Exception:
    _py_message_factory = None
    _cpp_message = None


def _patch_getprototype(factory_obj):
    if factory_obj is None:
        return
    if (not hasattr(factory_obj, "GetPrototype")) and hasattr(
        factory_obj, "GetMessageClass"
    ):
        try:
            setattr(factory_obj, "GetPrototype", factory_obj.GetMessageClass)
        except Exception:
            pass


try:
    if _cpp_message is not None and hasattr(_cpp_message, "default_pool"):
        pool = _cpp_message.default_pool
        if hasattr(pool, "FindMessageTypeByName"):
            pass
    if _cpp_message is not None and hasattr(_cpp_message, "MessageFactory"):
        _patch_getprototype(_cpp_message.MessageFactory())
except Exception:
    pass

try:
    if _py_message_factory is not None and hasattr(
        _py_message_factory, "MessageFactory"
    ):
        cls = _py_message_factory.MessageFactory
        if (not hasattr(cls, "GetPrototype")) and hasattr(cls, "GetMessageClass"):
            try:
                cls.GetPrototype = cls.GetMessageClass
            except Exception:
                pass
        _patch_getprototype(cls())
except Exception:
    pass

import tensorflow as tf
import tensorflow.keras.layers as L

tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]



## === cell 2
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
UNK = token2int["."]  # safe fallback (shouldn't be used for clean data)


def preprocess_inputs(df, cols=["sequence", "structure", "predicted_loop_type"]):
    """
    Returns int array of shape (n, seq_len, 3) where last dim corresponds to the 3 text columns.
    """
    arr = (
        df[cols]
        .applymap(lambda seq: [token2int.get(x, UNK) for x in seq])
        .values.tolist()
    )
    arr = np.array(arr)  # (n, 3, seq_len)
    arr = np.transpose(arr, (0, 2, 1))  # (n, seq_len, 3)
    return arr.astype(np.int32)




## === cell 3
def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True))


def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128):
    """
    Core logic preserved:
    - Input (seq_len, 3) token IDs
    - Embedding -> flatten last two dims -> 3 stacked BiGRU -> truncate -> Dense(5)
    """
    inputs = L.Input(shape=(seq_len, 3), dtype=tf.int32)

    embed = L.Embedding(input_dim=len(token2int), output_dim=embed_dim)(inputs)
    reshaped = L.Reshape((seq_len, 3 * embed_dim))(embed)

    hidden = gru_layer(hidden_dim, dropout)(reshaped)
    hidden = gru_layer(hidden_dim, dropout)(hidden)
    hidden = gru_layer(hidden_dim, dropout)(hidden)

    truncated = hidden[:, :pred_len]
    out = L.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(tf.keras.optimizers.Adam(), loss="mse")
    return model




## === cell 4
train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
sample_df = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")

print("train shape:", train.shape)
print("test shape:", test.shape)
print("sample_submission shape:", sample_df.shape)
print("test seq_length unique:", sorted(test["seq_length"].unique().tolist()))



## === cell 5
train_inputs = preprocess_inputs(train)
train_labels = (
    np.array(train[pred_cols].values.tolist()).transpose((0, 2, 1)).astype(np.float32)
)  # (n, 68, 5)

print("train_inputs:", train_inputs.shape, train_inputs.dtype)
print("train_labels:", train_labels.shape, train_labels.dtype)



## === cell 6
model = build_model(seq_len=107, pred_len=68)
model.summary()



## === cell 7
history = model.fit(
    train_inputs,
    train_labels,
    batch_size=64,
    epochs=60,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(),
        tf.keras.callbacks.ModelCheckpoint(
            "model.weights.h5", save_weights_only=True, save_best_only=False
        ),
    ],
    validation_split=0.3,
    verbose=2,
)



## === cell 8
print("Final train loss:", history.history["loss"][-1])
print("Final val loss:", history.history["val_loss"][-1])



## === cell 9
test_df = test.copy()
test_inputs = preprocess_inputs(test_df)

if os.path.exists("model.weights.h5"):
    model.load_weights("model.weights.h5")

test_preds_68 = model.predict(test_inputs, batch_size=128, verbose=0)  # (n, 68, 5)
print("test_preds_68:", test_preds_68.shape)



## === cell 10
seq_len_full = int(test_df["seq_length"].iloc[0])
pred_len = test_preds_68.shape[1]
assert seq_len_full == 107, f"Expected seq_length 107, got {seq_len_full}"
assert pred_len == 68, f"Expected pred_len 68, got {pred_len}"

last_pos = test_preds_68[:, -1:, :]  # (n,1,5)
pad = np.repeat(last_pos, repeats=seq_len_full - pred_len, axis=1)  # (n,39,5)
test_preds_107 = np.concatenate([test_preds_68, pad], axis=1)  # (n,107,5)
print("test_preds_107:", test_preds_107.shape)



## === cell 11
preds_ls = []
for i, uid in enumerate(test_df.id.values):
    single_pred = test_preds_107[i]  # (107,5)
    single_df = pd.DataFrame(single_pred, columns=pred_cols)
    single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
    preds_ls.append(single_df)

preds_df = pd.concat(preds_ls, ignore_index=True)
print("preds_df:", preds_df.shape)
print(preds_df.head())



## === cell 12
submission = sample_df[["id_seqpos"]].merge(preds_df, on="id_seqpos", how="left")

missing = submission[pred_cols].isna().any(axis=1).sum()
if missing:
    submission[pred_cols] = submission[pred_cols].fillna(0.0)
    print(f"Warning: {missing} rows missing after merge; filled with 0.0")

submission = submission[["id_seqpos"] + pred_cols]
assert (
    submission.shape[0] == sample_df.shape[0]
), "Row count mismatch vs sample_submission"
assert (
    list(submission.columns) == ["id_seqpos"] + pred_cols
), "Submission columns mismatch"

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
