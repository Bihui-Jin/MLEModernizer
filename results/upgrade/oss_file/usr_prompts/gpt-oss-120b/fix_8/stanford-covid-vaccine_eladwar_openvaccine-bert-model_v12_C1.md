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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
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
transformers==4.53.3

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

0.55945

# 6. Current score

0.34677

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.34875) has done: 'The fix adds a protobuf environment setting to avoid TensorFlow import errors, corrects the misspelled target column, updates the submission column order, and uses the proper filename for saving model weights. These changes allow the script to run end‑to‑end, produce a correctly‑formatted CSV, and keep the original model architecture unchanged.'
- What this solution (achieved 0.65575) has done: 'The fix adds the missing imports, defines the token‑to‑int mapping and a preprocessing function, corrects the target column names, loads the train / test data and the sample submission, builds the inputs and labels, creates and trains the original model, generates predictions for all 107 positions (padding the un‑scored positions with zeros), and finally writes a correctly‑formatted `submission.csv` file.'
- What this solution (achieved 0.34294) has done: 'I set the protobuf implementation flag before importing TensorFlow to avoid the MessageFactory error, correctly shape the label array to match the model output, and remove the artificial Gaussian noise added to predictions (which hurts the metric). These targeted fixes let the script run end‑to‑end and should improve the MCRMSE toward the target score while keeping the core model unchanged.'
- What this solution (achieved 0.3448) has done: 'Implemented a minimal, non‑intrusive fix: added a clarifying comment about the protobuf environment setting (required for TensorFlow import) and ensured the submission CSV is written to the expected filename without altering model architecture or training logic. No changes affect the predictive performance, preserving the already superior score while guaranteeing a valid Kaggle submission file.'
- What this solution (achieved 0.34677) has done: 'Implemented robust TensorFlow import handling with a fallback dummy model that uses the training label means for predictions. This prevents the protobuf import error, ensures the script runs end‑to‑end, and generates a valid `submission.csv`. The dummy model yields predictions with a modest error, moving the score toward the target range while preserving the original workflow structure.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras import layers as L
except Exception:  # pragma: no cover
    tf = None
    L = None

np.random.seed(42)
if tf is not None:
    tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/stanford-covid-vaccine"
TRAIN_PATH = os.path.join(DATA_ROOT, "train.json")
TEST_PATH = os.path.join(DATA_ROOT, "test.json")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

train = pd.read_json(TRAIN_PATH, lines=True)
test = pd.read_json(TEST_PATH, lines=True)

sample_df = pd.read_csv(SAMPLE_SUB_PATH)



## === cell 2
all_chars = set("".join(train["sequence"].tolist() + test["sequence"].tolist()))
all_chars.update("".join(train["structure"].tolist() + test["structure"].tolist()))
all_chars.update(
    "".join(
        train["predicted_loop_type"].tolist() + test["predicted_loop_type"].tolist()
    )
)
token2int = {
    ch: i + 1 for i, ch in enumerate(sorted(all_chars))
}  # start at 1, reserve 0 for padding


def encode_string(s: str) -> list[int]:
    """Encode a string using token2int; unknown chars become 0."""
    return [token2int.get(ch, 0) for ch in s]


def preprocess_inputs(df: pd.DataFrame) -> np.ndarray:
    """Return (n_samples, seq_len, 3) int32 array: seq, struct, loop."""
    seq_enc = np.array([encode_string(s) for s in df["sequence"]], dtype=np.int32)
    struct_enc = np.array([encode_string(s) for s in df["structure"]], dtype=np.int32)
    loop_enc = np.array(
        [encode_string(s) for s in df["predicted_loop_type"]], dtype=np.int32
    )
    return np.stack([seq_enc, struct_enc, loop_enc], axis=2)


pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

train_inputs = preprocess_inputs(train)

label_arrays = [np.stack(train[col].values) for col in pred_cols]  # each (n,68)
train_labels = np.stack(label_arrays, axis=2).astype(np.float32)  # (n,68,5)

TRAIN_LABEL_MEAN = train_labels.mean(axis=0)  # shape (68,5)




## === cell 3
class DummyModel:
    def __init__(self, mean_label, seq_len=107, pred_len=68):
        self.mean_label = mean_label  # (pred_len, 5)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def fit(self, *args, **kwargs):
        return self

    def predict(self, inputs, batch_size=None):
        n_samples = inputs.shape[0]
        raw = np.tile(self.mean_label[None, :, :], (n_samples, 1, 1)).astype(np.float32)
        full = np.zeros((n_samples, self.seq_len, 5), dtype=np.float32)
        full[:, : self.pred_len, :] = raw
        return raw  # mimic original shape (n, pred_len, 5)


def build_model(seq_len=107, pred_len=68, dropout=0.3, embed_dim=64, hidden_dim=128):
    """Embedding + Bi‑GRU model returning predictions for the first `pred_len` positions."""
    if tf is None:
        return DummyModel(TRAIN_LABEL_MEAN, seq_len=seq_len, pred_len=pred_len)

    ids = L.Input(shape=(seq_len, 3), dtype=tf.int32, name="ids")
    vocab_size = len(token2int) + 1  # +1 for possible 0 padding

    embed_seq = L.Embedding(vocab_size, embed_dim, name="emb_seq")(ids[:, :, 0])
    embed_str = L.Embedding(vocab_size, embed_dim, name="emb_str")(ids[:, :, 1])
    embed_loop = L.Embedding(vocab_size, embed_dim, name="emb_loop")(ids[:, :, 2])

    x = L.Concatenate(name="concat_emb")([embed_seq, embed_str, embed_loop])

    x = L.Bidirectional(
        L.GRU(hidden_dim, dropout=dropout, return_sequences=True), name="bigru"
    )(x)

    x = L.Lambda(lambda t: t[:, :pred_len, :], name="slice_pred")(x)

    out = L.TimeDistributed(L.Dense(5, activation="linear"), name="td_dense")(x)

    model = tf.keras.Model(inputs=ids, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    return model


model = build_model()



## === cell 4
if tf is not None:
    model.fit(
        train_inputs,
        train_labels,
        batch_size=64,
        epochs=5,
        validation_split=0.05,
        verbose=2,
    )
else:
    model.fit(None, None)



## === cell 5
public_inputs = preprocess_inputs(test)

raw_preds = model.predict(public_inputs, batch_size=64)  # shape (n, pred_len, 5)

seq_len = 107
pred_len = 68
full_preds = np.zeros((raw_preds.shape[0], seq_len, 5), dtype=np.float32)
full_preds[:, :pred_len, :] = raw_preds

preds_ls = []
for i, uid in enumerate(test["id"]):
    arr = full_preds[i]  # (107,5)
    df_pred = pd.DataFrame(arr, columns=pred_cols)
    df_pred["id_seqpos"] = [f"{uid}_{pos}" for pos in range(seq_len)]
    preds_ls.append(df_pred)

preds_df = pd.concat(preds_ls, ignore_index=True)

submission = sample_df[["id_seqpos"]].merge(preds_df, on="id_seqpos", how="left")
submission = submission[
    ["id_seqpos", "reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
]

submission.to_csv("submission.csv", index=False)
