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
joblib==1.5.2
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

0.54171

# 6. Current score

0.47624

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.47909) has done: 'I fix the protobuf/TensorFlow import crash by removing the unused `ResNet50` import (it triggers a protobuf incompatibility in this environment) and ensure GPU config uses TF2-safe APIs. I also fix the missing `bpps/` directory issue by falling back to a deterministic, minimal feature construction from `sequence/structure/predicted_loop_type` that matches the model’s expected `(130,130)` float32 input, so training/prediction can run end-to-end. Next, I fix the `object` dtype errors by explicitly stacking into numeric `float32` arrays and ensuring target `y` has shape `(n, 68, 5)` to match the model output. Finally, I generate predictions for all test rows, pad to full `seq_length`, align exactly to `sample_submission.csv`’s `id_seqpos` ordering, and write a valid `submission.csv`.'
- What this solution (achieved 0.47624) has done: 'I fix the TensorFlow/protobuf crash by importing TensorFlow only after forcing the Python protobuf implementation and by removing the unused `sklearn` import that triggers protobuf message factory initialization in this environment. I keep the model and feature logic unchanged, but make the GPU memory-growth setup TF2.18-safe and add deterministic settings to avoid run-to-run instability. I also correct a subtle training/target alignment risk by ensuring `y` is built as a contiguous `float32` array with the exact `(n, 68, 5)` shape the model outputs. Finally, I keep the same submission assembly logic but add a strict sanity check to guarantee all `id_seqpos` are present and the output CSV is valid.'
- What this solution (achieved 0.47624) has done: 'I fix the protobuf/TensorFlow crash that happens before any training by forcing the compatible pure-Python protobuf runtime and (crucially) importing `google.protobuf` before importing TensorFlow, then clearing TensorFlow’s cached protobuf modules so the setting actually takes effect. I keep the model/feature/training logic unchanged, but I also remove the GPU logical-device reconfiguration (which can throw or create unstable setups) and instead use the standard TF2 memory-growth path only. Finally, I keep the exact same submission-building logic, but add one extra alignment assert to ensure every `id_seqpos` in the sample submission got a prediction (no silent all-zero merges due to ID formatting), producing a valid `submission.csv`.'
- What this solution (achieved 0.47624) has done: 'I fix the TensorFlow/protobuf runtime crash by removing the forced pure-Python protobuf override and the manual module-clearing that’s now incompatible with the installed protobuf/TensorFlow combination (this is what triggers the `MessageFactory.GetPrototype` error before training starts). I keep your model, features, training loop, and submission-building logic unchanged, only making the import order safe and adding a small fallback so the script still runs even if no GPU is present. Since your current score is already better than the target (lower-is-better), I avoid any score-improving changes and focus strictly on correctness/stability and producing a valid `submission.csv`. The rest of the pipeline (data loading, feature building, training, inference, and CSV formatting/alignment) remains as-is.'
- What this solution (achieved 0.47624) has done: 'We fix the crash happening before training by pinning protobuf to the pure-Python implementation *before* TensorFlow is imported (this avoids the `MessageFactory.GetPrototype` incompatibility in this environment) while leaving your model/feature/training/submission logic unchanged. We also add a small safety fallback so the code still runs if TensorFlow is imported in a different order by the runtime. Finally, we keep the same merge/alignment assertions so the produced `submission.csv` is guaranteed valid and complete.'
- What this solution (achieved 0.47624) has done: 'I fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf override that’s incompatible with this environment and instead using a safe, minimal import sequence for TensorFlow. I also fix a downstream shape/ID alignment bug: your test set here uses `seq_length=107` for all rows, but the current code builds `id_seqpos` using the model’s output length (107) while the required submission has 107 positions per id; to keep semantics identical and avoid missing IDs, we always generate exactly `seq_length` rows per test id and then merge to `sample_submission.csv`. These changes are stability/correctness oriented and should keep performance in the same ballpark (no model/feature/training logic changes), producing a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.47624) has done: 'We fix the TensorFlow/protobuf import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before* TensorFlow is imported, which is the minimal change needed to unblock training in this environment. We keep your model/feature/training/prediction logic identical, only adjusting import order and adding a small safety guard to ensure the protobuf setting is applied early enough. Finally, we keep the same strict submission alignment checks so the notebook always writes a valid `submission.csv` that matches `sample_submission.csv` exactly.'
- What this solution (achieved 0.47624) has done: 'I fix the protobuf/TensorFlow import crash that prevents the pipeline from starting by removing the forced pure-Python protobuf override (it’s incompatible with this environment’s protobuf 6.x + TF 2.18 and triggers the `MessageFactory.GetPrototype` error). I keep the model, features, training loop, and submission assembly logic unchanged so the score behavior stays in the same ballpark (and still better than the target). I also add a small, safe GPU-memory-growth guard and keep deterministic seeding as-is. The script then run end-to-end and write a valid `submission.csv` matching `sample_submission.csv`’s required `id_seqpos` rows/columns.'
- What this solution (achieved 0.47624) has done: 'I fix the TensorFlow/protobuf import crash (`MessageFactory.GetPrototype`) by pinning protobuf to the pure-Python implementation *before* TensorFlow is imported, which is the minimal change needed to make the notebook run in this environment. I keep your feature construction, model definition, training loop, and submission assembly logic unchanged to preserve scoring behavior (and since your current score is already better than the target for a lower-is-better metric). I also add a small safety guard so the protobuf setting is applied early and doesn’t break if Kaggle preloads something unexpectedly. Finally, the script still write a fully aligned `submission.csv` matching `sample_submission.csv`.'
- What this solution (achieved 0.47624) has done: 'I fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf override (it’s incompatible with protobuf 6.x + TF 2.18 here and triggers the `MessageFactory.GetPrototype` error). I keep your model, feature construction, training loop, and submission-building logic the same, only making the import sequence safe and adding a small GPU-memory-growth guard that won’t change results. Since your current score (0.47624, lower-is-better) is already better than the target (0.54171), I won’t make any score-improving changes—just ensure the notebook runs end-to-end and writes a valid `submission.csv`. The rest of the pipeline (data load → X/y shapes → fit → predict → merge to sample) remains identical.'

# 9. Code solution

## === cell 0
import os
import gc
import sys
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SEED = 42
np.random.seed(SEED)

import tensorflow as tf
import tensorflow.keras.layers as L

tf.random.set_seed(SEED)
try:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

gc.collect()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("Could not set GPU memory growth:", e)

gc.collect()



## === cell 2
BASE_INPUT = "/kaggle/input/stanford-covid-vaccine"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/data/stanford-covid-vaccine"

TRAIN_PATH = os.path.join(BASE_INPUT, "train.json")
TEST_PATH = os.path.join(BASE_INPUT, "test.json")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

print("Using BASE_INPUT:", BASE_INPUT)
print(
    "Exists train/test/sample:",
    os.path.exists(TRAIN_PATH),
    os.path.exists(TEST_PATH),
    os.path.exists(SAMPLE_SUB_PATH),
)



## === cell 3
train_df = pd.read_json(TRAIN_PATH, lines=True)
test_df = pd.read_json(TEST_PATH, lines=True)
sample_df = pd.read_csv(SAMPLE_SUB_PATH)

train_df["id_hash"] = train_df["id"].apply(lambda x: x.split("_")[1])
test_df["id_hash"] = test_df["id"].apply(lambda x: x.split("_")[1])

print(train_df.shape, test_df.shape, sample_df.shape)



## === cell 4
target_columns = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

y_list = []
for c in target_columns:
    arr = np.stack(train_df[c].values).astype(np.float32, copy=False)
    y_list.append(arr)
y = np.stack(y_list, axis=-1).astype(np.float32, copy=False)
y = np.ascontiguousarray(y)

print("y:", y.shape, y.dtype)



## === cell 5
SEQ_LEN_FULL = 130  # model input expects 130x130
PRED_LEN = 68

seq_vocab = {"A": 0, "C": 1, "G": 2, "U": 3}
struct_vocab = {".": 0, "(": 1, ")": 2}
loop_vocab = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def _encode_string(s, vocab, max_len):
    arr = np.zeros((max_len,), dtype=np.int32)
    for i, ch in enumerate(s[:max_len]):
        arr[i] = vocab.get(ch, 0)
    return arr


def build_feature_matrix(sequence, structure, loop_type, seq_len_full=SEQ_LEN_FULL):
    seq_enc = _encode_string(sequence, seq_vocab, seq_len_full).astype(np.float32)
    st_enc = _encode_string(structure, struct_vocab, seq_len_full).astype(np.float32)
    lp_enc = _encode_string(loop_type, loop_vocab, seq_len_full).astype(np.float32)

    pos = np.arange(seq_len_full, dtype=np.float32) / max(1.0, (seq_len_full - 1.0))

    a = seq_enc / 3.0  # [0,1]
    b = st_enc / 2.0  # [0,1]
    c = lp_enc / 6.0  # [0,1]
    p = pos

    mat = (
        0.25 * (a[:, None] + a[None, :])
        + 0.20 * (b[:, None] + b[None, :])
        + 0.20 * (c[:, None] + c[None, :])
        + 0.20 * (p[:, None] + p[None, :])
        + 0.15 * (a[:, None] * a[None, :])
    ).astype(np.float32)

    mat = np.clip(mat, 0.0, 1.0)
    return mat


def make_X(df):
    X = np.stack(
        [
            build_feature_matrix(seq, struct, loop)
            for seq, struct, loop in zip(
                df["sequence"].values,
                df["structure"].values,
                df["predicted_loop_type"].values,
            )
        ],
        axis=0,
    ).astype(np.float32)
    return np.ascontiguousarray(X)


X_train = make_X(train_df)
X_test = make_X(test_df)

print("X_train:", X_train.shape, X_train.dtype)
print("X_test :", X_test.shape, X_test.dtype)




## === cell 6
def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128):
    image_tensor = L.Input(shape=(130, 130), dtype=tf.float32)

    im = L.Activation("linear")(image_tensor)
    im = L.Activation("linear")(im)
    im = L.Activation("linear")(im)
    truncated = im[:, :pred_len, :]
    out = L.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=image_tensor, outputs=out)
    model.compile(tf.keras.optimizers.Adam(), loss="mse")
    return model


try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("Could not enable XLA JIT:", e)

model = build_model(pred_len=PRED_LEN)
model.summary()



## === cell 7
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(),
    tf.keras.callbacks.ModelCheckpoint(
        "model.weights.h5", save_weights_only=True, save_best_only=False
    ),
]

device_name = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
print("Training on", device_name)

with tf.device(device_name):
    history = model.fit(
        X_train,
        y,
        batch_size=64,
        epochs=100,
        validation_split=0.05,
        callbacks=callbacks,
        verbose=2,
    )



## === cell 8
train_pred = model.predict(X_train, batch_size=64, verbose=0)
mae = np.mean(
    np.abs(train_pred.reshape(train_pred.shape[0], -1) - y.reshape(y.shape[0], -1))
)
print("Train MAE (flattened):", float(mae))



## === cell 9
model_short = build_model(seq_len=107, pred_len=107)
model_long = build_model(seq_len=130, pred_len=130)

if os.path.exists("model.weights.h5"):
    model_short.load_weights("model.weights.h5")
    model_long.load_weights("model.weights.h5")
else:
    print("WARNING: model.weights.h5 not found; using in-memory trained weights.")
    model_short.set_weights(model.get_weights())
    model_long.set_weights(model.get_weights())

print("Unique test seq_length:", sorted(test_df["seq_length"].unique().tolist()))



## === cell 10
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

all_parts = []
for seq_len, df_part in test_df.groupby("seq_length", sort=False):
    X_part = X_test[df_part.index.values]
    ids_part = df_part["id_hash"].values

    if int(seq_len) <= 107:
        preds = model_short.predict(X_part, batch_size=64, verbose=0)
    else:
        preds = model_long.predict(X_part, batch_size=64, verbose=0)

    out_rows = []
    for i, uid in enumerate(ids_part):
        single = preds[i]  # (pred_len, 5)
        if single.shape[0] >= int(seq_len):
            single = single[: int(seq_len), :]
        else:
            pad = np.zeros((int(seq_len) - single.shape[0], 5), dtype=single.dtype)
            single = np.concatenate([single, pad], axis=0)

        single_df = pd.DataFrame(single, columns=pred_cols)
        single_df["id_seqpos"] = [f"id_{uid}_{x}" for x in range(int(seq_len))]
        out_rows.append(single_df)

    part_df = (
        pd.concat(out_rows, axis=0, ignore_index=True)
        if out_rows
        else pd.DataFrame(columns=["id_seqpos"] + pred_cols)
    )
    all_parts.append(part_df)

preds_df = pd.concat(all_parts, axis=0, ignore_index=True)
print("preds_df:", preds_df.shape)
print(preds_df.head())



## === cell 11
submission = sample_df[["id_seqpos"]].merge(preds_df, on="id_seqpos", how="left")

missing = submission["reactivity"].isna().sum()
print("Rows with missing predictions after merge:", int(missing))

for c in pred_cols:
    submission[c] = submission[c].astype(np.float32).fillna(0.0)

submission = submission[["id_seqpos"] + pred_cols]

assert (
    submission.shape[0] == sample_df.shape[0]
), "Row count mismatch vs sample submission."
assert submission["id_seqpos"].isna().sum() == 0, "Missing id_seqpos."
assert submission[pred_cols].isna().sum().sum() == 0, "NaNs in predictions."
assert (
    missing == 0
), "Some id_seqpos did not receive predictions (ID formatting/alignment issue)."

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path, "shape:", submission.shape)
print(submission.head())
