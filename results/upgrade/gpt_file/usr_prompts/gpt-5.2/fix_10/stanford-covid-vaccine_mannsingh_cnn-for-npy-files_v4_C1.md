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

0.69003

# 6. Current score

0.40418

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.35427) has done: 'I fix the missing `bpps/` directory issue by removing the dependency on unavailable `.npy` files and instead build simple per-position features directly from `sequence`, `structure`, and `predicted_loop_type` (still using a small Conv2D model and the same MSE loss). I also resolve the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, and fix the Adam optimizer argument (`lr` → `learning_rate`) so the model compiles in TF 2.18. Finally, I correct the target shape to be per-base (68×5) and generate a submission with one row per `id_seqpos` covering all 107 positions, filling positions beyond `seq_scored` with the last predicted value to keep output valid.'
- What this solution (achieved 0.40627) has done: 'Your current run doesn’t yield a Kaggle score, so the most direct improvement is to make the pipeline reliably produce a valid `submission.csv` without crashing in the provided environment. The main blocker is the protobuf auto-downgrade/install+`execv` logic, which is brittle in Kaggle; we remove that and instead rely on the already-installed protobuf while keeping the pure-Python implementation flag for safety. To move the score in the right direction (lower MCRMSE) with minimal semantic change, we (1) train only on the 3 scored targets while still emitting all 5 columns (fill the 2 unscored with a stable value), and (2) avoid the “copy last scored position to all unscored positions” heuristic by filling positions beyond `seq_scored` with per-sample mean of predicted scored positions (not evaluated, but reduces pathological outputs and stabilizes training/prediction). These changes keep the same feature extraction, the same Conv2D→GAP→Dense core model, the same loss (MSE), and still write a correct submission file.'
- What this solution (achieved 0.40596) has done: 'I fix the TensorFlow import crash caused by the protobuf version mismatch by switching to the C++ protobuf runtime (removing the forced pure-Python setting) and ensuring the environment is set before TensorFlow is imported. This is a runtime-only change that should not alter the model or training semantics, but unblock training/inference and allow a valid `submission.csv` to be written. I also add a small safety fallback to load data from either `../input/stanford-covid-vaccine` or `../input` (as you already intended) and keep the rest of the feature extraction, model architecture, loss, and submission formatting identical. With the pipeline running reliably again, your score should move back toward (lower) values because the model actually train and generate predictions instead of failing.'
- What this solution (achieved 0.40344) has done: 'We fix the TensorFlow/protobuf crash that prevents the pipeline from running by forcing the pure-Python protobuf runtime *before* importing TensorFlow, which avoids the `MessageFactory.GetPrototype` AttributeError in this environment. We also make the input path resolution robust for both `../input/...` and `/kaggle/input/...` layouts without changing any modeling logic. Finally, we keep the same model, targets, training loop, and submission formatting, only adding small safety checks so a valid `submission.csv` is always produced.'
- What this solution (achieved 0.40644) has done: 'We fix the TensorFlow import crash by switching protobuf to the pure-Python implementation *before* importing TensorFlow (your current `"cpp"` setting is what triggers the missing `_message` error here). Then we make the script run end-to-end by keeping the same data loading, feature construction, Conv2D model, and MSE training loop, only updating the failing imports/ordering so `Sequential/ReduceLROnPlateau/history/model` are defined. Finally, we ensure a valid `submission.csv` is always written in the required column order and shape (one row per `id_seqpos`, all 5 targets), preserving your current scored-target training and fill logic.'
- What this solution (achieved 0.40418) has done: 'We fix the TensorFlow import crash (`MessageFactory` / protobuf incompatibility) by switching to the supported protobuf API implementation flag that works with TF 2.18 in this environment, and do it before TensorFlow is imported. We also make data loading robust to both directory layouts by checking `stanford-covid-vaccine/` as well as the base input folder, without changing any modeling logic. Finally, we keep the same feature construction, model, and training semantics, but add a small safety check ensuring the submission rows are filled for all 5 targets and written as `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(32)




## === cell 1
def resolve_base_input():
    candidates = [
        "../input/stanford-covid-vaccine",
        "../input",
        "/kaggle/input/stanford-covid-vaccine",
        "/kaggle/input",
        "/kaggle/data/stanford-covid-vaccine",
        "/kaggle/data",
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, "train.json")) and os.path.exists(
            os.path.join(p, "test.json")
        ):
            return p

        nested = os.path.join(p, "stanford-covid-vaccine")
        if os.path.exists(os.path.join(nested, "train.json")) and os.path.exists(
            os.path.join(nested, "test.json")
        ):
            return nested

    if os.path.exists("../input"):
        return "../input"

    raise FileNotFoundError(
        "Could not locate train.json/test.json. Checked: " + ", ".join(candidates)
    )


BASE_INPUT = resolve_base_input()
train_path = os.path.join(BASE_INPUT, "train.json")
test_path = os.path.join(BASE_INPUT, "test.json")
ss_path = os.path.join(BASE_INPUT, "sample_submission.csv")

train = pd.read_json(train_path, lines=True)
test = pd.read_json(test_path, lines=True)
ss = pd.read_csv(ss_path)

train = train.set_index("index")
test = test.set_index("index")

print("BASE_INPUT:", BASE_INPUT)
print("train:", train.shape, "test:", test.shape, "sample_submission:", ss.shape)
print("sample_submission columns:", ss.columns.tolist())



## === cell 2
targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

scored_targets = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
scored_target_idx = [targets.index(t) for t in scored_targets]



## === cell 3
train = train[
    ["id", "sequence", "structure", "predicted_loop_type", "seq_length", "seq_scored"]
    + targets
].copy()
test = test[
    ["id", "sequence", "structure", "predicted_loop_type", "seq_length", "seq_scored"]
].copy()

assert train["seq_length"].nunique() == 1, "Unexpected varying seq_length in train"
assert test["seq_length"].nunique() == 1, "Unexpected varying seq_length in test"
SEQ_LEN = int(train["seq_length"].iloc[0])
SCORED_LEN = int(train["seq_scored"].iloc[0])
print("SEQ_LEN:", SEQ_LEN, "SCORED_LEN:", SCORED_LEN)



## === cell 4
BASES = ["A", "C", "G", "U"]
STRUCT = ["(", ")", "."]
LOOPS = ["S", "M", "I", "B", "H", "E", "X"]

base_to_i = {b: i for i, b in enumerate(BASES)}
struct_to_i = {s: i for i, s in enumerate(STRUCT)}
loop_to_i = {l: i for i, l in enumerate(LOOPS)}


def one_hot_seq(s, vocab_map, vocab_size, L):
    arr = np.zeros((L, vocab_size), dtype=np.float32)
    for i, ch in enumerate(s[:L]):
        j = vocab_map.get(ch, None)
        if j is not None:
            arr[i, j] = 1.0
    return arr


def build_feature_map(sequence, structure, loop_type, L=107):
    seq_oh = one_hot_seq(sequence, base_to_i, len(BASES), L)  # (L,4)
    str_oh = one_hot_seq(structure, struct_to_i, len(STRUCT), L)  # (L,3)
    loop_oh = one_hot_seq(loop_type, loop_to_i, len(LOOPS), L)  # (L,7)

    feats = []
    for k in range(seq_oh.shape[1]):
        v = seq_oh[:, k]
        feats.append(np.outer(v, v).astype(np.float32))
    for k in range(str_oh.shape[1]):
        v = str_oh[:, k]
        feats.append(np.outer(v, v).astype(np.float32))
    for k in range(loop_oh.shape[1]):
        v = loop_oh[:, k]
        feats.append(np.outer(v, v).astype(np.float32))

    x = np.stack(feats, axis=-1)  # (L, L, 14)
    return x


X_train_all = np.stack(
    [
        build_feature_map(r.sequence, r.structure, r.predicted_loop_type, L=SEQ_LEN)
        for r in train.itertuples()
    ],
    axis=0,
).astype(np.float32)

X_test_all = np.stack(
    [
        build_feature_map(r.sequence, r.structure, r.predicted_loop_type, L=SEQ_LEN)
        for r in test.itertuples()
    ],
    axis=0,
).astype(np.float32)

print("X_train_all:", X_train_all.shape, "X_test_all:", X_test_all.shape)



## === cell 5
y_list = []
for r in train.itertuples():
    ys = []
    for t in scored_targets:
        arr = np.array(getattr(r, t), dtype=np.float32)
        if arr.shape[0] != SCORED_LEN:
            raise ValueError(
                f"Target {t} length mismatch: got {arr.shape[0]}, expected {SCORED_LEN}"
            )
        ys.append(arr)
    y_pos = np.stack(ys, axis=-1)  # (68, 3)
    y_list.append(y_pos)

y_all = np.stack(y_list, axis=0)  # (n, 68, 3)
y_all = np.nan_to_num(y_all, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
print("y_all:", y_all.shape)



## === cell 6
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_all, y_all, test_size=0.1, random_state=32, shuffle=True
)
print("Train split:", X_tr.shape, y_tr.shape, "Val split:", X_val.shape, y_val.shape)



## === cell 7
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.layers import (
    BatchNormalization,
    Activation,
    MaxPooling2D,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback

tf.random.set_seed(32)


def mcrmse_np(y_true, y_pred, eps=1e-12):
    mse = np.mean((y_true - y_pred) ** 2, axis=0)  # (68, n_targets)
    rmse_col = np.sqrt(np.mean(mse, axis=0) + eps)  # (n_targets,)
    return float(np.mean(rmse_col))


class ValMCRMSE(Callback):
    def __init__(self, Xv, yv, batch_size=32):
        super().__init__()
        self.Xv = Xv
        self.yv = yv
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.Xv, batch_size=self.batch_size, verbose=0)
        pred = pred.reshape((-1, SCORED_LEN, len(scored_targets))).astype(np.float32)
        score = mcrmse_np(self.yv, pred)
        if logs is not None:
            logs["val_mcrmse"] = score
        print(f" - val_mcrmse: {score:.6f}")




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
n_channels = X_tr.shape[-1]

model = Sequential()
model.add(
    Conv2D(64, (3, 3), padding="same", input_shape=(SEQ_LEN, SEQ_LEN, n_channels))
)
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(GlobalAveragePooling2D())

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.25))

model.add(Dense(SCORED_LEN * len(scored_targets), activation="linear"))

opt = Adam(learning_rate=0.005)
model.compile(optimizer=opt, loss="mean_squared_error")
model.summary()



## === cell 9
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=2, min_lr=1e-5, mode="auto"
)
val_mcrmse_cb = ValMCRMSE(X_val, y_val, batch_size=32)
callbacks = [reduce_lr, val_mcrmse_cb]

history = model.fit(
    x=X_tr,
    y=y_tr.reshape((-1, SCORED_LEN * len(scored_targets))),
    epochs=50,
    validation_data=(X_val, y_val.reshape((-1, SCORED_LEN * len(scored_targets)))),
    callbacks=callbacks,
    verbose=2,
)



## === cell 10
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.legend()
plt.title("MSE loss")

plt.subplot(1, 2, 2)
plt.plot(np.log10(np.maximum(history.history["loss"], 1e-12)), label="train log10")
plt.plot(np.log10(np.maximum(history.history["val_loss"], 1e-12)), label="val log10")
plt.legend()
plt.title("log10(loss)")
plt.tight_layout()



## === cell 11
pred_test_flat = model.predict(X_test_all, batch_size=32, verbose=0)
pred_test_scored = pred_test_flat.reshape((-1, SCORED_LEN, len(scored_targets))).astype(
    np.float32
)
print("pred_test_scored:", pred_test_scored.shape)



## === cell 12
sub = ss.copy()
sub_targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
assert sub_targets == targets, "Target column order mismatch vs expected"

id_to_idx = {rid: i for i, rid in enumerate(test["id"].values)}
assert len(id_to_idx) == len(
    test
), "Non-unique ids in test; submission mapping would break."

out = np.zeros((len(sub), len(targets)), dtype=np.float32)

pred_scored_mean = pred_test_scored.mean(axis=1)  # (n_test, 3)

for row_i, id_seqpos in enumerate(sub["id_seqpos"].values):
    rid, pos_str = id_seqpos.rsplit("_", 1)
    pos = int(pos_str)
    ti = id_to_idx[rid]

    if pos < SCORED_LEN:
        scored_vals = pred_test_scored[ti, pos, :]
    else:
        scored_vals = pred_scored_mean[ti, :]

    for k, col_idx in enumerate(scored_target_idx):
        out[row_i, col_idx] = scored_vals[k]

row_mean = out[:, scored_target_idx].mean(axis=1)
out[:, targets.index("deg_pH10")] = row_mean
out[:, targets.index("deg_50C")] = row_mean

for j, col in enumerate(targets):
    sub[col] = out[:, j]

sub = sub[ss.columns.tolist()]
assert sub.shape[0] == ss.shape[0]
assert sub.columns.tolist() == ss.columns.tolist()

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "shape:", sub.shape)
print(sub.head())
