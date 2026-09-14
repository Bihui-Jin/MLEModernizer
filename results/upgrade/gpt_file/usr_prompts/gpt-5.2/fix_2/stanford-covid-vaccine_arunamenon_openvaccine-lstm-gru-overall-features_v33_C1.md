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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

0.3763091623635361

# 6. Current score

0.42231

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.42231) has done: 'I fix the two blockers preventing end-to-end execution: (1) the TensorFlow import crash caused by an incompatible `protobuf` runtime, by avoiding TensorFlow entirely (it’s only needed for model inference here), and (2) missing `bpps/*.npy` files by safely substituting zero BPPS-based features when those files aren’t present. Then I replace the weight-loading/inference section with a deterministic, lightweight baseline that still produces correctly-shaped per-position predictions and writes a valid `.csv` submission matching `sample_submission.csv`. These fixes are score-neutral in intent (they mainly ensure a valid submission is produced), but the baseline uses simple train-set averages per position which should score meaningfully better than arbitrary constants.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import json

np.random.seed(42)



## === cell 1
TRAIN_PATH = "/kaggle/input/stanford-covid-vaccine/train.json"
TEST_PATH = "/kaggle/input/stanford-covid-vaccine/test.json"
SAMPLE_SUB_PATH = "/kaggle/input/stanford-covid-vaccine/sample_submission.csv"
BPPS_DIR = (
    "/kaggle/input/stanford-covid-vaccine/bpps"  # may not exist in this environment
)



## === cell 2
train_data = pd.read_json(TRAIN_PATH, lines=True)
test_data = pd.read_json(TEST_PATH, lines=True)
submission_format = pd.read_csv(SAMPLE_SUB_PATH, encoding="utf-8-sig")

print(
    "train:",
    train_data.shape,
    "test:",
    test_data.shape,
    "sample_sub:",
    submission_format.shape,
)
print("BPPS dir exists?", os.path.isdir(BPPS_DIR))



## === cell 3
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

SEQ_LEN = int(train_data["seq_length"].mode()[0])  # expected 107
SCORED_LEN = int(train_data["seq_scored"].mode()[0])  # expected 68

print("SEQ_LEN:", SEQ_LEN, "SCORED_LEN:", SCORED_LEN)



## === cell 4
from collections import Counter as count


def get_bases(data):
    bases = []
    for j in range(len(data)):
        counts = dict(count(data.iloc[j]["sequence"]))
        bases.append(
            (
                counts.get("A", 0) / SEQ_LEN,
                counts.get("G", 0) / SEQ_LEN,
                counts.get("C", 0) / SEQ_LEN,
                counts.get("U", 0) / SEQ_LEN,
            )
        )
    return pd.DataFrame(
        bases, columns=["A_percent", "G_percent", "C_percent", "U_percent"]
    )


def get_pairs_rate(data):
    pairs_rate = []
    for j in range(len(data)):
        res = dict(count(data.iloc[j]["structure"]))
        pairs_rate.append(res.get("(", 0) / 53.5)
    return pd.DataFrame(pairs_rate, columns=["pairs_rate"])


def get_pairs(data):
    pairs = []
    for j in range(len(data)):
        pairs_dict = {}
        queue = []
        struct = data.iloc[j]["structure"]
        seq = data.iloc[j]["sequence"]
        for i in range(len(struct)):
            if struct[i] == "(":
                queue.append(i)
            elif struct[i] == ")":
                if not queue:
                    continue
                first = queue.pop()
                key = (seq[first], seq[i])
                pairs_dict[key] = pairs_dict.get(key, 0) + 1

        pairs_num = sum(pairs_dict.values()) if pairs_dict else 0
        pairs_unique = [
            ("U", "G"),
            ("C", "G"),
            ("U", "A"),
            ("G", "C"),
            ("A", "U"),
            ("G", "U"),
        ]
        row = []
        for item in pairs_unique:
            row.append((pairs_dict.get(item, 0) / pairs_num) if pairs_num > 0 else 0.0)
        pairs.append(row)

    return pd.DataFrame(pairs, columns=["U-G", "C-G", "U-A", "G-C", "A-U", "G-U"])


def get_loops(data):
    loops = []
    available = ["E", "S", "H", "B", "X", "I", "M"]
    for j in range(len(data)):
        counts = dict(count(data.iloc[j]["predicted_loop_type"]))
        row = []
        for item in available:
            row.append(counts.get(item, 0) / SEQ_LEN)
        loops.append(row)
    return pd.DataFrame(loops, columns=available)




## === cell 5
def _safe_load_bpps(mol_id):
    path = os.path.join(BPPS_DIR, f"{mol_id}.npy")
    if os.path.isfile(path):
        return np.load(path)
    return None


def read_bpps_sum(df):
    arr = []
    for mol_id in df.id.to_list():
        bpps = _safe_load_bpps(mol_id)
        if bpps is None:
            arr.append(np.zeros(SEQ_LEN, dtype=np.float32))
        else:
            arr.append(bpps.sum(axis=1).astype(np.float32))
    return arr


def read_bpps_max(df):
    arr = []
    for mol_id in df.id.to_list():
        bpps = _safe_load_bpps(mol_id)
        if bpps is None:
            arr.append(np.zeros(SEQ_LEN, dtype=np.float32))
        else:
            arr.append(bpps.max(axis=1).astype(np.float32))
    return arr


def read_bpps_nb(df):
    bpps_nb_mean = 0.077522
    bpps_nb_std = 0.08914
    arr = []
    for mol_id in df.id.to_list():
        bpps = _safe_load_bpps(mol_id)
        if bpps is None:
            arr.append(np.zeros(SEQ_LEN, dtype=np.float32))
        else:
            bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
            bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
            arr.append(bpps_nb.astype(np.float32))
    return arr


train_data["bpps_sum"] = read_bpps_sum(train_data)
test_data["bpps_sum"] = read_bpps_sum(test_data)
train_data["bpps_max"] = read_bpps_max(train_data)
test_data["bpps_max"] = read_bpps_max(test_data)
train_data["bpps_nb"] = read_bpps_nb(train_data)
test_data["bpps_nb"] = read_bpps_nb(test_data)

print(
    "Added BPPS columns. Example lens:",
    len(train_data["bpps_sum"].iloc[0]),
    len(train_data["bpps_max"].iloc[0]),
    len(train_data["bpps_nb"].iloc[0]),
)



## === cell 6
train_data = pd.concat(
    [
        train_data,
        get_bases(train_data),
        get_pairs(train_data),
        get_loops(train_data),
        get_pairs_rate(train_data),
    ],
    axis=1,
)
test_data = pd.concat(
    [
        test_data,
        get_bases(test_data),
        get_pairs(test_data),
        get_loops(test_data),
        get_pairs_rate(test_data),
    ],
    axis=1,
)

print(
    "Feature-augmented train columns include:",
    [c for c in ["A_percent", "pairs_rate", "bpps_sum"] if c in train_data.columns],
)



## === cell 7

good_train = train_data.loc[train_data["signal_to_noise"] > 1].copy()
print("Good train rows:", good_train.shape)

Y = np.stack(
    [np.stack(good_train[col].values, axis=0) for col in target_cols], axis=-1
).astype(
    np.float32
)  # (n, 68, 5)
pos_mean = np.mean(Y, axis=0)  # (68, 5)

full_pos_mean = np.vstack(
    [pos_mean, np.repeat(pos_mean[-1:, :], SEQ_LEN - SCORED_LEN, axis=0)]
)  # (107, 5)
print("Baseline mean shape:", full_pos_mean.shape)




## === cell 8
def make_test_predictions(test_df):
    preds = []
    for uid in test_df["id"].values:
        single_df = pd.DataFrame(full_pos_mean, columns=target_cols)
        single_df["id_seqpos"] = [f"{uid}_{i}" for i in range(SEQ_LEN)]
        preds.append(single_df)
    return pd.concat(preds, ignore_index=True)


preds_df = make_test_predictions(test_data)
print("Preds df:", preds_df.shape, preds_df.head())



## === cell 9
submission = submission_format[["id_seqpos"]].merge(
    preds_df, how="left", on="id_seqpos"
)

for c in target_cols:
    if submission[c].isna().any():
        submission[c] = submission[c].fillna(0.0)

print("Submission shape:", submission.shape)
print(submission.head())



## === cell 10
os.makedirs("/kaggle/working", exist_ok=True)
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Columns:", submission.columns.tolist())
print("Any NaNs?", submission.isna().any().any())
