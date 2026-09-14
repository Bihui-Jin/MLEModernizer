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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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

0.3518779703353691

# 6. Current score

0.41185

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I fix the file‑loading errors by reading the provided `sample_submission.csv` from the correct data directory, guard against missing paths, and remove the stray cells that reference an undefined `df`. The script now loads the test JSON (unused here), loads a valid baseline submission, optionally fills any missing values, and writes a proper `submission.csv` with the required columns, ensuring the notebook runs end‑to‑end and produces a correctly formatted CSV.'
- What this solution (achieved 0.63824) has done: 'I replace the placeholder submission with a simple baseline that uses the per‑position average of each target from the training data. By loading train.json, computing the mean value for every position (0‑67) of the five scored columns, and then filling the sample_submission rows with those means (or 0 for positions beyond 68), the predictions become data‑driven instead of all zeros, which should lower the MCRMSE toward the target score while keeping the original workflow unchanged.'
- What this solution (achieved 0.42166) has done: 'I fixed the indexing error by assigning mean values only for positions < 68, preventing out‑of‑bounds access, and improved the baseline by computing per‑position means using only high‑quality training samples (SN_filter = 1 and signal_to_noise > 1). The script now safely builds the submission and writes it as submission.csv.'
- What this solution (achieved 0.63824) has done: 'The update adds a simple sequence‑aware baseline: for each of the five target columns we compute per‑position means separated by nucleotide (A, C, G, U) using the high‑quality training samples. When building the submission we look up the nucleotide of the test RNA at the required position and use the corresponding mean value, falling back to the overall per‑position mean for any missing cases. This small, data‑driven tweak keeps the original workflow but is expected to lower the MCRMSE toward the target score.'
- What this solution (achieved 0.41187) has done: 'The update fixes the NameError in the per‑base mean calculation by iterating over each target column first, then over positions and nucleotides, correctly filling `mean_base` with the nucleotide‑specific averages (fall‑back to the overall per‑position mean when a nucleotide is absent). The prediction loop now uses these accurate means, which should lower the MCRMSE toward the target while keeping the original workflow unchanged.'
- What this solution (achieved 0.41207) has done: 'I blend the nucleotide‑position means computed from the high‑quality subset with the same means computed from all training samples (using a 0.7 / 0.3 weight). This adds a small amount of extra data to reduce variance while keeping the original per‑base logic, which should lower the MCRMSE toward the target score.'
- What this solution (achieved 0.41185) has done: 'I increase the reliance on the high‑quality subset by raising its blending weight from 0.7 to 0.85 (and lowering the all‑data weight accordingly). This small adjustment keeps the original per‑base mean logic while likely improving the MCRMSE, moving the score closer to the target. The rest of the pipeline remains unchanged, and the script still writes a valid submission.csv.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from pathlib import Path

test_path = Path("../input/stanford-covid-vaccine/test.json")
if not test_path.is_file():
    test_path = Path("data/test.json")  # fallback
df_test = pd.read_json(test_path, lines=True)

sample_path_candidates = [
    Path("../input/stanford-covid-vaccine/sample_submission.csv"),
    Path("data/sample_submission.csv"),
    Path("../input/sample_submission.csv"),
    Path("input/sample_submission.csv"),
]
sample_path = next((p for p in sample_path_candidates if p.is_file()), None)
if sample_path is None:
    raise FileNotFoundError("sample_submission.csv not found in any expected location.")
df_sub = pd.read_csv(sample_path)

train_path = Path("../input/stanford-covid-vaccine/train.json")
if not train_path.is_file():
    train_path = Path("data/train.json")  # fallback
df_train = pd.read_json(train_path, lines=True)

df_good = df_train[(df_train["SN_filter"] == 1) & (df_train["signal_to_noise"] > 1)]

target_cols = [
    "reactivity",
    (
        "deg_Mg_p10".replace("_p10", "_p10")
        if "deg_Mg_p10" in df_train.columns
        else "deg_Mg_pH10"
    ),
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]

max_scored = 68
bases = ["A", "C", "G", "U"]
base_to_idx = {b: i for i, b in enumerate(bases)}

mean_vectors = {}
for col in target_cols:
    arr = np.stack(df_good[col].values)  # shape (n_good, 68)
    mean_vectors[col] = arr.mean(axis=0)  # (68,)

seq_array_good = np.array([list(s) for s in df_good["sequence"]])  # (n_good, 107)
mean_base = {col: np.zeros((max_scored, len(bases))) for col in target_cols}
for col in target_cols:
    arr = np.stack(df_good[col].values)
    overall_pos_mean = arr.mean(axis=0)
    for pos in range(max_scored):
        for b_idx, b in enumerate(bases):
            mask = seq_array_good[:, pos] == b
            if mask.any():
                mean_base[col][pos, b_idx] = arr[mask, pos].mean()
            else:
                mean_base[col][pos, b_idx] = overall_pos_mean[pos]

mean_vectors_all = {}
for col in target_cols:
    arr_all = np.stack(df_train[col].values)
    mean_vectors_all[col] = arr_all.mean(axis=0)

seq_array_all = np.array([list(s) for s in df_train["sequence"]])  # (n_all, 107)
mean_base_all = {col: np.zeros((max_scored, len(bases))) for col in target_cols}
for col in target_cols:
    arr_all = np.stack(df_train[col].values)
    overall_pos_mean_all = arr_all.mean(axis=0)
    for pos in range(max_scored):
        for b_idx, b in enumerate(bases):
            mask = seq_array_all[:, pos] == b
            if mask.any():
                mean_base_all[col][pos, b_idx] = arr_all[mask, pos].mean()
            else:
                mean_base_all[col][pos, b_idx] = overall_pos_mean_all[pos]

id_to_seq = dict(zip(df_test["id"], df_test["sequence"]))

id_series = df_sub["id_seqpos"].str.rsplit("_", n=1).str[0]
pos_series = df_sub["id_seqpos"].str.rsplit("_", n=1).str[-1].astype(int)

base_list = []
for full_id, pos in zip(id_series, pos_series):
    seq = id_to_seq.get(full_id, None)
    if seq is not None and pos < len(seq):
        base_list.append(seq[pos])
    else:
        base_list.append(None)

base_idx_arr = np.array([base_to_idx.get(b, -1) for b in base_list])

w_high = 0.85
w_all = 0.15

for col in target_cols:
    preds = np.zeros(len(df_sub), dtype=float)  # default 0 for unscored positions
    mask = (pos_series < max_scored) & (base_idx_arr >= 0)
    if mask.any():
        high_vals = mean_base[col][pos_series[mask].values, base_idx_arr[mask]]
        all_vals = mean_base_all[col][pos_series[mask].values, base_idx_arr[mask]]
        preds[mask] = w_high * high_vals + w_all * all_vals
    fallback_mask = (pos_series < max_scored) & (base_idx_arr == -1)
    if fallback_mask.any():
        high_overall = mean_vectors[col][pos_series[fallback_mask].values]
        all_overall = mean_vectors_all[col][pos_series[fallback_mask].values]
        preds[fallback_mask] = w_high * high_overall + w_all * all_overall
    df_sub[col] = preds



## === cell 1
output_path = Path("submission.csv")
df_sub.to_csv(output_path, index=False)
