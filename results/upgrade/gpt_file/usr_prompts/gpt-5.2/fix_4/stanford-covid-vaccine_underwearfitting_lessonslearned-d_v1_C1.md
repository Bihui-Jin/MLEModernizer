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

0.3647779140395166

# 6. Current score

0.42418

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'The current notebook fails because it tries to read multiple external submissions from `../input/...` that do not exist in your environment, so none of the `sub*` DataFrames are created and downstream cells error out. To keep the ensemble-style “blend submissions then write CSV” core logic intact while making it runnable, I replace missing inputs with a safe fallback baseline submission built directly from `sample_submission.csv` (constant zeros), and I load any of the referenced files only if they actually exist. I also make the blending robust to missing `sub*` inputs by averaging only over successfully loaded submissions, preserving the original weights where possible. Finally, it writes a valid `.csv` submission with the exact required columns.'
- What this solution (achieved 0.63824) has done: 'Your current score is poor because almost all ensemble inputs are missing in this environment, so the blend collapses to an all-zeros baseline. To move toward the target (lower is better), the smallest legitimate improvement is to replace that baseline with a simple per-position mean predictor learned from `train.json` (computed only from the training targets), while keeping the ensemble/blending logic intact. This preserves the notebook’s “blend submissions then write CSV” core behavior but ensures a much stronger fallback when external submission files aren’t available. The submission is still aligned to `sample_submission.csv` and written with the exact required columns.'
- What this solution (achieved 0.42418) has done: 'I fix the IndexError in the baseline-per-position mean fallback by safely indexing only the scored positions (0–67) and filling the remaining positions (68–106) with an overall mean, instead of attempting `mu[seqpos]` for out-of-range indices. This preserves your core “fallback baseline + optional ensemble blend” logic and only changes the baseline value construction so the notebook runs end-to-end. I also make the `id_seqpos` parsing slightly more robust and ensure the baseline remains aligned to the sample submission order. These changes are score-improving relative to the current broken run while keeping everything else intact.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os



## === cell 1
SUB_COLS = [
    "id_seqpos",
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]


def _find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


sample_path = _find_first_existing(
    [
        "/kaggle/input/sample_submission.csv",
        "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
        "../input/sample_submission.csv",
        "../input/stanford-covid-vaccine/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
        "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
    ]
)

if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected Kaggle paths."
    )

sample_sub = pd.read_csv(sample_path)
missing_cols = [c for c in SUB_COLS if c not in sample_sub.columns]
if missing_cols:
    raise ValueError(f"sample_submission.csv missing required columns: {missing_cols}")



## === cell 2
train_path = _find_first_existing(
    [
        "/kaggle/input/train.json",
        "/kaggle/input/stanford-covid-vaccine/train.json",
        "../input/train.json",
        "../input/stanford-covid-vaccine/train.json",
        "/kaggle/data/train.json",
        "/kaggle/data/stanford-covid-vaccine/train.json",
    ]
)

baseline = sample_sub[SUB_COLS].copy()

if train_path is None:
    for c in SUB_COLS[1:]:
        baseline[c] = 0.0
else:
    train_df = pd.read_json(train_path, lines=True)

    targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    per_pos_means = {}
    for t in targets:
        arr = np.vstack(train_df[t].values).astype("float64")  # (n_train, 68)
        per_pos_means[t] = np.nanmean(arr, axis=0)  # (68,)

    seqpos = (
        baseline["id_seqpos"]
        .astype(str)
        .str.rsplit("_", n=1, expand=True)[1]
        .astype(int)
        .to_numpy()
    )

    for t in targets:
        mu = per_pos_means[t]
        overall = float(np.nanmean(mu))

        vals = np.full(seqpos.shape[0], overall, dtype="float64")
        mask = (seqpos >= 0) & (seqpos < len(mu))
        vals[mask] = mu[seqpos[mask]]
        baseline[t] = vals




## === cell 3
def load_submission_or_none(path, name):
    """Load a submission if it exists and has required columns, else return None."""
    if path is None or (not os.path.exists(path)):
        return None
    df = pd.read_csv(path)
    if "id_seqpos" not in df.columns:
        return None
    if all(c in df.columns for c in SUB_COLS):
        df = df[SUB_COLS].copy()
    else:
        return None
    return df


sub1 = load_submission_or_none(
    "../input/ae-pretrained-local/submission_ae_nonpretrained_local.csv", "sub1"
)  # @sin
sub2 = load_submission_or_none(
    "../input/ae-pretrained-local/submission_ae_pretrained_local.csv", "sub2"
)  # @sin
sub3 = load_submission_or_none(
    "../input/ae-pretrained-local/submission_ae_pretrained_local2.csv", "sub3"
)  # @sin
sub4 = load_submission_or_none(
    "../input/openvaccine-pytorch-ae-pretrain/submission.csv", "sub4"
)  # @public



## === cell 4
sub5 = load_submission_or_none(
    "../input/24551-ae-gcn/submission(3).csv", "sub5"
)  # @sin
sub6 = load_submission_or_none(
    "../input/hawkey-ae-pretrained-gcn-3ensemble/submission(4).csv", "sub6"
)  # @hawkey



## === cell 5
sub7 = load_submission_or_none(
    "../input/lstm-gru-5fold-2seeds-knncv/submission.csv", "sub7"
)  # @sin
sub8 = load_submission_or_none(
    "../input/hrunic-lstm-25177/submission(6).csv", "sub8"
)  # @hrunic
sub9 = load_submission_or_none(
    "../input/hrunic-gru-25365/submission(7).csv", "sub9"
)  # @hrunic
sub10 = load_submission_or_none(
    "../input/gru-lstm-with-feature-engineering-and-augmentation/submission.csv",
    "sub10",
)  # @public



## === cell 6
sub11 = load_submission_or_none(
    "../input/hawkey-gcn-only-25301/submission(5).csv", "sub11"
)  # @hawkey



## === cell 7
final_index = baseline.set_index("id_seqpos").index


def align_to_final(df):
    if df is None:
        return None
    df = df.drop_duplicates("id_seqpos")
    df = df.set_index("id_seqpos").reindex(final_index)
    for c in SUB_COLS[1:]:
        if c not in df.columns:
            df[c] = 0.0
    df = df[SUB_COLS[1:]].astype("float64")
    df = df.fillna(0.0)
    return df


sub1 = align_to_final(sub1)
sub2 = align_to_final(sub2)
sub3 = align_to_final(sub3)
sub4 = align_to_final(sub4)
sub5 = align_to_final(sub5)
sub6 = align_to_final(sub6)
sub7 = align_to_final(sub7)
sub8 = align_to_final(sub8)
sub9 = align_to_final(sub9)
sub10 = align_to_final(sub10)
sub11 = align_to_final(sub11)

baseline_aligned = align_to_final(baseline)




## === cell 8
def mean_of_available(dfs):
    dfs = [d for d in dfs if d is not None]
    if len(dfs) == 0:
        return None
    out = dfs[0].copy()
    for d in dfs[1:]:
        out = out + d
    return out / len(dfs)


block_a = mean_of_available([sub1, sub2, sub3, sub4])
block_b = mean_of_available([sub5, sub6])

if block_a is None and block_b is None:
    ac_pretrained = baseline_aligned
elif block_a is None:
    ac_pretrained = block_b
elif block_b is None:
    ac_pretrained = block_a
else:
    ac_pretrained = block_a * 0.5 + block_b * 0.5

block_c = mean_of_available([sub7, sub8, sub9, sub10])
block_d = sub11

if block_c is None and block_d is None:
    non_ac_pretrained = baseline_aligned
elif block_c is None:
    non_ac_pretrained = block_d
elif block_d is None:
    non_ac_pretrained = block_c
else:
    non_ac_pretrained = block_c * 0.6 + block_d * 0.4



## === cell 9
final_pred = ac_pretrained * 0.5 + non_ac_pretrained * 0.5

final_pred = final_pred.replace([np.inf, -np.inf], np.nan).fillna(0.0)

final = pd.DataFrame({"id_seqpos": final_index})
for c in SUB_COLS[1:]:
    final[c] = final_pred[c].values



## === cell 10
out_path = "final_v1.csv"
final.to_csv(out_path, index=False)

assert out_path.endswith(".csv")
assert list(final.columns) == SUB_COLS
assert len(final) == len(sample_sub)
assert final["id_seqpos"].iloc[0] == sample_sub["id_seqpos"].iloc[0]
