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

0.3589839418734281

# 6. Current score

0.4215

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'The current notebook fails because it tries to read a non-existent CSV (`../input/worst-submission/ensemble52.csv`), so `df` is never created and all later cells error. Since we don’t have your external ensemble file, the safest minimal fix is to generate a valid submission directly from the provided competition `sample_submission.csv`, which guarantees correct row ordering and required columns. To keep the original “calibration” intent of scaling predictions (without changing model logic, since none exists here), we keep the existing divide-by-1.08 step but apply it to the sample submission values (they are already valid placeholders). This run end-to-end and write `submission.csv` with the correct schema.'
- What this solution (achieved 0.42294) has done: 'Your current script is effectively submitting near-all-zeros (because `sample_submission.csv` targets are blanks/NaNs that you coerce to 0), which is why the score is far from the target (lower is better). To move the score down toward the target with minimal logic change and no ML training, we generate simple, legitimate baseline predictions from `train.json`: per-position means for each of the 5 targets, and then fill the test submission rows by their `seqpos`. This keeps the same submission schema, preserves your existing calibration step (`/= 1.08`) to avoid overshooting improvements too aggressively, and still runs fast under constraints. The result should improve MCRMSE substantially from 0.63824 toward the 0.3589 target.'
- What this solution (achieved 0.42203) has done: 'To move the MCRMSE down toward your target with minimal risk, I keep your current “per-position mean from train.json” baseline but make it more representative by (1) computing means using only the first `seq_scored` positions (as intended) and then (2) filling positions beyond `seq_scored` with the per-target *global* mean (instead of clipping them to position 67). I also add a tiny, deterministic shrinkage of per-position means toward the global mean to reduce variance/overfitting to noisy positions, which often improves this competition’s leaderboard with simple baselines. Finally, I tune (slightly) your existing calibration divisor (still a single scalar) so predictions are not uniformly pushed too low, which can hurt RMSE; this is a minimal post-processing change that should reduce the gap toward 0.3589.'
- What this solution (achieved 0.42154) has done: 'Your current baseline is already close to the target but still worse (0.42203 vs 0.35898, lower is better), so we should make only a small, low-risk improvement. I keep the exact same “per-position mean from filtered train.json” core logic, but tune the two existing smoothing/calibration knobs in a metric-aligned way: slightly reduce shrinkage toward the global mean (to preserve useful per-position signal) and slightly reduce the post-scaling divisor (so predictions aren’t uniformly pushed too low). These are minimal post-processing changes that commonly yield a modest RMSE drop without changing the approach or output format. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.42148) has done: 'We keep your exact baseline approach (SN_filter==1, per-position means for the first `seq_scored` positions, global mean for unscored positions) and only adjust the two existing post-processing “knobs” that affect MCRMSE: shrinkage toward the global mean and the uniform calibration divisor. Since your current score (0.42154) is still worse than the target (0.35898, lower is better), we make a small, low-risk move toward stronger per-position signal (slightly less shrinkage) and slightly less down-scaling to avoid underpredicting. These are minimal changes that don’t alter the modeling logic or submission semantics, just the final calibrated values. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.4215) has done: 'Your current score (0.42148) is still worse than the target (0.35898; lower is better), so we should make a small, low-risk improvement without changing the baseline approach. The biggest legitimate gain for this competition’s simple baselines is usually filtering training rows to match the test distribution (the public test was filtered for higher signal/noise), so we additionally restrict training to higher `signal_to_noise` while keeping `SN_filter==1` as-is. To avoid overfitting risk from a hard cutoff, we keep the same per-position mean + global-mean for unscored positions logic and only slightly adjust the existing shrinkage/callibration knobs (minimal post-processing) to align with the new filtered means. This keeps the same core logic, runs fast, and still produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
sample_path_candidates = [
    "/kaggle/input/sample_submission.csv",
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
    "../input/sample_submission.csv",
    "../input/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
]
sample_path = next((p for p in sample_path_candidates if os.path.exists(p)), None)
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations: "
        + ", ".join(sample_path_candidates)
    )

df = pd.read_csv(sample_path)
df.head()



## === cell 2
sequences = list(set(["_".join(v.split("_")[:-1]) for v in df.id_seqpos.values]))
sequences[-10:]



## === cell 3
train_path_candidates = [
    "/kaggle/input/train.json",
    "/kaggle/input/stanford-covid-vaccine/train.json",
    "../input/train.json",
    "../input/stanford-covid-vaccine/train.json",
    "/kaggle/data/train.json",
    "/kaggle/data/stanford-covid-vaccine/train.json",
]
train_path = next((p for p in train_path_candidates if os.path.exists(p)), None)
if train_path is None:
    raise FileNotFoundError(
        "Could not find train.json in expected locations: "
        + ", ".join(train_path_candidates)
    )

train = pd.read_json(train_path, lines=True)

target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
missing_train = [c for c in target_cols if c not in train.columns]
if missing_train:
    raise ValueError(f"train.json is missing required target columns: {missing_train}")

train_use = train.copy()
if "SN_filter" in train_use.columns:
    train_use = train_use.loc[train_use["SN_filter"] == 1].copy()
    if len(train_use) == 0:
        train_use = train.copy()

if "signal_to_noise" in train_use.columns:
    snr_thr = 1.0
    train_snr = train_use.loc[train_use["signal_to_noise"] >= snr_thr].copy()
    if len(train_snr) >= 200:
        train_use = train_snr

seq_scored = int(train_use["seq_scored"].iloc[0])

means_by_pos = {}
global_means = {}

shrink_alpha = 0.975  # was 0.965

for c in target_cols:
    arr = np.stack(train_use[c].values).astype(np.float32)  # (n, seq_scored)
    arr = arr[:, :seq_scored]
    pos_mean = arr.mean(axis=0)  # (seq_scored,)
    glob_mean = float(arr.mean())
    means_by_pos[c] = (
        shrink_alpha * pos_mean + (1.0 - shrink_alpha) * glob_mean
    ).astype(np.float32)
    global_means[c] = np.float32(glob_mean)

seqpos = df["id_seqpos"].str.rsplit("_", n=1).str[-1].astype(int).values
is_scored_pos = seqpos < seq_scored
seqpos_scored = np.clip(seqpos, 0, seq_scored - 1)

for c in target_cols:
    pred = np.empty(len(df), dtype=np.float32)
    pred[is_scored_pos] = means_by_pos[c][seqpos_scored[is_scored_pos]]
    pred[~is_scored_pos] = global_means[c]
    df[c] = pred

df.head()



## === cell 4
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
missing = [c for c in target_cols if c not in df.columns]
if missing:
    raise ValueError(f"Submission is missing required columns: {missing}")

for c in target_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

df[target_cols] /= 1.004  # was 1.008

df[target_cols] = df[target_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)

df.head()



## === cell 5
assert df.shape[1] == 6, f"Expected 6 columns, got {df.shape[1]}"
assert (
    list(df.columns) == ["id_seqpos"] + target_cols
), f"Unexpected column order: {df.columns.tolist()}"
assert df["id_seqpos"].isna().sum() == 0, "id_seqpos contains NaNs"
df.describe(include="all").head()



## === cell 6
df.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(df.head())
print("Rows:", len(df), "Cols:", df.shape[1])
