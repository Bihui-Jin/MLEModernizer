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

0.41631

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'We replace the missing “ensemble52.csv” load with a simple baseline that computes overall mean values from the training data and fills the sample submission template with those means, ensuring a correctly‑named *.csv* file is written. This fixes the FileNotFoundError, defines `df`, and produces a valid submission while keeping the core logic unchanged.'
- What this solution (achieved 0.63824) has done: 'The fix changes JSON loading to handle line‑separated records, computes per‑position means (with a global fallback for positions beyond the scored region) and uses those means to fill the submission file based on each row’s sequence position. This resolves the JSONDecodeError, defines the mean variables before they are used, and improves the baseline score by providing more targeted predictions while keeping the core mean‑based logic unchanged.'
- What this solution (achieved 0.42418) has done: 'We fix the out‑of‑bounds indexing by extending each per‑position mean array to cover the full range of sequence positions that appear in the submission (positions 0‑max). The arrays are padded with the global mean, so positions beyond the scored region use the global fallback safely. This removes the `IndexError` and lets the script write a proper `submission.csv`, preserving the original mean‑based prediction logic while nudging the score toward the target.'
- What this solution (achieved 0.42608) has done: 'We blend each per‑position mean with its overall global mean (using a small weight α≈0.15). This simple smoothing reduces the impact of noisy position‑specific estimates while keeping the original mean‑based logic intact, so the predictions become slightly more regularized and should lower the MCRMSE toward the target score.'
- What this solution (achieved 0.43036) has done: 'Implemented a more accurate global mean calculation (using total sums/counts rather than mean‑of‑means) and increased the smoothing factor α to 0.30. This stronger blend with the overall mean reduces noisy per‑position estimates, aiming to lower the MCRMSE toward the target while keeping the original baseline logic untouched.'
- What this solution (achieved 0.42518) has done: 'I reduce the smoothing factor α from 0.30 to 0.10 so the predictions rely more on the per‑position means (which are usually more informative than the global mean). This small adjustment is expected to lower the MCRMSE toward the target while keeping the original baseline logic unchanged.'
- What this solution (achieved 0.63824) has done: 'I add per‑nucleotide, per‑position mean calculations and use the test sequences to look up the nucleotide at each position. Predictions now come from these more specific means (with a tiny blend α = 0.05 toward the overall global mean) and a fallback to the original position‑wise mean when a nucleotide‑specific value is missing. This keeps the original mean‑based approach while providing a modest, targeted improvement expected to lower the MCRMSE toward the target.'
- What this solution (achieved 0.42289) has done: 'I fixed the NaN‑filling logic for the per‑base mean arrays (so the mask matches dimensions) and corrected the variable name error. I also increased the smoothing factor α to 0.30 to blend more with the global mean, which reduces noisy per‑position estimates and moves the MCRMSE closer to the target while keeping the original mean‑based approach.'
- What this solution (achieved 0.43347) has done: 'I keep the overall mean‑based baseline unchanged but increase the smoothing weight `alpha` from 0.30 to 0.50, giving a stronger blend with the global mean. This stronger regularisation should reduce noise in the per‑position/base predictions and move the MCRMSE lower toward the target while preserving the original logic and output format.'
- What this solution (achieved 0.4182) has done: 'The update reduces the blending weight `alpha` from 0.50 to 0.15 so predictions rely more on the per‑position/base means that capture actual patterns in the training data, which should lower the MCRMSE toward the target while keeping the original mean‑based logic unchanged.'
- What this solution (achieved 0.63824) has done: 'I replace the fixed‑alpha blending with an adaptive shrinkage that weighs each per‑position or per‑base estimate by its observed count (so rare noisy estimates are pulled more toward the global mean). This keeps the overall mean‑based approach unchanged while likely reducing MCRMSE, moving the score closer to the target.'
- What this solution (achieved 0.41631) has done: 'I corrected the typo that prevented the per‑base mean for `deg_50C` from being created and increased the blending parameter `k` to 20.0 so the predictions are weighted more toward the global mean, which should reduce noise and lower the MCRMSE toward the target while preserving the original mean‑based logic.'

# 9. Code solution

## === cell 0
import json
import numpy as np
import pandas as pd

train_path = "../input/train.json"
train_data = []
with open(train_path, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            train_data.append(json.loads(line))

max_len = max(rec.get("seq_scored", 0) for rec in train_data)

base_idx_map = {"A": 0, "C": 1, "G": 2, "U": 3}
n_bases = len(base_idx_map)

reactivity_sum = np.zeros(max_len, dtype=float)
reactivity_cnt = np.zeros(max_len, dtype=int)
deg_mg_ph10_sum = np.zeros(max_len, dtype=float)
deg_mg_ph10_cnt = np.zeros(max_len, dtype=int)
deg_ph10_sum = np.zeros(max_len, dtype=float)
deg_ph10_cnt = np.zeros(max_len, dtype=int)
deg_mg_50c_sum = np.zeros(max_len, dtype=float)
deg_mg_50c_cnt = np.zeros(max_len, dtype=int)
deg_50c_sum = np.zeros(max_len, dtype=float)
deg_50c_cnt = np.zeros(max_len, dtype=int)

reactivity_sum_base = np.zeros((max_len, n_bases), dtype=float)
reactivity_cnt_base = np.zeros((max_len, n_bases), dtype=int)
deg_mg_ph10_sum_base = np.zeros((max_len, n_bases), dtype=float)
deg_mg_ph10_cnt_base = np.zeros((max_len, n_bases), dtype=int)
deg_ph10_sum_base = np.zeros((max_len, n_bases), dtype=float)
deg_ph10_cnt_base = np.zeros((max_len, n_bases), dtype=int)
deg_mg_50c_sum_base = np.zeros((max_len, n_bases), dtype=float)
deg_mg_50c_cnt_base = np.zeros((max_len, n_bases), dtype=int)
deg_50c_sum_base = np.zeros((max_len, n_bases), dtype=float)
deg_50c_cnt_base = np.zeros((max_len, n_bases), dtype=int)

for rec in train_data:
    seq = rec.get("sequence", "")
    for i, val in enumerate(rec.get("reactivity", [])):
        reactivity_sum[i] += val
        reactivity_cnt[i] += 1
    for i, val in enumerate(rec.get("deg_Mg_pH10", [])):
        deg_mg_ph10_sum[i] += val
        deg_mg_ph10_cnt[i] += 1
    for i, val in enumerate(rec.get("deg_pH10", [])):
        deg_ph10_sum[i] += val
        deg_ph10_cnt[i] += 1
    for i, val in enumerate(rec.get("deg_Mg_50C", [])):
        deg_mg_50c_sum[i] += val
        deg_mg_50c_cnt[i] += 1
    for i, val in enumerate(rec.get("deg_50C", [])):
        deg_50c_sum[i] += val
        deg_50c_cnt[i] += 1

    for i, base in enumerate(seq[:max_len]):
        bidx = base_idx_map.get(base)
        if bidx is None:
            continue
        if i < len(rec.get("reactivity", [])):
            reactivity_sum_base[i, bidx] += rec["reactivity"][i]
            reactivity_cnt_base[i, bidx] += 1
        if i < len(rec.get("deg_Mg_pH10", [])):
            deg_mg_ph10_sum_base[i, bidx] += rec["deg_Mg_pH10"][i]
            deg_mg_ph10_cnt_base[i, bidx] += 1
        if i < len(rec.get("deg_pH10", [])):
            deg_ph10_sum_base[i, bidx] += rec["deg_pH10"][i]
            deg_ph10_cnt_base[i, bidx] += 1
        if i < len(rec.get("deg_Mg_50C", [])):
            deg_mg_50c_sum_base[i, bidx] += rec["deg_Mg_50C"][i]
            deg_mg_50c_cnt_base[i, bidx] += 1
        if i < len(rec.get("deg_50C", [])):
            deg_50c_sum_base[i, bidx] += rec["deg_50C"][i]
            deg_50c_cnt_base[i, bidx] += 1

reactivity_mean_pos = np.where(
    reactivity_cnt > 0, reactivity_sum / reactivity_cnt, np.nan
)
deg_mg_ph10_mean_pos = np.where(
    deg_mg_ph10_cnt > 0, deg_mg_ph10_sum / deg_mg_ph10_cnt, np.nan
)
deg_ph10_mean_pos = np.where(deg_ph10_cnt > 0, deg_ph10_sum / deg_ph10_cnt, np.nan)
deg_mg_50c_mean_pos = np.where(
    deg_mg_50c_cnt > 0, deg_mg_50c_sum / deg_mg_50c_cnt, np.nan
)
deg_50c_mean_pos = np.where(deg_50c_cnt > 0, deg_50c_sum / deg_50c_cnt, np.nan)

reactivity_mean_global = (
    reactivity_sum.sum() / reactivity_cnt.sum() if reactivity_cnt.sum() > 0 else 0.0
)
deg_mg_ph10_mean_global = (
    deg_mg_ph10_sum.sum() / deg_mg_ph10_cnt.sum() if deg_mg_ph10_cnt.sum() > 0 else 0.0
)
deg_ph10_mean_global = (
    deg_ph10_sum.sum() / deg_ph10_cnt.sum() if deg_ph10_cnt.sum() > 0 else 0.0
)
deg_mg_50c_mean_global = (
    deg_mg_50c_sum.sum() / deg_mg_50c_cnt.sum() if deg_mg_50c_cnt.sum() > 0 else 0.0
)
deg_50c_mean_global = (
    deg_50c_sum.sum() / deg_50c_cnt.sum() if deg_50c_cnt.sum() > 0 else 0.0
)

reactivity_mean_pos = np.where(
    np.isnan(reactivity_mean_pos), reactivity_mean_global, reactivity_mean_pos
)
deg_mg_ph10_mean_pos = np.where(
    np.isnan(deg_mg_ph10_mean_pos), deg_mg_ph10_mean_global, deg_mg_ph10_mean_pos
)
deg_ph10_mean_pos = np.where(
    np.isnan(deg_ph10_mean_pos), deg_ph10_mean_global, deg_ph10_mean_pos
)
deg_mg_50c_mean_pos = np.where(
    np.isnan(deg_mg_50c_mean_pos), deg_mg_50c_mean_global, deg_mg_50c_mean_pos
)
deg_50c_mean_pos = np.where(
    np.isnan(deg_50c_mean_pos), deg_50c_mean_global, deg_50c_mean_pos
)


def base_mean_or_pos(mean_base, mean_pos):
    """Fill missing per‑base values with the per‑position mean."""
    filled = np.where(np.isnan(mean_base), mean_pos[:, None], mean_base)
    return filled


reactivity_mean_base = base_mean_or_pos(
    np.where(
        reactivity_cnt_base > 0, reactivity_sum_base / reactivity_cnt_base, np.nan
    ),
    reactivity_mean_pos,
)
deg_mg_ph10_mean_base = base_mean_or_pos(
    np.where(
        deg_mg_ph10_cnt_base > 0,
        deg_mg_ph10_sum_base / deg_mg_ph10_cnt_base,
        np.nan,
    ),
    deg_mg_ph10_mean_pos,
)
deg_ph10_mean_base = base_mean_or_pos(
    np.where(deg_ph10_cnt_base > 0, deg_ph10_sum_base / deg_ph10_cnt_base, np.nan),
    deg_ph10_mean_pos,
)
deg_mg_50c_mean_base = base_mean_or_pos(
    np.where(
        deg_mg_50c_cnt_base > 0,
        deg_mg_50c_sum_base / deg_mg_50c_cnt_base,
        np.nan,
    ),
    deg_mg_50c_mean_pos,
)
deg_50c_mean_base = base_mean_or_pos(
    np.where(deg_50c_cnt_base > 0, deg_50c_sum_base / deg_50c_cnt_base, np.nan),
    deg_50c_mean_pos,
)



## === cell 1
test_path = "../input/test.json"
test_data = []
with open(test_path, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            test_data.append(json.loads(line))

id_to_seq = {rec["id"]: rec["sequence"] for rec in test_data}

sample_sub_path = "../input/sample_submission.csv"
df = pd.read_csv(sample_sub_path)

df["seqpos"] = df["id_seqpos"].apply(lambda x: int(x.split("_")[-1]))
df["sample_id"] = df["id_seqpos"].apply(lambda x: x.rsplit("_", 1)[0])

df["base"] = df.apply(
    lambda row: id_to_seq.get(row["sample_id"], "A")[row["seqpos"]], axis=1
)

pos = df["seqpos"].values
base_codes = df["base"].map(base_idx_map).values
max_pos_needed = int(pos.max())


def extend_mean(arr, global_mean, needed_len):
    """Extend 1‑D array to cover the required length with the global mean."""
    if needed_len + 1 > len(arr):
        extended = np.full(needed_len + 1, global_mean, dtype=arr.dtype)
        extended[: len(arr)] = arr
        return extended
    return arr


def extend_mean_2d(arr, global_mean, needed_len):
    """Extend 2‑D array (position × base) similarly."""
    if needed_len + 1 > arr.shape[0]:
        extended = np.full((needed_len + 1, arr.shape[1]), global_mean, dtype=arr.dtype)
        extended[: arr.shape[0], :] = arr
        return extended
    return arr


reactivity_ext = extend_mean(
    reactivity_mean_pos, reactivity_mean_global, max_pos_needed
)
deg_mg_ph10_ext = extend_mean(
    deg_mg_ph10_mean_pos, deg_mg_ph10_mean_global, max_pos_needed
)
deg_ph10_ext = extend_mean(deg_ph10_mean_pos, deg_ph10_mean_global, max_pos_needed)
deg_mg_50c_ext = extend_mean(
    deg_mg_50c_mean_pos, deg_mg_50c_mean_global, max_pos_needed
)
deg_50c_ext = extend_mean(deg_50c_mean_pos, deg_50c_mean_global, max_pos_needed)

reactivity_cnt_ext = extend_mean(reactivity_cnt, 0, max_pos_needed)
deg_mg_ph10_cnt_ext = extend_mean(deg_mg_ph10_cnt, 0, max_pos_needed)
deg_ph10_cnt_ext = extend_mean(deg_ph10_cnt, 0, max_pos_needed)
deg_mg_50c_cnt_ext = extend_mean(deg_mg_50c_cnt, 0, max_pos_needed)
deg_50c_cnt_ext = extend_mean(deg_50c_cnt, 0, max_pos_needed)

reactivity_base_ext = extend_mean_2d(
    reactivity_mean_base, reactivity_mean_global, max_pos_needed
)
deg_mg_ph10_base_ext = extend_mean_2d(
    deg_mg_ph10_mean_base, deg_mg_ph10_mean_global, max_pos_needed
)
deg_ph10_base_ext = extend_mean_2d(
    deg_ph10_mean_base, deg_ph10_mean_global, max_pos_needed
)
deg_mg_50c_base_ext = extend_mean_2d(
    deg_mg_50c_mean_base, deg_mg_50c_mean_global, max_pos_needed
)
deg_50c_base_ext = extend_mean_2d(
    deg_50c_mean_base, deg_50c_mean_global, max_pos_needed
)

reactivity_cnt_base_ext = extend_mean_2d(reactivity_cnt_base, 0, max_pos_needed)
deg_mg_ph10_cnt_base_ext = extend_mean_2d(deg_mg_ph10_cnt_base, 0, max_pos_needed)
deg_ph10_cnt_base_ext = extend_mean_2d(deg_ph10_cnt_base, 0, max_pos_needed)
deg_mg_50c_cnt_base_ext = extend_mean_2d(deg_mg_50c_cnt_base, 0, max_pos_needed)
deg_50c_cnt_base_ext = extend_mean_2d(deg_50c_cnt_base, 0, max_pos_needed)

k = 20.0


def adaptive_blend(mean_arr, cnt_arr, global_mean):
    """Blend per‑position/base mean with the global mean using count‑based weighting."""
    weight = cnt_arr / (cnt_arr + k)
    return weight * mean_arr + (1.0 - weight) * global_mean


reactivity_blend = adaptive_blend(
    reactivity_ext, reactivity_cnt_ext, reactivity_mean_global
)
deg_mg_ph10_blend = adaptive_blend(
    deg_mg_ph10_ext, deg_mg_ph10_cnt_ext, deg_mg_ph10_mean_global
)
deg_ph10_blend = adaptive_blend(deg_ph10_ext, deg_ph10_cnt_ext, deg_ph10_mean_global)
deg_mg_50c_blend = adaptive_blend(
    deg_mg_50c_ext, deg_mg_50c_cnt_ext, deg_mg_50c_mean_global
)
deg_50c_blend = adaptive_blend(deg_50c_ext, deg_50c_cnt_ext, deg_50c_mean_global)

reactivity_blend_base = adaptive_blend(
    reactivity_base_ext, reactivity_cnt_base_ext, reactivity_mean_global
)
deg_mg_ph10_blend_base = adaptive_blend(
    deg_mg_ph10_base_ext, deg_mg_ph10_cnt_base_ext, deg_mg_ph10_mean_global
)
deg_ph10_blend_base = adaptive_blend(
    deg_ph10_base_ext, deg_ph10_cnt_base_ext, deg_ph10_mean_global
)
deg_mg_50c_blend_base = adaptive_blend(
    deg_mg_50c_base_ext, deg_mg_50c_cnt_base_ext, deg_mg_50c_mean_global
)
deg_50c_blend_base = adaptive_blend(
    deg_50c_base_ext, deg_50c_cnt_base_ext, deg_50c_mean_global
)

df["reactivity"] = reactivity_blend_base[pos, base_codes]
df["deg_Mg_pH10"] = deg_mg_ph10_blend_base[pos, base_codes]
df["deg_pH10"] = deg_ph10_blend_base[pos, base_codes]
df["deg_Mg_50C"] = deg_mg_50c_blend_base[pos, base_codes]
df["deg_50C"] = deg_50c_blend_base[pos, base_codes]

df = df.drop(columns=["seqpos", "sample_id", "base"])



## === cell 2
df.to_csv("submission.csv", index=False)
