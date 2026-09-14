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

0.70156

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'The fix adds the missing pandas import, uses the correct absolute paths for the Kaggle data, and simplifies the workflow to just load the test set and the provided sample submission before writing it out as `submission.csv`. No model changes are introduced, ensuring the code runs end‑to‑end and produces a valid submission file.'
- What this solution (achieved 0.42418) has done: 'I add a lightweight baseline that uses the mean values of each target at every sequence position from the training set. This replaces the placeholder “copy‑sample‑submission” approach with a simple, deterministic prediction that is known to improve MCRMSE. The script now (1) loads the training data, (2) computes per‑position means for the three scored targets, (3) maps those means to each test row based on the position extracted from `id_seqpos`, and (4) writes the final `submission.csv`. No complex modeling is introduced, preserving the original workflow while moving the score closer to the target.'
- What this solution (achieved 0.47906) has done: 'I enhance the baseline by conditioning the per‑position means on the nucleotide at that position. For each of the three scored targets I compute mean values for each base (A, C, G, U) from the training data and use these calibrated predictions for the test rows. When a base‑specific mean is unavailable (e.g., positions beyond 68 or rare bases) the code falls back to the simple per‑position overall mean, preserving the original logic. This modest adjustment is expected to lower the MCRMSE toward the target without altering the core workflow.'
- What this solution (achieved 0.47906) has done: 'I keep the overall workflow unchanged but add a small smoothing step when using the base‑specific means: predictions be a weighted blend of the per‑base mean and the plain per‑position mean, where the weight depends on how many training examples support that base‑position pair. This reduces over‑fitting on sparse base‑position combos and should lower the MCRMSE toward the target. I also fill the remaining three submission columns (deg_pH10, deg_50C) with their per‑position averages for completeness.'
- What this solution (achieved 0.47906) has done: 'I add base‑specific smoothing for the remaining two scored targets (`deg_pH10` and `deg_50C`). This mirrors the existing logic for the other three targets, giving the model a little more signal without changing its overall structure. The new per‑base means are computed together with the existing ones, and the predictions for these columns now use the same `smooth_predict` blending function. This small enrichment should lower the MCRMSE toward the target value.'
- What this solution (achieved 0.47906) has done: 'I lower the smoothing factor and require at least two observations before trusting a base‑specific mean, so predictions rely more on the richer per‑base statistics that usually improve MCRMSE. I also remove the hard‑coded zero‑override for a specific id prefix, which can hurt performance. These minimal adjustments keep the overall workflow unchanged while moving the score closer to the target.'
- What this solution (achieved 0.48075) has done: 'I keep the overall workflow unchanged but make two focused adjustments:  
1. Increase the influence of base‑specific statistics for the reactivity prediction by lowering the smoothing constant to 0.5 (still respecting the minimum count).  
2. Predict the remaining three scored targets by scaling the reactivity prediction with per‑position ratios derived from the training data (target / reactivity means). This simple linear calibration keeps the core logic intact while providing a more informative baseline, expected to lower the MCRMSE toward the target.'
- What this solution (achieved 0.47906) has done: 'I replace the linear scaling of the other targets with the same base‑specific smoothing used for reactivity. By predicting each scored target directly from per‑position and per‑base means (with the existing smoothing factor), the model stays structurally unchanged but should lower the MCRMSE, moving the score toward the target.'
- What this solution (achieved 0.70156) has done: 'The update keeps the original workflow but refines the baseline: it lowers the minimum count for using base‑specific statistics, reduces the smoothing factor so base‑specific means have more influence, and replaces the separate per‑position means for the four non‑reactivity targets with a simple linear scaling of the predicted reactivity using per‑position ratios derived from the training data. This small calibration is expected to lower the MCRMSE and move the score closer to the target while preserving the core logic.'
- What this solution (achieved 0.47906) has done: 'I replace the linear‑ratio scaling of the four non‑reactivity targets with the same base‑specific smoothing used for the reactivity prediction. By directly blending per‑position means with per‑base means we exploit more signal from the training data, which should lower the MCRMSE toward the target while keeping the overall workflow unchanged.'
- What this solution (achieved 0.48075) has done: 'I keep the overall workflow unchanged but improve the baseline by (1) trusting base‑specific statistics a bit more (reduce SMOOTH_K to 0.1) and (2) predicting the four non‑reactivity targets via a simple linear scaling of the reactivity prediction using per‑position ratios derived from the training data. This adds a modest, well‑grounded calibration that should lower the MCRMSE toward the target while preserving the core logic.'
- What this solution (achieved 0.47906) has done: 'I lower the smoothing constant so base‑specific statistics have a bit more influence, and I replace the linear scaling of the non‑reactivity targets with the same smooth‑predict blending used for reactivity. This keeps the core workflow unchanged while giving each target a more informative baseline, which should reduce the MCRMSE toward the target score.'
- What this solution (achieved 0.70156) has done: 'I keep the original workflow but replace the loosely‑calibrated per‑position/base smoothing for the four non‑reactivity targets with a simple linear scaling using the mean ratio of each target to reactivity observed in the training set. This adds a small, data‑driven calibration that typically improves the correlation between targets and therefore lowers the MCRMSE, moving the score closer to the target while leaving the core model unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

BASE_INPUT = "/kaggle/input/stanford-covid-vaccine"

df_test = pd.read_json(os.path.join(BASE_INPUT, "test.json"), lines=True)
df_sub = pd.read_csv(os.path.join(BASE_INPUT, "sample_submission.csv"))
df_train = pd.read_json(os.path.join(BASE_INPUT, "train.json"), lines=True)

reactivity_means = np.mean(np.vstack(df_train["reactivity"].values), axis=0)  # (68,)
deg_mg_ph10_means = np.mean(np.vstack(df_train["deg_Mg_pH10"].values), axis=0)
deg_mg_50c_means = np.mean(np.vstack(df_train["deg_Mg_50C"].values), axis=0)
deg_pH10_means = np.mean(np.vstack(df_train["deg_pH10"].values), axis=0)
deg_50c_means = np.mean(np.vstack(df_train["deg_50C"].values), axis=0)

bases = ["A", "C", "G", "U"]
base_to_idx = {b: i for i, b in enumerate(bases)}
num_pos = 68
num_bases = len(bases)

sums_react = np.zeros((num_pos, num_bases))
sums_deg_ph10 = np.zeros((num_pos, num_bases))
sums_deg_50c = np.zeros((num_pos, num_bases))
sums_deg_pH10 = np.zeros((num_pos, num_bases))
sums_deg_50C = np.zeros((num_pos, num_bases))
counts = np.zeros((num_pos, num_bases))

for _, row in df_train.iterrows():
    seq = row["sequence"]
    react = row["reactivity"]
    deg_ph10 = row["deg_Mg_pH10"]
    deg_50c = row["deg_Mg_50C"]
    deg_pH10 = row["deg_pH10"]
    deg_50C = row["deg_50C"]
    for p in range(len(react)):
        base = seq[p]
        idx = base_to_idx.get(base, -1)
        if idx >= 0:
            sums_react[p, idx] += react[p]
            sums_deg_ph10[p, idx] += deg_ph10[p]
            sums_deg_50c[p, idx] += deg_50c[p]
            sums_deg_pH10[p, idx] += deg_pH10[p]
            sums_deg_50C[p, idx] += deg_50C[p]
            counts[p, idx] += 1

means_react_base = np.divide(
    sums_react, counts, out=np.zeros_like(sums_react), where=counts != 0
)
means_deg_ph10_base = np.divide(
    sums_deg_ph10, counts, out=np.zeros_like(sums_deg_ph10), where=counts != 0
)
means_deg_50c_base = np.divide(
    sums_deg_50c, counts, out=np.zeros_like(sums_deg_50c), where=counts != 0
)
means_deg_pH10_base = np.divide(
    sums_deg_pH10, counts, out=np.zeros_like(sums_deg_pH10), where=counts != 0
)
means_deg_50C_base = np.divide(
    sums_deg_50C, counts, out=np.zeros_like(sums_deg_50C), where=counts != 0
)


def extract_pos(id_seqpos):
    try:
        return int(id_seqpos.split("_")[-1])
    except:
        return -1


def extract_id(id_seqpos):
    try:
        sample_part = id_seqpos.rsplit("_", 1)[0]
        return sample_part[3:] if sample_part.startswith("id_") else sample_part
    except:
        return ""


positions = df_sub["id_seqpos"].apply(extract_pos)
sample_ids = df_sub["id_seqpos"].apply(extract_id)

id_to_seq = dict(zip(df_test["id"], df_test["sequence"]))

overall_reactivity_mean = reactivity_means.mean()
overall_deg_mg_ph10_mean = deg_mg_ph10_means.mean()
overall_deg_mg_50c_mean = deg_mg_50c_means.mean()
overall_deg_pH10_mean = deg_pH10_means.mean()
overall_deg_50c_mean = deg_50c_means.mean()

SMOOTH_K = 0.05  # a small weight for base‑specific means
MIN_COUNT = 1


def smooth_predict(pos, sample_id, overall_mean, per_pos_means, per_base_means):
    """Blend base‑specific mean with the per‑position mean."""
    if pos < 0 or pos >= num_pos:
        return overall_mean
    seq = id_to_seq.get(sample_id, "")
    if len(seq) <= pos:
        return overall_mean
    base = seq[pos]
    idx = base_to_idx.get(base, -1)
    if idx >= 0 and counts[pos, idx] >= MIN_COUNT:
        w = counts[pos, idx] / (counts[pos, idx] + SMOOTH_K)
        return w * per_base_means[pos, idx] + (1 - w) * per_pos_means[pos]
    else:
        return per_pos_means[pos]


df_sub["reactivity"] = [
    smooth_predict(p, sid, overall_reactivity_mean, reactivity_means, means_react_base)
    for p, sid in zip(positions, sample_ids)
]

react_arr = np.vstack(df_train["reactivity"].values)
deg_mg_ph10_arr = np.vstack(df_train["deg_Mg_pH10"].values)
deg_pH10_arr = np.vstack(df_train["deg_pH10"].values)
deg_mg_50c_arr = np.vstack(df_train["deg_Mg_50C"].values)
deg_50c_arr = np.vstack(df_train["deg_50C"].values)

ratio_deg_mg_ph10 = np.where(react_arr != 0, deg_mg_ph10_arr / react_arr, 0.0)
ratio_deg_pH10 = np.where(react_arr != 0, deg_pH10_arr / react_arr, 0.0)
ratio_deg_mg_50c = np.where(react_arr != 0, deg_mg_50c_arr / react_arr, 0.0)
ratio_deg_50c = np.where(react_arr != 0, deg_50c_arr / react_arr, 0.0)

ratio_deg_mg_ph10_means = np.mean(ratio_deg_mg_ph10, axis=0)
ratio_deg_pH10_means = np.mean(ratio_deg_pH10, axis=0)
ratio_deg_mg_50c_means = np.mean(ratio_deg_mg_50c, axis=0)
ratio_deg_50c_means = np.mean(ratio_deg_50c, axis=0)

overall_ratio_deg_mg_ph10 = (
    overall_deg_mg_ph10_mean / overall_reactivity_mean
    if overall_reactivity_mean != 0
    else 0
)
overall_ratio_deg_pH10 = (
    overall_deg_pH10_mean / overall_reactivity_mean
    if overall_reactivity_mean != 0
    else 0
)
overall_ratio_deg_mg_50c = (
    overall_deg_mg_50c_mean / overall_reactivity_mean
    if overall_reactivity_mean != 0
    else 0
)
overall_ratio_deg_50c = (
    overall_deg_50c_mean / overall_reactivity_mean
    if overall_reactivity_mean != 0
    else 0
)


def ratio_scaled(pos, react_pred, ratio_means, overall_ratio):
    """Scale reactivity prediction by the learned per‑position ratio."""
    if 0 <= pos < num_pos:
        return react_pred * ratio_means[pos]
    else:
        return react_pred * overall_ratio


react_preds = df_sub["reactivity"].values
pos_array = positions.values

df_sub["deg_Mg_pH10"] = [
    ratio_scaled(p, r, ratio_deg_mg_ph10_means, overall_ratio_deg_mg_ph10)
    for p, r in zip(pos_array, react_preds)
]

df_sub["deg_pH10"] = [
    ratio_scaled(p, r, ratio_deg_pH10_means, overall_ratio_deg_pH10)
    for p, r in zip(pos_array, react_preds)
]

df_sub["deg_Mg_50C"] = [
    ratio_scaled(p, r, ratio_deg_mg_50c_means, overall_ratio_deg_mg_50c)
    for p, r in zip(pos_array, react_preds)
]

df_sub["deg_50C"] = [
    ratio_scaled(p, r, ratio_deg_50c_means, overall_ratio_deg_50c)
    for p, r in zip(pos_array, react_preds)
]



## === cell 1
output_path = "submission.csv"
df_sub.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}, shape: {df_sub.shape}")
