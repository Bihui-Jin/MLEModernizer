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

0.41973

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.47906) has done: 'I replace the missing ensemble CSV load with a simple baseline that predicts the mean values of the three scored targets from the training data for every position in each test sample, and then write a correctly formatted `submission.csv`. This fixes the FileNotFoundError, removes undefined variables, and ensures a valid submission file is produced while keeping the core logic unchanged.'
- What this solution (achieved 0.42418) has done: 'I replace the single global‑mean baseline with a per‑position mean baseline for the three scored targets (reactivity, deg_Mg_pH10, deg_Mg_50C). For each position that is within the scored region of a test sample I use the corresponding mean value computed from the training set; for positions outside the scored region I keep the existing zero placeholder. This adds only a small amount of computation, preserves the overall workflow, and is expected to lower the MCRMSE toward the target score.'
- What this solution (achieved 0.4437) has done: 'I replace the global‑mean baseline with a per‑position median baseline calculated only on the high‑quality training samples (`SN_filter == 1`). Medians are more robust to outliers and should reduce the MCRMSE, moving the score closer to the target while keeping the overall workflow unchanged. I also fill the not‑scored targets (`deg_pH10`, `deg_50C`) with their per‑position medians for consistency.'
- What this solution (achieved 0.42166) has done: 'Implemented missing imports, switched to using only high‑quality training samples (`SN_filter == 1`) for the per‑position means (which improves the baseline toward the target score), and kept the original submission‑generation logic. The script now runs end‑to‑end and writes a correctly formatted `submission.csv`.'
- What this solution (achieved 0.42418) has done: 'I replace the high‑quality‑only mean calculations with means computed over the full training set. Using all available samples provides a more stable per‑position estimate, which should reduce the MCRMSE and move the score closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.41641) has done: 'We keep the same overall workflow but replace the simple per‑position global‑mean baseline with a per‑position, nucleotide‑aware mean. For each of the three scored targets (reactivity, deg_Mg_pH10, deg_Mg_50C) we compute the average value for every position 0‑67 conditioned on the base (A, C, G, U) found in the training sequences. At prediction time we look up the test sample’s base at that position and use the corresponding conditional mean, falling back to the overall position mean when a base‑specific average is unavailable. This small, data‑driven tweak is expected to reduce the MCRMSE and move the score closer to the target while preserving the original pipeline logic.'
- What this solution (achieved 0.41187) has done: 'The update filters the training data to keep only high‑quality samples (where `SN_filter == 1`) before computing all per‑position and nucleotide‑aware statistics. This reduces noise in the learned baselines, so predictions should be closer to the true values and the MCRMSE move toward the target lower score while preserving the original workflow.'
- What this solution (achieved 0.41325) has done: 'I keep the same high‑quality filtering and per‑nucleotide statistics but add a tiny smoothing step: compute a 3‑position moving‑average of the per‑position means and blend it (70 % nucleotide‑specific + 30 % smoothed position mean). This uses only the existing data, adds negligible overhead, and is expected to reduce noise in the baseline predictions, moving the MCRMSE closer to the target value.'
- What this solution (achieved 0.41187) has done: 'We tighten the baseline by relying primarily on the per‑nucleotide statistics (which capture sequence‑specific effects) and fall back to the plain per‑position mean only when a nucleotide‑specific value is missing. This removes the smoothing blend that diluted the nucleotide signal, keeping the core workflow unchanged while shifting predictions toward lower MCRMSE.'
- What this solution (achieved 0.41257) has done: 'I keep the same data loading, filtering, and per‑position + per‑nucleotide statistics, but blend the nucleotide‑specific mean with the overall position mean (70 % nucleotide‑specific, 30 % position mean). This smooths noisy nucleotide‑specific estimates while preserving the core logic, and should lower the MCRMSE a bit, moving the score toward the target.'
- What this solution (achieved 0.41973) has done: 'I remove the high‑quality filter so the statistics use all training samples (giving more stable estimates) and lower the blending factor `ALPHA` to rely more on the robust per‑position mean. These minor tweaks keep the same workflow while expected to reduce the MCRMSE toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

train_path = "../input/stanford-covid-vaccine/train.json"
df_train = pd.read_json(train_path, lines=True)


MAX_POS = 68
nuc_to_idx = {"A": 0, "C": 1, "G": 2, "U": 3}
num_nucs = len(nuc_to_idx)

sum_reac = np.zeros((MAX_POS, num_nucs))
cnt_reac = np.zeros((MAX_POS, num_nucs))

sum_mg_pH10 = np.zeros((MAX_POS, num_nucs))
cnt_mg_pH10 = np.zeros((MAX_POS, num_nucs))

sum_mg_50C = np.zeros((MAX_POS, num_nucs))
cnt_mg_50C = np.zeros((MAX_POS, num_nucs))

reactivity_pos_mean = np.zeros(MAX_POS)
deg_mg_pH10_pos_mean = np.zeros(MAX_POS)
deg_mg_50C_pos_mean = np.zeros(MAX_POS)

for _, row in df_train.iterrows():
    seq = row["sequence"]
    seq_scored = row["seq_scored"]
    reac_vals = row["reactivity"]
    mg_pH10_vals = row["deg_Mg_pH10"]
    mg_50C_vals = row["deg_Mg_50C"]
    for pos in range(seq_scored):
        nuc = seq[pos]
        idx = nuc_to_idx.get(nuc)
        if idx is None:
            continue
        sum_reac[pos, idx] += reac_vals[pos]
        cnt_reac[pos, idx] += 1

        sum_mg_pH10[pos, idx] += mg_pH10_vals[pos]
        cnt_mg_pH10[pos, idx] += 1

        sum_mg_50C[pos, idx] += mg_50C_vals[pos]
        cnt_mg_50C[pos, idx] += 1

        reactivity_pos_mean[pos] += reac_vals[pos]
        deg_mg_pH10_pos_mean[pos] += mg_pH10_vals[pos]
        deg_mg_50C_pos_mean[pos] += mg_50C_vals[pos]

n_samples = len(df_train)
reactivity_pos_mean /= n_samples
deg_mg_pH10_pos_mean /= n_samples
deg_mg_50C_pos_mean /= n_samples

with np.errstate(divide="ignore", invalid="ignore"):
    reac_nuc_mean = np.where(cnt_reac > 0, sum_reac / cnt_reac, np.nan)
    mg_pH10_nuc_mean = np.where(cnt_mg_pH10 > 0, sum_mg_pH10 / cnt_mg_pH10, np.nan)
    mg_50C_nuc_mean = np.where(cnt_mg_50C > 0, sum_mg_50C / cnt_mg_50C, np.nan)

zero_placeholder = 0.0

ALPHA = 0.3




## === cell 1
test_path = "../input/stanford-covid-vaccine/test.json"
df_test = pd.read_json(test_path, lines=True)

submission_rows = []

for _, row in df_test.iterrows():
    sample_id = row["id"]
    seq_len = row["seq_length"]
    seq_scored = row["seq_scored"]
    sequence = row["sequence"]

    for pos in range(seq_len):
        id_seqpos = f"{sample_id}_{pos}"
        if pos < seq_scored:
            nuc = sequence[pos]
            idx = nuc_to_idx.get(nuc)

            if idx is not None and not np.isnan(reac_nuc_mean[pos, idx]):
                reactivity_val = (
                    ALPHA * reac_nuc_mean[pos, idx]
                    + (1 - ALPHA) * reactivity_pos_mean[pos]
                )
            else:
                reactivity_val = reactivity_pos_mean[pos]

            if idx is not None and not np.isnan(mg_pH10_nuc_mean[pos, idx]):
                deg_mg_pH10_val = (
                    ALPHA * mg_pH10_nuc_mean[pos, idx]
                    + (1 - ALPHA) * deg_mg_pH10_pos_mean[pos]
                )
            else:
                deg_mg_pH10_val = deg_mg_pH10_pos_mean[pos]

            if idx is not None and not np.isnan(mg_50C_nuc_mean[pos, idx]):
                deg_mg_50C_val = (
                    ALPHA * mg_50C_nuc_mean[pos, idx]
                    + (1 - ALPHA) * deg_mg_50C_pos_mean[pos]
                )
            else:
                deg_mg_50C_val = deg_mg_50C_pos_mean[pos]

            deg_pH10_val = zero_placeholder
            deg_50C_val = zero_placeholder
        else:
            reactivity_val = zero_placeholder
            deg_mg_pH10_val = zero_placeholder
            deg_mg_50C_val = zero_placeholder
            deg_pH10_val = zero_placeholder
            deg_50C_val = zero_placeholder

        submission_rows.append(
            {
                "id_seqpos": id_seqpos,
                "reactivity": reactivity_val,
                "deg_Mg_pH10": deg_mg_pH10_val,
                "deg_pH10": deg_pH10_val,
                "deg_Mg_50C": deg_mg_50C_val,
                "deg_50C": deg_50C_val,
            }
        )

submission_df = pd.DataFrame(submission_rows)




## === cell 2
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(
    f"Submission written to {os.path.abspath(submission_path)} "
    f"with shape {submission_df.shape}"
)
