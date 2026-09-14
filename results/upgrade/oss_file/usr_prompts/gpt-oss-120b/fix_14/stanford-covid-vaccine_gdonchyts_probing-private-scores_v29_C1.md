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

0.44468

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.47906) has done: 'The fix removes the missing “ensemble52.csv” load and replaces the broken logic with a simple baseline: compute overall mean values of each target from the training set and use those means as predictions for every position of every test sequence, then write a correctly‑formatted `submission.csv`.'
- What this solution (achieved 0.44603) has done: 'I replace the global‑mean baseline with a simple position‑wise mean model: for each of the three scored targets (reactivity, deg_Mg_pH10, deg_pH10) I compute the average value at every sequence position across the training set and use those per‑position averages for the first seq_scored bases of each test sample (the only bases that affect the leaderboard). For the remaining positions (up to the full length 107) I fall back to the overall mean, preserving the required output shape while giving more accurate predictions for the scored region, which should move the MCRMSE closer to the target score.'
- What this solution (achieved 0.4448) has done: 'I filter the training data to keep only high‑quality samples (SN_filter = 1 and signal_to_noise > 1.0) before computing the per‑position and overall means. This modest change preserves the original baseline logic but should yield more accurate average values for the scored positions, moving the MCRMSE closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.42166) has done: 'I add a per‑position mean for the scored target **deg_Mg_50C** (which was previously predicted with a global mean) and use it for the first `seq_scored` positions of each test sample. This small calibration should lower the MCRMSE and bring the score nearer the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.42166) has done: 'I add a per‑position mean for the fifth scored target (`deg_50C`) and use it for the first `seq_scored` positions of each test sample, falling back to the overall mean for the remaining positions. This small calibration keeps the original baseline logic while improving predictions for all three scored columns, which should move the MCRMSE closer to the target (lower is better).'
- What this solution (achieved 0.44369) has done: 'I replace the simple mean‑based baseline with a median‑based baseline, which is less sensitive to outliers and should lower the MCRMSE a bit while keeping the same overall structure. The code now computes per‑position medians for each target and uses the overall median as a fallback for positions beyond the scored region, then writes the submission file.'
- What this solution (achieved 0.42166) has done: 'Implemented a switch from per‑position **median** to per‑position **mean** for the three scored targets (reactivity, deg_Mg_pH10, deg_Mg_50C) while keeping the same filtering and fallback overall‑mean logic. This change aligns the baseline with the central tendency of the training data (means are less sensitive to outliers than medians) and is expected to lower the MCRMSE, moving the score closer to the target. The rest of the pipeline and submission format remain unchanged.'
- What this solution (achieved 0.43568) has done: 'I smooth the per‑position mean arrays with a tiny moving‑average (window = 3) before they are used for predictions. This keeps the overall baseline logic unchanged but reduces noise in the position‑wise estimates, which should lower the MCRMSE and move the score closer to the target.'
- What this solution (achieved 0.43719) has done: 'I blend the smoothed per‑position means with the overall means (using a high‑weight α≈0.85) to regularise noisy position estimates while keeping the same overall pipeline. This small adjustment is expected to lower the MCRMSE and move the score closer to the target without altering the core logic.'
- What this solution (achieved 0.44496) has done: 'I slightly increase the reliance on the position‑wise averages (raise α to 0.95) and use a wider smoothing window (size 5) to make the per‑position estimates more stable. I also switch the fallback overall means to those computed from the full training set (unfiltered) so non‑scored positions use the broader data distribution. These small tweaks keep the overall baseline logic intact while expectedly lowering the MCRMSE toward the target.'
- What this solution (achieved 0.44639) has done: 'I replace the simple un‑weighted per‑position means with a signal‑to‑noise‑weighted mean, then smooth and blend them using a slightly lower α (0.85) so the model is less prone to noisy position estimates. This keeps the overall baseline structure intact while improving the calibration of the predictions, which should lower the MCRMSE toward the target score.'
- What this solution (achieved 0.44451) has done: 'I replace the weighted per‑position means with simple un‑weighted averages (still using the filtered high‑quality subset) and remove the blending with the overall mean, keeping only a light smoothing step. This gives the model a clearer position‑wise signal for the scored region, which should lower the MCRMSE toward the target while preserving the overall pipeline.'
- What this solution (achieved 0.44468) has done: 'I replace the un‑weighted per‑position and overall averages with signal‑to‑noise‑weighted averages (using the filtered high‑quality training set). This keeps the same baseline structure while giving slightly more reliable estimates for the scored positions, which should lower the MCRMSE toward the target. The rest of the pipeline and file output remain unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import json
from pathlib import Path

train_path = Path("../input/stanford-covid-vaccine/train.json")
test_path = Path("../input/stanford-covid-vaccine/test.json")
df_train = pd.read_json(train_path, lines=True)
df_test = pd.read_json(test_path, lines=True)

df_train_filtered = df_train[
    (df_train["SN_filter"] == 1) & (df_train["signal_to_noise"] > 1.0)
]
if df_train_filtered.empty:
    df_train_filtered = df_train


def weighted_per_position_mean(series, weights):
    """Weighted mean at each sequence position."""
    stacked = np.stack(series.apply(np.array).values)  # (samples, seq_len)
    w = np.array(weights)[:, None]  # broadcast to (samples,1)
    weighted_sum = np.sum(stacked * w, axis=0)
    return weighted_sum / np.sum(w)


def overall_weighted_mean(series, weights):
    """Weighted overall mean across all positions and samples."""
    values = np.concatenate(series.values)
    repeats = [len(x) for x in series.values]
    w_rep = np.repeat(np.array(weights), repeats)
    return np.average(values, weights=w_rep)


def smooth(arr, window=5):
    """Simple moving‑average smoother."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


weights = df_train_filtered["signal_to_noise"]

pos_mean_reactivity = weighted_per_position_mean(
    df_train_filtered["reactivity"], weights
)
pos_mean_deg_Mg_pH10 = weighted_per_position_mean(
    df_train_filtered["deg_Mg_pH10"], weights
)
pos_mean_deg_pH10 = weighted_per_position_mean(df_train_filtered["deg_pH10"], weights)
pos_mean_deg_Mg_50C = weighted_per_position_mean(
    df_train_filtered["deg_Mg_50C"], weights
)
pos_mean_deg_50C = weighted_per_position_mean(df_train_filtered["deg_50C"], weights)

pos_mean_reactivity = smooth(pos_mean_reactivity, window=5)
pos_mean_deg_Mg_pH10 = smooth(pos_mean_deg_Mg_pH10, window=5)
pos_mean_deg_pH10 = smooth(pos_mean_deg_pH10, window=5)
pos_mean_deg_Mg_50C = smooth(pos_mean_deg_Mg_50C, window=5)
pos_mean_deg_50C = smooth(pos_mean_deg_50C, window=5)

mean_reactivity = overall_weighted_mean(
    df_train["reactivity"], df_train["signal_to_noise"]
)
mean_deg_Mg_pH10 = overall_weighted_mean(
    df_train["deg_Mg_pH10"], df_train["signal_to_noise"]
)
mean_deg_pH10 = overall_weighted_mean(df_train["deg_pH10"], df_train["signal_to_noise"])
mean_deg_Mg_50C = overall_weighted_mean(
    df_train["deg_Mg_50C"], df_train["signal_to_noise"]
)
mean_deg_50C = overall_weighted_mean(df_train["deg_50C"], df_train["signal_to_noise"])



## === cell 1
rows = []
for _, row in df_test.iterrows():
    seq_id = row["id"]
    seq_len = int(row["seq_length"])  # total length (107)
    seq_scored = int(row["seq_scored"])  # positions that are scored (usually 68)

    for pos in range(seq_len):
        if pos < seq_scored:
            reactivity_val = pos_mean_reactivity[pos]
            deg_Mg_pH10_val = pos_mean_deg_Mg_pH10[pos]
            deg_pH10_val = pos_mean_deg_pH10[pos]
            deg_Mg_50C_val = pos_mean_deg_Mg_50C[pos]
            deg_50C_val = pos_mean_deg_50C[pos]
        else:
            reactivity_val = mean_reactivity
            deg_Mg_pH10_val = mean_deg_Mg_pH10
            deg_pH10_val = mean_deg_pH10
            deg_Mg_50C_val = mean_deg_Mg_50C
            deg_50C_val = mean_deg_50C

        rows.append(
            {
                "id_seqpos": f"{seq_id}_{pos}",
                "reactivity": reactivity_val,
                "deg_Mg_pH10": deg_Mg_pH10_val,
                "deg_pH10": deg_pH10_val,
                "deg_Mg_50C": deg_Mg_50C_val,
                "deg_50C": deg_50C_val,
            }
        )

submission = pd.DataFrame(
    rows,
    columns=[
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ],
)



## === cell 2
submission_path = Path("submission.csv")
submission.to_csv(submission_path, index=False)
