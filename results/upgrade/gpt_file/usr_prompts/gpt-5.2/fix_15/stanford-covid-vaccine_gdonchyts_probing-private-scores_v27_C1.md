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

0.3684779204132506

# 6. Current score

0.44981

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I remove the dependency on the missing `../input/worst-submission/ensemble52.csv` by instead starting from the competition’s provided `sample_submission.csv`, which guarantees correct rows/columns. To keep your intended “score calibration” behavior, I preserve the same scaling step (`/= 1.1555`) but apply it to the sample submission’s numeric target columns. I also make file-path resolution robust by checking the known available locations under `/kaggle/...` so it runs in this environment without path errors. Finally, the script always write a valid `submission.csv` with the required 6 columns.'
- What this solution (achieved 0.43107) has done: 'Your current script submits a scaled version of the sample submission, which is essentially a constant baseline and is far from the target MCRMSE. To move the score toward your target with minimal core-logic change, I keep your “calibration divide” behavior but replace the all-zero baseline with a simple, legitimate per-position prior learned from `train.json` (mean target at each seq position). This stays within the same “no model training loop” spirit, uses only provided data, and typically yields a large RMSE improvement vs zeros. I also ensure predictions are generated for all 107 positions by filling unscored tail positions (68–106) with the last available mean, so the submission remains valid.'
- What this solution (achieved 0.44283) has done: 'To move your MCRMSE down toward the 0.3685 target (lower is better) with minimal logic change, I keep your same “per-position prior from train” approach but add two small, metric-aligned fixes: (1) compute the per-position means using only `SN_filter==1` rows (matching the competition’s high-quality subset, usually improves generalization) and (2) apply your existing `/= 1.1555` calibration only to the two unscored columns so it doesn’t accidentally hurt the three scored targets. I also add a tiny smoothing across neighboring positions (a 3-point moving average) to reduce noise in the per-position prior without changing the overall approach. The rest of the pipeline (paths, shape handling for 107 positions, submission format) is kept the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.47008) has done: 'You’re currently above the target (0.44283 vs 0.36848; lower is better), so we should make a small, legitimate change that is likely to improve generalization without changing the “per-position prior” core logic. I keep the exact pipeline (position-wise means from train, 3-pt smoothing, fill tail, same submission construction), but compute the per-position means with an error-weighted average using the provided `*_error_*` columns—this better matches measurement reliability and typically reduces RMSE. I also apply the same smoothing to the per-position error estimates before weighting to avoid unstable weights, and I keep your existing decision to only apply `/= 1.1555` to the two unscored columns. The script still run end-to-end and write a valid `submission.csv` with the required 6 columns.'
- What this solution (achieved 0.46471) has done: 'We should move the score down (lower is better) from 0.47008 toward the 0.36848 target, but with minimal changes that keep your “per-position prior” core logic intact. The biggest low-risk issue in the current approach is that error-weighted means can overweight very small reported errors, making the prior noisier and less generalizable; we cap (floor) the per-position errors before forming weights to prevent extreme weights. We also add a tiny global shrinkage toward each target’s overall mean (a standard variance-reduction step for priors) to improve generalization without changing the modeling approach. Everything else (SN_filter==1, 3-pt smoothing, tail fill to 107, and submission format) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.4616) has done: 'Your current score (0.46471, lower-is-better) is still worse than the target (0.36848), so we should make a small, legitimate change that’s likely to improve generalization while keeping the same “per-position prior from train” core logic. The most impactful low-risk tweak here is to compute the per-position prior using only the scored positions (0–67) and then fill the unscored tail (68–106) with the last scored-position value; this prevents the unscored positions from being influenced by a prior computed on a different effective length and avoids any accidental distribution shift in the tail. Additionally, we keep your error-weighting, smoothing, and shrinkage unchanged, but we make the weighting numerically safer by clipping extreme weights via a slightly stronger per-position error floor (still derived from the data), which should reduce overfitting to tiny error values. The submission format, column order, and the existing “/1.1555 only on unscored columns” behavior are preserved.'
- What this solution (achieved 0.45647) has done: 'We should move the score down (lower is better) from 0.4616 toward the 0.3685 target, while keeping the same “per-position prior from train + smoothing + error-weighted mean + shrinkage + tail fill” core logic. The smallest likely gain is to tune the two regularization knobs you already have: (1) slightly stronger shrinkage toward the global mean to reduce variance/overfitting in the position prior, and (2) a slightly higher error floor to prevent a few tiny reported errors from dominating the weighted average. These are metric-aligned (reduce RMSE) and don’t change the approach or training semantics. Everything else (paths, SN_filter usage, 3-pt smoothing, and submission schema) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.4524) has done: 'You’re currently worse than the target (0.45647 vs 0.36848; lower is better), so we should make a very small variance-reduction tweak that tends to improve MCRMSE without changing the “per-position prior + smoothing + error-weighted mean + shrinkage + tail fill” core logic. I (1) slightly increase the shrinkage toward the global mean to reduce overfitting/noise in the position prior, and (2) slightly strengthen the error floor so a few tiny error values don’t dominate the weighted mean. These are the same knobs you already use, just nudged a bit toward stability/generalization. Everything else (data paths, SN_filter usage, smoothing, weighting formula, submission format, and writing `submission.csv`) remains unchanged.'
- What this solution (achieved 0.45021) has done: 'To move your MCRMSE down (lower is better) from 0.4524 toward the 0.3685 target without changing the core “per-position prior + error-weighted mean + smoothing + shrinkage + tail fill” approach, I only tune the two variance-control knobs you already have: increase the global-mean shrinkage slightly and raise the per-position error floor slightly to further prevent tiny-error domination. I also make the weighting numerically safer by clipping extremely large weights (equivalently bounding minimum effective error), which is a minimal stability tweak within the same weighted-mean logic. All paths, feature usage, smoothing, calibration on the two unscored columns, and the submission format/writing stay the same.'
- What this solution (achieved 0.4501) has done: 'We need to reduce your MCRMSE (lower-is-better) from 0.45021 toward the 0.36848 target, but with minimal changes that preserve the same “per-position prior + error-weighted mean + smoothing + shrinkage + tail fill” core logic. The most likely low-risk improvement is to cap *both* sides of the weighting: you already cap maximum weight (minimum effective error), but you don’t cap minimum weight (maximum effective error), so very noisy measurements still contribute and can degrade the position prior. I add a per-position upper error cap based on a robust statistic (smoothed 90th percentile), so extremely noisy samples are downweighted consistently without changing the approach. Everything else (SN_filter usage, smoothing, shrinkage value, tail fill, and submission format/writing) stays the same.'
- What this solution (achieved 0.4501) has done: 'Your current gap to target is still large (0.4501 vs 0.3685, lower is better), so we should make a small, legitimate generalization improvement without changing the overall “per-position prior from train + smoothing + error-weighted mean + shrinkage + tail fill” approach. The most likely low-risk win is to use only the highest-quality training rows (not just `SN_filter==1`, but also a reasonable `signal_to_noise` cutoff), because very noisy rows can skew the positional prior and hurt MCRMSE on the clean test set. I also make the weighting more robust by treating non-finite target values as missing before computing means/weighted means (the dataset can contain NaNs/inf-like artifacts), which prevents a few bad values from contaminating a position. Everything else (smoothing, shrinkage value, error floor/cap logic, calibration only on unscored columns, submission format) is preserved and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.44972) has done: 'We need to lower MCRMSE from 0.4501 toward the 0.3685 target (lower is better), so I make the smallest generalization-oriented change that stays within your existing “per-position prior + error-weighted mean + smoothing + shrinkage + tail fill” logic. The main adjustment is to slightly tighten the training-row quality filter by using a higher `signal_to_noise` cutoff (test is high-quality, so this usually improves priors without changing the approach). I also make the weight computation robust to remaining noisy rows by dropping rows whose per-sample mean error is extreme (winsorized via a simple percentile cutoff) before computing position means; this is still the same weighted-mean prior, just with obvious outliers removed. Everything else—targets, smoothing, weighting formula, shrinkage, tail fill to 107, calibration of only unscored columns, and submission writing—stays the same.'
- What this solution (achieved 0.45005) has done: 'I keep your current “per-position prior from train + error-weighted mean + 3pt smoothing + shrinkage + tail fill” core logic, but make two small, score-aligned tweaks aimed at better matching the (clean) test distribution. First, I tune the training-row quality filter slightly more toward high-SNR examples (a minimal generalization nudge, not a modeling change). Second, I slightly increase the shrinkage toward the global mean (variance reduction) while keeping all weighting, smoothing, and submission formatting identical—this is the smallest knob that typically improves MCRMSE for this kind of positional prior. The script still run end-to-end and write a valid `submission.csv` with the required 6 columns.'
- What this solution (achieved 0.44981) has done: 'To move MCRMSE down toward your 0.3685 target (lower is better) while keeping the exact same “per-position prior from train + error-weighted mean + 3pt smoothing + shrinkage + tail fill + unscored-column calibration” core logic, I’m only adjusting two generalization knobs that directly affect the variance/bias tradeoff of that prior. First, I slightly relax the `signal_to_noise` cutoff to keep more (still good) training rows, which typically stabilizes the per-position mean and reduces overfitting to a too-small high-SNR subset. Second, I slightly reduce the global-mean shrinkage (from 0.30 to 0.26) because the current stronger shrink can oversmooth and add bias on scored targets; this change is minimal and keeps the same shrinkage mechanism. Everything else (weighting formula, smoothing, error floor/cap logic, tail fill, submission schema and writing `submission.csv`) is preserved.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
CANDIDATE_SAMPLE_PATHS = [
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]

sample_path = None
for p in CANDIDATE_SAMPLE_PATHS:
    if os.path.exists(p):
        sample_path = p
        break

if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in any expected location. "
        f"Tried: {CANDIDATE_SAMPLE_PATHS}"
    )

sub = pd.read_csv(sample_path)

target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
missing = [c for c in target_cols + ["id_seqpos"] if c not in sub.columns]
if missing:
    raise ValueError(f"sample_submission is missing required columns: {missing}")

sub = sub[["id_seqpos"] + target_cols].copy()
sub.head(), sub.shape, sub.columns.tolist()



## === cell 2
CANDIDATE_TRAIN_PATHS = [
    "/kaggle/input/stanford-covid-vaccine/train.json",
    "/kaggle/input/train.json",
    "/kaggle/data/stanford-covid-vaccine/train.json",
    "/kaggle/data/train.json",
]
train_path = None
for p in CANDIDATE_TRAIN_PATHS:
    if os.path.exists(p):
        train_path = p
        break

if train_path is None:
    raise FileNotFoundError(
        "Could not find train.json in any expected location. "
        f"Tried: {CANDIDATE_TRAIN_PATHS}"
    )

train = pd.read_json(train_path, lines=True)

seq_scored = int(train["seq_scored"].mode().iloc[0])
seq_length = int(train["seq_length"].mode().iloc[0])

if "SN_filter" in train.columns:
    train = train.loc[train["SN_filter"].astype(int) == 1].reset_index(drop=True)

if "signal_to_noise" in train.columns:
    snr = pd.to_numeric(train["signal_to_noise"], errors="coerce")
    train_snr = train.loc[snr >= 1.60].reset_index(drop=True)  # was 1.75
    if len(train_snr) > 0:
        train = train_snr


def smooth_3pt(x: np.ndarray) -> np.ndarray:
    """
    Keep: 3-point moving average smoothing across sequence positions
    to reduce noise in the per-position prior without changing the core 'position mean' logic.
    """
    x = x.astype(np.float32, copy=False)
    if x.size < 3:
        return x
    y = x.copy()
    y[1:-1] = (x[:-2] + x[1:-1] + x[2:]) / 3.0
    y[0] = (x[0] + x[1]) / 2.0
    y[-1] = (x[-2] + x[-1]) / 2.0
    return y


ERROR_MAP = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}


def weighted_pos_mean(values_2d: np.ndarray, errors_2d: np.ndarray) -> np.ndarray:
    """
    Compute per-position weighted mean with weights=1/(err^2 + eps).
    Falls back to unweighted mean if errors are unusable.
    """
    v = values_2d.astype(np.float32, copy=False)
    e = errors_2d.astype(np.float32, copy=False)

    v = np.where(np.isfinite(v), v, np.nan).astype(np.float32, copy=False)
    e = np.where(np.isfinite(e) & (e > 0), e, np.nan).astype(np.float32, copy=False)

    eps = np.float32(1e-6)
    w = 1.0 / (e * e + eps)

    w = np.minimum(w, np.float32(1.0 / (0.06 * 0.06))).astype(np.float32, copy=False)

    num = np.nansum(w * v, axis=0)
    den = np.nansum(w, axis=0)
    out = num / den
    out = out.astype(np.float32, copy=False)
    return out


def filter_noisy_rows_by_error(train_df: pd.DataFrame, seq_scored: int) -> pd.DataFrame:
    err_cols = [c for c in ERROR_MAP.values() if c in train_df.columns]
    if not err_cols:
        return train_df
    per_row_means = []
    for c in err_cols:
        e = np.vstack(train_df[c].values).astype(np.float32)[:, :seq_scored]
        e = np.where(np.isfinite(e) & (e > 0), e, np.nan).astype(np.float32, copy=False)
        per_row_means.append(np.nanmean(e, axis=1))
    row_err = np.nanmean(np.vstack(per_row_means), axis=0).astype(np.float32)
    if not np.isfinite(row_err).any():
        return train_df
    thr = np.nanquantile(row_err, 0.95)
    keep = row_err <= thr
    if keep.mean() < 0.7:
        return train_df
    return train_df.loc[keep].reset_index(drop=True)


train = filter_noisy_rows_by_error(train, seq_scored)

pos_means = {}
for c in target_cols:
    arr = np.vstack(train[c].values).astype(np.float32)[
        :, :seq_scored
    ]  # (n, seq_scored)
    arr = np.where(np.isfinite(arr), arr, np.nan).astype(np.float32, copy=False)

    err_col = ERROR_MAP.get(c)
    if err_col is not None and err_col in train.columns:
        err = np.vstack(train[err_col].values).astype(np.float32)[:, :seq_scored]
        err = np.where(np.isfinite(err) & (err > 0), err, np.nan).astype(
            np.float32, copy=False
        )

        med_err = np.nanmedian(err, axis=0).astype(np.float32)
        med_err = smooth_3pt(med_err)

        q10 = np.nanquantile(err, 0.10, axis=0).astype(np.float32)
        q10 = smooth_3pt(q10)

        q90 = np.nanquantile(err, 0.90, axis=0).astype(np.float32)
        q90 = smooth_3pt(q90)

        err_cap = np.maximum(np.float32(0.20), np.float32(1.25) * q90).astype(
            np.float32
        )
        err_floor = np.maximum(np.float32(0.07), np.float32(1.00) * q10).astype(
            np.float32
        )

        err = 0.7 * err + 0.3 * med_err[None, :]
        err = np.maximum(err, err_floor[None, :]).astype(np.float32)
        err = np.minimum(err, err_cap[None, :]).astype(np.float32)

        m = weighted_pos_mean(arr, err)
        m_fallback = np.nanmean(arr, axis=0).astype(np.float32)
        m = np.where(np.isfinite(m), m, m_fallback).astype(np.float32)
    else:
        m = np.nanmean(arr, axis=0).astype(np.float32)

    m = smooth_3pt(m)

    overall = np.nanmean(arr).astype(np.float32)

    shrink = np.float32(0.26)  # was 0.30
    m = (1.0 - shrink) * m + shrink * overall

    if seq_length > seq_scored:
        tail = np.full((seq_length - seq_scored,), m[-1], dtype=np.float32)
        m_full = np.concatenate([m, tail], axis=0)
    else:
        m_full = m[:seq_length]
    pos_means[c] = m_full

assert len(next(iter(pos_means.values()))) == seq_length
seq_scored, seq_length, {
    k: (float(np.nanmin(v)), float(np.nanmax(v))) for k, v in pos_means.items()
}



## === cell 3
id_seqpos = sub["id_seqpos"].astype(str)
seqpos = id_seqpos.str.split("_").str[-1].astype(int).values

for c in target_cols:
    sub[c] = pos_means[c][seqpos]

sub[target_cols] = sub[target_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

unscored_cols = ["deg_pH10", "deg_50C"]
sub[unscored_cols] /= 1.1555

sub[target_cols] = sub[target_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
sub.head()



## === cell 4
sub = sub[["id_seqpos"] + target_cols]
assert sub.shape[1] == 6
assert sub.columns.tolist() == [
    "id_seqpos",
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]
assert sub.isna().sum().sum() == 0

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv", sub.shape)
print(sub.head())
