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

0.43714

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'Your notebook fails because it tries to load a non-existent baseline CSV (`../input/worst-submission/ensemble52.csv`), so `df` is never created and later cells error. I remove that dependency by generating a valid submission directly from `sample_submission.csv`, which guarantees correct row count/column names and end-to-end execution. I also make the “private set seq_length==130” logic safe (this dataset uses length 107) and guard the optional manual override so it won’t crash if the id prefix doesn’t exist. This produce a valid `submission.csv`; since no current score exists, the priority is correctness and producing a runnable baseline.'
- What this solution (achieved 0.42418) has done: 'Your current score is far worse than the target (lower is better), and the main reason is that the code is not doing any learning/prediction at all—it mostly outputs the sample submission (plus a harmful manual override). To move the score toward the target with minimal core-logic change, I replace the harmful override with a simple, legitimate baseline that uses only the training data: per-position means of each target across train, applied to every test sequence position. This keeps the pipeline lightweight (pandas/numpy only), produces a valid submission with correct row order/columns, and should substantially reduce MCRMSE versus all-zeros/sample values. I also make the file paths robust to both `../input/...` and your provided `/kaggle/...` layout while keeping the same dataset.'
- What this solution (achieved 0.42166) has done: 'Your current score (0.42418, lower-is-better) is worse than the target (0.35188), so we should make a small, legitimate improvement without changing the basic “simple baseline from training statistics” approach. I keep the per-position mean baseline, but compute those means using only higher-quality training rows (`SN_filter == 1`), which usually reduces noise and improves MCRMSE with minimal code change. I also handle missing/NaN target values safely when averaging (using `nanmean`) to avoid contaminating position means. The rest of the pipeline (paths, submission row alignment, output columns) stays the same and still write a valid `submission.csv`.'
- What this solution (achieved 0.45727) has done: 'We keep your “per-position mean baseline” core logic, but reduce noise by (1) filtering the training rows more strictly using both `SN_filter==1` and a modest `signal_to_noise` threshold, and (2) using per-position weighted means where weights are inverse measurement-error (so high-uncertainty labels contribute less). This is still the same baseline family (training statistics → position means → fill submission by `seqpos`), but it usually improves MCRMSE for this competition because label noise is heteroscedastic. We also ensure NaNs/inf in errors don’t break weighting and keep the same submission format/paths. These are minimal, legitimate changes expected to move your 0.42166 score downward toward the 0.35188 target.'
- What this solution (achieved 0.45754) has done: 'We need to move your score down (lower-is-better) from 0.45727 toward 0.35188, and your current approach is still a simple per-position baseline, so the safest improvement is to keep that core logic but make the averaging more robust to outliers/noisy labels. I keep the same “training statistics → position means → fill by seqpos” pipeline, but (1) winsorize (clip) training targets per position using robust percentiles, and (2) compute a weighted mean on the clipped values using your existing inverse-error weights. This typically improves MCRMSE for this competition without changing the overall method or adding new models. All paths, submission schema, and row alignment remain unchanged, and it still write a valid `submission.csv`.'
- What this solution (achieved 0.45754) has done: 'We need to move your score down (lower-is-better) from 0.45754 toward the 0.35188 target, without changing the core “per-position training statistics baseline” approach. The smallest, high-leverage adjustment is to stop clamping all non-scored positions (68–106) to position 67, and instead learn separate means for the full `seq_length` (107) by padding train targets to length 107 and computing per-position statistics; this keeps semantics identical for scored positions while removing a systematic distribution shift for the tail. To keep the change minimal and stable, we keep your SN_filter + signal_to_noise filtering, winsorization, and inverse-error weighting exactly as-is, but extend them to 107 positions and use a safe per-position fallback when some padded positions have fewer valid samples. The submission format, paths, and output columns remain unchanged and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.45754) has done: 'We need to move the score down (lower-is-better) from 0.45754 toward 0.35188, and the safest way is to keep your exact “per-position training statistics baseline” core logic but make the training aggregation better aligned to the metric. Specifically, we compute position means using only the first `seq_scored` positions for each training row (instead of padding 68→107 with NaNs and then quantiling across a NaN-heavy tail), and we compute separate tail (68–106) means from the actual distribution of the last scored position to avoid noisy/undefined quantiles while keeping scored positions unchanged. We also adjust winsorization to be computed only on the scored region where real labels exist, then extend a stable constant to the tail; this is a minimal change that targets the main weakness (tail statistics) without changing the overall approach. Submission formatting, paths, filtering, inverse-error weighting, and writing `submission.csv` remain the same.'
- What this solution (achieved 0.45754) has done: 'Your current score (0.45754, lower-is-better) is still far from the target (0.35188), so we should make a small, metric-aligned improvement while keeping the exact same “per-position training-statistics baseline” core logic. The biggest safe win here is to better match the competition scoring: compute the per-position means using only the three scored targets (reactivity, deg_Mg_pH10, deg_Mg_50C) and then derive the two unscored targets from those (simple averages), instead of independently learning noisy means for unscored columns. This keeps the same pipeline (train aggregation → per-position values → fill by seqpos) and preserves training approach/loss/architecture (none), but typically reduces error on scored columns by reducing noise/inter-target inconsistency. All paths, filtering, winsorization, inverse-error weighting, row alignment, and submission schema remain unchanged, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.45754) has done: 'We need to move your score down (lower-is-better) from 0.45754 toward 0.35188, and the core logic must remain a simple per-position training-statistics baseline. The smallest metric-aligned change with high leverage is to stop predicting a constant tail for positions 68–106 (currently copied from position 67), and instead learn a smooth position-dependent tail by fitting a simple linear trend on the last few scored positions (per target) and extrapolating to 107; this does not change any scored-position predictions used for training aggregation, but makes the full-length submission more realistic and avoids a hard discontinuity at 68. I keep your existing SN_filter + signal_to_noise filtering, winsorization, inverse-error weighting, and the “derive unscored columns from scored columns” strategy unchanged. The submission schema, row alignment, and output file path remain identical and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.45991) has done: 'We need to reduce MCRMSE from 0.45754 toward 0.35188 (lower-is-better) while keeping your core “per-position training-statistics baseline” intact. The most impactful minimal fix is that you currently filter training rows with `signal_to_noise >= 1.0`, which matches the public test filtering but removes a lot of training signal and can hurt generalization; we keep `SN_filter==1` but drop the extra SNR threshold to use more labeled data. Next, we align the weighting with label reliability better by weighting with inverse *variance* but adding a small per-target floor based on the typical error scale (so weights don’t explode for tiny errors), which stabilizes the weighted mean. Everything else (winsorization, per-position aggregation, tail extrapolation, deriving unscored columns, submission formatting/paths) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.4503) has done: 'Your current score (0.45991; lower-is-better) is still worse than the target (0.35188), so we should make a small, safe improvement that keeps the exact same “per-position weighted mean baseline + winsorization + tail extrapolation + derived unscored columns” core logic. The highest-leverage minimal fix is to compute the per-position statistics using *all* training rows (not only `SN_filter==1`), but weight each sequence’s contribution by its `signal_to_noise` so low-quality rows contribute less rather than being hard-dropped; this usually improves generalization because it restores training diversity while still downweighting noise. Concretely, we multiply your existing per-position inverse-variance weights by a per-row SNR weight (clipped to a safe range) without changing any loops, model structure, or targets. Everything else (paths, alignment, submission schema, writing `submission.csv`) remains unchanged.'
- What this solution (achieved 0.45916) has done: 'We need to move your score down (lower-is-better) from 0.4503 toward the 0.35188 target, but keep the exact same “per-position weighted mean baseline + winsorization + tail extrapolation + derived unscored columns” core logic. The smallest likely win is to fix a mismatch between what we’re optimizing and what’s scored: your aggregation currently uses all training rows equally in position space, but the competition score only evaluates the first `seq_scored` positions and test is filtered for quality—so we (a) use `SN_filter==1` rows only (matching test distribution) and (b) compute winsorization and means only over the scored region, then extrapolate tail as you already do. This is a minimal change (just the row filter and slightly safer/cleaner quantile computation) that typically reduces MCRMSE without changing the overall approach or adding any modeling. Submission formatting/paths stay identical and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.43714) has done: 'Your current score (0.45916, lower-is-better) is worse than the target (0.35188), so we should make a small, low-risk improvement while keeping the exact same “per-position weighted mean baseline + winsorization + tail extrapolation + derived unscored columns” core logic. The most likely issue is that we’re filtering to `SN_filter==1` and then *also* applying `signal_to_noise` weighting—this can over-emphasize a subset and hurt generalization—so we keep the `SN_filter==1` match-to-test filter but remove the extra SNR weighting (set it to 1.0). Additionally, to better match the evaluation (MCRMSE) under heteroscedastic noise, we soften the inverse-variance weighting slightly by using `1/(e_eff^p)` with `p=1.0` instead of `p=2.0` (still the same weighted-mean approach, just less extreme weights). Everything else (paths, winsorization, extrapolation, deriving unscored targets, submission formatting) remains unchanged and it still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd




## === cell 1
def _resolve_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {candidates}")


train_path = _resolve_path(
    [
        "../input/stanford-covid-vaccine/train.json",
        "/kaggle/input/stanford-covid-vaccine/train.json",
        "/kaggle/data/stanford-covid-vaccine/train.json",
        "/kaggle/input/train.json",
        "/kaggle/data/train.json",
    ]
)
test_path = _resolve_path(
    [
        "../input/stanford-covid-vaccine/test.json",
        "/kaggle/input/stanford-covid-vaccine/test.json",
        "/kaggle/data/stanford-covid-vaccine/test.json",
        "/kaggle/input/test.json",
        "/kaggle/data/test.json",
    ]
)
sample_sub_path = _resolve_path(
    [
        "../input/stanford-covid-vaccine/sample_submission.csv",
        "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
        "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)

df_train = pd.read_json(train_path, lines=True)
df_test = pd.read_json(test_path, lines=True)
df = pd.read_csv(sample_sub_path)

required_cols = [
    "id_seqpos",
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"sample_submission.csv missing required columns: {missing}")

print(
    "Loaded:", {"train": df_train.shape, "test": df_test.shape, "sample_sub": df.shape}
)



## === cell 2
scored_targets = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
all_targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

err_cols = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

df_train_use = df_train.copy()
if "SN_filter" in df_train_use.columns:
    df_train_use = df_train_use.loc[
        pd.to_numeric(df_train_use["SN_filter"], errors="coerce").fillna(0).astype(int)
        == 1
    ].reset_index(drop=True)

SEQ_LEN = (
    int(df_train_use["seq_length"].iloc[0])
    if "seq_length" in df_train_use.columns and len(df_train_use) > 0
    else 107
)
SEQ_LEN = 107 if SEQ_LEN <= 0 else SEQ_LEN  # safety

SCORED_LEN = (
    int(df_train_use["seq_scored"].iloc[0])
    if "seq_scored" in df_train_use.columns and len(df_train_use) > 0
    else 68
)
SCORED_LEN = 68 if SCORED_LEN <= 0 else min(SCORED_LEN, SEQ_LEN)


def _pad_to_len(list_of_arrays, L: int) -> np.ndarray:
    out = np.full((len(list_of_arrays), L), np.nan, dtype=np.float32)
    for i, a in enumerate(list_of_arrays):
        a = np.asarray(a, dtype=np.float32)
        n = min(len(a), L)
        out[i, :n] = a[:n]
    return out


def _weighted_pos_mean(y_2d: np.ndarray, w_2d: np.ndarray) -> np.ndarray:
    y = y_2d.astype(np.float32, copy=False)
    w = w_2d.astype(np.float32, copy=False)

    valid = np.isfinite(y) & np.isfinite(w) & (w > 0)
    y = np.where(valid, y, 0.0).astype(np.float32, copy=False)
    w = np.where(valid, w, 0.0).astype(np.float32, copy=False)

    wsum = np.sum(w, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(wsum > 0, np.sum(w * y, axis=0) / wsum, np.nan).astype(
            np.float32
        )
    return out


def _winsorize_per_position(y_2d: np.ndarray, low_q=0.02, high_q=0.98) -> np.ndarray:
    y = y_2d.astype(np.float32, copy=False)
    lo = np.nanquantile(y, low_q, axis=0).astype(np.float32)
    hi = np.nanquantile(y, high_q, axis=0).astype(np.float32)
    lo = np.where(np.isfinite(lo), lo, -np.inf).astype(np.float32)
    hi = np.where(np.isfinite(hi), hi, np.inf).astype(np.float32)
    return np.clip(y, lo[None, :], hi[None, :]).astype(np.float32, copy=False)


def _extrapolate_tail_linear(
    m_scored: np.ndarray, full_len: int, scored_len: int, k: int = 8
) -> np.ndarray:
    m_scored = np.asarray(m_scored, dtype=np.float32)
    if full_len <= scored_len:
        return m_scored[:full_len].astype(np.float32, copy=False)

    k = int(max(2, min(k, scored_len)))
    x = np.arange(scored_len - k, scored_len, dtype=np.float32)
    y = m_scored[scored_len - k : scored_len].astype(np.float32, copy=False)

    mask = np.isfinite(y)
    if mask.sum() < 2:
        tail_val = np.float32(m_scored[scored_len - 1])
        return np.concatenate(
            [m_scored, np.full((full_len - scored_len,), tail_val, dtype=np.float32)],
            axis=0,
        )

    x = x[mask]
    y = y[mask]

    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if not np.isfinite(denom) or denom <= 0:
        slope = np.float32(0.0)
    else:
        slope = np.float32(np.sum((x - x_mean) * (y - y_mean)) / denom)
    intercept = np.float32(y_mean - slope * x_mean)

    x_tail = np.arange(scored_len, full_len, dtype=np.float32)
    y_tail = intercept + slope * x_tail

    return np.concatenate([m_scored, y_tail.astype(np.float32)], axis=0)


n_rows = len(df_train_use)

snr_w_2d = np.ones((n_rows, 1), dtype=np.float32)

ERR_POWER = np.float32(1.0)

pos_means = {}

for t in scored_targets:
    y_scored = _pad_to_len(df_train_use[t].values, SCORED_LEN)  # (n, SCORED_LEN)
    y_scored_clip = _winsorize_per_position(y_scored, low_q=0.02, high_q=0.98)

    ec = err_cols.get(t)
    if ec in df_train_use.columns:
        e_scored = _pad_to_len(df_train_use[ec].values, SCORED_LEN)

        e_valid = e_scored[np.isfinite(e_scored) & (e_scored > 0)]
        if e_valid.size > 0:
            e_floor = np.float32(np.nanmedian(e_valid) * 0.5)
            if not np.isfinite(e_floor) or e_floor <= 0:
                e_floor = np.float32(1e-3)
        else:
            e_floor = np.float32(1e-3)

        e_scored = np.where(
            np.isfinite(e_scored) & (e_scored > 0), e_scored, np.nan
        ).astype(np.float32, copy=False)
        e_eff = np.maximum(e_scored, e_floor).astype(np.float32, copy=False)

        w_scored = (1.0 / np.power(e_eff, ERR_POWER)) * snr_w_2d
        m_scored = _weighted_pos_mean(y_scored_clip, w_scored)

        fallback_pos = np.nanmean(y_scored_clip, axis=0).astype(np.float32)
        global_fallback = np.nanmean(fallback_pos).astype(np.float32)
        m_scored = np.where(np.isfinite(m_scored), m_scored, fallback_pos).astype(
            np.float32
        )
        m_scored = np.where(np.isfinite(m_scored), m_scored, global_fallback).astype(
            np.float32
        )
    else:
        w_scored = (
            np.where(np.isfinite(y_scored_clip), 1.0, 0.0).astype(np.float32) * snr_w_2d
        )
        m_scored = _weighted_pos_mean(y_scored_clip, w_scored)
        fallback_pos = np.nanmean(y_scored_clip, axis=0).astype(np.float32)
        global_fallback = np.nanmean(fallback_pos).astype(np.float32)
        m_scored = np.where(np.isfinite(m_scored), m_scored, fallback_pos).astype(
            np.float32
        )
        m_scored = np.where(np.isfinite(m_scored), m_scored, global_fallback).astype(
            np.float32
        )

    m_full = _extrapolate_tail_linear(
        m_scored, full_len=SEQ_LEN, scored_len=SCORED_LEN, k=8
    )
    pos_means[t] = m_full

pos_means["deg_pH10"] = (
    0.5 * (pos_means["deg_Mg_pH10"] + pos_means["reactivity"])
).astype(np.float32)
pos_means["deg_50C"] = (
    0.5 * (pos_means["deg_Mg_50C"] + pos_means["reactivity"])
).astype(np.float32)

for t in all_targets:
    if pos_means[t].shape[0] != SEQ_LEN:
        raise ValueError(f"Unexpected mean shape for {t}: {pos_means[t].shape}")

print(
    "Using train rows for mean baseline:",
    len(df_train_use),
    "of",
    len(df_train),
    "| SEQ_LEN:",
    SEQ_LEN,
    "| SCORED_LEN:",
    SCORED_LEN,
    "| ERR_POWER:",
    float(ERR_POWER),
)



## === cell 3
id_seqpos = df["id_seqpos"].astype(str).values
seqpos = np.array([int(s.rsplit("_", 1)[-1]) for s in id_seqpos], dtype=np.int32)

seqpos_clamped = np.clip(seqpos, 0, SEQ_LEN - 1)

for t in all_targets:
    df[t] = pos_means[t][seqpos_clamped].astype(np.float32)



## === cell 4
for c in required_cols[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
df[required_cols[1:]] = (
    df[required_cols[1:]].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)
df = df[required_cols]



## === cell 5
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print("Head:\n", df.head())
