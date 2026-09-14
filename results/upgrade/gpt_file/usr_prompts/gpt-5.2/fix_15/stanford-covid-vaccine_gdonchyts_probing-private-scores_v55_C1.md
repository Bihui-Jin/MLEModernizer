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

0.41905

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I remove the dependency on the missing `../input/worst-submission/ensemble52.csv` and instead build a submission directly from the provided `sample_submission.csv`, which guarantees the correct row count and column names. I also fix the logic bug where you tried to select `seq_length == 130` (this dataset uses 107) and eliminate downstream `NameError`s caused by `df` never being created. Since no valid submission was previously generated, this focuses on correctness/stability rather than score tuning; it produce a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.63824) has done: 'Your current code intentionally injects a huge incorrect value (`reactivity = 100`) for one specific test id prefix, which dramatically worsen MCRMSE (lower is better) and explains the poor 0.63824 score. I remove that sabotage line while keeping the rest of your submission-building logic identical (still based on `sample_submission.csv` for correct formatting/row count). I also add a small safety check to ensure all predictions are finite numeric values before writing, without changing your approach or introducing any new modeling. This minimal change should move your score substantially down toward the target 0.3519.'
- What this solution (achieved 0.42166) has done: 'Your current pipeline always submits the all-zeros baseline copied from `sample_submission.csv`, which is why the score is stuck around ~0.64 (this competition’s baseline). To move the score down toward the target (lower is better), the smallest legitimate improvement that preserves your “no ML model” core logic is to replace the zeros with a simple per-position prior computed from the training set targets (mean target value at each `seqpos`). This keeps the exact same submission-building flow and format checks, but uses information available in `train.json` to produce more realistic predictions for each base position. I also fill positions 68–106 (not scored but required) with the last scored position’s mean to keep output well-defined and stable.'
- What this solution (achieved 0.3912) has done: 'Your current approach (per-position mean priors) is stable but leaves score on the table because it ignores sample-specific signals already present in test.json (sequence/structure/loop type). To move your MCRMSE down toward the target with minimal logic change, I keep the same “predict by seqpos priors” core, but add small additive adjustments learned from train.json based on (a) nucleotide identity at each position and (b) loop-type at each position. This remains a simple closed-form baseline (no model training loops), is fully deterministic, and still fills unscored positions 68–106 with the last scored position’s estimate. The submission format/row order remains anchored to sample_submission.csv to guarantee a valid file.'
- What this solution (achieved 0.3891) has done: 'Your current score (0.3912) is worse than the target (0.3519) for a lower-is-better metric, so we should make a small, low-risk improvement. Keeping your same “positional mean + additive base and loop residual effects” core logic, I add a tiny amount of regularization (shrinkage) to the base/loop residuals to reduce noise/overfitting, which typically improves generalization on this dataset. I also add a lightweight clipping of predictions to the training target quantile range per column to avoid rare extreme values that can hurt RMSE, without changing the overall approach. Submission format and row ordering remain anchored to `sample_submission.csv`, and runtime stays well under the limit.'
- What this solution (achieved 0.41711) has done: 'Your current score (0.3891) is worse than the target (0.3519) for a lower-is-better metric, so we should make a small generalization-focused improvement without changing the core “positional mean + additive base/loop residual effects” approach. The simplest low-risk gain here is to add a third residual correction for the `structure` character at each position (paired/unpaired context), learned from train.json the same way as base/loop, with the same shrinkage to avoid overfitting. This leverages an already-available per-position signal in both train and test and typically improves MCRMSE while preserving identical evaluation semantics and keeping runtime small. Everything else (format anchored to sample_submission, filling unscored positions with last scored mean, clipping) remains the same.'
- What this solution (achieved 0.41629) has done: 'Your current score (0.41711) is worse than the target (0.35188) for a lower-is-better metric, so we should make a small, low-risk generalization improvement while keeping the same “positional mean + additive base/loop/structure residual effects” core logic. The biggest issue is likely over/under-shrinkage from using the same K for all targets and all effect types; we can tune shrinkage per target (still the same closed-form residual approach) to reduce error without changing evaluation semantics. I also switch clipping from global quantiles to per-seqpos quantiles (computed from train) for scored positions, which is still a simple deterministic post-processing step but better matches position-dependent target distributions and typically reduces RMSE. Everything else (format anchored to sample_submission.csv, filling positions 68–106 with last scored estimates, deterministic processing) stays the same.'
- What this solution (achieved 0.41876) has done: 'Your current residual-additive baseline is being held back mainly by miscalibrated effect shrinkage and overly aggressive per-position clipping, which can bias predictions and inflate RMSE. I keep the exact same core logic (positional mean + additive base/loop/structure residual effects) but (1) tune shrinkage down slightly (less shrinkage) for the scored targets only, and (2) soften clipping by moving from 0.1/99.9% to 0.05/99.95% per-seqpos quantiles (still deterministic, still learned only from train). I also fix one small post-processing issue: the final global clip currently forces *all* positions (including scored ones) into the seqpos-67 range, which can unnecessarily distort scored predictions; instead we only apply that clip to the unscored tail (>=68) while leaving scored positions clipped by their own seqpos ranges. These are minimal changes that should reduce error (lower-is-better) and move your 0.41629 closer toward the 0.35188 target.'
- What this solution (achieved 0.41059) has done: 'We’re currently worse than the target (0.41876 vs 0.35188, lower-is-better), so we should make a small, low-risk generalization tweak without changing your core “positional mean + additive base/loop/structure residual effects” approach. The biggest likely issue is that you’re applying residuals learned on raw targets even though the competition is noisy; a tiny amount of additional shrinkage applied uniformly to the *sum* of residuals (not changing what residuals are, just scaling them) often improves RMSE on this dataset. I keep your per-target/per-effect shrinkage exactly as-is, but introduce a single residual scaling factor `RESID_SCALE < 1` applied after adding base+loop+struct residuals (scored positions only), and leave clipping logic unchanged. This is minimal, deterministic, and should move MCRMSE down toward the target without altering submission format or runtime.'
- What this solution (achieved 0.41077) has done: 'We’re currently above the target (0.41059 vs 0.35188, lower-is-better), so we should make a small, low-risk generalization improvement while keeping your exact baseline form: positional mean plus additive (base/loop/structure) residuals with shrinkage and clipping. The main tweak is to calibrate the residual scaling per target (instead of one shared `RESID_SCALE`), because the three scored targets often benefit from different residual strengths; this is a minimal change that can reduce RMSE without changing the approach. I also make the residual scaling apply only to the scored targets (leave the two unscored columns conservative) to avoid destabilizing columns that don’t affect the metric. Everything else (data reading, SN_filter use, long-format construction, effects, clipping, submission format/row order) stays the same.'
- What this solution (achieved 0.40802) has done: 'Your current baseline is already deterministic and stable but is underperforming because it treats the three residual sources (base/loop/structure) as equally reliable everywhere. I keep the exact same “positional mean + additive residuals + shrinkage + clipping” core, but add a tiny, closed-form reliability weighting that down-weights residuals when a (seqpos, category) has low support in train (using the already-computed counts). This reduces overconfident residual corrections (a common RMSE driver) without changing the approach or adding any training loops. I also restrict residual application to canonical categories only (A/C/G/U for base, known loop/struct symbols), treating unknowns as zero-effect, which avoids noisy reindex artifacts. Submission format, row order, clipping logic, and file output remain unchanged.'
- What this solution (achieved 0.42134) has done: 'Your score (0.40802) is still worse than the target (0.35188) for a lower-is-better metric, so we should make a small, low-risk improvement without changing your core “positional mean + additive base/loop/structure residuals + shrinkage + clipping” approach. The main issue is that the current residuals are learned and applied on raw targets, even though the competition labels have per-position measurement uncertainty; a standard, minimal upgrade is to compute residual means with inverse-variance weights derived from the provided `*_error_*` columns (still closed-form, no training loops). This typically improves MCRMSE by down-weighting noisy labels while keeping the exact same prediction form and test-time features. I keep your existing shrinkage/reliability weighting and clipping logic intact, only swapping the residual aggregation to weighted means and leaving everything else (format anchored to sample_submission, tail fill, determinism) unchanged.'
- What this solution (achieved 0.42057) has done: 'Your current score (0.42134) is worse than the target (0.35188) for a lower-is-better metric, so we should make a small improvement without changing the core “positional mean + additive base/loop/structure residuals + shrinkage/reliability + clipping” approach. The biggest low-risk fix is that your per-position priors and residual effects are computed using all five targets equally, even though only three targets are scored; we can legitimately tune only the scored targets to reduce MCRMSE while leaving the unscored ones conservative and stable. Concretely, we (1) apply slightly less shrinkage (smaller K) and slightly stronger residual scaling for the three scored targets only, and (2) slightly relax clipping for scored targets (wider quantiles) while keeping unscored targets’ clipping unchanged. This keeps identical data sources, no training loops/models, preserves your semantics, and should move the score downward toward the target.'
- What this solution (achieved 0.41905) has done: 'To move your MCRMSE down toward the target (lower-is-better) while keeping your exact “positional mean + additive base/loop/structure residuals + shrinkage/reliability + clipping” core, I’m making two minimal, low-risk changes. First, I compute the per-position priors and residual means using a slightly more conservative inverse-variance weighting (a higher `EPS` and lower `W_MAX`) to reduce over-trusting very small reported errors that can overfit and hurt generalization. Second, I slightly re-balance residual strength for the three scored targets only (tiny RESID_SCALE nudges), leaving unscored targets unchanged, which should improve scored columns without destabilizing the rest. Everything else (data sources, features, additive residual form, clipping logic, submission format/row order, runtime) remains the same.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os



## === cell 1
BASE = "/kaggle/input/stanford-covid-vaccine"
train_path = os.path.join(BASE, "train.json")
test_path = os.path.join(BASE, "test.json")
sample_sub_path = os.path.join(BASE, "sample_submission.csv")

df_train = pd.read_json(train_path, lines=True)
df_test = pd.read_json(test_path, lines=True)
sample_sub = pd.read_csv(sample_sub_path)

print(
    "train rows:",
    df_train.shape,
    "test rows:",
    df_test.shape,
    "sample_sub rows:",
    sample_sub.shape,
)
print("sample_sub columns:", sample_sub.columns.tolist())



## === cell 2
seq_lengths = df_test["seq_length"].value_counts().sort_index()
print("test seq_length distribution:\n", seq_lengths)

sequences = sorted(df_test.loc[df_test["seq_length"] == 107, "id"].unique().tolist())
print("num sequences with length 107:", len(sequences))
print("last 10 ids:", sequences[-10:])



## === cell 3
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
scored_target_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]  # scored by Kaggle

err_col_by_target = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

if "SN_filter" in df_train.columns:
    df_train_use = df_train[df_train["SN_filter"] == 1].copy()
    if len(df_train_use) == 0:
        df_train_use = df_train.copy()
else:
    df_train_use = df_train.copy()

records = []
for _, row in df_train_use.iterrows():
    seq_scored = int(row["seq_scored"])
    L = min(seq_scored, 68)
    seq = str(row["sequence"])
    loop = str(row["predicted_loop_type"])
    struct = str(row["structure"])
    for pos in range(L):
        rec = {"seqpos": pos}
        rec["base"] = seq[pos] if pos < len(seq) else np.nan
        rec["loop_type"] = loop[pos] if pos < len(loop) else np.nan
        rec["struct_char"] = struct[pos] if pos < len(struct) else np.nan
        for c in target_cols:
            v = row[c][pos] if pos < len(row[c]) else np.nan
            rec[c] = v
            ec = err_col_by_target[c]
            ev = row[ec][pos] if (ec in row and pos < len(row[ec])) else np.nan
            rec[f"{c}__err"] = ev
        records.append(rec)

train_long = pd.DataFrame.from_records(records)

for c in target_cols:
    train_long[c] = pd.to_numeric(train_long[c], errors="coerce")
    train_long[f"{c}__err"] = pd.to_numeric(train_long[f"{c}__err"], errors="coerce")

EPS = 3e-6
W_MAX = 3e3

for c in target_cols:
    e = train_long[f"{c}__err"].to_numpy(dtype=np.float32)
    w = 1.0 / (e * e + EPS)
    w = np.clip(w, 0.0, W_MAX).astype(np.float32)
    w[~np.isfinite(w)] = 0.0
    train_long[f"{c}__w"] = w


def weighted_mean_by_group(df_in: pd.DataFrame, group_cols, value_cols, weight_cols):
    g = df_in.groupby(group_cols, sort=False)
    out = {}
    for v, w in zip(value_cols, weight_cols):
        num = g.apply(
            lambda x: np.nansum(
                (x[v].to_numpy(dtype=np.float32)) * (x[w].to_numpy(dtype=np.float32))
            )
        )
        den = g.apply(lambda x: np.nansum(x[w].to_numpy(dtype=np.float32)))
        m = (num / den).replace([np.inf, -np.inf], np.nan)
        out[v] = m
    return pd.DataFrame(out)


pos_means = weighted_mean_by_group(
    train_long,
    ["seqpos"],
    target_cols,
    [f"{c}__w" for c in target_cols],
)

global_means = pd.Series(
    {
        c: np.nansum(
            train_long[c].to_numpy(dtype=np.float32)
            * train_long[f"{c}__w"].to_numpy(dtype=np.float32)
        )
        / max(np.nansum(train_long[f"{c}__w"].to_numpy(dtype=np.float32)), 1e-12)
        for c in target_cols
    }
)

for pos in range(68):
    if pos not in pos_means.index:
        pos_means.loc[pos] = global_means
pos_means = pos_means.sort_index()

resid = train_long.copy()
for c in target_cols:
    resid[c] = resid[c] - resid["seqpos"].map(pos_means[c])

base_mean_resid = weighted_mean_by_group(
    resid,
    ["seqpos", "base"],
    target_cols,
    [f"{c}__w" for c in target_cols],
)
base_cnt = resid.groupby(["seqpos", "base"]).size().rename("n").astype(np.float32)

loop_mean_resid = weighted_mean_by_group(
    resid,
    ["seqpos", "loop_type"],
    target_cols,
    [f"{c}__w" for c in target_cols],
)
loop_cnt = resid.groupby(["seqpos", "loop_type"]).size().rename("n").astype(np.float32)

struct_mean_resid = weighted_mean_by_group(
    resid,
    ["seqpos", "struct_char"],
    target_cols,
    [f"{c}__w" for c in target_cols],
)
struct_cnt = (
    resid.groupby(["seqpos", "struct_char"]).size().rename("n").astype(np.float32)
)

K_BASE_BY_TARGET = {
    "reactivity": 24.0,
    "deg_Mg_pH10": 32.0,
    "deg_pH10": 45.0,
    "deg_Mg_50C": 32.0,
    "deg_50C": 45.0,
}
K_LOOP_BY_TARGET = {
    "reactivity": 17.0,
    "deg_Mg_pH10": 24.0,
    "deg_pH10": 35.0,
    "deg_Mg_50C": 24.0,
    "deg_50C": 35.0,
}
K_STRUCT_BY_TARGET = {
    "reactivity": 17.0,
    "deg_Mg_pH10": 24.0,
    "deg_pH10": 35.0,
    "deg_Mg_50C": 24.0,
    "deg_50C": 35.0,
}

base_effect = {}
loop_effect = {}
struct_effect = {}
for c in target_cols:
    bw = (base_cnt / (base_cnt + float(K_BASE_BY_TARGET[c]))).astype(np.float32)
    lw = (loop_cnt / (loop_cnt + float(K_LOOP_BY_TARGET[c]))).astype(np.float32)
    sw = (struct_cnt / (struct_cnt + float(K_STRUCT_BY_TARGET[c]))).astype(np.float32)

    base_effect[c] = base_mean_resid[c].mul(bw, axis=0)
    loop_effect[c] = loop_mean_resid[c].mul(lw, axis=0)
    struct_effect[c] = struct_mean_resid[c].mul(sw, axis=0)

REL_K_BASE = 12.0
REL_K_LOOP = 12.0
REL_K_STRUCT = 12.0
base_rel = (base_cnt / (base_cnt + np.float32(REL_K_BASE))).astype(np.float32)
loop_rel = (loop_cnt / (loop_cnt + np.float32(REL_K_LOOP))).astype(np.float32)
struct_rel = (struct_cnt / (struct_cnt + np.float32(REL_K_STRUCT))).astype(np.float32)

CLIP_Q_LO_SCORED = 0.0002
CLIP_Q_HI_SCORED = 0.9998
CLIP_Q_LO_UNSCORED = 0.0005
CLIP_Q_HI_UNSCORED = 0.9995

pos_q_lo = {}
pos_q_hi = {}
for c in target_cols:
    if c in scored_target_cols:
        qlo, qhi = CLIP_Q_LO_SCORED, CLIP_Q_HI_SCORED
    else:
        qlo, qhi = CLIP_Q_LO_UNSCORED, CLIP_Q_HI_UNSCORED
    pos_q_lo[c] = train_long.groupby("seqpos")[c].quantile(qlo)
    pos_q_hi[c] = train_long.groupby("seqpos")[c].quantile(qhi)

pos_q_lo = pd.DataFrame(pos_q_lo)
pos_q_hi = pd.DataFrame(pos_q_hi)

global_q_lo = {}
global_q_hi = {}
for c in target_cols:
    if c in scored_target_cols:
        qlo, qhi = CLIP_Q_LO_SCORED, CLIP_Q_HI_SCORED
    else:
        qlo, qhi = CLIP_Q_LO_UNSCORED, CLIP_Q_HI_UNSCORED
    global_q_lo[c] = float(train_long[c].quantile(qlo))
    global_q_hi[c] = float(train_long[c].quantile(qhi))
global_q_lo = pd.Series(global_q_lo)
global_q_hi = pd.Series(global_q_hi)

for pos in range(68):
    if pos not in pos_q_lo.index:
        pos_q_lo.loc[pos] = global_q_lo
        pos_q_hi.loc[pos] = global_q_hi
pos_q_lo = pos_q_lo.sort_index()
pos_q_hi = pos_q_hi.sort_index()

print("Per-position means (head):")
print(pos_means.head())
print("Clip ranges example (seqpos 0):")
print(pd.DataFrame({"lo": pos_q_lo.loc[0], "hi": pos_q_hi.loc[0]}))



## === cell 4
test_lookup = df_test.set_index("id")[
    ["sequence", "predicted_loop_type", "structure"]
].to_dict(orient="index")

df = sample_sub.copy()
seqpos = df["id_seqpos"].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
ids = df["id_seqpos"].astype(str).str.rsplit("_", n=1).str[0]

df["_seqpos"] = seqpos
df["_id"] = ids

bases = np.empty(len(df), dtype=object)
loops = np.empty(len(df), dtype=object)
structs = np.empty(len(df), dtype=object)

for i, (rid, sp_i) in enumerate(zip(df["_id"].to_numpy(), df["_seqpos"].to_numpy())):
    info = test_lookup.get(rid, None)
    if info is None:
        bases[i] = np.nan
        loops[i] = np.nan
        structs[i] = np.nan
        continue
    s = info["sequence"]
    lt = info["predicted_loop_type"]
    st = info["structure"]
    bases[i] = s[sp_i] if sp_i < len(s) else np.nan
    loops[i] = lt[sp_i] if sp_i < len(lt) else np.nan
    structs[i] = st[sp_i] if sp_i < len(st) else np.nan

df["_base"] = bases
df["_loop_type"] = loops
df["_struct_char"] = structs

sp = df["_seqpos"].to_numpy()
scored_mask = sp < 68
unscored_mask = ~scored_mask

last_scored_mean = pos_means.loc[67]

RESID_SCALE_BY_TARGET = {
    "reactivity": 1.00,  # was 0.98
    "deg_Mg_pH10": 0.97,  # was 0.95
    "deg_pH10": 0.92,  # unscored: keep conservative/stable
    "deg_Mg_50C": 0.97,  # was 0.95
    "deg_50C": 0.92,  # unscored: keep conservative/stable
}

VALID_BASES = set(["A", "C", "G", "U"])
VALID_LOOP = set(list("SMIBHEX"))  # bpRNA loop types
VALID_STRUCT = set(["(", ")", "."])

for c in target_cols:
    vals = np.empty(len(df), dtype=np.float32)
    vals.fill(np.float32(last_scored_mean[c]))

    if scored_mask.any():
        sp_sc = df.loc[scored_mask, "_seqpos"].to_numpy()
        base_sc = df.loc[scored_mask, "_base"].to_numpy()
        loop_sc = df.loc[scored_mask, "_loop_type"].to_numpy()
        struct_sc = df.loc[scored_mask, "_struct_char"].to_numpy()

        base_sc = np.array(
            [b if b in VALID_BASES else np.nan for b in base_sc], dtype=object
        )
        loop_sc = np.array(
            [l if l in VALID_LOOP else np.nan for l in loop_sc], dtype=object
        )
        struct_sc = np.array(
            [s if s in VALID_STRUCT else np.nan for s in struct_sc], dtype=object
        )

        base_pred = pos_means.loc[sp_sc, c].to_numpy(dtype=np.float32)

        idx_base = pd.MultiIndex.from_arrays([sp_sc, base_sc], names=["seqpos", "base"])
        be = base_effect[c].reindex(idx_base).to_numpy()
        be = np.where(np.isfinite(be), be, 0.0).astype(np.float32)
        br = base_rel.reindex(idx_base).to_numpy()
        br = np.where(np.isfinite(br), br, 0.0).astype(np.float32)

        idx_loop = pd.MultiIndex.from_arrays(
            [sp_sc, loop_sc], names=["seqpos", "loop_type"]
        )
        le = loop_effect[c].reindex(idx_loop).to_numpy()
        le = np.where(np.isfinite(le), le, 0.0).astype(np.float32)
        lr = loop_rel.reindex(idx_loop).to_numpy()
        lr = np.where(np.isfinite(lr), lr, 0.0).astype(np.float32)

        idx_struct = pd.MultiIndex.from_arrays(
            [sp_sc, struct_sc], names=["seqpos", "struct_char"]
        )
        se = struct_effect[c].reindex(idx_struct).to_numpy()
        se = np.where(np.isfinite(se), se, 0.0).astype(np.float32)
        sr = struct_rel.reindex(idx_struct).to_numpy()
        sr = np.where(np.isfinite(sr), sr, 0.0).astype(np.float32)

        resid_sum = be * br + le * lr + se * sr
        pred = base_pred + np.float32(RESID_SCALE_BY_TARGET.get(c, 0.92)) * resid_sum

        lo = pos_q_lo.loc[sp_sc, c].to_numpy(dtype=np.float32)
        hi = pos_q_hi.loc[sp_sc, c].to_numpy(dtype=np.float32)
        pred = np.minimum(np.maximum(pred, lo), hi)

        vals[scored_mask] = pred

    if unscored_mask.any():
        lo_last = np.float32(pos_q_lo.loc[67, c])
        hi_last = np.float32(pos_q_hi.loc[67, c])
        vals[unscored_mask] = np.clip(vals[unscored_mask], lo_last, hi_last)

    df[c] = vals

df = df.drop(columns=["_seqpos", "_id", "_base", "_loop_type", "_struct_char"])

print(df.head())



## === cell 5
mask = df["id_seqpos"].astype(str).str.startswith("id_ff4593941_")
n = int(mask.sum())
print("rows matching id_ff4593941_*:", n)



## === cell 6
expected_cols = ["id_seqpos"] + target_cols
missing = [c for c in expected_cols if c not in df.columns]
extra = [c for c in df.columns if c not in expected_cols]
if missing:
    raise ValueError(f"Submission missing required columns: {missing}")
if extra:
    df = df[expected_cols]

if df.shape[0] != sample_sub.shape[0]:
    raise ValueError(
        f"Row count mismatch: df has {df.shape[0]} rows, sample has {sample_sub.shape[0]} rows"
    )

for c in target_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    if not np.isfinite(df[c].to_numpy()).all():
        raise ValueError(f"Non-finite values found in column: {c}")

print("submission shape:", df.shape)
print(df.describe(include="all").transpose().head(10))



## === cell 7
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
