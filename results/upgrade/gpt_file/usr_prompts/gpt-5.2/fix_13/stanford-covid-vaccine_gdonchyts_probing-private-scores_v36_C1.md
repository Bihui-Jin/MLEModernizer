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

0.46757

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I remove the dependency on the missing `../input/worst-submission/ensemble52.csv` file and instead build a valid submission directly from `sample_submission.csv`, which is guaranteed to exist in your environment. This fixes the immediate `FileNotFoundError` and the downstream `NameError` where `df` was never created. I keep your existing “set reactivity to 0 for a specific id prefix” logic intact, but guard it so it won’t crash if no matching rows exist. Finally, the script always write a correctly shaped `submission.csv` with the required columns.'
- What this solution (achieved 0.41187) has done: 'Your current score is far worse than the target (0.63824 vs 0.35188; lower is better), and the main reason is that you’re essentially submitting the sample submission with almost-all-constant predictions. With the available packages (no deep learning libs), the smallest legitimate improvement is to replace the constant predictions with per-position averages learned from the training set, and to use a simple sequence-based adjustment (average by nucleotide at each position) to better match the evaluation targets while keeping the overall approach very lightweight. This keeps the “core logic” of generating a submission from provided files, but makes the predictions data-driven using train.json. The script still produce a valid `submission.csv` with the exact required columns and row alignment via `id_seqpos`.'
- What this solution (achieved 0.40047) has done: 'Your current score (0.41187; lower is better) is worse than the target (0.35188), so we should improve predictions while keeping your simple “train-derived per-position means with nucleotide adjustment” core intact. The smallest likely gain is to (1) train means only on the *scored* target columns (3 columns) to reduce noise and then copy those predictions into the two unscored columns, and (2) add a light structure-based adjustment at each position (paired/unpaired from `structure`) in the same spirit as your nucleotide adjustment. This keeps the same overall approach (no new models/loops/training paradigm), but uses more of the provided inputs to better match the metric. The submission format and row alignment via `sample_submission.csv` are preserved exactly, still writing `submission.csv`.'
- What this solution (achieved 0.3957) has done: 'Your current score (0.40047; lower is better) is worse than the target (0.35188), so we should make small, low-risk improvements to reduce MCRMSE without changing the overall “train-derived per-position means + simple adjustments” approach. The minimal gain likely comes from (1) adding the provided `predicted_loop_type` as another lightweight categorical adjustment per position (similar to your nucleotide and paired/unpaired adjustments), and (2) smoothing the final prediction as a convex combination with the global per-position mean to reduce variance/overfitting to sparse categories. This keeps the same core logic (no new model class, no new training loop), but uses one more informative input already present in both train/test and adds a stability regularizer. The submission format, alignment by `id_seqpos`, and writing `submission.csv` remain unchanged.'
- What this solution (achieved 0.40674) has done: 'You’re currently worse than the target (0.3957 vs 0.3519; lower is better), so we make the smallest safe change that should reduce MCRMSE without changing your overall “per-position mean + categorical adjustments” approach. The main improvement is to make the shrinkage adaptive: categories (nt/pair/loop) with low support at a given position are noisy, so we shrink them more toward the robust per-position mean using count-based weights learned from train. This keeps the same features and linear mixing, but replaces fixed alphas with reliability-weighted alphas, which usually improves generalization on this dataset. We keep submission alignment via `sample_submission.csv` identical and still write a valid `submission.csv`.'
- What this solution (achieved 0.46515) has done: 'We’re currently worse than the target (0.40674 vs 0.35188; lower is better), so we should make a small, low-risk improvement that keeps your same “per-position mean + categorical adjustments with shrinkage” core intact. The biggest issue in your current logic is that the `reactivity`, `deg_Mg_pH10`, and `deg_Mg_50C` targets have different noise levels, but you’re treating all training rows equally; we can reduce MCRMSE by using the provided per-position measurement errors as weights when computing the means (more reliable measurements get more influence). This preserves the same feature set and combination formula, but swaps unweighted averages for inverse-variance-weighted averages in the precomputed tables (position mean and the nt/pair/loop conditional means). The submission format, alignment via `sample_submission.csv`, and the “special id prefix => reactivity=0” rule remain unchanged.'
- What this solution (achieved 0.45287) has done: 'Your current score (0.46515; lower is better) is still worse than the target (0.35188), so we should improve generalization without changing your core “weighted per-position mean + categorical adjustments with shrinkage” logic. The biggest correctness issue is in how you compute the nucleotide/pair/loop conditional means: you accumulate numerators with per-target weights `W[c]`, but you divide by a *different* denominator (`w_support` averaged across targets), which biases these conditional means; fixing this to use per-target denominators is a minimal, directly relevant change. I also keep your adaptive shrinkage structure intact, but base each feature’s shrinkage on the correct per-target effective weight sum (so reliability is consistent by target). Everything else (inputs, outputs, columns, special id rule, and writing `submission.csv`) remains the same.'
- What this solution (achieved 0.45287) has done: 'We keep your existing “weighted per-position mean + (nt/pair/loop) conditional adjustments with adaptive shrinkage” core intact, but make one minimal change that usually improves MCRMSE: don’t hard-copy the unscored columns from the scored ones. Instead, we also compute the same kind of weighted per-position means (and the same adjusted predictions) for `deg_pH10` and `deg_50C`, which is consistent with your approach and uses only train.json. This shouldn’t affect the metric directly (those columns aren’t scored), but in this competition it often slightly improves overall submission sanity and can indirectly help if any evaluation tooling checks all columns; it’s also a low-risk improvement. Additionally, we replace the “fallback for p>=68 uses last mean” with the more appropriate “use the last available scored position mean for p<107” only when needed, while still ensuring all 107 positions get a prediction and preserving submission row alignment.'
- What this solution (achieved 0.45287) has done: 'Your current score (0.45287; lower is better) is still far from the target (0.35188), so we should make the smallest, low-risk change that improves generalization while preserving your exact “weighted per-position mean + (nt/pair/loop) conditional means with adaptive shrinkage” core. The most direct improvement is to reduce noisy contributions from very-low-support categories by adding a tiny “effective weight floor” when computing the reliability weights (w_nt/w_pair/w_loop), which stabilizes shrinkage behavior without changing the model form. In addition, we make the fallback for p>=68 slightly more faithful to the data by using per-position extrapolation via the last two scored positions (linear trend) instead of a flat last value, while still keeping predictions bounded and deterministic. These are minimal, metric-relevant tweaks and keep I/O paths and submission schema unchanged.'
- What this solution (achieved 0.44651) has done: 'Your current score (0.45287; lower is better) is still worse than the target (0.35188), so we should make a small, low-risk improvement without changing your overall “weighted per-position mean + categorical conditional means with adaptive shrinkage” approach. The most direct issue left is that the inverse-variance weights can become extremely large for tiny reported errors, which makes some conditional tables overly dominated by a few measurements and hurts generalization; we cap the maximum weight to stabilize all computed means while keeping the same weighting logic. This is a minimal change (one new hyperparameter + one clip) that preserves semantics and should reduce MCRMSE toward the target. All I/O paths, feature usage (nt/pair/loop), blending/shrinkage structure, and submission formatting remain unchanged.'
- What this solution (achieved 0.46724) has done: 'We should move your score down (lower is better) toward 0.3519 from 0.4465, so the smallest likely gain is to keep your exact “weighted per-position mean + (nt/pair/loop) conditional means with adaptive shrinkage” logic but fix a reliability mismatch: you compute conditional means using inverse-variance weights, yet your shrinkage “support” uses raw weight sums which can still be dominated by clipped high-weights and behave unlike an actual sample size. I add a minimal “effective sample size” (ESS) calculation per (position, category) and use that ESS only for shrinkage strength (not for the conditional means), which typically improves generalization without changing the core model form. Everything else—features, weighting for the means, blending formula, tail extrapolation, id-prefix rule, and submission format/path—stays unchanged, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.46757) has done: 'Your current score (0.46724; lower is better) is still worse than the target (0.35188), so we should make a small, low-risk change that improves generalization without changing your core “weighted per-position mean + (nt/pair/loop) conditional means with adaptive shrinkage” approach. The most likely remaining issue is that the inverse-variance weighting and ESS-based shrinkage can still produce overly strong category effects; adding a tiny amount of *extra* global shrinkage toward the robust per-position mean at prediction time reduces variance and usually improves MCRMSE on this dataset. I also clamp in-scoring-range predictions to a conservative range to avoid rare outliers hurting RMSE, while leaving your tail extrapolation logic intact. All I/O paths, features used, weighting tables, blending structure, and submission formatting remain unchanged, and the script still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

df_train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
df_test = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")

required_cols = [
    "id_seqpos",
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]
missing = [c for c in required_cols if c not in sub.columns]
if missing:
    raise ValueError(f"sample_submission.csv missing required columns: {missing}")

scored_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
unscored_cols = ["deg_pH10", "deg_50C"]
all_target_cols = scored_cols + unscored_cols

SEQ_LEN = int(df_test["seq_length"].iloc[0])
assert SEQ_LEN == 107, f"Unexpected seq_length in test: {SEQ_LEN}"



## === cell 1
SCORING_LEN = int(df_train["seq_scored"].mode().iloc[0])  # should be 68
assert SCORING_LEN == 68, f"Unexpected seq_scored in train: {SCORING_LEN}"


def _stack_targets(df, col):
    return np.vstack(df[col].values).astype(np.float32)


def _stack_errors(df, col):
    return np.vstack(df[col].values).astype(np.float32)


err_col_map = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_pH10": "deg_error_pH10",
    "deg_50C": "deg_error_50C",
}

df_train_f = df_train[df_train["SN_filter"] == 1].reset_index(drop=True)
if len(df_train_f) == 0:
    df_train_f = df_train.copy()

Y = {c: _stack_targets(df_train_f, c) for c in all_target_cols}
E = {c: _stack_errors(df_train_f, err_col_map[c]) for c in all_target_cols}

eps = 1e-3

max_inv_var = 2500.0  # equivalent to min effective error ~0.02; keeps weights bounded

W = {}
for c in all_target_cols:
    e = np.clip(E[c], eps, 10.0).astype(np.float32)
    inv_var = (1.0 / (e * e)).astype(np.float32)
    W[c] = np.clip(inv_var, 0.0, max_inv_var).astype(np.float32)

pos_mean = {}
for c in all_target_cols:
    wsum = W[c].sum(axis=0)  # (68,)
    wsum = np.where(wsum > 0, wsum, 1.0).astype(np.float32)
    pos_mean[c] = (W[c] * Y[c]).sum(axis=0) / wsum  # shape (68,)

seqs = df_train_f["sequence"].astype(str).values
nt_map = {"A": 0, "C": 1, "G": 2, "U": 3}
K = 4

pos_nt_wsum = {c: np.zeros((SCORING_LEN, K), dtype=np.float64) for c in all_target_cols}
pos_nt_wcnt = {
    c: np.zeros((SCORING_LEN, K), dtype=np.float64) for c in all_target_cols
}  # sum(w)
pos_nt_w2 = {
    c: np.zeros((SCORING_LEN, K), dtype=np.float64) for c in all_target_cols
}  # sum(w^2)

for i, s in enumerate(seqs):
    if len(s) < SCORING_LEN:
        continue
    for p in range(SCORING_LEN):
        k = nt_map.get(s[p], None)
        if k is None:
            continue
        for c in all_target_cols:
            w = float(W[c][i, p])
            pos_nt_wcnt[c][p, k] += w
            pos_nt_w2[c][p, k] += w * w
            pos_nt_wsum[c][p, k] += float(W[c][i, p] * Y[c][i, p])

pos_nt_mean = {}
pos_nt_ess = {}
for c in all_target_cols:
    mean = np.zeros((SCORING_LEN, K), dtype=np.float32)
    ess = np.zeros((SCORING_LEN, K), dtype=np.float32)
    for p in range(SCORING_LEN):
        for k in range(K):
            denom = pos_nt_wcnt[c][p, k]
            if denom > 0:
                mean[p, k] = pos_nt_wsum[c][p, k] / denom
                w2 = pos_nt_w2[c][p, k]
                ess[p, k] = (denom * denom / w2) if w2 > 0 else 0.0
            else:
                mean[p, k] = pos_mean[c][p]
                ess[p, k] = 0.0
    pos_nt_mean[c] = mean
    pos_nt_ess[c] = ess

structures = df_train_f["structure"].astype(str).values

pair_wcnt = {
    c: np.zeros((SCORING_LEN, 2), dtype=np.float64) for c in all_target_cols
}  # sum(w)
pair_wsum = {c: np.zeros((SCORING_LEN, 2), dtype=np.float64) for c in all_target_cols}
pair_w2 = {
    c: np.zeros((SCORING_LEN, 2), dtype=np.float64) for c in all_target_cols
}  # sum(w^2)

for i, st in enumerate(structures):
    if len(st) < SCORING_LEN:
        continue
    for p in range(SCORING_LEN):
        ch = st[p]
        g = 1 if (ch == "(" or ch == ")") else 0
        for c in all_target_cols:
            w = float(W[c][i, p])
            pair_wcnt[c][p, g] += w
            pair_w2[c][p, g] += w * w
            pair_wsum[c][p, g] += float(W[c][i, p] * Y[c][i, p])

pair_mean = {}
pair_ess = {}
for c in all_target_cols:
    m = np.zeros((SCORING_LEN, 2), dtype=np.float32)
    ess = np.zeros((SCORING_LEN, 2), dtype=np.float32)
    for p in range(SCORING_LEN):
        for g in range(2):
            denom = pair_wcnt[c][p, g]
            if denom > 0:
                m[p, g] = pair_wsum[c][p, g] / denom
                w2 = pair_w2[c][p, g]
                ess[p, g] = (denom * denom / w2) if w2 > 0 else 0.0
            else:
                m[p, g] = pos_mean[c][p]
                ess[p, g] = 0.0
    pair_mean[c] = m
    pair_ess[c] = ess

loop_types_train = df_train_f["predicted_loop_type"].astype(str).values
loop_set = set()
for lt in loop_types_train:
    for ch in lt[:SCORING_LEN]:
        loop_set.add(ch)
loop_list = sorted(loop_set)
loop_map = {ch: i for i, ch in enumerate(loop_list)}
L = len(loop_list)

loop_wcnt = {
    c: np.zeros((SCORING_LEN, L), dtype=np.float64) for c in all_target_cols
}  # sum(w)
loop_wsum = {c: np.zeros((SCORING_LEN, L), dtype=np.float64) for c in all_target_cols}
loop_w2 = {
    c: np.zeros((SCORING_LEN, L), dtype=np.float64) for c in all_target_cols
}  # sum(w^2)

for i, lt in enumerate(loop_types_train):
    if len(lt) < SCORING_LEN:
        continue
    for p in range(SCORING_LEN):
        ch = lt[p]
        li = loop_map.get(ch, None)
        if li is None:
            continue
        for c in all_target_cols:
            w = float(W[c][i, p])
            loop_wcnt[c][p, li] += w
            loop_w2[c][p, li] += w * w
            loop_wsum[c][p, li] += float(W[c][i, p] * Y[c][i, p])

loop_mean = {}
loop_ess = {}
for c in all_target_cols:
    m = np.zeros((SCORING_LEN, L), dtype=np.float32)
    ess = np.zeros((SCORING_LEN, L), dtype=np.float32)
    for p in range(SCORING_LEN):
        for li in range(L):
            denom = loop_wcnt[c][p, li]
            if denom > 0:
                m[p, li] = loop_wsum[c][p, li] / denom
                w2 = loop_w2[c][p, li]
                ess[p, li] = (denom * denom / w2) if w2 > 0 else 0.0
            else:
                m[p, li] = pos_mean[c][p]
                ess[p, li] = 0.0
    loop_mean[c] = m
    loop_ess[c] = ess

base_alpha_nt = 0.50
base_alpha_pair = 0.25
base_alpha_loop = 0.25

tau_nt = 200.0
tau_pair = 400.0
tau_loop = 800.0

global_shrink_to_mean = 0.05

support_floor = 5.0

pos_tail_slope = {}
for c in all_target_cols:
    last = float(pos_mean[c][SCORING_LEN - 1])
    prev = float(pos_mean[c][SCORING_LEN - 2])
    pos_tail_slope[c] = last - prev

extra_pred_shrink = (
    0.05  # keeps core blending the same; just nudges toward robust pos_mean
)

clip_lo, clip_hi = -0.75, 2.75



## === cell 2
test_ids = df_test["id"].astype(str).values
test_seqs = df_test["sequence"].astype(str).values
test_structs = df_test["structure"].astype(str).values
test_loops = df_test["predicted_loop_type"].astype(str).values

id_to_seq = dict(zip(test_ids, test_seqs))
id_to_struct = dict(zip(test_ids, test_structs))
id_to_loop = dict(zip(test_ids, test_loops))

id_seqpos = sub["id_seqpos"].astype(str)
id_part = id_seqpos.str.rsplit("_", n=1).str[0]
pos_part = id_seqpos.str.rsplit("_", n=1).str[1].astype(int).values

pred = {c: np.zeros(len(sub), dtype=np.float32) for c in all_target_cols}

for idx in range(len(sub)):
    rid = id_part.iloc[idx]
    p = int(pos_part[idx])

    s = id_to_seq.get(rid, "")
    st = id_to_struct.get(rid, "")
    lt = id_to_loop.get(rid, "")

    if p < SCORING_LEN and len(s) > p:
        k = nt_map.get(s[p], None)
        if k is None:
            nt_val = {c: float(pos_mean[c][p]) for c in all_target_cols}
            sup_nt = {c: 0.0 for c in all_target_cols}
        else:
            nt_val = {c: float(pos_nt_mean[c][p, k]) for c in all_target_cols}
            sup_nt = {c: float(pos_nt_ess[c][p, k]) for c in all_target_cols}

        if len(st) > p:
            chs = st[p]
            g = 1 if (chs == "(" or chs == ")") else 0
            pair_val = {c: float(pair_mean[c][p, g]) for c in all_target_cols}
            sup_pair = {c: float(pair_ess[c][p, g]) for c in all_target_cols}
        else:
            pair_val = {c: float(pos_mean[c][p]) for c in all_target_cols}
            sup_pair = {c: 0.0 for c in all_target_cols}

        if len(lt) > p and L > 0:
            chl = lt[p]
            li = loop_map.get(chl, None)
            if li is None:
                loop_val = {c: float(pos_mean[c][p]) for c in all_target_cols}
                sup_loop = {c: 0.0 for c in all_target_cols}
            else:
                loop_val = {c: float(loop_mean[c][p, li]) for c in all_target_cols}
                sup_loop = {c: float(loop_ess[c][p, li]) for c in all_target_cols}
        else:
            loop_val = {c: float(pos_mean[c][p]) for c in all_target_cols}
            sup_loop = {c: 0.0 for c in all_target_cols}

        for c in all_target_cols:
            eff_nt = max(0.0, sup_nt[c])
            eff_pair = max(0.0, sup_pair[c])
            eff_loop = max(0.0, sup_loop[c])

            w_nt = eff_nt / (eff_nt + tau_nt + support_floor)
            w_pair = eff_pair / (eff_pair + tau_pair + support_floor)
            w_loop = eff_loop / (eff_loop + tau_loop + support_floor)

            a_nt = base_alpha_nt * w_nt
            a_pair = base_alpha_pair * w_pair
            a_loop = base_alpha_loop * w_loop
            a_mean = max(0.0, 1.0 - (a_nt + a_pair + a_loop))

            combined = (
                a_nt * nt_val[c]
                + a_pair * pair_val[c]
                + a_loop * loop_val[c]
                + a_mean * float(pos_mean[c][p])
            )
            val = (
                1.0 - global_shrink_to_mean
            ) * combined + global_shrink_to_mean * float(pos_mean[c][p])

            val = (1.0 - extra_pred_shrink) * val + extra_pred_shrink * float(
                pos_mean[c][p]
            )

            pred[c][idx] = np.float32(np.clip(val, clip_lo, clip_hi))
    else:
        t = max(0, p - (SCORING_LEN - 1))
        for c in all_target_cols:
            base = float(pos_mean[c][SCORING_LEN - 1])
            val = base + float(pos_tail_slope[c]) * float(t)
            pred[c][idx] = np.float32(np.clip(val, -1.0, 3.0))



## === cell 3
out = sub.copy()

for c in all_target_cols:
    out[c] = pred[c]

out["id_seqpos"] = out["id_seqpos"].astype(str)
mask = out.id_seqpos.str.startswith("id_c739ea3bf")
if mask.any():
    out.loc[mask, "reactivity"] = 0.0

out = out[required_cols].copy()
target_cols_all = required_cols[1:]
out[target_cols_all] = (
    out[target_cols_all].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)



## === cell 4
out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print(out.head())
print("Columns:", list(out.columns))
