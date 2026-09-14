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

0.3520523385748121

# 6. Current score

0.42245

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I remove the hardcoded dependency on a missing `../input/worst-submission/ensemble52.csv` and instead build a valid submission directly from the provided competition files. Since no model code exists here, the safest minimal fix is to start from `sample_submission.csv` and (optionally) apply your existing one-off adjustment to the `reactivity` column for the specified `id`. This run end-to-end in the given environment and always write a correctly formatted `submission.csv`. The score won’t match your target without a real model, but it at least yield a valid submission and eliminate the runtime errors.'
- What this solution (achieved 0.42166) has done: 'Your current score is far worse than the target (lower is better), and the code is essentially submitting the sample submission (all zeros), so the smallest legitimate improvement is to replace those constant predictions with a simple baseline learned from `train.json`. To preserve core logic (no model/training loop exists), we compute per-position means for each target from the training set (optionally restricted to `SN_filter==1` for cleaner labels) and use those means as predictions for each `seqpos`. We still output all 5 required columns, and for the two unscored columns we also use their per-position means (harmless and valid). This keeps the pipeline simple, fast (<600s), avoids leakage, and should move the score substantially toward your target compared with all-zeros.'
- What this solution (achieved 0.48245) has done: 'Your current score (0.42166; lower is better) is worse than the target (0.35205), so we should improve the baseline while keeping the same overall “per-position mean from train.json” core approach. The smallest, most reliable upgrade is to compute the per-position means using sample quality weights (derived from the provided per-position measurement errors), so cleaner measurements contribute more without changing the modeling paradigm. We also remove the hardcoded override that forces one id’s reactivity to 0.0 (this is very likely hurting score). Finally, we keep the same submission formatting and ensure the output CSV remains valid.'
- What this solution (achieved 0.42176) has done: 'We keep your “per-position baseline from train.json” core logic, but remove the inverse-variance weighting, because this particular weighting is likely hurting leaderboard generalization (it can over-trust noisy/biased low-error measurements) and your score (0.48245) is far from the target (0.35205, lower is better). Instead, we compute simple per-position means on the same `SN_filter==1` subset, which is a minimal change and often a stronger baseline for this competition. We also add a small safety clamp on targets to a reasonable range to prevent rare outliers from inflating RMSE, without changing the approach or requiring any new packages. Submission format/path remains identical and the script still runs end-to-end producing `submission.csv`.'
- What this solution (achieved 0.42467) has done: 'We keep your current “per-position mean baseline from train.json” approach, but make it slightly stronger and more robust by (1) using a trimmed mean per position to reduce the influence of outlier/noisy training measurements and (2) optionally blending in a global mean as shrinkage to improve generalization without changing the core logic. This should move your score down (lower is better) toward the 0.352 target while staying within the same simple statistical baseline family and keeping runtime fast. We also ensure the non-scored positions (>= seq_scored) are filled consistently and keep the same submission format and path. No model architecture/training loop/loss changes are introduced.'
- What this solution (achieved 0.4233) has done: 'We keep your exact “per-position baseline from train.json” core logic, but adjust two robustness knobs that directly affect leaderboard MCRMSE: (1) apply the competition’s standard train filtering of low-quality samples using `signal_to_noise > 1` in addition to `SN_filter==1`, and (2) slightly retune the trimmed-mean and shrinkage strengths so the baseline generalizes better (your current settings look a bit too aggressive on trimming and too strong on shrinkage for this dataset). We also make the fill rule for unscored positions use a global mean (instead of repeating the last scored position), which is a small change but often reduces error for positions >68 that are still required in the submission. The submission format, paths, and overall pipeline remain identical and it still write a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.42346) has done: 'We keep your current “per-position baseline from train.json” approach, but make one small, metric-aligned improvement: compute position-wise means only from training rows with *valid labels* (finite values) at that position and target, instead of letting rare NaNs/outliers influence the trimmed mean and the global mean. We also slightly reduce shrinkage (toward the global mean) because your current configuration likely over-smooths genuine positional patterns, which can hurt MCRMSE when test differs from train in a structured way. Everything else (paths, filtering by SN_filter and signal_to_noise, trimmed mean robustness, clipping, submission format) remains unchanged and still runs end-to-end writing `submission.csv`.'
- What this solution (achieved 0.42245) has done: 'We keep your exact “per-position baseline from train.json” core logic, but make two small, metric-aligned tweaks that typically improve MCRMSE without changing the approach: (1) compute trimmed means using only the *central* (1−2q) fraction via sorting (more stable than quantiles on small samples) and (2) slightly retune the robustness knobs (a bit less trimming, a bit less shrinkage) to preserve positional signal while still damping outliers. We also make the clipping bounds data-driven from the filtered training labels (same for all targets) instead of fixed [-0.5, 2.0], which reduces unnecessary saturation that can inflate RMSE. Paths, filtering (SN_filter + signal_to_noise), prediction filling rules, and submission formatting remain unchanged, and it still writes a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.42245) has done: 'Your current score (0.42245, lower is better) is worse than the target (0.35205), so we should make a small, low-risk improvement while preserving the same core “per-position baseline from train.json” approach. The most direct metric-aligned tweak is to compute per-position means separately for different `predicted_loop_type` contexts (loop-type stratification), then fall back to the global per-position mean when a loop-type bucket is too small; this keeps the same baseline idea but uses an extra provided feature that correlates with reactivity/degradation. We keep your existing filtering (`SN_filter==1` and `signal_to_noise>1`), trimming, shrinkage, and clipping logic, and we only change how the per-position mean is estimated (now conditional on loop type). Submission format/paths remain unchanged and the script still writes a valid `submission.csv` end-to-end.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

np.random.seed(0)



## === cell 1
candidate_paths = [
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "../input/stanford-covid-vaccine/sample_submission.csv",
    "../input/sample_submission.csv",
]

sample_path = next((p for p in candidate_paths if os.path.exists(p)), None)
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected Kaggle paths. "
        f"Tried: {candidate_paths}"
    )

sub = pd.read_csv(sample_path)

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

sub.head(), sub.shape



## === cell 2
sequences = list(set(["_".join(v.split("_")[:-1]) for v in sub.id_seqpos.values]))
sequences[-10:], len(sequences)



## === cell 3
train_candidate_paths = [
    "/kaggle/input/stanford-covid-vaccine/train.json",
    "/kaggle/input/train.json",
    "/kaggle/data/stanford-covid-vaccine/train.json",
    "/kaggle/data/train.json",
    "../input/stanford-covid-vaccine/train.json",
    "../input/train.json",
]
train_path = next((p for p in train_candidate_paths if os.path.exists(p)), None)
if train_path is None:
    raise FileNotFoundError(
        "Could not find train.json in expected Kaggle paths. "
        f"Tried: {train_candidate_paths}"
    )

train = pd.read_json(train_path, lines=True)

targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
for t in targets:
    if t not in train.columns:
        raise ValueError(f"train.json missing target column: {t}")

train_used = train.copy()
if "SN_filter" in train_used.columns:
    train_used = train_used.loc[train_used["SN_filter"].astype(int) == 1]
if "signal_to_noise" in train_used.columns:
    train_used = train_used.loc[
        pd.to_numeric(train_used["signal_to_noise"], errors="coerce") > 1.0
    ]
train_used = train_used.reset_index(drop=True)

if len(train_used) < 50:
    train_used = (
        train.loc[train["SN_filter"].astype(int) == 1].reset_index(drop=True)
        if "SN_filter" in train.columns
        else train
    )

seq_scored = int(train_used["seq_scored"].iloc[0])
if seq_scored <= 0:
    raise ValueError(f"Unexpected seq_scored={seq_scored}")


def _trimmed_mean_1d(x: np.ndarray, trim_q: float) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if trim_q <= 0.0:
        return float(np.mean(x))
    xs = np.sort(x.astype(np.float32, copy=False))
    k = int(np.floor(trim_q * xs.size))
    if 2 * k >= xs.size:
        return float(np.mean(xs))
    return float(np.mean(xs[k : xs.size - k]))


TRIM_Q = 0.02
SHRINK_ALPHA = 0.04
MIN_GROUP = (
    80  # small safety threshold; below this we fall back to global per-position means
)

all_finite = []
for t in targets:
    y_tmp = np.asarray(train_used[t].tolist(), dtype=np.float32)
    if y_tmp.ndim != 2 or y_tmp.shape[1] != seq_scored:
        raise ValueError(
            f"Unexpected shape for {t}: {y_tmp.shape}, expected (*, {seq_scored})"
        )
    all_finite.append(y_tmp[np.isfinite(y_tmp)])
all_finite = (
    np.concatenate(all_finite) if len(all_finite) else np.array([], dtype=np.float32)
)

if all_finite.size:
    clip_lo = float(np.quantile(all_finite, 0.001))
    clip_hi = float(np.quantile(all_finite, 0.999))
    clip_lo = max(clip_lo, -1.0)
    clip_hi = min(clip_hi, 3.0)
else:
    clip_lo, clip_hi = -0.5, 2.0

pos_means_global = {}
global_means = {}
for t in targets:
    y = np.asarray(train_used[t].tolist(), dtype=np.float32)
    y_finite = y[np.isfinite(y)]
    global_mu = np.mean(y_finite) if y_finite.size else 0.0
    global_mu = np.float32(global_mu)

    mu = np.empty((seq_scored,), dtype=np.float32)
    for j in range(seq_scored):
        mu[j] = _trimmed_mean_1d(y[:, j], TRIM_Q)

    mu = np.where(np.isfinite(mu), mu, global_mu).astype(np.float32)
    mu = ((1.0 - SHRINK_ALPHA) * mu + SHRINK_ALPHA * global_mu).astype(np.float32)

    pos_means_global[t] = mu
    global_means[t] = float(global_mu)

pos_means_by_loop = {t: {} for t in targets}
if "predicted_loop_type" in train_used.columns:
    train_loops = train_used["predicted_loop_type"].astype(str).values
    unique_loops = np.unique(train_loops)

    for lt in unique_loops:
        idx = np.where(train_loops == lt)[0]
        if idx.size < MIN_GROUP:
            continue  # too small; will use global
        for t in targets:
            y = np.asarray(train_used.loc[idx, t].tolist(), dtype=np.float32)
            y_finite = y[np.isfinite(y)]
            loop_mu = (
                np.mean(y_finite) if y_finite.size else np.float32(global_means[t])
            )
            loop_mu = np.float32(loop_mu)

            mu = np.empty((seq_scored,), dtype=np.float32)
            for j in range(seq_scored):
                mu[j] = _trimmed_mean_1d(y[:, j], TRIM_Q)

            mu = np.where(np.isfinite(mu), mu, loop_mu).astype(np.float32)
            mu = ((1.0 - SHRINK_ALPHA) * mu + SHRINK_ALPHA * loop_mu).astype(np.float32)
            pos_means_by_loop[t][lt] = mu

test_candidate_paths = [
    "/kaggle/input/stanford-covid-vaccine/test.json",
    "/kaggle/input/test.json",
    "/kaggle/data/stanford-covid-vaccine/test.json",
    "/kaggle/data/test.json",
    "../input/stanford-covid-vaccine/test.json",
    "../input/test.json",
]
test_path = next((p for p in test_candidate_paths if os.path.exists(p)), None)
if test_path is None:
    raise FileNotFoundError(
        "Could not find test.json in expected Kaggle paths. "
        f"Tried: {test_candidate_paths}"
    )
test = pd.read_json(test_path, lines=True)

if "id" not in test.columns or "predicted_loop_type" not in test.columns:
    raise ValueError(
        "test.json missing required columns: id and/or predicted_loop_type"
    )

id_to_loop = dict(
    zip(test["id"].astype(str).values, test["predicted_loop_type"].astype(str).values)
)

id_part = sub["id_seqpos"].astype(str).str.rsplit("_", n=1).str[0].values
seqpos = sub["id_seqpos"].astype(str).str.rsplit("_", n=1).str[-1].astype(int).values

loop_strs = np.array([id_to_loop.get(i, "UNK") for i in id_part], dtype=object)
loop_char = np.array(
    [
        s[p] if (isinstance(s, str) and (0 <= p < len(s))) else "UNK"
        for s, p in zip(loop_strs, seqpos)
    ],
    dtype=object,
)

n = sub.shape[0]
for t in targets:
    base_global = pos_means_global[t]
    gmu = np.float32(global_means[t])

    pred = np.empty((n,), dtype=np.float32)
    pred[:] = np.where(
        seqpos < seq_scored, base_global[np.minimum(seqpos, seq_scored - 1)], gmu
    ).astype(np.float32)

    if len(pos_means_by_loop[t]) > 0:
        scored_mask = seqpos < seq_scored
        for lt, mu_lt in pos_means_by_loop[t].items():
            m = scored_mask & (loop_char == lt)
            if np.any(m):
                pred[m] = mu_lt[seqpos[m]]

    sub[t] = pred

for t in targets:
    sub[t] = sub[t].clip(lower=clip_lo, upper=clip_hi)

for c in required_cols[1:]:
    sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(0.0)

sub.head()



## === cell 4
assert sub.shape[0] > 0
assert list(sub.columns) == required_cols
assert sub["id_seqpos"].isna().sum() == 0
assert sub[required_cols[1:]].isna().sum().sum() == 0

if not sub["id_seqpos"].is_unique:
    sub = sub.drop_duplicates(subset=["id_seqpos"], keep="first").reset_index(drop=True)

sub.describe(include="all").T



## === cell 5
out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print(f"Wrote {out_path} with shape={sub.shape} to current working directory.")
print(sub.head())
