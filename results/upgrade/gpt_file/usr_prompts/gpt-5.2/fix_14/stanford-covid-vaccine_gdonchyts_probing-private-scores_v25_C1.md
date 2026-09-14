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

0.3704880744110533

# 6. Current score

0.48319

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'Your notebook fails because it tries to read a missing `../input/worst-submission/ensemble52.csv`, so `df` is never created and all later cells crash. To make it run end-to-end and produce a valid `submission.csv`, I replace that dependency by starting from the competition’s provided `sample_submission.csv` and (to keep the original “calibration divide” logic intact) apply the same `/= 1.17` scaling to all prediction columns. I also add a tiny, safe path resolver so it works with either `/kaggle/input/...` or the provided `/kaggle/data/...` layout, and I ensure columns/order match the required submission format.'
- What this solution (achieved 0.42166) has done: 'I fix the JSON loading error by reading `train.json` with `lines=True` (this dataset is newline-delimited JSON), which currently prevents `target_cols`/`pos_means` from being created and causes all later NameError/column-mismatch failures. Then I keep the existing core logic (a per-position mean baseline, expanded to full `seq_length`) but ensure the generated submission has all required columns in the exact order expected by `sample_submission.csv`. Finally, I add a small fallback so that if any row in `train.json` has missing/short target arrays, it won’t crash; this is score-neutral but makes the pipeline robust and ensures a valid `submission.csv` is always written.'
- What this solution (achieved 0.48258) has done: 'Your current solution is a strong “per-position mean” baseline; the safest way to move the MCRMSE down toward your target without changing core logic is to compute those per-position means with better weighting and less noise. I keep the exact same prediction construction (lookup by `seqpos`, fill unscored positions with the last mean), but change the mean estimator to a per-position **inverse-variance weighted mean** using the provided `*_error_*` columns (this is still “mean per position”, just more statistically efficient). Because only 3 targets are scored, I keep predicting all 5 but only apply weighting where error columns exist; this typically improves leaderboard score for this competition while preserving semantics. I also add a tiny safety floor on errors to avoid division issues and keep the same submission validation/writing.'
- What this solution (achieved 0.4748) has done: 'To move your MCRMSE down toward the target while preserving the exact “per-position mean baseline” core logic, I make the mean estimator slightly more robust and better aligned with the metric. Specifically, I (1) compute weighted means only over the scored region but also filter out extreme-noise/invalid training sequences more consistently using `signal_to_noise` (a standard competition filter that improves this baseline without changing its nature), and (2) apply a small shrinkage of the per-position means toward the global mean (still a mean-baseline, just lower-variance), which typically reduces error on noisy positions. The prediction construction, submission alignment, columns, and file writing remain unchanged.'
- What this solution (achieved 0.43924) has done: 'You’re already using a per-position mean baseline, so the smallest safe way to reduce MCRMSE toward the target is to make that same estimator better aligned to what’s scored and less biased by noisy training rows. I keep the identical prediction construction (lookup by seqpos; fill unscored positions with the last mean; output all 5 columns), but adjust the training filter to the competition-standard `signal_to_noise > 1` **or** `SN_filter==1` rather than requiring both (which can remove too much useful data). I also change the weighted mean to use the provided per-position error as an **inverse-std** weight (1/e) instead of inverse-variance (1/e²), which is a mild, stability-oriented reweighting that often improves this baseline under RMSE without changing the core “weighted mean per position” logic. Finally, I apply the same small shrinkage but compute the global mean from the same filtered rows for consistency.'
- What this solution (achieved 0.44377) has done: 'To move your MCRMSE down from 0.43924 toward the 0.37049 target (lower is better) while keeping the same “per-position weighted mean baseline” core logic, I make two minimal estimator tweaks that reduce variance/bias without changing the prediction construction. First, I compute the shrinkage “global mean” using the same inverse-error weights (instead of an unweighted mean), so shrinkage is consistent with the weighted per-position means. Second, I slightly tune the shrinkage strength `alpha` upward (a small change) to stabilize noisy positions, which typically improves this baseline on this competition. Everything else (data loading, filtering rule, per-position lookup, unscored fill with last mean, submission alignment) remains unchanged and it still write a valid `submission.csv`.'
- What this solution (achieved 0.44519) has done: 'Your current score (0.44377, lower-is-better) is still above the target (0.37049), so we should cautiously improve while keeping the same “per-position weighted mean baseline” logic. The smallest likely win is to (1) tune the shrinkage strength slightly downward (your last change increased alpha and worsened score), and (2) compute the per-position means and global mean only from the highest-quality training rows (`SN_filter==1`) to better match the public/private test distribution. This preserves the exact modeling approach (still just weighted means per position, same lookup and fill) but reduces noise/bias from low-quality measurements. Everything else (paths, JSON loading, column order, submission writing) remains unchanged.'
- What this solution (achieved 0.4445) has done: 'I keep your same per-position weighted-mean baseline and submission construction, but adjust the training-row filter to use both `SN_filter==1` (high-quality) and a mild `signal_to_noise` threshold when available, which typically reduces noise and should move MCRMSE down toward your target. I also make the shrinkage strength `alpha` slightly smaller (it was previously tuned upward and hurt), which preserves the same “shrink toward global mean” logic but reduces bias. Finally, I keep all paths, column ordering, and the `id_seqpos` alignment checks unchanged to ensure a valid `submission.csv` is produced.'
- What this solution (achieved 0.44422) has done: 'We keep your exact “per-position (weighted) mean + global-mean shrinkage + fill unscored with last mean” baseline, but tune the two parts that most directly affect MCRMSE: the training-row quality filter and the shrinkage strength `alpha`. Because your current score (0.4445, lower-is-better) is worse than the target (0.37049), we should cautiously improve; the smallest likely win is to (1) relax the `signal_to_noise` cutoff back toward the competition-standard `>1.0` while still requiring `SN_filter==1`, and (2) slightly reduce shrinkage (your current `alpha=0.04` can add bias when the per-position estimate is already good). These are minimal, metric-aligned changes that preserve core logic and keep end-to-end submission generation unchanged.'
- What this solution (achieved 0.48314) has done: 'Your score (0.44422, lower-is-better) is still worse than the target (0.37049), so we should make a very small, metric-aligned improvement without changing the baseline’s nature. I keep the same “per-position weighted mean + global-mean shrinkage + fill unscored with last mean” logic, but adjust the weighting to match RMSE better by using inverse-variance weights (1/e²) instead of inverse-std (1/e). I also slightly reduce the error floor (so genuinely high-quality measurements can contribute more) while keeping a safe minimum to avoid divide-by-zero. Everything else (data loading, filtering, submission schema/order, and writing `submission.csv`) remains unchanged.'
- What this solution (achieved 0.48314) has done: 'Your current score (0.48314, lower-is-better) is still meaningfully worse than the target (0.37049), so we should make a minimal, low-risk change that improves the same per-position weighted-mean baseline rather than altering the approach. The biggest likely gain within the same logic is to compute the per-position means on a higher-quality subset that better matches the filtered test distribution: use `SN_filter==1` alone (dropping the extra `signal_to_noise > 1` constraint, which can discard useful rows and make estimates noisier). I keep the exact same inverse-variance weighting, shrinkage form, and “fill unscored positions with last mean” behavior. Everything else (paths, submission alignment/validation, writing `submission.csv`) remains unchanged.'
- What this solution (achieved 0.48319) has done: 'Your current baseline is still above the target (0.48314 vs 0.37049, lower-is-better), so we should make a small, metric-aligned improvement without changing the core “per-position weighted mean + shrinkage + fill unscored with last mean” logic. The most likely low-risk win is to (1) compute the per-position means using only the **scored region** (as you already do) but additionally (2) apply a gentle, robust **winsorization** of training targets per position before taking the weighted mean, which reduces the impact of heavy-tailed/noisy measurements that inflate RMSE. This keeps the estimator as a per-position (weighted) mean, just with minimal outlier-robustness, and does not change the prediction construction or submission schema. Everything else (paths, filtering by `SN_filter==1`, inverse-variance weights, shrinkage form, unscored fill, and CSV writing) remains unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


def _first_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {paths}")




## === cell 1
sample_path = _first_existing_path(
    [
        "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)
train_path = _first_existing_path(
    [
        "/kaggle/input/stanford-covid-vaccine/train.json",
        "/kaggle/input/train.json",
        "/kaggle/data/stanford-covid-vaccine/train.json",
        "/kaggle/data/train.json",
    ]
)

sub = pd.read_csv(sample_path)

expected_cols = [
    "id_seqpos",
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]
missing = [c for c in expected_cols if c not in sub.columns]
if missing:
    raise ValueError(f"sample_submission.csv missing columns: {missing}")

sub = sub[expected_cols].copy()
sub.head()



## === cell 2
train = pd.read_json(train_path, lines=True)

target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
need = ["seq_scored"] + target_cols
missing = [c for c in need if c not in train.columns]
if missing:
    raise ValueError(f"train.json missing columns: {missing}")

has_sn_filter = "SN_filter" in train.columns
has_snr = "signal_to_noise" in train.columns

if has_sn_filter:
    train = train.loc[train["SN_filter"].values == 1].reset_index(drop=True)
elif has_snr:
    train = train.loc[train["signal_to_noise"].values > 1.0].reset_index(drop=True)

seq_scored = int(train["seq_scored"].iloc[0])
if not (1 <= seq_scored <= 107):
    raise ValueError(f"Unexpected seq_scored value: {seq_scored}")


def _stack_to_len(values, L):
    """Stack list-of-arrays into (n, L), padding with NaN if any row is short."""
    out = np.full((len(values), L), np.nan, dtype=np.float64)
    for i, v in enumerate(values):
        if v is None:
            continue
        a = np.asarray(v, dtype=np.float64)
        n = min(L, a.shape[0])
        out[i, :n] = a[:n]
    return out


def _winsorize_per_position(y, lo_q=0.01, hi_q=0.99):
    """
    Clip each position (column) of y to [q_lo, q_hi] computed ignoring NaNs.
    """
    q_lo = np.nanquantile(y, lo_q, axis=0)
    q_hi = np.nanquantile(y, hi_q, axis=0)
    q_lo = np.where(np.isfinite(q_lo), q_lo, -1e9)
    q_hi = np.where(np.isfinite(q_hi), q_hi, +1e9)
    return np.clip(y, q_lo, q_hi)


error_col_map = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

ERR_FLOOR = 1e-4  # prevents divide-by-zero / overweighting tiny reported errors

pos_means = {}
for col in target_cols:
    y = _stack_to_len(train[col].values, seq_scored)  # (n_samples, seq_scored)

    y = _winsorize_per_position(y, lo_q=0.01, hi_q=0.99)

    err_col = error_col_map.get(col)
    if err_col is not None and err_col in train.columns:
        e = _stack_to_len(train[err_col].values, seq_scored)
        e = np.where(np.isfinite(e), e, np.nan)
        e = np.maximum(e, ERR_FLOOR)

        w = 1.0 / (e * e)

        num = np.nansum(w * y, axis=0)
        den = np.nansum(w, axis=0)
        m = np.divide(num, den, out=np.zeros_like(num), where=(den > 0))

        wsum = np.nansum(w)
        global_mean = (
            float(np.nansum(w * y) / wsum) if wsum > 0 else float(np.nanmean(y))
        )
    else:
        m = np.nanmean(y, axis=0)
        global_mean = float(np.nanmean(y))

    m = np.where(np.isfinite(m), m, 0.0)
    if not np.isfinite(global_mean):
        global_mean = 0.0

    alpha = 0.02
    m = (1.0 - alpha) * m + alpha * global_mean

    pos_means[col] = m

{k: (v.shape, float(np.min(v)), float(np.max(v))) for k, v in pos_means.items()}



## === cell 3
ids = sub["id_seqpos"].astype(str).values
seqpos = np.array([int(x.rsplit("_", 1)[1]) for x in ids], dtype=np.int64)

pred = pd.DataFrame({"id_seqpos": ids})
for col in target_cols:
    m = pos_means[col]
    full = np.empty_like(seqpos, dtype=np.float64)

    mask_scored = seqpos < seq_scored
    full[mask_scored] = m[seqpos[mask_scored]]
    full[~mask_scored] = m[-1]

    pred[col] = full

pred = pred[expected_cols].copy()
pred.head()



## === cell 4
if pred.shape[0] != sub.shape[0]:
    raise ValueError(
        f"Row count mismatch vs sample_submission: {pred.shape[0]} vs {sub.shape[0]}"
    )
if list(pred.columns) != expected_cols:
    raise ValueError(
        f"Column mismatch: got {list(pred.columns)} expected {expected_cols}"
    )
if pred["id_seqpos"].duplicated().any():
    raise ValueError("Duplicate id_seqpos found in submission")
if not (pred["id_seqpos"].values == sub["id_seqpos"].values).all():
    raise ValueError(
        "id_seqpos order mismatch vs sample_submission (would misalign predictions)"
    )

pred_cols = expected_cols[1:]
pred[pred_cols] = pred[pred_cols].astype(np.float64)
pred[pred_cols] = pred[pred_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

pred.describe()



## === cell 5
pred.to_csv("submission.csv", index=False)

with open("submission.csv", "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))
