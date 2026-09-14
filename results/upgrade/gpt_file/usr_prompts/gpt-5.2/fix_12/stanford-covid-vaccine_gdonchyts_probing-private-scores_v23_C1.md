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

0.3677278275718549

# 6. Current score

0.42286

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'Your notebook fails because it tries to load a non-existent `../input/worst-submission/ensemble52.csv`, so `df` is never created and all later cells error. To keep the core “calibration-by-scaling” logic intact, I instead read the provided competition `sample_submission.csv` and use it as the base submission template. Since there is no model here to generate predictions, we create a valid, deterministic baseline by filling predictions with zeros (a safe default) and then apply your existing `/= 1.15` scaling step (which is score-neutral here but preserves your intended logic). Finally, we write `submission.csv` with the exact required columns.'
- What this solution (achieved 0.43073) has done: 'Your current score is far worse than the target (lower is better), and the main issue is that the notebook produces an “all zeros” submission, which is a very weak baseline. To improve toward the target with minimal core-logic disruption and without adding any model training, I keep your exact “template + optional scaling + write submission.csv” structure but replace the zero-filling with a simple, legitimate training-data prior: predict the per-position mean from `train.json` for each target. This typically gives a large MCRMSE improvement versus zeros because it matches the marginal distribution and position effects. I also keep predictions constant for all `id`s (no leakage) and still output all 107 positions as required by the submission format.'
- What this solution (achieved 0.42592) has done: 'Your current baseline uses per-position means from all training rows, but this competition’s train set contains many low-quality (low signal/noise) samples that can hurt those priors; using the provided `SN_filter==1` subset is a minimal, legitimate way to make the same “position-mean prior” cleaner and should move MCRMSE down toward your target. I keep your exact pipeline structure (template → compute per-position priors → fill submission → optional scaling → write CSV) and only change the mean-estimation step to prefer filtered data when available, with a safe fallback to all rows if needed. I also make the scaling act only on prediction columns (not `id_seqpos`) to avoid any accidental dtype/coercion issues while preserving your calibration intent. The output remains a valid `submission.csv` with the required columns and 25680 rows.'
- What this solution (achieved 0.49589) has done: 'To move your MCRMSE down toward the target with minimal disruption, I keep your exact “position-mean prior from train → fill sample_submission template → optional scaling → write submission.csv” core logic. The only scoring-relevant tweak is to compute the per-position means as *inverse-variance weighted means* using the provided per-target error arrays (still using `SN_filter==1` when available), which is a small but legitimate refinement of the same prior and typically improves this competition’s metric. I also make the scaling factor optional but default to your existing 1.15 so evaluation semantics remain the same unless you later decide to tune it. The output remains a valid `submission.csv` with the exact required columns and row count.'
- What this solution (achieved 0.46109) has done: 'Your current score (0.49589, lower is better) is worse than the target (0.3677), so we should improve it with the smallest legitimate change while keeping your “position-prior → fill sample_submission → scale → write” logic intact. The most likely issue is that inverse-variance weighting can overfit to noisy/underestimated error values; a minimal fix is to add a small error-floor and (optionally) lightly shrink the weighted means toward the unweighted means to stabilize the prior without changing the overall approach. This keeps the same data sources, targets, and submission semantics, but typically reduces MCRMSE by avoiding extreme weights. Everything else (template loading, per-position fill for 107 positions, scaling, and CSV output) remains the same.'
- What this solution (achieved 0.42567) has done: 'Your current solution is a “position prior” baseline, but it still uses all sequences equally; in this competition, downweighting noisy/low-SNR samples usually improves MCRMSE without changing the core approach. I keep the exact pipeline (load template → compute per-position priors from train → fill all 107 positions → scale → write submission.csv) but change the mean computation to use **signal_to_noise-based sample weights** (with SN_filter==1 still preferred) and use a **stable, per-position weighted mean** (no inverse-variance error weighting, which can create extreme weights and hurt). This is a minimal scoring-relevant change that tends to move your 0.46109 closer to the 0.3677 target. Everything else (columns, row alignment by seqpos, scaling factor, output file) stays identical.'
- What this solution (achieved 0.4256) has done: 'Your current baseline is already close to the best you can get with a pure “position prior”, so the most likely remaining gap to the 0.3677 target is from a slight mismatch between how we compute the position means and the evaluation distribution. To improve with minimal logic change, I keep your exact pipeline (template → compute per-position priors from train → fill → scale → write), but (1) use a more standard SNR weighting (linear `signal_to_noise` instead of `sqrt`) to better downweight noisy samples, and (2) add a tiny amount of shrinkage of the weighted mean toward the unweighted mean to stabilize against overweighting a subset. Everything else (same data sources, same targets, same fill for 107 positions, same scaling step, same submission schema) remains the same.'
- What this solution (achieved 0.43093) has done: 'To move your score down toward the 0.3677 target while keeping the same “position prior → fill sample_submission → scale → write” core logic, I’m only changing how the position prior is estimated. Specifically, I (1) compute the prior using **only the first 68 scored positions** (unchanged) but with a **robust per-position trimmed mean** (to reduce the impact of outlier/noisy train rows that remain even after `SN_filter==1`), and (2) keep your existing SNR-based weighting + small shrinkage, but apply trimming before weighting to stabilize the weighted mean. Everything else (paths, columns, seqpos alignment, scaling by 1.15, and writing `submission.csv`) remains identical.'
- What this solution (achieved 0.42528) has done: 'Your current score (0.43093, lower is better) is worse than the target (0.3677), so we should make a small, low-risk improvement to the same “position prior → fill template → scale → write” pipeline. The biggest lever left without changing core logic is the global scaling: your fixed `SCALE=1.15` is likely miscalibrated for this prior, and a slightly different value can reduce MCRMSE without changing the prediction shape or data usage. I keep everything else identical, but tune the single scalar `SCALE` using an internal cross-validated objective on the training set (only the 3 scored targets, positions 0–67), then apply that tuned scale to the submission. This preserves your model-free prior approach and training semantics, just calibrating it more appropriately to the metric.'
- What this solution (achieved 0.42286) has done: 'We keep your exact “position prior → fill sample_submission → scale → write” pipeline, but make the scale calibration match the metric more closely by optimizing it via a closed-form least-squares fit on the scored targets (reactivity, deg_Mg_pH10, deg_Mg_50C) instead of a coarse grid. This is a minimal change (only the SCALE selection) that typically nudges MCRMSE down without changing your prior construction, trimming, or weighting logic. We also compute that scale using the same robust HQ subset you already use and clip it to a conservative range to avoid destabilizing the submission. Everything else (paths, columns, row alignment, 107-position fill, and writing `submission.csv`) remains the same.'
- What this solution (achieved 0.42286) has done: 'Your current score (0.42286, lower-is-better) is still worse than the target (0.36773), so we should make a small improvement without changing the overall “position prior → fill sample_submission → global scale → write” pipeline. The least disruptive lever left is to make the global scaling match the competition metric more closely by fitting **separate** closed-form scales per scored target (reactivity, deg_Mg_pH10, deg_Mg_50C) instead of one shared scale; this keeps identical prediction shape and semantics but improves calibration. We keep your existing robust/weighted prior construction unchanged, only changing the scaling step and leaving unscored targets at scale=1.0 to avoid unnecessary drift. The output remains a valid `submission.csv` with the required columns and row count.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
possible_paths = [
    "/kaggle/input/sample_submission.csv",
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
]

sample_path = None
for p in possible_paths:
    if os.path.exists(p):
        sample_path = p
        break

if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected Kaggle locations. "
        f"Tried: {possible_paths}"
    )

df = pd.read_csv(sample_path)
df.head()



## === cell 2
train_paths = [
    "/kaggle/input/train.json",
    "/kaggle/input/stanford-covid-vaccine/train.json",
    "/kaggle/data/train.json",
    "/kaggle/data/stanford-covid-vaccine/train.json",
]
train_path = None
for p in train_paths:
    if os.path.exists(p):
        train_path = p
        break
if train_path is None:
    raise FileNotFoundError(
        "Could not find train.json in expected Kaggle locations. "
        f"Tried: {train_paths}"
    )

train = pd.read_json(train_path, lines=True)

targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
seq_scored = int(train["seq_scored"].iloc[0])  # 68 in this competition
seq_length = int(train["seq_length"].iloc[0])  # 107 in this competition

if "SN_filter" in train.columns:
    train_hq = train[train["SN_filter"] == 1].copy()
    if len(train_hq) < 100:  # safety fallback in case of unexpected data issues
        train_hq = train
else:
    train_hq = train

if "signal_to_noise" in train_hq.columns:
    sn = train_hq["signal_to_noise"].to_numpy(dtype=np.float32)
    sn = np.nan_to_num(
        sn, nan=np.float32(0.0), posinf=np.float32(0.0), neginf=np.float32(0.0)
    )
    sn = np.clip(sn, np.float32(0.0), np.float32(10.0))
    sample_w = (sn + np.float32(1e-3)).astype(np.float32)  # (n,)
else:
    sample_w = np.ones(len(train_hq), dtype=np.float32)


def _trimmed_mean_per_pos(y_2d: np.ndarray, trim_q: float = 0.08) -> np.ndarray:
    """
    y_2d: (n, 68) float32 with possible NaNs.
    Returns per-position trimmed mean (68,) float32.
    """
    y = y_2d.astype(np.float32, copy=False)
    lo = np.nanquantile(y, trim_q, axis=0).astype(np.float32)
    hi = np.nanquantile(y, 1.0 - trim_q, axis=0).astype(np.float32)
    keep = (y >= lo[None, :]) & (y <= hi[None, :]) & np.isfinite(y)
    y_kept = np.where(keep, y, np.nan)
    m = np.nanmean(y_kept, axis=0).astype(np.float32)
    m_fallback = np.nanmean(y, axis=0).astype(np.float32)
    m = np.where(np.isfinite(m), m, m_fallback).astype(np.float32)
    return m


pos_means = {}
SHRINK_ALPHA = np.float32(0.08)  # keep your existing small stabilization
TRIM_Q = 0.08  # minimal, conservative trimming

for t in targets:
    y_list = train_hq[t].values
    y = np.stack(y_list).astype(np.float32)  # (n, 68)

    m_u_robust = _trimmed_mean_per_pos(y, trim_q=TRIM_Q)

    lo = np.nanquantile(y, TRIM_Q, axis=0).astype(np.float32)
    hi = np.nanquantile(y, 1.0 - TRIM_Q, axis=0).astype(np.float32)
    y_wins = np.clip(y, lo[None, :], hi[None, :]).astype(np.float32)

    y_is_nan = ~np.isfinite(y_wins)
    if y_is_nan.any():
        y_wins = np.nan_to_num(
            y_wins, nan=np.float32(0.0), posinf=np.float32(0.0), neginf=np.float32(0.0)
        )

    w = sample_w[:, None].astype(np.float32)  # (n,1) -> broadcast to (n,68)
    if y_is_nan.any():
        w = w.copy()
        w[y_is_nan] = np.float32(0.0)

    w_sum = np.sum(w, axis=0)  # (68,)
    with np.errstate(divide="ignore", invalid="ignore"):
        m_w = np.sum(w * y_wins, axis=0) / w_sum  # weighted mean

    if np.any(~np.isfinite(m_w)):
        m_w = np.where(np.isfinite(m_w), m_w, m_u_robust)

    m = (np.float32(1.0) - SHRINK_ALPHA) * m_w + SHRINK_ALPHA * m_u_robust
    pos_means[t] = m.astype(np.float32)

pos_full = {}
for t in targets:
    full = np.empty(seq_length, dtype=np.float32)
    full[:seq_scored] = pos_means[t]
    full[seq_scored:] = pos_means[t][-1]
    pos_full[t] = full



## === cell 3
scored_targets = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

Y_true = np.stack(
    [np.stack(train_hq[t].values).astype(np.float32) for t in scored_targets], axis=-1
)  # (n,68,3)

Y_pred_base = np.stack(
    [pos_means[t].astype(np.float32) for t in scored_targets], axis=-1
)[
    None, :, :
]  # (1,68,3)
Y_pred_base = np.repeat(Y_pred_base, repeats=Y_true.shape[0], axis=0)  # (n,68,3)

best_scales = {}
for k, t in enumerate(scored_targets):
    y = Y_true[:, :, k].reshape(-1).astype(np.float64)
    b = Y_pred_base[:, :, k].reshape(-1).astype(np.float64)

    den = float(np.dot(b, b))
    if den <= 0.0 or not np.isfinite(den):
        s_opt = 1.15
    else:
        z = float(np.dot(y, b) / den)  # z = 1/s
        if not np.isfinite(z) or z == 0.0:
            s_opt = 1.15
        else:
            s_opt = float(1.0 / z)

    s_opt = float(np.clip(s_opt, 0.9, 1.4))
    best_scales[t] = np.float32(s_opt)

SCALE = float(np.mean([best_scales[t] for t in scored_targets]))


def mcrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = np.mean((y_true - y_pred) ** 2, axis=0)  # (68,3)
    rmse_per_target = np.sqrt(np.mean(mse, axis=0))  # (3,)
    return float(np.mean(rmse_per_target))


Y_pred_scaled = Y_pred_base.copy().astype(np.float32)
for k, t in enumerate(scored_targets):
    Y_pred_scaled[:, :, k] = Y_pred_scaled[:, :, k] / best_scales[t]

cv_score = mcrmse(Y_true, Y_pred_scaled)
print(
    "Chosen per-target SCALE via closed-form LS (scored targets only): "
    + ", ".join([f"{t}={float(best_scales[t]):.4f}" for t in scored_targets])
    + f" (mean={SCALE:.4f}; train MCRMSE={cv_score:.6f})"
)



## === cell 4
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
    raise ValueError(
        f"Missing required columns in sample submission template: {missing}"
    )

df = df[required_cols].copy()

seqpos = df["id_seqpos"].str.rsplit("_", n=1).str[-1].astype(int).to_numpy()
for t in targets:
    df[t] = pos_full[t][seqpos]



## === cell 5
per_col_scale = {
    "reactivity": best_scales["reactivity"],
    "deg_Mg_pH10": best_scales["deg_Mg_pH10"],
    "deg_Mg_50C": best_scales["deg_Mg_50C"],
    "deg_pH10": np.float32(1.0),
    "deg_50C": np.float32(1.0),
}
for c in required_cols[1:]:
    df[c] = df[c].astype(np.float32) / per_col_scale[c]



## === cell 6
if df.isna().any().any():
    raise ValueError("Submission contains NaNs after processing.")

for c in required_cols[1:]:
    df[c] = df[c].astype(np.float32)

df.head()



## === cell 7
out_path = "submission.csv"
df.to_csv(out_path, index=False)

print(f"Wrote {out_path} with shape {df.shape} and columns {list(df.columns)}")
print(df.head(3))
