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

0.43536

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42418) has done: 'I remove the dependency on the missing external baseline file and instead start from the provided `sample_submission.csv`, which guarantees the correct row count and column names. Then I build a simple, deterministic baseline prediction by taking per-position means of the training targets (for each of the 5 output columns) and filling the submission accordingly; this runs end-to-end with only numpy/pandas and yields a valid, nontrivial score (instead of failing to produce a CSV). I also keep your later “override a specific id’s reactivity to 100” step intact but guard it so it doesn’t crash if the id isn’t present. Finally, I ensure `submission.csv` is written in the working directory.'
- What this solution (achieved 0.4437) has done: 'We keep your “per-position mean” baseline intact, but reduce error by (1) filtering training rows to those that pass `SN_filter==1` (the competition’s high-quality subset), and (2) using per-position medians instead of means to be more robust to outliers/negative/noisy measurements. We also make the tail (positions 68–106) fill value be the median of the last few scored positions instead of only the last position, which slightly stabilizes extrapolation while preserving the same core approach. Finally, we keep your id-based override but set it to a neutral value (the learned baseline) rather than an extreme 100.0, because the current override almost certainly worsens MCRMSE.'
- What this solution (achieved 0.4437) has done: 'We keep your per-position median baseline and SN_filter==1 filtering (same core logic), but make two small, metric-aligned tweaks to reduce MCRMSE toward the target: (1) clip predictions to a reasonable range learned from the filtered training targets (to avoid occasional extreme medians/tail fills hurting RMSE), and (2) fill the unscored tail (positions 68–106) with a more stable statistic: the median of the last `tail_k` medians (as you do) but also clipped by the same learned bounds. These changes don’t alter the model approach, only post-processing/calibration, and should improve score without adding dependencies. The submission format/row alignment remains driven by `sample_submission.csv`, so it stay valid.'
- What this solution (achieved 0.4497) has done: 'Your current baseline is already valid but is likely underperforming because it uses a single global per-position statistic and then extrapolates the unscored tail with a constant value. To move the MCRMSE down toward the target with minimal semantic change, I keep the same “per-position robust center” approach but switch from pure median to a small shrinkage blend (median + mean) to better approximate squared-error optimal predictions while remaining robust. I also make the clipping bounds slightly less aggressive (0.5%–99.5%) so we don’t over-clip and bias predictions, and I smooth the per-position centers with a tiny window to reduce noisy position-to-position jitter that hurts RMSE. The submission format, row alignment via `sample_submission.csv`, and your guarded id override are kept intact.'
- What this solution (achieved 0.4437) has done: 'We keep your same per-position robust-center baseline, but remove the extra smoothing and the mean/median blending because those two tweaks likely biased the per-position estimates away from the squared-error optimum for this dataset and worsened MCRMSE versus a pure robust center. We also revert the clipping to be a bit more protective against outliers (1%–99%) while still learned from `SN_filter==1` data, since extreme values can disproportionately hurt RMSE. Finally, we keep your tail-fill logic and guarded id-override intact, only ensuring predictions remain clipped and aligned to `sample_submission.csv` for a valid submission.'
- What this solution (achieved 0.42166) has done: 'Your current score (0.4437, lower-is-better) is still far from the target (0.3519), so we need a small but meaningful improvement while keeping the same per-position-statistic baseline. I keep your SN_filter==1 filtering and per-position robust centers, but switch the estimator from per-position median to per-position mean (the squared-error optimal center) while still clipping to quantile-learned bounds for outlier protection. I also change the unscored tail (positions 68–106) fill from a constant to a simple linear extrapolation based on the last few scored positions’ trend, which is still the same “use per-position aggregate stats” core logic but usually reduces error versus a flat tail. Submission alignment via `sample_submission.csv` and your guarded id override remain intact.'
- What this solution (achieved 0.4804) has done: 'Your current baseline is a per-position mean with clipping plus a linear tail extrapolation, which is still quite far from the target (need to *lower* MCRMSE). To improve toward the target while keeping the same “per-position aggregate statistics” core logic, I (1) compute the tail trend using a least-squares line fit over the last `tail_k` scored positions (more stable than the current covariance-based slope), and (2) blend that extrapolated tail with a constant “last-k average” tail anchor to reduce extrapolation overshoot while preserving the same approach. I also compute the centers as a simple weighted mean using the provided per-position measurement errors as inverse-variance weights, which is still the same per-position mean idea but better aligned to RMSE under heteroskedastic noise. Submission alignment via `sample_submission.csv` and your guarded id override remain intact, and the script still writes `submission.csv` end-to-end.'
- What this solution (achieved 0.44317) has done: 'To move your MCRMSE down toward the target while preserving the same “per-position aggregate statistics + tail extrapolation + clipping” core logic, I make two small, metric-aligned changes. First, I shrink your inverse-variance weighting slightly (use 1/(e+eps) instead of 1/(e+eps)^2) so extremely small reported errors don’t over-dominate and destabilize the per-position mean. Second, I switch the tail slope fit to be *anchored at the last scored point* (fit on y - y_last vs steps), which reduces bias/overshoot in the extrapolated tail while keeping the same linear trend approach and deterministic blending with the anchor. Everything else (SN_filter==1 filtering, clipping by quantiles, submission alignment to sample_submission.csv, and the guarded override) stays intact and the script still writes `submission.csv` end-to-end.'
- What this solution (achieved 0.40285) has done: 'Your current baseline is already valid and deterministic, but it likely underperforms because it ignores simple sample-level stratification that correlates with targets (e.g., loop context) while still using the same per-position aggregate-statistics approach. I keep your exact training/prediction flow (SN_filter==1 filtering, inverse-error weighted per-position centers, linear tail extrapolation, clipping, and submission alignment), and add one minimal refinement: compute per-position centers *separately by predicted loop type (S/M/I/B/H/E/X)* and use the appropriate center for each test row’s loop at that position, falling back to the global center when a loop group is missing. This preserves the “aggregate stats per position” core logic (just conditioned on an existing provided categorical feature) and should reduce MCRMSE toward the target without changing the metric semantics. The tail logic and clipping remain identical, just applied per loop-group center.'
- What this solution (achieved 0.43536) has done: 'We need to lower MCRMSE from 0.40285 toward 0.35188, so we keep your exact “per-position aggregate stats + loop-type conditioning + tail extrapolation + clipping + sample_submission alignment” pipeline, but make the weighting more robust and better aligned to squared error. Specifically, we (1) switch from inverse-error to inverse-variance weighting (1/(err^2)) with a floor on errors to prevent a few tiny errors from dominating, and (2) add a very small empirical-Bayes shrinkage of each loop-specific per-position center toward the global center based on that loop’s effective sample size per position (stabilizes sparse loop categories). Both changes preserve the same core logic (still deterministic per-position weighted means, just more stable) and should reduce RMSE without changing I/O or submission format.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd



## === cell 1
df_train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)
df_test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)

df = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")

TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
ERRORS = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}



## === cell 2
df_train_f = df_train[df_train["SN_filter"] == 1].reset_index(drop=True)
if len(df_train_f) == 0:
    df_train_f = df_train.copy()

seq_scored = int(df_train_f["seq_scored"].iloc[0])  # expected 68
seq_len = int(df_train_f["seq_length"].iloc[0])  # expected 107

clip_q_lo, clip_q_hi = 0.01, 0.99

clip_bounds = {}
for t in TARGETS:
    arr = np.stack(df_train_f[t].values).astype(np.float32)  # (n_samples, 68)
    flat = arr.reshape(-1)
    lo = float(np.quantile(flat, clip_q_lo))
    hi = float(np.quantile(flat, clip_q_hi))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or lo >= hi:
        lo, hi = float(np.min(flat)), float(np.max(flat))
    clip_bounds[t] = (lo, hi)

eps = 1e-6
err_floor = 0.02  # small, stabilizing floor in target units; keeps core weighted-mean logic intact

loop_vocab = list("SMIBHEX")  # observed bpRNA loop types in this competition
pos_center_by_loop = {t: {} for t in TARGETS}
pos_effn_by_loop = {
    t: {} for t in TARGETS
}  # effective sample size per position (for shrinkage)

train_loop_str = df_train_f["predicted_loop_type"].astype(str).values

for t in TARGETS:
    y_all = np.stack(df_train_f[t].values).astype(np.float32)  # (n, 68)
    e_all = np.stack(df_train_f[ERRORS[t]].values).astype(np.float32)  # (n, 68)
    e_all = np.maximum(e_all, err_floor).astype(np.float32)

    lo, hi = clip_bounds[t]

    for L in loop_vocab:
        mask = np.array(
            [[ch == L for ch in s[:seq_scored]] for s in train_loop_str], dtype=bool
        )
        if not mask.any():
            continue

        w = 1.0 / (e_all * e_all + eps)  # inverse-variance
        wy = w * y_all

        w_sum = (w * mask).sum(axis=0)
        wy_sum = (wy * mask).sum(axis=0)

        mu = np.full((seq_scored,), np.nan, dtype=np.float32)
        ok = w_sum > 0
        mu[ok] = (wy_sum[ok] / (w_sum[ok] + 1e-12)).astype(np.float32)
        mu = np.clip(mu, lo, hi).astype(np.float32)
        pos_center_by_loop[t][L] = mu

        w2_sum = ((w * mask) ** 2).sum(axis=0)
        effn = np.zeros((seq_scored,), dtype=np.float32)
        ok2 = (w_sum > 0) & (w2_sum > 0)
        effn[ok2] = (w_sum[ok2] ** 2) / (w2_sum[ok2] + 1e-12)
        pos_effn_by_loop[t][L] = effn

pos_center_global = {}
for t in TARGETS:
    y = np.stack(df_train_f[t].values).astype(np.float32)  # (n, 68)
    e = np.stack(df_train_f[ERRORS[t]].values).astype(np.float32)  # (n, 68)
    e = np.maximum(e, err_floor).astype(np.float32)

    w = 1.0 / (e * e + eps)  # inverse-variance
    mu = (w * y).sum(axis=0) / (w.sum(axis=0) + 1e-12)

    lo, hi = clip_bounds[t]
    pos_center_global[t] = np.clip(mu, lo, hi).astype(np.float32)

full_center_by_loop = {t: {} for t in TARGETS}
full_center_global = {}

tail_k = 5
alpha_trend = 0.65  # unchanged


def build_full_from_scored(
    y_scored, lo, hi, seq_len, seq_scored, tail_k=5, alpha_trend=0.65
):
    y = y_scored.astype(np.float32)
    k = int(min(tail_k, len(y)))
    tail_slice = y[-k:].astype(np.float32)

    y_last = float(y[-1])
    steps = np.arange(-(k - 1), 1, dtype=np.float32)  # [-k+1,...,0]
    dy = tail_slice - y_last
    denom = float((steps**2).sum())
    if denom > 0:
        b = float((steps * dy).sum() / denom)
    else:
        b = 0.0

    tail_steps = np.arange(1, (seq_len - seq_scored) + 1, dtype=np.float32)
    tail_trend = (y_last + b * tail_steps).astype(np.float32)

    anchor = float(tail_slice.mean())
    tail_anchor = np.full_like(tail_trend, anchor, dtype=np.float32)

    tail = (alpha_trend * tail_trend + (1.0 - alpha_trend) * tail_anchor).astype(
        np.float32
    )
    tail = np.clip(tail, lo, hi).astype(np.float32)
    return np.concatenate([y, tail]).astype(np.float32)


for t in TARGETS:
    lo, hi = clip_bounds[t]

    full_center_global[t] = build_full_from_scored(
        pos_center_global[t],
        lo,
        hi,
        seq_len,
        seq_scored,
        tail_k=tail_k,
        alpha_trend=alpha_trend,
    )

    tau = 8.0  # small prior strength in "effective samples"; keeps loop conditioning but reduces noise
    for L, mu in pos_center_by_loop[t].items():
        mu_filled = mu.copy()
        nan_mask = ~np.isfinite(mu_filled)
        if nan_mask.any():
            mu_filled[nan_mask] = pos_center_global[t][nan_mask]

        effn = pos_effn_by_loop[t].get(L, np.zeros((seq_scored,), dtype=np.float32))
        shrink = (effn / (effn + tau)).astype(np.float32)  # in [0,1]
        mu_shrunk = (shrink * mu_filled + (1.0 - shrink) * pos_center_global[t]).astype(
            np.float32
        )
        mu_shrunk = np.clip(mu_shrunk, lo, hi).astype(np.float32)

        full_center_by_loop[t][L] = build_full_from_scored(
            mu_shrunk,
            lo,
            hi,
            seq_len,
            seq_scored,
            tail_k=tail_k,
            alpha_trend=alpha_trend,
        )



## === cell 3
seqpos = df["id_seqpos"].str.rsplit("_", n=1).str[-1].astype(int).values

test_id_to_loop = dict(
    zip(df_test["id"].values, df_test["predicted_loop_type"].astype(str).values)
)

ids = df["id_seqpos"].str.rsplit("_", n=1).str[0].values
loops = np.array([test_id_to_loop.get(i, "") for i in ids], dtype=object)

for t in TARGETS:
    lo, hi = clip_bounds[t]
    preds = np.empty(len(df), dtype=np.float32)

    preds[:] = full_center_global[t][seqpos]

    for L, fullL in full_center_by_loop[t].items():
        m = np.fromiter(
            (len(s) > sp and s[sp] == L for s, sp in zip(loops, seqpos)),
            dtype=bool,
            count=len(df),
        )
        if m.any():
            preds[m] = fullL[seqpos[m]]

    df[t] = np.clip(preds, lo, hi)



## === cell 4
sequences = list(set(df_test[df_test.seq_length != 130].id))
sequences.sort()



## === cell 5
sequences[-10:]



## === cell 6
mask_check = df.id_seqpos.str.startswith("id_0fd2f39fa")
df.loc[mask_check, "reactivity"].values[:5] if mask_check.any() else np.array([])



## === cell 7
mask_override = df.id_seqpos.str.startswith("id_fb7dfccef")
if mask_override.any():
    lo, hi = clip_bounds["reactivity"]
    df.loc[mask_override, "reactivity"] = np.clip(
        df.loc[mask_override, "reactivity"].values.astype(np.float32), lo, hi
    )



## === cell 8
df.loc[mask_override, "reactivity"].values[:5] if mask_override.any() else np.array([])



## === cell 9
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print("Columns:", list(df.columns))
