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

0.42796

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I remove the dependency on a missing external CSV (`../input/worst-submission/ensemble52.csv`) and instead build the submission from the provided `sample_submission.csv`, which guarantees the required row count and columns. I keep the original intent of the notebook (a simple post-processing edit) by applying the same `id_c5e79dcb9` reactivity override on the base submission. I also fix the cell numbering to be valid and add small safety checks so the script always writes a correct `submission.csv` even if some expected IDs are absent. This is primarily a correctness/runtime fix; it should yield a valid (not necessarily good) score rather than failing to submit.'
- What this solution (achieved 0.42166) has done: 'Your current approach is essentially submitting the sample submission (all zeros) with one manual reactivity override, which explains the weak score; the smallest legitimate improvement is to replace the zeros with a simple per-position baseline learned from the training targets (no architecture/training loop changes needed). We compute the mean target value at each of the first 68 positions across training (optionally filtered by `SN_filter==1` to better match test quality), then fill the submission rows accordingly by parsing `seqpos` from `id_seqpos`. For positions > 67 (not scored), we safely fill with 0.0 to keep the file valid. This keeps your “post-processing submission builder” core logic intact while moving the score down toward the target.'
- What this solution (achieved 0.48245) has done: 'We keep your “learn a simple baseline from train then fill the sample_submission” core logic, but make two small, metric-relevant adjustments to better match the public test distribution and reduce MCRMSE toward the target. First, we compute position-wise means using **sample weights derived from the provided per-position measurement errors**, which is a legitimate way to downweight noisy labels without changing the modeling approach. Second, because only 3 targets are scored, we avoid injecting unnecessary noise into the unscored targets by setting them to a conservative constant (the global mean) instead of position means; this tends to slightly reduce overall instability while keeping semantics identical for the scored columns.'
- What this solution (achieved 0.48245) has done: 'We keep your “position-wise baseline from train + fill sample_submission” core logic, but adjust it in a metric-relevant way to better match what’s actually scored. Specifically, we compute the position-wise means using only the first `seq_scored` positions and only for the 3 scored targets, and we stop forcing the unscored targets to a global mean (we instead fill them with the same position-wise means as the scored ones, which reduces distribution shift and typically helps MCRMSE indirectly). We also remove the hardcoded `id_c5e79dcb9` reactivity override because it injects arbitrary error into a scored column and is very likely harming MCRMSE. All I/O paths and submission formatting/row-count checks remain unchanged.'
- What this solution (achieved 0.4488) has done: 'Your current baseline is already valid but is likely underfitting because it predicts only simple (weighted) per-position means and uses a single global fallback for non-scored positions, which can create distribution mismatch. To move the (lower-is-better) MCRMSE down toward your target with minimal logic change, I keep the same “learn from train then fill sample_submission” approach but (1) blend the error-weighted mean with the unweighted mean to reduce overconfidence in potentially miscalibrated error weights, and (2) use per-position means for all 107 positions by extending positions 68–106 with the last scored position’s mean (rather than a single global constant). These are small, metric-relevant post-processing changes that preserve your core method and should improve generalization without changing I/O paths or the submission schema. The script still run end-to-end and write a correct `submission.csv`.'
- What this solution (achieved 0.45066) has done: 'We keep your exact “learn per-position baseline from train then fill `sample_submission.csv`” logic, but make two small, metric-relevant fixes that usually reduce MCRMSE on this competition without changing the approach. First, we compute the position means only for the scored positions (0–67) as you already do, but we also **clip training targets to a reasonable range** (based on training quantiles) before averaging to reduce the impact of extreme/noisy outliers that can skew a mean baseline. Second, instead of extending positions 68–106 with the last scored value (which can create a hard step), we extend using the **global mean** for that target, which is typically a safer, lower-variance fill for unscored positions while keeping submission validity. These are minimal changes, preserve your core method, and are aimed at moving the score down from 0.4488 toward the 0.3519 target.'
- What this solution (achieved 0.4428) has done: 'We keep your “position-wise baseline learned from train then filled into `sample_submission.csv`” core logic intact, but make two small, metric-relevant adjustments aimed at reducing the MCRMSE from 0.45066 toward 0.35188. First, we compute the per-position averages using a robust **Huber-style iterative reweighting** on residuals (on top of your existing error-based weights), which reduces the influence of mislabeled/outlier positions without changing the fundamental averaging approach. Second, we tune the blend between weighted and unweighted means slightly toward stability (less dependence on error weights), which often helps when error magnitudes are imperfectly calibrated. All paths, output schema, and the end-to-end submission writing remain unchanged.'
- What this solution (achieved 0.43599) has done: 'We keep your exact “learn per-position baseline from train then fill `sample_submission.csv`” approach, but tighten it in two metric-relevant, minimal ways to move MCRMSE down from 0.4428 toward 0.3519. First, we compute means using only the three scored targets as “drivers” by calibrating the unscored targets (`deg_pH10`, `deg_50C`) via a simple linear shrinkage toward the scored analogs (`deg_Mg_pH10`, `deg_Mg_50C`), reducing unnecessary error propagation without changing any training loop/model. Second, we slightly retune the robustness/weight blending to rely a bit more on stable unweighted means (less sensitivity to imperfect error calibration) while keeping your robust IRLS mean computation intact. Output paths, schema, and row alignment remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.43759) has done: 'We’re currently worse than the target (0.43599 vs 0.35188, lower is better), so we want a small, legitimate improvement without changing the “position-wise baseline from train then fill sample_submission” core logic. The biggest low-risk gain here is to stop estimating the per-position means on a noisy subset: use all training rows but weight them by `signal_to_noise` (and keep your existing per-position error weights + IRLS), which better matches the test distribution while preserving the same averaging approach. Second, because only the first 68 positions are scored, we reduce unnecessary distribution shift by filling positions 68–106 with the average of the last few scored positions (a smoother continuation than a single global mean) for all targets. These two changes are minimal, metric-relevant, and should move MCRMSE down toward your target band.'
- What this solution (achieved 0.42796) has done: 'We’re currently worse than the target (0.43759 vs 0.35188, lower is better), so we want a small, legitimate improvement without changing your core “robust per-position averaging + fill sample_submission” logic. The most likely low-risk gain is to compute the per-position baseline using only higher-quality training rows (`SN_filter==1`) to better match the test set filtering, while falling back to all rows if that filter would remove too much data. Second, we slightly retune the two stability knobs you already have (blend toward unweighted mean and IRLS Huber cutoff) to reduce sensitivity to imperfect error/SNR calibration—this keeps the same approach but often lowers MCRMSE for this competition. All paths, submission schema, and the end-to-end CSV writing remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
train_path = "../input/stanford-covid-vaccine/train.json"
df_train = pd.read_json(train_path, lines=True)

if "SN_filter" in df_train.columns:
    df_hq = df_train[df_train["SN_filter"].astype(int) == 1].copy()
    df_train_use = df_hq if len(df_hq) >= 500 else df_train.copy()
else:
    df_train_use = df_train.copy()

if "signal_to_noise" in df_train_use.columns:
    snr = (
        pd.to_numeric(df_train_use["signal_to_noise"], errors="coerce")
        .fillna(1.0)
        .values.astype(np.float32)
    )
    snr = np.clip(snr, 0.0, 10.0)  # cap extreme leverage
    snr_w = snr / (np.mean(snr) + 1e-6)  # normalize around 1
else:
    snr_w = np.ones(len(df_train_use), dtype=np.float32)



## === cell 2
sample_sub_path = "../input/stanford-covid-vaccine/sample_submission.csv"
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



## === cell 3
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
error_cols = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

seq_scored = (
    int(df_train_use["seq_scored"].iloc[0])
    if "seq_scored" in df_train_use.columns
    else 68
)
seq_scored = min(seq_scored, 68)  # scoring positions are 68

eps = 1e-6
pos_means = {}
global_means = {}

blend_alpha = 0.25

clip_q_low, clip_q_high = 0.01, 0.99
clip_bounds = {}
for col in target_cols:
    y_all = np.concatenate(df_train_use[col].values).astype(np.float32)
    lo = float(np.quantile(y_all, clip_q_low))
    hi = float(np.quantile(y_all, clip_q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = -1e9, 1e9
    clip_bounds[col] = (lo, hi)


def _robust_weighted_mean_per_position(y, w_base, n_iter=2, c=2.0):
    """
    Huber-style IRLS on residuals to reduce outlier impact while preserving the same
    'compute per-position mean' core logic.
    """
    if w_base is None:
        w_eff = np.ones_like(y, dtype=np.float32)
    else:
        w_eff = w_base.astype(np.float32, copy=True)

    wsum = w_eff.sum(axis=0)
    mu = (w_eff * y).sum(axis=0) / np.maximum(wsum, eps)

    for _ in range(max(0, int(n_iter))):
        r = y - mu[None, :]
        mad = np.median(np.abs(r), axis=0).astype(np.float32)
        s = 1.4826 * mad + 1e-3  # avoid zero scale

        abs_r = np.abs(r)
        thresh = (c * s)[None, :]
        w_rob = np.where(abs_r <= thresh, 1.0, thresh / np.maximum(abs_r, 1e-6)).astype(
            np.float32
        )

        w_eff2 = w_eff * w_rob
        wsum2 = w_eff2.sum(axis=0)
        mu = (w_eff2 * y).sum(axis=0) / np.maximum(wsum2, eps)

    mu_global = float((w_eff * y).sum() / np.maximum(w_eff.sum(), eps))
    return mu.astype(np.float32), np.float32(mu_global)


for col in target_cols:
    y = np.vstack(df_train_use[col].values).astype(np.float32)
    if y.shape[1] != seq_scored:
        y = y[:, :seq_scored]

    lo, hi = clip_bounds[col]
    y = np.clip(y, lo, hi)

    mu_pos_unw = y.mean(axis=0)
    mu_global_unw = float(y.mean())

    w_base = None
    err_col = error_cols.get(col)
    if err_col in df_train_use.columns:
        e = np.vstack(df_train_use[err_col].values).astype(np.float32)
        if e.shape[1] != seq_scored:
            e = e[:, :seq_scored]
        w_base = 1.0 / (np.square(e) + eps)
        w_base = np.clip(w_base, 0.0, 1e6).astype(np.float32)

    snr_w_col = snr_w.astype(np.float32)[:, None]
    if w_base is None:
        w_base = snr_w_col * np.ones_like(y, dtype=np.float32)
    else:
        w_base = w_base * snr_w_col

    mu_pos_w, mu_global_w = _robust_weighted_mean_per_position(
        y, w_base, n_iter=2, c=2.0
    )
    mu_pos = blend_alpha * mu_pos_w + (1.0 - blend_alpha) * mu_pos_unw.astype(
        np.float32
    )
    mu_global = np.float32(
        blend_alpha * float(mu_global_w) + (1.0 - blend_alpha) * mu_global_unw
    )

    pos_means[col] = mu_pos.astype(np.float32)
    global_means[col] = np.float32(mu_global)


def _fit_shrink(a_pos, b_pos):
    a = a_pos.astype(np.float32)
    b = b_pos.astype(np.float32)
    am = float(a.mean())
    bm = float(b.mean())
    av = float(((a - am) ** 2).mean())
    if not np.isfinite(av) or av < 1e-8:
        k = 1.0
        d = bm - k * am
    else:
        cov = float(((a - am) * (b - bm)).mean())
        k = cov / av
        d = bm - k * am
    if not np.isfinite(k) or not np.isfinite(d):
        k, d = 1.0, 0.0
    return float(k), float(d)


shrink = 0.35  # keep existing tuning

k_pH10, d_pH10 = _fit_shrink(pos_means["deg_Mg_pH10"], pos_means["deg_pH10"])
mapped_pH10 = (k_pH10 * pos_means["deg_Mg_pH10"] + d_pH10).astype(np.float32)
pos_means["deg_pH10"] = (1.0 - shrink) * pos_means["deg_pH10"] + shrink * mapped_pH10
global_means["deg_pH10"] = np.float32(
    (1.0 - shrink) * float(global_means["deg_pH10"])
    + shrink * float(k_pH10 * global_means["deg_Mg_pH10"] + d_pH10)
)

k_50C, d_50C = _fit_shrink(pos_means["deg_Mg_50C"], pos_means["deg_50C"])
mapped_50C = (k_50C * pos_means["deg_Mg_50C"] + d_50C).astype(np.float32)
pos_means["deg_50C"] = (1.0 - shrink) * pos_means["deg_50C"] + shrink * mapped_50C
global_means["deg_50C"] = np.float32(
    (1.0 - shrink) * float(global_means["deg_50C"])
    + shrink * float(k_50C * global_means["deg_Mg_50C"] + d_50C)
)

seqpos = df["id_seqpos"].astype(str).str.rsplit("_", n=1).str[-1].astype(int).values
full_len = int(seqpos.max()) + 1  # should be 107

tail_k = 8
tail_k = int(np.clip(tail_k, 1, seq_scored))

for col in target_cols:
    mu_scored = pos_means[col]  # length seq_scored (<=68)

    tail_fill = (
        float(mu_scored[-tail_k:].mean())
        if seq_scored > 0
        else float(global_means[col])
    )

    mu_full = np.empty(full_len, dtype=np.float32)
    mu_full[:seq_scored] = mu_scored
    if full_len > seq_scored:
        mu_full[seq_scored:] = tail_fill

    clipped_seqpos = np.clip(seqpos, 0, full_len - 1)
    df[col] = mu_full[clipped_seqpos]



## === cell 4
for c in required_cols[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df[required_cols[1:]] = (
    df[required_cols[1:]].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)
df = df[required_cols]



## === cell 5
out_path = "submission.csv"
df.to_csv(out_path, index=False)

if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
    raise RuntimeError("Failed to write a non-empty submission.csv")

if len(df) != 25680:
    raise RuntimeError(f"Unexpected submission row count: {len(df)} (expected 25680)")
