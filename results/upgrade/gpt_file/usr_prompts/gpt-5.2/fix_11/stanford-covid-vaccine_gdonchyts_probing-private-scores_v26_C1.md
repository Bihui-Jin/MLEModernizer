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

0.3684094342822526

# 6. Current score

0.48258

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I remove the hard dependency on the missing `../input/worst-submission/ensemble52.csv` by instead loading the provided competition `sample_submission.csv` and using it as the base. To keep the spirit of your original approach (a simple global scaling), I apply the same `/= 1.155` scaling to the prediction columns (they’re all zeros in the sample, so this remains score-neutral but fixes execution). I also keep the intermediate `sequences` extraction working by deriving it from `id_seqpos` as you intended. Finally, I ensure a valid `submission.csv` is written with the required columns.'
- What this solution (achieved 0.42418) has done: 'Your current notebook is effectively submitting (near-)all zeros, which explains the poor 0.63824 MCRMSE; to move toward the 0.3684 target (lower is better), the smallest legitimate improvement is to replace the zero baseline with a simple, leakage-free baseline learned from the training labels. I keep the overall “no model, just generate a submission” core approach, but compute per-position mean targets from `train.json` for the 5 columns and use those as predictions for every test sequence position (and fill positions >68 by copying the last scored position mean). This is a standard baseline for this competition and should substantially reduce error without changing evaluation semantics. I also keep your column checks and ensure the produced CSV exactly matches `sample_submission.csv` row order and required columns.'
- What this solution (achieved 0.42166) has done: 'You’re currently using per-position means computed over all training rows, which includes many low-quality (low SNR / failed filter) samples; a minimal, competition-standard improvement is to compute those same per-position means using only `SN_filter == 1` rows (and optionally require reasonable `signal_to_noise`) to better match the test distribution and reduce MCRMSE. I keep the exact same “no model, just per-position mean baseline and copy last scored position for 69–107” core logic and submission row order. I also add a tiny robustness step to ignore any NaNs in the training target arrays when taking means, so the baseline doesn’t get skewed by missing values. This should move the score down from 0.42418 toward your 0.3684 target without changing evaluation semantics or adding a new modeling approach.'
- What this solution (achieved 0.42862) has done: 'To move your MCRMSE down toward the 0.3684 target (lower is better) with minimal risk, I keep the exact same “per-position mean baseline” core logic but make the baseline better match the public/private test distribution. Specifically: (1) compute means using only high-quality training rows (SN_filter==1 and a higher signal_to_noise threshold), and (2) additionally clip training targets to the competition’s typical value range before averaging to reduce the influence of extreme/noisy outliers. These are small, standard baseline refinements for this dataset and should improve score without changing the overall approach or submission format. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.42166) has done: 'Your current baseline is already valid but a bit mismatched to the competition scoring: only the first 68 positions are scored, and for positions >68 you currently copy the 68th mean which can create unnecessary error on the unscored tail without helping the score. I keep the same per-position-mean core approach, but (1) compute the means with a slightly stricter, competition-standard quality filter (`SN_filter==1` and `signal_to_noise>=1.0` only, no fallback), and (2) stop clipping targets (the clip can bias the mean away from true values and has likely hurt your score). Finally, for positions >68, I fill with the per-target global mean of the scored positions (instead of repeating position 67), which is score-neutral (unscored) and typically yields more reasonable outputs while preserving submission format.'
- What this solution (achieved 0.48258) has done: 'Your current approach (per-position mean baseline) is already stable and fairly close to the target, so the smallest improvement likely to reduce MCRMSE toward 0.3684 is to weight the per-position averages by measurement reliability instead of treating all training rows equally. Concretely, we keep the same filtering (`SN_filter==1` and `signal_to_noise>=1.0`) and the same “compute a 68-length mean vector then extend to 107” logic, but compute *weighted* means using the provided per-position error columns (`*_error_*`) as inverse-variance weights. This better matches the competition’s noise model and typically improves this baseline without changing the overall semantics or adding a new model. The rest of the pipeline (row order, id_seqpos parsing, required columns, writing `submission.csv`) remains unchanged.'
- What this solution (achieved 0.48258) has done: 'Your current score (0.48258, lower-is-better) is still worse than the target (0.36841), so we should make a small, legitimate improvement that keeps the same “per-position mean baseline” core logic. The most impactful minimal fix here is to **predict the two unscored targets (`deg_pH10`, `deg_50C`) as linear functions of the scored ones**, fitted on the training set (still leakage-free), because those columns are required in the submission and their error contributes to public LB even if not in the official metric description you provided. We keep your weighted-per-position means exactly as-is for the scored targets, and only add a tiny ridge-stabilized linear mapping (fit on `SN_filter==1 & signal_to_noise>=1.0`) from `(reactivity, deg_Mg_pH10, deg_Mg_50C)` → each unscored target per position. This should reduce overall submission error with minimal code change and without changing the overall approach or output format.'
- What this solution (achieved 0.48258) has done: 'We keep your existing “weighted per-position mean baseline + per-position ridge mapping for the two required-but-unscored columns” core logic, but fix the most likely reason your score regressed: the ridge mapping currently ignores measurement reliability, so it can learn noisy relationships and harm the scored columns indirectly via correlated calibration. The minimal, metric-aligned change is to fit those per-position ridge maps with **inverse-variance sample weights** derived from the provided error columns, while keeping the same features, per-position fitting, and ridge stabilization. We also add a small safety clamp on the ridge-predicted unscored columns to a reasonable range based on the filtered training distribution (percentile-based) to prevent occasional extreme values that can inflate RMSE. All paths, output format, and the scored-target computation remain unchanged, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.48258) has done: 'We keep your exact baseline structure (SN-filtered training, inverse-variance weighted per-position means, and the per-position weighted ridge mapping for the two required columns), but fix two small issues that can easily hurt MCRMSE: (1) the ridge features currently use raw means while the ridge was trained on raw per-sample targets—adding a simple per-feature standardization (fit on the same filtered train) stabilizes the mapping without changing the approach, and (2) your `robust_clip_to_train` clips using global percentiles over all positions, which can over/under-clip certain positions; switching to **per-position** percentile clipping is still the same “safety clamp” idea but better aligned to the per-position nature of the metric. These are minimal changes that should move the score down from 0.4826 toward the 0.3684 target while preserving evaluation semantics and producing the same submission format.'
- What this solution (achieved 0.48258) has done: 'We keep your exact “SN-filtered, inverse-variance weighted per-position mean baseline + per-position weighted ridge for deg_pH10/deg_50C” approach, but fix the most likely reason it’s performing poorly: the ridge mapping is currently applied to the **means** of the scored targets, whereas it was trained to map **per-sample** scored targets to unscored ones; this distribution mismatch can easily worsen MCRMSE. The minimal, metric-aligned correction is to fit the ridge on per-sample standardized features and then apply it to the **per-position weighted means of those same standardized features**, not to the raw means. We also make the ridge regularization slightly stronger (still tiny) to reduce overfitting noise in low-data positions; this is a small calibration/regularization tweak, not a modeling change. Everything else (paths, filtering, weighted means for scored targets, per-position clip safety, submission schema/order) stays the same and it still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd



## === cell 1
SAMPLE_PATHS = [
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
]
sample_path = None
for p in SAMPLE_PATHS:
    try:
        with open(p, "r"):
            sample_path = p
            break
    except FileNotFoundError:
        pass

if sample_path is None:
    raise FileNotFoundError(
        f"Could not find sample_submission.csv in any of: {SAMPLE_PATHS}"
    )

sub = pd.read_csv(sample_path)
sub.head()



## === cell 2
sequences = list(set(["_".join(v.split("_")[:-1]) for v in sub.id_seqpos.values]))
sequences[-10:]



## === cell 3
TRAIN_PATHS = [
    "/kaggle/input/train.json",
    "/kaggle/data/train.json",
    "/kaggle/input/stanford-covid-vaccine/train.json",
    "/kaggle/data/stanford-covid-vaccine/train.json",
]
train_path = None
for p in TRAIN_PATHS:
    try:
        with open(p, "r"):
            train_path = p
            break
    except FileNotFoundError:
        pass

if train_path is None:
    raise FileNotFoundError(f"Could not find train.json in any of: {TRAIN_PATHS}")

train = pd.read_json(train_path, lines=True)

target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
error_cols = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}
seq_scored = int(train["seq_scored"].iloc[0])  # 68
seq_length = int(train["seq_length"].iloc[0])  # 107

train_used = train.copy()
if "SN_filter" in train_used.columns:
    train_used = train_used[train_used["SN_filter"].astype(int) == 1].copy()
if "signal_to_noise" in train_used.columns:
    train_used = train_used[train_used["signal_to_noise"].astype(float) >= 1.0].copy()

if len(train_used) == 0:
    raise ValueError("After filtering, no training rows remain; cannot build baseline.")


def weighted_pos_mean(values_2d: np.ndarray, errors_2d: np.ndarray) -> np.ndarray:
    """
    Reliability-weighted mean per position using inverse-variance weights.
    Kept unchanged to preserve your current core baseline behavior for scored targets.
    """
    v = values_2d.astype(np.float32, copy=False)
    e = errors_2d.astype(np.float32, copy=False)

    eps = np.float32(1e-3)
    w = 1.0 / np.square(np.maximum(e, eps))

    bad = ~np.isfinite(v) | ~np.isfinite(w)
    if bad.any():
        w = w.copy()
        w[bad] = 0.0
        v = np.where(np.isfinite(v), v, 0.0)

    num = np.sum(w * v, axis=0)
    den = np.sum(w, axis=0)

    out = np.empty((v.shape[1],), dtype=np.float32)
    zero_den = den <= 0
    if np.any(zero_den):
        out[~zero_den] = (num[~zero_den] / den[~zero_den]).astype(np.float32)
        out[zero_den] = np.nanmean(values_2d[:, zero_den].astype(np.float32), axis=0)
    else:
        out = (num / den).astype(np.float32)
    return out


pos_means = {}
for c in target_cols:
    arr = np.vstack(train_used[c].values)  # (n_train_used, 68)
    err_c = error_cols.get(c, None)

    if err_c is not None and err_c in train_used.columns:
        err = np.vstack(train_used[err_c].values)  # (n_train_used, 68)
        pos_means[c] = weighted_pos_mean(arr, err)  # (68,)
    else:
        pos_means[c] = np.nanmean(arr.astype(np.float32), axis=0)  # (68,)


def fit_weighted_ridge_map_per_pos(
    X: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray | None = None,
    alpha: float = 3e-3,
) -> np.ndarray:
    """
    Fit y ~= b0 + X @ b (ridge), per position, optionally weighted by inverse-variance from y_err.
    Returns coef [b0, b1, b2, b3] for each position.
    X: (n, 68, 3), y: (n, 68), y_err: (n, 68) or None

    Change (score-relevant, minimal): slightly stronger ridge (alpha) to reduce noise overfit
    in per-position fits, while keeping the same approach/semantics.
    """
    n, L, k = X.shape
    coefs = np.zeros((L, k + 1), dtype=np.float32)
    I = np.eye(k + 1, dtype=np.float32)
    I[0, 0] = 0.0  # don't regularize intercept
    eps = np.float32(1e-3)

    for p in range(L):
        Xp = X[:, p, :].astype(np.float32, copy=False)
        yp = y[:, p].astype(np.float32, copy=False)

        ok = np.isfinite(yp) & np.all(np.isfinite(Xp), axis=1)
        if y_err is not None:
            ep = y_err[:, p].astype(np.float32, copy=False)
            ok = ok & np.isfinite(ep)

        if ok.sum() < (k + 2):
            coefs[p, 0] = np.nanmean(yp[ok]) if ok.any() else np.nanmean(yp)
            continue

        Xp = Xp[ok]
        yp = yp[ok]

        A = np.concatenate(
            [np.ones((Xp.shape[0], 1), dtype=np.float32), Xp], axis=1
        )  # (m, 4)

        if y_err is None:
            ATA = A.T @ A
            ATy = A.T @ yp
        else:
            ep = y_err[:, p].astype(np.float32, copy=False)[ok]
            w = 1.0 / np.square(np.maximum(ep, eps))  # (m,)
            Aw = A * w[:, None]
            ATA = A.T @ Aw
            ATy = A.T @ (w * yp)

        beta = np.linalg.solve(ATA + (alpha * I), ATy).astype(np.float32)
        coefs[p] = beta
    return coefs


def apply_ridge_map(coefs: np.ndarray, Xpred: np.ndarray) -> np.ndarray:
    """
    coefs: (68, 4), Xpred: (68, 3) -> ypred: (68,)
    """
    Xp = Xpred.astype(np.float32, copy=False)
    return (coefs[:, 0] + np.sum(coefs[:, 1:] * Xp, axis=1)).astype(np.float32)


X_train_raw = np.stack(
    [
        np.vstack(train_used["reactivity"].values),
        np.vstack(train_used["deg_Mg_pH10"].values),
        np.vstack(train_used["deg_Mg_50C"].values),
    ],
    axis=2,
).astype(
    np.float32
)  # (n, 68, 3)

mu_X = np.nanmean(X_train_raw, axis=(0, 1)).astype(np.float32)  # (3,)
sd_X = np.nanstd(X_train_raw, axis=(0, 1)).astype(np.float32)  # (3,)
sd_X = np.maximum(sd_X, np.float32(1e-3))
X_train = (X_train_raw - mu_X[None, None, :]) / sd_X[None, None, :]

y_deg_pH10 = np.vstack(train_used["deg_pH10"].values).astype(np.float32)
y_deg_50C = np.vstack(train_used["deg_50C"].values).astype(np.float32)

err_deg_pH10 = (
    np.vstack(train_used["deg_error_pH10"].values).astype(np.float32)
    if "deg_error_pH10" in train_used.columns
    else None
)
err_deg_50C = (
    np.vstack(train_used["deg_error_50C"].values).astype(np.float32)
    if "deg_error_50C" in train_used.columns
    else None
)

coef_deg_pH10 = fit_weighted_ridge_map_per_pos(
    X_train, y_deg_pH10, y_err=err_deg_pH10, alpha=3e-3
)
coef_deg_50C = fit_weighted_ridge_map_per_pos(
    X_train, y_deg_50C, y_err=err_deg_50C, alpha=3e-3
)

X_mean_std = np.empty((seq_scored, 3), dtype=np.float32)
for j, (col, err_col) in enumerate(
    [
        ("reactivity", "reactivity_error"),
        ("deg_Mg_pH10", "deg_error_Mg_pH10"),
        ("deg_Mg_50C", "deg_error_Mg_50C"),
    ]
):
    arr = np.vstack(train_used[col].values).astype(np.float32)  # (n,68)
    arr_std = (arr - mu_X[j]) / sd_X[j]
    if err_col in train_used.columns:
        err = np.vstack(train_used[err_col].values).astype(np.float32)
        X_mean_std[:, j] = weighted_pos_mean(arr_std, err)
    else:
        X_mean_std[:, j] = np.nanmean(arr_std, axis=0).astype(np.float32)

deg_pH10_pred = apply_ridge_map(coef_deg_pH10, X_mean_std)
deg_50C_pred = apply_ridge_map(coef_deg_50C, X_mean_std)


def robust_clip_to_train_per_pos(
    y_train_2d: np.ndarray, y_pred_1d: np.ndarray
) -> np.ndarray:
    """
    Clip per position rather than globally (kept as-is).
    """
    ytr = y_train_2d.astype(np.float32, copy=False)
    lo = np.nanpercentile(ytr, 1.0, axis=0).astype(np.float32)  # (68,)
    hi = np.nanpercentile(ytr, 99.0, axis=0).astype(np.float32)  # (68,)
    return np.clip(y_pred_1d.astype(np.float32, copy=False), lo, hi).astype(np.float32)


pos_means["deg_pH10"] = robust_clip_to_train_per_pos(y_deg_pH10, deg_pH10_pred)
pos_means["deg_50C"] = robust_clip_to_train_per_pos(y_deg_50C, deg_50C_pred)

pos_means_full = {}
for c in target_cols:
    v68 = pos_means[c]
    tail_fill = float(np.nanmean(v68))  # unscored tail; keep simple/robust
    v107 = np.empty((seq_length,), dtype=np.float32)
    v107[:seq_scored] = v68
    v107[seq_scored:] = tail_fill
    pos_means_full[c] = v107

seqpos = sub["id_seqpos"].str.rsplit("_", n=1).str[-1].astype(np.int16).values

for c in target_cols:
    sub[c] = pos_means_full[c][seqpos]

sub.head()



## === cell 4
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
    raise ValueError(f"Submission is missing required columns: {missing}")

if sub["id_seqpos"].isna().any():
    raise ValueError("Found NaN in id_seqpos, submission would be invalid.")

pred_cols = [c for c in sub.columns if c != "id_seqpos"]
sub[pred_cols] = sub[pred_cols].astype(np.float32)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
