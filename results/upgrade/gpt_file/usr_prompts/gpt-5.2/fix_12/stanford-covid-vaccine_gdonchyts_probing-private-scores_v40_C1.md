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

0.48627

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42418) has done: 'You’re getting an IndexError because `pos_mean[t]` has length 68 (0–67), but you index it with `seqpos` values up to 106; `np.where` still evaluates both branches, so it crashes even though you intend to use `overall_mean` for the tail. I fix this by safely indexing `pos_mean` only for scored positions and filling the rest with `overall_mean`, preserving the same baseline logic. This also allow downstream cells to run so `df` exists, making your optional tweak and sanity checks execute without NameErrors. Finally, the script write a valid `submission.csv` with the correct columns and row count.'
- What this solution (achieved 0.42166) has done: 'Your current approach is a global-position mean baseline; to move the score closer to the 0.3519 target (lower is better) with minimal changes, we keep the exact same core “position mean + overall mean for tail” logic but make it a bit more robust to noisy training rows. Specifically, we compute the per-position means using only higher-quality training samples (the provided `SN_filter==1` rows), which is a standard competition-safe data cleaning step and typically improves this baseline without changing the model family. We also remove the hard-coded single-id tweak (it’s effectively arbitrary and can only hurt generalization), while keeping all I/O and submission formatting identical. The rest of the pipeline (targets, masking, tail fill, and CSV writing) remains unchanged.'
- What this solution (achieved 0.4216) has done: 'To move your score down toward the 0.3519 target (lower is better) with minimal risk, I keep the same “per-position mean for first 68 + overall mean for tail” baseline but make the mean estimates more reliable. Concretely: (1) winsorize (clip) training target values per position using robust quantiles computed on the SN_filter==1 subset, reducing the impact of noisy/outlier measurements without changing the modeling family; (2) apply a tiny shrinkage of per-position means toward the global mean to reduce variance from limited sample effects. Submission formatting, paths, and the prediction construction logic remain identical.'
- What this solution (achieved 0.48411) has done: 'Your current baseline already has the right idea (position-wise means + tail fill), but it’s still held back by noisy/low-quality training rows. To move the score down toward the 0.3519 target with minimal semantic change, I keep the exact same prediction construction and submission format, and only make the mean estimates more representative by (1) filtering to `SN_filter==1` (already done) and additionally requiring a modest `signal_to_noise` threshold, and (2) computing the per-position means as inverse-variance weighted means using the provided `*_error_*` arrays (this is still “a mean baseline”, just a better mean). I keep your winsorization and tiny shrinkage intact (same logic), applying them after weighting so outliers still don’t dominate. This should improve MCRMSE without changing any core modeling approach or adding new dependencies.'
- What this solution (achieved 0.4833) has done: 'You’re currently above the target (0.48411 vs 0.35188, lower is better), so we should improve the baseline cautiously without changing its core “position-wise mean for first 68 + overall mean for tail” logic. The smallest lever with good payoff here is to tune the signal-to-noise filtering (your current `>=1.0` is too permissive vs what the organizers used for high-quality data) and slightly adjust the shrinkage toward the global mean to reduce residual noise. I keep your weighting-by-error, winsorization, and submission formatting exactly the same, and only (1) pick a modestly higher `signal_to_noise` threshold with a safe fallback and (2) slightly reduce shrinkage so we trust improved per-position estimates more. This should move MCRMSE downward toward the 0.3519 target while remaining stable and within the same modeling semantics.'
- What this solution (achieved 0.48335) has done: 'We keep your exact “weighted per-position mean for first 68 + overall mean for tail” baseline, but adjust one lever that’s likely holding you back: the `signal_to_noise` threshold is currently too strict and can leave you averaging over too few sequences, hurting generalization. I make the threshold adaptive by picking a quantile-based cutoff (with a minimum of 1.0) so you use a stable high-quality subset without starving data, and keep the same safe fallbacks. Everything else (error-weighting, winsorization, shrinkage, submission formatting, and file paths) remains unchanged to preserve evaluation semantics while nudging MCRMSE downward toward the target.'
- What this solution (achieved 0.48268) has done: 'Your current baseline is already the right “family” (position-wise mean for first 68 + overall mean for tail), but the score suggests your mean estimates are still too noisy. I keep the exact same prediction construction and submission formatting, and only adjust how the per-position/overall means are estimated: compute winsorization quantiles using the high-quality subset only (so outlier clipping isn’t distorted by low-quality rows), and slightly retune the SNR quantile cutoff to use a bit more stable data without becoming too permissive. These are minimal changes that preserve the same evaluation semantics and should move MCRMSE downward toward the 0.3519 target. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.48598) has done: 'To move your MCRMSE down toward the 0.3519 target while preserving your exact “position-wise mean for first 68 + overall mean for tail” baseline, I only adjust how the means are estimated, not the prediction construction. Concretely, I keep your SN_filter/SNR selection, error-weighted means, winsorization, and shrinkage, but switch the winsorization quantiles to be *weighted by inverse-variance* (so noisy points don’t distort clipping thresholds). I also prevent extreme weights from dominating by using a slightly tighter cap (still the same weighting logic), which usually stabilizes the weighted mean under this dataset’s error distributions. Submission formatting, paths, and row alignment remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.48654) has done: 'Your current score (0.48598, lower-is-better) is still far from the target (0.35188), so we should improve the *mean-baseline* estimates without changing the overall “per-position mean for first 68 + overall mean for tail” prediction construction. The smallest high-impact fix is to compute means only on *scored* training rows with acceptable measurement quality by additionally filtering on the per-row `*_error_*` arrays (dropping rows with extreme average error), which reduces noise while keeping the same weighted-mean logic. I also retune the weight cap slightly upward (still capped) because the current cap can underuse legitimately precise measurements after your weighted winsorization, and this often improves MCRMSE for this competition. Submission formatting and row alignment remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.48627) has done: 'I keep your exact “weighted per-position mean for first 68 + overall mean for tail” baseline and only adjust the *data selection for estimating those means*, since your current filtering is likely leaving too little (and thus noisy) data and hurting generalization. Concretely, I replace the single fixed SNR quantile cutoff with a small grid of candidate SNR quantiles and pick the one that minimizes an internal CV MCRMSE on the scored targets/positions—this preserves the same model family and prediction construction, but tunes the one lever that most affects your mean estimates. I also compute the error-based row filter on the same candidate subset inside CV (so it’s consistent), but keep your weighting, weighted winsorization, shrinkage, tail fill, and submission formatting unchanged. This should move the score downward toward the 0.3519 target with minimal semantic change and still finish fast.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os



## === cell 1
DATA_DIR = "../input/stanford-covid-vaccine"

train_path = os.path.join(DATA_DIR, "train.json")
test_path = os.path.join(DATA_DIR, "test.json")
sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

df_train = pd.read_json(train_path, lines=True)
df_test = pd.read_json(test_path, lines=True)
df_sub = pd.read_csv(sub_path)

df_train.shape, df_test.shape, df_sub.shape



## === cell 2
TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
ERR_COL = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

seq_scored = int(df_train["seq_scored"].iloc[0])  # 68
seq_length = int(df_train["seq_length"].iloc[0])  # 107

eps = np.float32(1e-6)


def weighted_quantile(values, quantiles, sample_weight, axis=0):
    """
    Compute weighted quantiles along `axis` for an array.
    values: np.ndarray
    sample_weight: same shape as values
    quantiles: scalar or 1d array in [0,1]
    """
    values = np.asarray(values)
    sample_weight = np.asarray(sample_weight)
    q = np.atleast_1d(quantiles).astype(np.float64)

    if axis != 0:
        values = np.swapaxes(values, axis, 0)
        sample_weight = np.swapaxes(sample_weight, axis, 0)

    n = values.shape[0]
    rest = values.shape[1:]
    out = np.empty((len(q),) + rest, dtype=np.float64)

    v2 = values.reshape(n, -1).astype(np.float64)
    w2 = sample_weight.reshape(n, -1).astype(np.float64)

    for j in range(v2.shape[1]):
        v = v2[:, j]
        ww = w2[:, j]
        m = np.isfinite(v) & np.isfinite(ww) & (ww > 0)
        if not np.any(m):
            out[:, j] = np.nan
            continue
        v = v[m]
        ww = ww[m]
        idx = np.argsort(v)
        v = v[idx]
        ww = ww[idx]
        cdf = np.cumsum(ww)
        if cdf[-1] <= 0:
            out[:, j] = np.nan
            continue
        cdf = cdf / cdf[-1]
        out[:, j] = np.interp(q, cdf, v)

    out = out.reshape((len(q),) + rest)
    if axis != 0:
        out = np.swapaxes(out, axis, 0)
    if np.isscalar(quantiles):
        return out[0]
    return out


def mcrmse(y_true, y_pred, eps=1e-12):
    mse = np.mean((y_true - y_pred) ** 2, axis=0)  # (seq_scored, n_targets)
    rmse_per_pos = np.sqrt(np.maximum(mse, eps))  # (seq_scored, n_targets)
    rmse_per_target = np.mean(rmse_per_pos, axis=0)  # (n_targets,)
    return float(np.mean(rmse_per_target))


def fit_means_on_df(df_fit, weight_cap=3e5, q_low=0.02, q_high=0.98, shrink=0.03):
    train_y = {
        t: np.stack(df_fit[t].values).astype(np.float32)[:, :seq_scored]
        for t in TARGETS
    }
    train_err = {
        t: np.stack(df_fit[ERR_COL[t]].values).astype(np.float32)[:, :seq_scored]
        for t in TARGETS
    }

    w = {}
    for t in TARGETS:
        wi = 1.0 / (np.maximum(train_err[t], eps) ** 2)
        w[t] = np.clip(wi, 0.0, weight_cap).astype(np.float32)

    train_y_clipped = {}
    for t in TARGETS:
        lo = weighted_quantile(train_y[t], q_low, w[t], axis=0).astype(np.float32)
        hi = weighted_quantile(train_y[t], q_high, w[t], axis=0).astype(np.float32)

        bad = ~np.isfinite(lo) | ~np.isfinite(hi) | (hi <= lo)
        if np.any(bad):
            lo_u = np.quantile(train_y[t], q_low, axis=0).astype(np.float32)
            hi_u = np.quantile(train_y[t], q_high, axis=0).astype(np.float32)
            lo = np.where(bad, lo_u, lo).astype(np.float32)
            hi = np.where(bad, hi_u, hi).astype(np.float32)

        train_y_clipped[t] = np.clip(train_y[t], lo[None, :], hi[None, :]).astype(
            np.float32
        )

    pos_mean = {}
    overall_mean = {}
    for t in TARGETS:
        wt = w[t]
        yt = train_y_clipped[t]
        denom_pos = np.maximum(wt.sum(axis=0), eps)
        pos_mean[t] = (wt * yt).sum(axis=0) / denom_pos

        denom_all = np.maximum(wt.sum(), eps)
        overall_mean[t] = float((wt * yt).sum() / denom_all)

    for t in TARGETS:
        pos_mean[t] = (1.0 - shrink) * pos_mean[t] + shrink * overall_mean[t]
        pos_mean[t] = pos_mean[t].astype(np.float32)

    return pos_mean, overall_mean


def build_oof_pred(df_all, train_idx, valid_idx, snr_thresh, err_q=0.95, min_rows=200):
    df_train_part = df_all.iloc[train_idx].copy()

    base_mask = (df_train_part["SN_filter"] == 1) & (
        df_train_part["signal_to_noise"] >= snr_thresh
    )
    df_base = df_train_part.loc[base_mask].copy()
    if len(df_base) == 0:
        df_base = df_train_part.loc[df_train_part["SN_filter"] == 1].copy()
    if len(df_base) == 0:
        df_base = df_train_part.copy()

    err_means = []
    for t in TARGETS:
        e = np.stack(df_base[ERR_COL[t]].values).astype(np.float32)[:, :seq_scored]
        err_means.append(np.nanmean(e, axis=1))
    err_mean_all = np.nanmean(np.stack(err_means, axis=1), axis=1)

    err_cut = float(np.nanquantile(err_mean_all, err_q))
    mask_err = np.isfinite(err_mean_all) & (err_mean_all <= err_cut)
    df_mean = df_base.loc[mask_err].reset_index(drop=True)

    if len(df_mean) < min_rows:
        err_cut = float(np.nanquantile(err_mean_all, min(0.98, max(err_q, 0.95))))
        mask_err = np.isfinite(err_mean_all) & (err_mean_all <= err_cut)
        df_mean = df_base.loc[mask_err].reset_index(drop=True)
    if len(df_mean) == 0:
        df_mean = df_base.reset_index(drop=True)

    pos_mean, overall_mean = fit_means_on_df(df_mean)

    yhat = np.zeros((len(valid_idx), seq_scored, len(TARGETS)), dtype=np.float32)
    for k, t in enumerate(TARGETS):
        yhat[:, :, k] = pos_mean[t][None, :]

    return yhat


SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
scored_target_idx = [TARGETS.index(t) for t in SCORED_TARGETS]

y_all = np.zeros((len(df_train), seq_scored, len(TARGETS)), dtype=np.float32)
for k, t in enumerate(TARGETS):
    y_all[:, :, k] = np.stack(df_train[t].values).astype(np.float32)[:, :seq_scored]

rng = np.random.RandomState(0)
idx_all = np.arange(len(df_train))
rng.shuffle(idx_all)
folds = np.array_split(idx_all, 5)

snr_series = df_train.loc[df_train["SN_filter"] == 1, "signal_to_noise"].astype(float)
if len(snr_series) == 0:
    cand_snr = [1.0]
else:
    cand_q = [0.30, 0.40, 0.50, 0.60]
    cand_snr = sorted({float(max(1.0, snr_series.quantile(q))) for q in cand_q})

best_snr = None
best_cv = None

for snr_th in cand_snr:
    oof = np.zeros((len(df_train), seq_scored, len(TARGETS)), dtype=np.float32)
    for f in range(len(folds)):
        valid_idx = folds[f]
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != f])
        oof_pred = build_oof_pred(
            df_train, train_idx, valid_idx, snr_thresh=snr_th, err_q=0.95, min_rows=200
        )
        oof[valid_idx] = oof_pred

    cv = mcrmse(y_all[:, :, scored_target_idx], oof[:, :, scored_target_idx])
    if (best_cv is None) or (cv < best_cv):
        best_cv = cv
        best_snr = snr_th

base_mask = (df_train["SN_filter"] == 1) & (df_train["signal_to_noise"] >= best_snr)
df_train_base = df_train.loc[base_mask].copy()
if len(df_train_base) == 0:
    df_train_base = df_train.loc[df_train["SN_filter"] == 1].copy()
if len(df_train_base) == 0:
    df_train_base = df_train.copy()

err_means = []
for t in TARGETS:
    e = np.stack(df_train_base[ERR_COL[t]].values).astype(np.float32)[:, :seq_scored]
    err_means.append(np.nanmean(e, axis=1))
err_mean_all = np.nanmean(np.stack(err_means, axis=1), axis=1)

err_cut = float(np.nanquantile(err_mean_all, 0.95))
mask_err = np.isfinite(err_mean_all) & (err_mean_all <= err_cut)
df_train_mean = df_train_base.loc[mask_err].reset_index(drop=True)
if len(df_train_mean) < 200:
    err_cut = float(np.nanquantile(err_mean_all, 0.98))
    mask_err = np.isfinite(err_mean_all) & (err_mean_all <= err_cut)
    df_train_mean = df_train_base.loc[mask_err].reset_index(drop=True)
if len(df_train_mean) == 0:
    df_train_mean = df_train_base.reset_index(drop=True)

pos_mean, overall_mean = fit_means_on_df(df_train_mean)

print("Selected SNR threshold:", best_snr, "| internal CV (scored targets):", best_cv)
pos_mean["reactivity"][:5], overall_mean["reactivity"]



## === cell 3
id_seqpos = df_sub["id_seqpos"].astype(str).values
seqpos = np.array([int(s.split("_")[-1]) for s in id_seqpos], dtype=np.int32)

pred = {}
scored_mask = seqpos < seq_scored
scored_pos = seqpos[scored_mask]

for t in TARGETS:
    vals = np.full(shape=(len(seqpos),), fill_value=overall_mean[t], dtype=np.float32)
    vals[scored_mask] = pos_mean[t][scored_pos].astype(np.float32)
    pred[t] = vals

df = df_sub.copy()
for t in TARGETS:
    df[t] = pred[t]

df.head()



## === cell 4
pass



## === cell 5
expected_cols = ["id_seqpos"] + TARGETS
assert list(df.columns) == expected_cols, f"Unexpected columns: {df.columns.tolist()}"
assert len(df) == len(df_sub), "Row count mismatch vs sample_submission"
assert df[TARGETS].isna().sum().sum() == 0, "NaNs found in predictions"

df.describe(include="all")



## === cell 6
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head(3).to_string(index=False))
