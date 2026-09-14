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

0.42235

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.47906) has done: 'I remove the dependency on the missing `../input/worst-submission/ensemble52.csv` and instead build a valid submission directly from the provided `sample_submission.csv` so the notebook runs end-to-end and always writes `submission.csv`. To keep the core intent (producing a submission) with minimal changes and without introducing a new modeling approach, I fill predictions with simple, deterministic constants derived from the training-set means for each target. I also keep your existing post-processing line (setting one specific id’s `reactivity` to 0) but guard it so it doesn’t error if the id doesn’t exist. Finally, I add basic format checks to ensure the output has the correct columns and row count.'
- What this solution (achieved 0.42418) has done: 'Your current solution predicts per-position constants from global training means, which is stable but leaves score on the table because it ignores strong per-position patterns and the fact that only the first 68 positions are scored. To move the score toward the target with minimal change in approach, I keep the same “simple deterministic baseline from training statistics” core logic, but compute position-wise means (0–67) for each target and use those for scored positions. For the remaining positions (68–106, unscored), we keep using the global means to preserve the same semantics and avoid overfitting oddities. This should reduce MCRMSE substantially versus a single constant while still being lightweight and fully deterministic, and it still write a valid `submission.csv`.'
- What this solution (achieved 0.42178) has done: 'Your current baseline is position-wise means over all training rows, which is decent but still diluted by low-quality/noisy training samples. To move the MCRMSE down toward the target with minimal logic change, I keep the same “predict per-position training statistics” approach but compute those statistics using only `SN_filter==1` rows (the competition’s recommended high-quality subset) and also use `signal_to_noise`-weighted means to better match the test distribution. The rest of the pipeline (parsing `seqpos`, filling 0–67 with per-position means and 68–106 with global means, writing `submission.csv`) stays the same to preserve semantics and ensure a valid submission.'
- What this solution (achieved 0.42549) has done: 'You’re currently above the target (0.42178 vs 0.35188, lower is better), so we should make a small, low-risk change that plausibly improves MCRMSE without changing the core “predict training statistics per position” logic. The minimal gap-reducing tweak here is to compute per-position means using a trimmed/robust mean over `SN_filter==1` (dropping extreme values per position) instead of a pure weighted mean, since this dataset has known noisy/outlier measurements and trimming typically helps a constant-per-position baseline. We keep the same prediction shape, the same handling of unscored positions (global mean), and still write `submission.csv` with the exact required columns/rows. This is still deterministic, uses only training statistics, and stays within the same baseline approach.'
- What this solution (achieved 0.42475) has done: 'We keep your exact “predict deterministic training statistics per position” baseline, but make a small change to better match the test distribution: compute per-position means only from the high-quality subset (`SN_filter==1`) and additionally drop rows whose `signal_to_noise` is in the bottom tail, which often contain noisy labels that hurt a mean-based baseline. This is still the same core logic (no model, no new features, no new training loop), just a slightly cleaner subset for the same statistic you already use. We also compute the trimming bounds in a weighted way (using `signal_to_noise` as weights) to keep the robustness idea while respecting the better-quality samples more. Everything else (scored vs unscored positions handling, output columns/rows, `submission.csv` writing) remains unchanged.'
- What this solution (achieved 0.42235) has done: 'You’re currently worse than the target (0.42475 vs 0.35188, lower is better), so we should make the smallest change that plausibly improves MCRMSE while keeping the exact “predict deterministic training statistics per position” baseline. The main issue is that your weighted trimming can over-trim and slightly bias means; a minimal improvement is to keep your same high-quality filtering, but replace the trimming statistic with a simple, deterministic winsorization (clip) per position before taking the same S2N-weighted mean. This keeps identical semantics (still per-position weighted training statistics; no model, no new features), but is typically more stable than dropping points entirely. Everything else (scored vs unscored handling, submission formatting, and `submission.csv` writing) remains unchanged.'
- What this solution (achieved 0.42235) has done: 'You’re currently worse than the target (0.42235 vs 0.35188; lower is better), so we should make a small, low-risk improvement within the same “deterministic per-position training statistics” baseline. The least invasive boost is to stop predicting all 5 targets independently: instead, keep your exact per-position computation for the 3 scored targets, and for the 2 unscored targets (deg_pH10, deg_50C) predict them via a simple per-position linear calibration from the scored counterparts (deg_Mg_pH10 and deg_Mg_50C). This typically reduces overall noise and slightly improves the shared representation, while not changing the core approach (still position-wise statistics from train, no model training loop). Everything else (SN_filter filtering, s2n weighting, winsorization, scored vs unscored position handling, and writing a valid `submission.csv`) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing



## === cell 1
df_train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)
df = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")



## === cell 2
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

SEQ_SCORED = int(df_train["seq_scored"].iloc[0])  # expected 68
SEQ_LEN = int(df_train["seq_length"].iloc[0])  # expected 107

df_train_use = df_train[df_train["SN_filter"] == 1].copy()
if df_train_use.shape[0] == 0:
    df_train_use = df_train.copy()

if "signal_to_noise" in df_train_use.columns:
    s2n = pd.to_numeric(df_train_use["signal_to_noise"], errors="coerce")
    s2n = s2n.replace([np.inf, -np.inf], np.nan)
    S2N_Q_LOW = 0.10
    thr = (
        float(np.nanquantile(s2n.values, S2N_Q_LOW))
        if np.isfinite(s2n.values).any()
        else -np.inf
    )
    keep = (s2n.values >= thr) & np.isfinite(s2n.values)
    if keep.sum() >= 50:  # guard: ensure we still have enough rows
        df_train_use = df_train_use.loc[keep].copy()

WINSOR_Q_LOW = 0.05
WINSOR_Q_HIGH = 0.95


def weighted_quantile(values, quantiles, sample_weight):
    """
    Deterministic weighted quantile for 1D arrays.
    values: 1D
    quantiles: float or array-like in [0,1]
    sample_weight: 1D non-negative
    """
    values = np.asarray(values, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    m = np.isfinite(values) & np.isfinite(sample_weight) & (sample_weight > 0)
    if not np.any(m):
        q = np.atleast_1d(quantiles).astype(np.float64)
        return np.full_like(q, np.nan, dtype=np.float64) if q.ndim else np.nan
    v = values[m]
    w = sample_weight[m]
    sorter = np.argsort(v)
    v = v[sorter]
    w = w[sorter]
    cw = np.cumsum(w)
    cw /= cw[-1]
    q = np.atleast_1d(quantiles).astype(np.float64)
    out = np.interp(q, cw, v)
    return out if np.ndim(quantiles) else float(out[0])


train_global_means = {}
train_pos_means = {}

if "signal_to_noise" in df_train_use.columns:
    base_w = (
        pd.to_numeric(df_train_use["signal_to_noise"], errors="coerce")
        .astype(np.float64)
        .values
    )
    base_w = np.where(np.isfinite(base_w) & (base_w > 0), base_w, 0.0)
else:
    base_w = np.ones((df_train_use.shape[0],), dtype=np.float64)


def compute_pos_mean(mat, base_w):
    """Compute per-position winsorized, s2n-weighted mean (core logic unchanged)."""
    pos_mean = np.empty((mat.shape[1],), dtype=np.float64)
    for j in range(mat.shape[1]):
        v = mat[:, j]
        m = np.isfinite(v)
        vj = v[m]
        wj = base_w[m]
        if vj.size == 0:
            pos_mean[j] = np.nan
            continue

        if vj.size >= 20 and np.any(wj > 0):
            lo = weighted_quantile(vj, WINSOR_Q_LOW, wj)
            hi = weighted_quantile(vj, WINSOR_Q_HIGH, wj)
            if np.isfinite(lo) and np.isfinite(hi) and lo <= hi:
                vj2 = np.clip(vj, lo, hi)
                wj2 = wj
            else:
                vj2, wj2 = vj, wj
        else:
            vj2, wj2 = vj, wj

        sw = np.sum(wj2)
        if sw > 0:
            pos_mean[j] = float(np.sum(vj2 * wj2) / sw)
        else:
            pos_mean[j] = float(np.mean(vj2))

    if np.isnan(pos_mean).any():
        fallback = np.nanmean(mat, axis=0).astype(np.float64)
        pos_mean = np.where(np.isnan(pos_mean), fallback, pos_mean)
    return pos_mean.astype(np.float64)


scored_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
for col in scored_cols + ["deg_pH10", "deg_50C"]:
    arrs = df_train_use[col].values  # each is length 68 list/array
    mat = np.vstack([np.asarray(a, dtype=np.float64) for a in arrs])  # (n_samples, 68)
    pos_mean = compute_pos_mean(mat, base_w)
    train_pos_means[col] = pos_mean
    train_global_means[col] = float(np.nanmean(pos_mean))


def compute_pos_linear_map(x_mat, y_mat, base_w, eps=1e-12):
    """
    Per-position weighted linear regression y ~= a*x + b using closed-form weighted moments.
    Returns a_pos, b_pos (each length L).
    """
    x_mat = np.asarray(x_mat, dtype=np.float64)
    y_mat = np.asarray(y_mat, dtype=np.float64)
    w = np.asarray(base_w, dtype=np.float64).reshape(-1, 1)

    m = np.isfinite(x_mat) & np.isfinite(y_mat) & np.isfinite(w) & (w > 0)
    w_eff = np.where(m, w, 0.0)

    sw = np.sum(w_eff, axis=0)
    sw = np.where(sw > 0, sw, np.nan)

    mx = np.nansum(w_eff * x_mat, axis=0) / sw
    my = np.nansum(w_eff * y_mat, axis=0) / sw

    cov = np.nansum(w_eff * (x_mat - mx) * (y_mat - my), axis=0) / sw
    varx = np.nansum(w_eff * (x_mat - mx) * (x_mat - mx), axis=0) / sw

    a = cov / np.where(np.isfinite(varx) & (varx > eps), varx, np.nan)
    b = my - a * mx

    x_all = x_mat[np.isfinite(x_mat) & np.isfinite(y_mat)]
    y_all = y_mat[np.isfinite(x_mat) & np.isfinite(y_mat)]
    if x_all.size >= 10:
        vx = np.var(x_all)
        if vx > eps:
            a_glob = float(np.cov(x_all, y_all, bias=True)[0, 1] / vx)
        else:
            a_glob = 0.0
        b_glob = float(np.mean(y_all) - a_glob * np.mean(x_all))
    else:
        a_glob, b_glob = 0.0, float(np.nanmean(y_all)) if y_all.size else 0.0

    a = np.where(np.isfinite(a), a, a_glob)
    b = np.where(np.isfinite(b), b, b_glob)
    return a.astype(np.float64), b.astype(np.float64)


arr_mg_ph10 = np.vstack(
    [np.asarray(a, dtype=np.float64) for a in df_train_use["deg_Mg_pH10"].values]
)
arr_ph10 = np.vstack(
    [np.asarray(a, dtype=np.float64) for a in df_train_use["deg_pH10"].values]
)
a_ph10, b_ph10 = compute_pos_linear_map(arr_mg_ph10, arr_ph10, base_w)

arr_mg_50c = np.vstack(
    [np.asarray(a, dtype=np.float64) for a in df_train_use["deg_Mg_50C"].values]
)
arr_50c = np.vstack(
    [np.asarray(a, dtype=np.float64) for a in df_train_use["deg_50C"].values]
)
a_50c, b_50c = compute_pos_linear_map(arr_mg_50c, arr_50c, base_w)

train_pos_means["deg_pH10"] = (a_ph10 * train_pos_means["deg_Mg_pH10"] + b_ph10).astype(
    np.float64
)
train_pos_means["deg_50C"] = (a_50c * train_pos_means["deg_Mg_50C"] + b_50c).astype(
    np.float64
)

train_global_means["deg_pH10"] = float(np.nanmean(train_pos_means["deg_pH10"]))
train_global_means["deg_50C"] = float(np.nanmean(train_pos_means["deg_50C"]))

train_global_means



## === cell 3
id_seqpos = df["id_seqpos"].astype(str)
seqpos = id_seqpos.str.rsplit("_", n=1, expand=True)[1].astype(int)

for col in target_cols:
    pred = np.full(
        shape=(len(df),), fill_value=train_global_means[col], dtype=np.float64
    )
    scored_mask = seqpos.values < SEQ_SCORED
    pred[scored_mask] = train_pos_means[col][seqpos.values[scored_mask]]
    df[col] = pred

mask = df["id_seqpos"].astype(str).str.startswith("id_1f6d9e2dc")
if mask.any():
    df.loc[mask, "reactivity"] = 0.0



## === cell 4
required_cols = ["id_seqpos"] + target_cols
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Submission missing required columns: {missing}")

df = df[required_cols]

for col in target_cols:
    df[col] = (
        pd.to_numeric(df[col], errors="coerce")
        .fillna(train_global_means[col])
        .astype(np.float32)
    )

expected_rows = pd.read_csv(
    "../input/stanford-covid-vaccine/sample_submission.csv"
).shape[0]
if df.shape[0] != expected_rows:
    raise ValueError(
        f"Unexpected submission row count: got {df.shape[0]}, expected {expected_rows}"
    )

df.head()



## === cell 5
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print("Columns:", list(df.columns))
