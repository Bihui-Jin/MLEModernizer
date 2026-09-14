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

0.48565

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'The current notebook fails immediately because it tries to load a non-existent external submission file (`../input/worst-submission/ensemble52.csv`), so `df` is never created and all later cells error. To make this run end-to-end without changing the competition semantics, I instead load the provided `sample_submission.csv` from the competition dataset and use it as a valid baseline submission. I keep the existing “set reactivity to 0 for a specific id prefix” line (but guard it so it won’t error if the prefix isn’t present). Finally, I write `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.42418) has done: 'Your current score is far above (worse than) the target for a lower-is-better metric, so we need a legitimate but still minimal improvement beyond the all-zeros-style baseline. Keeping the “submission-from-template” core flow, I fit a very small baseline model: per-position means of the 5 targets from `train.json`, and use those means as predictions for each `seqpos` in the sample submission. This preserves simple semantics (no architecture/training loops), runs fast, and typically improves MCRMSE substantially versus zeros by matching the empirical marginal distributions per position. I also keep your special-case `id_b9c266213_` reactivity override exactly as you had it.'
- What this solution (achieved 0.4437) has done: 'Your current score (0.42418, lower-is-better) is worse than the target (0.35188), so we should make a small, legitimate improvement without changing the overall “per-position baseline” core approach. The biggest low-risk gain here is to stop training noise/outliers from dominating the per-position means by (1) filtering to SN_filter==1 (high-quality measurements) and (2) using a per-position median instead of mean, which is still the same basic “constant per position” model but more robust. We keep your submission construction, required columns, and the special-case `id_b9c266213_` reactivity override exactly as before. This should generally reduce MCRMSE versus the raw mean baseline while staying simple and fast.'
- What this solution (achieved 0.48654) has done: 'To move your lower-is-better MCRMSE score closer to the 0.35188 target (from 0.4437), we keep the same “per-position constant” baseline but improve it in two minimal, legitimate ways. First, we compute per-position **weighted** robust central tendency using the provided per-position measurement errors (inverse-variance weights) instead of an unweighted median, still using only `train.json`. Second, we add a very small amount of shrinkage toward the global (across-position) weighted mean to reduce variance from sparse/noisy positions, which typically helps this competition while preserving the same overall modeling semantics. All I/O paths and the submission schema remain unchanged, and we still write a valid `submission.csv`.'
- What this solution (achieved 0.48258) has done: 'Your current score (0.48654, lower-is-better) is worse than the target (0.35188), so we should make a small, legitimate improvement while keeping the same “per-seqpos constant” baseline. The lowest-risk gain is to tune the existing shrinkage strength `alpha` using an internal cross-validation on the training set (still the same model: weighted per-position mean shrunk toward a global mean), selecting the `alpha` that minimizes validation MCRMSE on the 3 scored targets. This doesn’t change architecture/training loops (there are none), doesn’t add approximations, and typically improves leaderboard score by choosing a better bias/variance tradeoff for this dataset. All I/O paths and the submission schema remain unchanged, and we still write a valid `submission.csv`.'
- What this solution (achieved 0.48258) has done: 'We’re currently worse than the target (0.48258 vs 0.35188, lower-is-better), so we should make a small, legitimate improvement while keeping your exact “per-seqpos constant, error-weighted mean + global shrinkage” core logic. The main issue is the CV split is random by RNA id, but the metric is evaluated only on the first `seq_scored` positions; a more stable and representative CV here is to stratify the split by `signal_to_noise` (and keep SN_filter==1) so alpha is tuned on a validation set that better matches the high-quality distribution used in the public test filtering. This does not change the model family or introduce any approximations; it just improves the alpha selection stability, which should move MCRMSE down toward the target. I also keep all I/O paths and the special-case `id_b9c266213_` override unchanged and still write a valid `submission.csv`.'
- What this solution (achieved 0.48258) has done: 'Your current MCRMSE (0.48258, lower-is-better) is still far worse than the target (0.35188), so we should make a small, legitimate improvement while keeping the same “per-seqpos constant prediction = error-weighted mean + global shrinkage” core logic. The biggest issue is that the shrinkage alpha is being tuned on only one split, which is noisy and can easily pick a poor alpha; I replace that with a small K-fold CV over ids (still just tuning alpha, no new model). I also make the CV objective match the competition slightly better by weighting per-position errors using the provided measurement-error weights for the scored targets (this stays within your existing weighting scheme and evaluation semantics). Everything else (data loading, stats computation, submission construction, and the id_b9c266213_ override) stays intact, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.48565) has done: 'Your current score (0.48258, lower-is-better) is still far from the target (0.35188), so we should make a small but meaningful improvement while keeping the same core “per-seqpos constant prediction = error-weighted mean + global shrinkage” approach. The biggest low-risk gain is to compute the per-position statistic on the *training* side as an error-weighted mean after lightly winsorizing target values within each seqpos/target (clipping extreme outliers using weighted quantiles), which reduces the impact of low-quality/outlier measurements without changing the model family. I keep your K-fold alpha tuning and submission construction intact, but tune alpha against the unweighted competition metric (to better match MCRMSE) while still fitting the same weighted+shrinkage statistics. These changes are minimal, fast, and directly aimed at lowering MCRMSE toward the target without altering I/O paths or submission schema.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
CANDIDATE_SAMPLE_PATHS = [
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "../input/stanford-covid-vaccine/sample_submission.csv",
    "../input/sample_submission.csv",
    "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]
sample_path = next((p for p in CANDIDATE_SAMPLE_PATHS if os.path.exists(p)), None)
if sample_path is None:
    raise FileNotFoundError(
        f"Could not find sample_submission.csv in any expected location: {CANDIDATE_SAMPLE_PATHS}"
    )

CANDIDATE_TRAIN_PATHS = [
    "/kaggle/input/stanford-covid-vaccine/train.json",
    "/kaggle/input/train.json",
    "../input/stanford-covid-vaccine/train.json",
    "../input/train.json",
    "/kaggle/data/stanford-covid-vaccine/train.json",
    "/kaggle/data/train.json",
]
train_path = next((p for p in CANDIDATE_TRAIN_PATHS if os.path.exists(p)), None)
if train_path is None:
    raise FileNotFoundError(
        f"Could not find train.json in any expected location: {CANDIDATE_TRAIN_PATHS}"
    )

sample_df = pd.read_csv(sample_path)
train_df = pd.read_json(train_path, lines=True)

print("sample_submission:", sample_df.shape, "train:", train_df.shape)
sample_df.head()



## === cell 2
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
error_cols = [
    "reactivity_error",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_error_Mg_50C",
    "deg_error_50C",
]
scored_targets = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
scored_idx = [target_cols.index(c) for c in scored_targets]

for c in target_cols:
    if c not in train_df.columns:
        raise ValueError(f"train.json missing expected target column: {c}")

seq_scored = int(train_df["seq_scored"].iloc[0])
seq_length = int(train_df["seq_length"].iloc[0])

if "SN_filter" in train_df.columns:
    train_df_f = train_df.loc[train_df["SN_filter"].astype(int) == 1].copy()
    if len(train_df_f) == 0:
        train_df_f = train_df
else:
    train_df_f = train_df

Y_all = np.stack([np.stack(train_df_f[c].values) for c in target_cols], axis=-1).astype(
    np.float64
)  # (n, 68, 5)
if Y_all.shape[1] != seq_scored:
    raise ValueError(
        f"Unexpected seq_scored length in train targets: got {Y_all.shape[1]}, expected {seq_scored}"
    )

have_all_errors = all(c in train_df_f.columns for c in error_cols)
if have_all_errors:
    E_all = np.stack(
        [np.stack(train_df_f[c].values) for c in error_cols], axis=-1
    ).astype(
        np.float64
    )  # (n, 68, 5)
    E_all = np.clip(E_all, 1e-3, None)
    W_all = 1.0 / (E_all**2)
else:
    W_all = np.ones_like(Y_all, dtype=np.float64)

print("Using train rows:", len(train_df_f), "of", len(train_df))
print("Using error-based weights:", have_all_errors)




## === cell 3
def _weighted_means(Y, W):
    w_sum = np.sum(W, axis=0)  # (68,5)
    wy_sum = np.sum(W * Y, axis=0)  # (68,5)
    pos_wmean = wy_sum / np.clip(w_sum, 1e-12, None)

    global_wmean = np.sum(W * Y, axis=(0, 1)) / np.clip(
        np.sum(W, axis=(0, 1)), 1e-12, None
    )
    return pos_wmean, global_wmean


def _mcrmse_unweighted(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2, axis=0)  # (68,k)
    rmse_col = np.sqrt(np.mean(mse, axis=0))  # (k,)
    return float(np.mean(rmse_col))


def _weighted_quantile_1d(x, w, q):
    """
    Compute weighted quantile for 1D arrays (q in [0,1]).
    Small helper to winsorize outliers per seqpos/target while keeping the
    same core constant-per-position model; this typically reduces MCRMSE.
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    x = x[m]
    w = w[m]
    if x.size == 1:
        return float(x[0])
    idx = np.argsort(x)
    xs = x[idx]
    ws = w[idx]
    cdf = np.cumsum(ws)
    cutoff = q * cdf[-1]
    j = int(np.searchsorted(cdf, cutoff, side="left"))
    j = min(max(j, 0), xs.size - 1)
    return float(xs[j])


def _winsorize_Y_per_pos(Y, W, low_q=0.02, high_q=0.98):
    """
    Change is directly score-motivated: clip extreme Y values within each
    (seqpos, target) using weighted quantiles from the training distribution.
    This keeps the model family identical (per-position constant) but makes
    the estimated constants more robust, usually lowering MCRMSE.
    """
    Yw = Y.copy()
    n, L, T = Y.shape
    for p in range(L):
        for t in range(T):
            lo = _weighted_quantile_1d(Y[:, p, t], W[:, p, t], low_q)
            hi = _weighted_quantile_1d(Y[:, p, t], W[:, p, t], high_q)
            if np.isfinite(lo) and np.isfinite(hi) and (hi >= lo):
                Yw[:, p, t] = np.clip(Yw[:, p, t], lo, hi)
    return Yw


Y_all_rob = _winsorize_Y_per_pos(Y_all, W_all, low_q=0.02, high_q=0.98)

rng = np.random.default_rng(0)
n = Y_all.shape[0]

alpha_grid = np.array(
    [0.00, 0.03, 0.06, 0.09, 0.12, 0.16, 0.20, 0.25, 0.32, 0.40], dtype=np.float64
)

K = 5
if "signal_to_noise" in train_df_f.columns:
    s2n = pd.to_numeric(train_df_f["signal_to_noise"], errors="coerce")
    s2n = s2n.fillna(s2n.median())
    s2n = np.asarray(s2n, dtype=np.float64)

    try:
        bins = pd.qcut(s2n, q=K, labels=False, duplicates="drop")
        bins = np.asarray(bins, dtype=int)
        unique_bins = np.unique(bins[~np.isnan(bins)])
    except Exception:
        bins = None
        unique_bins = []

    if bins is None or len(unique_bins) <= 1:
        perm = rng.permutation(n)
        fold_id = np.empty(n, dtype=int)
        fold_id[perm] = np.arange(n) % K
    else:
        fold_id = np.full(n, -1, dtype=int)
        for b in unique_bins:
            b_idx = np.where(bins == b)[0]
            b_perm = rng.permutation(b_idx)
            fold_id[b_perm] = np.arange(b_perm.size) % K

        leftovers = np.where(fold_id < 0)[0]
        if leftovers.size:
            fold_id[leftovers] = rng.integers(0, K, size=leftovers.size)
else:
    perm = rng.permutation(n)
    fold_id = np.empty(n, dtype=int)
    fold_id[perm] = np.arange(n) % K

best_alpha = None
best_score = None

for a in alpha_grid:
    fold_scores = []
    for k in range(K):
        val_idx = np.where(fold_id == k)[0]
        tr_idx = np.where(fold_id != k)[0]
        if val_idx.size == 0 or tr_idx.size == 0:
            continue

        Y_tr, W_tr = Y_all_rob[tr_idx], W_all[tr_idx]
        Y_val = Y_all[val_idx]  # use original values for metric computation

        pos_wmean_tr, global_wmean_tr = _weighted_means(Y_tr, W_tr)
        pos_stats_tr = (1.0 - a) * pos_wmean_tr + a * global_wmean_tr[None, :]  # (68,5)

        yhat_val = np.broadcast_to(pos_stats_tr[None, :, :], Y_val.shape)

        y_true_s = Y_val[:, :, scored_idx]
        y_pred_s = yhat_val[:, :, scored_idx]
        score_k = _mcrmse_unweighted(y_true_s, y_pred_s)
        fold_scores.append(score_k)

    score = float(np.mean(fold_scores)) if fold_scores else np.inf
    if (best_score is None) or (score < best_score):
        best_score = score
        best_alpha = float(a)

print(
    f"K-fold CV selected alpha={best_alpha:.3f} (avg val MCRMSE on scored targets={best_score:.6f})"
)

pos_wmean_all, global_wmean = _weighted_means(Y_all_rob, W_all)
alpha = best_alpha
pos_stats = (1.0 - alpha) * pos_wmean_all + alpha * global_wmean[None, :]

pos_stat_df = pd.DataFrame(pos_stats, columns=target_cols)
pos_stat_df["seqpos"] = np.arange(seq_scored, dtype=int)
pos_stat_df.head()



## === cell 4
if "id_seqpos" not in sample_df.columns:
    raise ValueError(
        f"Expected 'id_seqpos' in submission dataframe, got columns: {sample_df.columns.tolist()}"
    )

sub = sample_df[["id_seqpos"]].copy()
sub["seqpos"] = sub["id_seqpos"].astype(str).str.rsplit("_", n=1).str[-1].astype(int)

sub = sub.merge(pos_stat_df, on="seqpos", how="left")

for i, c in enumerate(target_cols):
    fill_val = float(global_wmean[i]) if np.isfinite(global_wmean[i]) else 0.0
    sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(fill_val)

sub.drop(columns=["seqpos"], inplace=True)
sub.head(), sub.shape



## === cell 5
mask = sub["id_seqpos"].astype(str).str.startswith("id_b9c266213_")
if mask.any():
    sub.loc[mask, "reactivity"] = 0.0

required_cols = ["id_seqpos"] + target_cols
missing = [c for c in required_cols if c not in sub.columns]
if missing:
    raise ValueError(
        f"Submission is missing required columns: {missing}. Found: {sub.columns.tolist()}"
    )

if sub["id_seqpos"].isna().any():
    raise ValueError("Found NA in id_seqpos, cannot submit.")

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape={sub.shape} and columns={sub.columns.tolist()}")
print(sub.head())
