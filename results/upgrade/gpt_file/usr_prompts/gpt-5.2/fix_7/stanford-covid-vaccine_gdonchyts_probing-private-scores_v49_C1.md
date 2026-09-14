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

0.46496

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I remove the dependency on the missing `../input/worst-submission/ensemble52.csv` by instead starting from the provided `sample_submission.csv`, which guarantees correct rows/columns and avoids the `FileNotFoundError`. I also fix the incorrect private-set detection logic (this dataset uses `seq_length==107`, not 130) and guard against index errors when slicing `sequences[-10:]`. Finally, I keep your intended single-id override (`id_ffda94f24`) but make it safe even if that id doesn’t exist, and ensure the code always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.42418) has done: 'Your current code is essentially writing the sample submission (all zeros) plus a single-id override, which score poorly; to move toward the target, the smallest legitimate improvement is to replace zeros with simple, data-driven priors computed from the training set. I keep your exact submission-building approach (start from `sample_submission.csv` to guarantee format) and your safe single-id override, but fill predictions using per-position means from `train.json` (a strong baseline for this competition). For unscored positions (beyond 68), we extend using the last available mean value to keep shapes valid. This preserves the overall approach (no model/architecture) while substantially reducing error versus all-zeros, moving the score closer to 0.3519.'
- What this solution (achieved 0.42166) has done: 'We keep your exact “per-position mean from train.json” baseline and submission-building flow, but make two small, metric-relevant improvements that should reduce MCRMSE toward the target: (1) compute the means only from higher-quality training rows (`SN_filter==1`) to better match the filtered test distribution, and (2) clip predictions to a reasonable range derived from the training distribution to reduce the impact of outliers (important for RMSE). We also remove the `id_ffda94f24` hard override because it can only worsen RMSE (it injects a huge error) and is not needed for submission validity. All I/O paths and output format remain unchanged, and the script still writes `submission.csv`.'
- What this solution (achieved 0.48245) has done: 'Your current solution is a solid “per-position mean” baseline but it still uses all SN_filter==1 rows equally; to move the score down toward the target, the smallest legitimate gain is to weight the per-position averages by measurement reliability using the provided per-position error columns (lower error ⇒ higher weight), which is directly aligned with reducing RMSE. I keep the same overall logic (train-derived priors filled into sample_submission by seqpos) and the same clipping idea, but compute *weighted* means for each target using its corresponding `*_error_*` arrays. This should improve calibration especially on harder/noisier positions and reduce MCRMSE without changing any modeling approach. The script still run end-to-end on the same paths and still write a valid `submission.csv` in the required format.'
- What this solution (achieved 0.47409) has done: 'Your current weighted-per-position mean baseline is reasonable but can be nudged closer to the target by better matching the train/test distribution and making the weighting more stable. I (1) further filter the training rows used for priors to the same “high-quality” regime as test by requiring `SN_filter==1` *and* a minimum `signal_to_noise` threshold, and (2) make the inverse-variance weights less extreme by clipping very small errors (prevents a few positions/rows from dominating), which typically reduces RMSE. I also align the clip bounds computation with the filtered training subset (so clipping matches what the priors were fit on), while keeping the same submission-building approach and output schema. Paths, core “train-derived priors filled by seqpos into sample_submission” logic, and runtime constraints remain unchanged.'
- What this solution (achieved 0.46496) has done: 'We keep your exact “train-derived per-position priors filled into `sample_submission.csv` by `seqpos`” approach, but nudge it closer to the target by better matching the public/private test filtering and stabilizing the weighted mean. Specifically, we (1) slightly raise the `signal_to_noise` threshold (test is curated to higher SNR, so this usually reduces MCRMSE), (2) use a safer, less-extreme inverse-variance weighting (cap max weight and use a slightly higher error floor) to avoid a few rows dominating, and (3) compute clipping bounds from a slightly wider quantile range (0.5%–99.5%) to reduce over-clipping bias while still preventing outliers from hurting RMSE. These are minimal changes that preserve your core logic and keep runtime well under the limit, while aiming to improve the current 0.47409 toward 0.35188.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
test_path = "../input/stanford-covid-vaccine/test.json"
train_path = "../input/stanford-covid-vaccine/train.json"
df_test = pd.read_json(test_path, lines=True)
df_train = pd.read_json(train_path, lines=True)

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
    raise ValueError(f"Sample submission missing required columns: {missing}")



## === cell 2
sequences = sorted(df_test["id"].unique().tolist())



## === cell 3
tail_n = min(10, len(sequences))
_ = sequences[-tail_n:]
print(_)



## === cell 4
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

N_SCORED = 68
N_TOTAL = 107

df_train_used = df_train.copy()
if "SN_filter" in df_train_used.columns and (df_train_used["SN_filter"] == 1).any():
    df_train_used = df_train_used.loc[df_train_used["SN_filter"] == 1]

if "signal_to_noise" in df_train_used.columns:
    df_train_used = df_train_used.loc[df_train_used["signal_to_noise"] >= 1.25]

df_train_used = df_train_used.reset_index(drop=True)
if len(df_train_used) == 0:
    df_train_used = df_train.reset_index(drop=True)


def _safe_stack(frame, col, n_scored=68):
    arrs = []
    for x in frame[col].values:
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
            a = np.asarray(x, dtype=np.float64)
        else:
            a = np.full((n_scored,), np.nan, dtype=np.float64)
        if a.shape[0] < n_scored:
            a = np.pad(a, (0, n_scored - a.shape[0]), constant_values=np.nan)
        elif a.shape[0] > n_scored:
            a = a[:n_scored]
        arrs.append(a)
    return np.vstack(arrs)


def _robust_clip_bounds(frame, col):
    mat = _safe_stack(frame, col, n_scored=N_SCORED)  # (n, 68)
    v = mat.reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -np.inf, np.inf
    lo = np.quantile(v, 0.005)
    hi = np.quantile(v, 0.995)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = np.min(v), np.max(v)
    return float(lo), float(hi)


error_map = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

pos_means = {}
clip_bounds = {}

EPS = 1e-6  # avoid division by zero

ERR_FLOOR = 0.03
MAX_W = 1.0 / (ERR_FLOOR**2)  # consistent cap relative to floor

for col in target_cols:
    y = _safe_stack(df_train_used, col, n_scored=N_SCORED)  # (n, 68)

    err_col = error_map.get(col, None)
    if err_col is not None and err_col in df_train_used.columns:
        e = _safe_stack(df_train_used, err_col, n_scored=N_SCORED)  # (n, 68)
        e = np.where(np.isfinite(e), np.maximum(e, ERR_FLOOR), np.nan)

        w = 1.0 / (np.square(e) + EPS)
        w = np.clip(w, 0.0, MAX_W)
        w[~np.isfinite(w)] = 0.0

        y_mask = np.isfinite(y)
        w = w * y_mask

        denom = np.sum(w, axis=0)  # (68,)
        numer = np.sum(w * np.nan_to_num(y, nan=0.0), axis=0)
        m = np.divide(numer, np.where(denom > 0, denom, np.nan))
    else:
        m = np.nanmean(y, axis=0)

    if not np.isfinite(m[-1]):
        overall = np.nanmean(m)
        if not np.isfinite(overall):
            overall = 0.0
        m = np.nan_to_num(m, nan=overall)

    m_full = np.concatenate(
        [m, np.full((N_TOTAL - N_SCORED,), m[-1], dtype=np.float64)]
    )
    m_full = np.nan_to_num(m_full, nan=0.0)
    pos_means[col] = m_full

    clip_bounds[col] = _robust_clip_bounds(df_train_used, col)

seqpos = df["id_seqpos"].astype(str).str.rsplit("_", n=1).str[-1]
seqpos = (
    pd.to_numeric(seqpos, errors="coerce").fillna(0).astype(int).clip(0, N_TOTAL - 1)
)

for col in target_cols:
    pred = pos_means[col][seqpos.values]
    lo, hi = clip_bounds[col]
    if np.isfinite(lo) and np.isfinite(hi):
        pred = np.clip(pred, lo, hi)
    df[col] = pred



## === cell 5
for c in required_cols[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
df["id_seqpos"] = df["id_seqpos"].astype(str)

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print("Columns:", df.columns.tolist())
