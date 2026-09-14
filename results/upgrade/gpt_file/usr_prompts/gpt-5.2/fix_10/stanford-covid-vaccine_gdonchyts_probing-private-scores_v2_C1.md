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

0.44834

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42418) has done: 'I remove the dependency on the missing `../input/worst-submission/ensemble52.csv` and instead load the provided `sample_submission.csv` plus `test.json` from the competition dataset paths that exist in your environment. Then I generate a valid baseline prediction by using the per-position mean targets computed from the training set (this is score-improving versus all-zeros, while keeping logic simple and stable). I also fix the incorrect row selection in the “probe_sequence” cell (it currently matches no rows because `id_seqpos` contains `id_..._pos`). Finally, I guarantee the submission has exactly the required 25680 rows/6 columns and write `submission.csv` in the working directory.'
- What this solution (achieved 0.42166) has done: 'We keep your per-position mean baseline (core logic) but remove the single-sequence “probe” zeroing, since it artificially worsens MCRMSE and moves you away from the lower-is-better target. We also compute the position means only from high-quality training samples (`SN_filter==1`), which is a minimal, competition-consistent change that typically improves generalization and should reduce the score gap toward 0.3519. Finally, we keep the submission format identical and still fill unscored tail positions deterministically, ensuring a valid `submission.csv` is always produced.'
- What this solution (achieved 0.42178) has done: 'We keep your per-position mean baseline (same core logic) but make two minimal, competition-consistent tweaks that usually reduce MCRMSE: (1) compute position means using a signal-to-noise weighted average on the high-quality (`SN_filter==1`) subset, so cleaner samples contribute more; and (2) clip predictions to a conservative range derived from the training distribution (per target) to reduce the impact of outliers, without changing the model approach. We also remove the now-unused probe cell’s effect (it already doesn’t change predictions) while keeping it harmless. The script still write a valid `submission.csv` with exactly the required rows/columns.'
- What this solution (achieved 0.42178) has done: 'We’re currently worse than the target (lower-is-better), so we should make a minimal, metric-consistent change that improves generalization without changing the core “per-position mean baseline” logic. The biggest safe gain usually comes from applying the competition’s known quality filtering more strictly by using only higher signal-to-noise examples (not just `SN_filter==1`) when computing the per-position means. I keep the same weighted-per-position averaging and the same submission construction, but restrict the averaging set to `SN_filter==1` and `signal_to_noise >= 1.0` (matching the dataset’s own quality criterion); if that filter becomes empty, we fall back safely to the prior behavior. Everything else (including clipping, tail fill, and output format) remains unchanged and it still write a valid `submission.csv`.'
- What this solution (achieved 0.45452) has done: 'We’re currently worse than the target (lower-is-better), so the smallest safe move is to improve the per-position mean baseline without changing its core logic. I keep the exact “per-position average over training” approach, but (1) compute means only over the *reliably measured* positions by weighting each sample-position using the provided per-position measurement errors (inverse-variance weighting), and (2) slightly refine the tail (unscored positions 68–106) fill by using the mean of the last few scored positions instead of just position 67 to reduce variance. Submission formatting, paths, targets, clipping, and the overall pipeline remain the same, and it still write a valid `submission.csv`.'
- What this solution (achieved 0.45112) has done: 'We’re currently worse than the target (lower-is-better), so the smallest score-improving change is to keep your exact per-position baseline but make the inverse-variance weights more robust by capping extremely small errors that otherwise dominate the weighted mean and can hurt generalization. This preserves the same core logic (positionwise averaging with SNR and error weighting), but reduces variance from outlier-weighted samples. I also compute clipping bounds from the same robust error-capped distribution (still quantile-based) to stay consistent with the new weighting stability. Everything else (paths, tail fill, submission format/rows/columns) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.44832) has done: 'We’re currently worse than the target (lower-is-better), so the smallest safe move is to reduce variance/overfitting in the per-position inverse-variance averaging without changing the overall “positionwise weighted mean” logic. I keep the exact pipeline, but make the error-weighting more robust by (1) flooring errors per-position (not globally) and (2) additionally clipping the maximum per-sample weight contribution to avoid a few near-zero-error measurements dominating a position. This should improve generalization and move MCRMSE down toward the target while preserving the same evaluation semantics and submission format. The script still run end-to-end and write a valid `submission.csv` with the required 25680 rows and 6 columns.'
- What this solution (achieved 0.44851) has done: 'We’re still worse than the target (lower-is-better), so the smallest score-improving change is to reduce noise/variance in the positionwise weighted-mean baseline without changing its core “per-position averaging” logic. I keep your SNR + inverse-variance weighting, but (1) winsorize (clip) training target values per position to a tight quantile band before averaging so extreme outliers don’t skew the mean, and (2) compute clipping bounds from the same high-quality subset but using slightly less aggressive quantiles to avoid over-clipping. Everything else (data loading, weighting structure, tail fill strategy, submission formatting and row order) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.44834) has done: 'We’re still worse than the target (lower-is-better), so we should make the smallest change that plausibly reduces MCRMSE without altering the core “per-position weighted mean baseline” logic. The current winsorization (1%–99% per position) is likely too aggressive and can bias the mean; I relax it to a very mild winsorization (0.1%–99.9%) to keep robustness while reducing bias. I also align the global prediction clipping bounds to the same slightly wider quantiles (0.1%–99.9%) so we don’t over-clip and hurt calibration. Everything else (HQ filtering, SNR+inverse-variance weighting, weight caps, tail fill, submission formatting) stays the same and still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import json
import numpy as np
import pandas as pd



## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/stanford-covid-vaccine",
    "/kaggle/data/stanford-covid-vaccine",
    "/kaggle/input",
    "/kaggle/data",
]


def find_existing(*relative_paths):
    for base in BASE_CANDIDATES:
        for rel in relative_paths:
            p = os.path.join(base, rel)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(
        f"Could not find any of: {relative_paths} under {BASE_CANDIDATES}"
    )


train_path = find_existing("train.json", "stanford-covid-vaccine/train.json")
test_path = find_existing("test.json", "stanford-covid-vaccine/test.json")
sample_sub_path = find_existing(
    "sample_submission.csv", "stanford-covid-vaccine/sample_submission.csv"
)

train = pd.read_json(train_path, lines=True)
test = pd.read_json(test_path, lines=True)
sample_sub = pd.read_csv(sample_sub_path)

train.shape, test.shape, sample_sub.shape



## === cell 2
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
ERROR_COLS = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

SEQ_LEN = int(test["seq_length"].iloc[0])  # 107
SCORED_LEN = int(train["seq_scored"].iloc[0])  # 68

train_hq = train[
    (train["SN_filter"] == 1) & (train["signal_to_noise"] >= 1.0)
].reset_index(drop=True)
if len(train_hq) == 0:
    train_hq = train[train["SN_filter"] == 1].reset_index(drop=True)
if len(train_hq) == 0:
    train_hq = train.reset_index(drop=True)  # final safety fallback

w_snr = train_hq["signal_to_noise"].astype(np.float32).to_numpy()
w_snr = np.nan_to_num(w_snr, nan=0.0, posinf=0.0, neginf=0.0)
w_snr = np.clip(w_snr, 0.0, None)
if float(w_snr.sum()) <= 0.0:
    w_snr = np.ones(len(train_hq), dtype=np.float32)

pos_means = {}

EPS = 1e-3
ERR_FLOOR_QUANTILE = 0.10  # conservative floor to reduce domination by tiny errors

W_CLIP_QUANTILE = 0.99  # cap per-position weights at this quantile
W_CLIP_MIN = np.float32(1.0)  # avoid capping below a sensible scale

Y_WINSOR_QLO = 0.001
Y_WINSOR_QHI = 0.999

for col in TARGET_COLS:
    y = np.asarray(train_hq[col].tolist(), dtype=np.float32)  # (n, 68)

    y_lo = np.quantile(y, Y_WINSOR_QLO, axis=0, method="linear").astype(np.float32)
    y_hi = np.quantile(y, Y_WINSOR_QHI, axis=0, method="linear").astype(np.float32)
    y = np.clip(y, y_lo[None, :], y_hi[None, :]).astype(np.float32)

    err_col = ERROR_COLS[col]
    if err_col in train_hq.columns:
        e = np.asarray(train_hq[err_col].tolist(), dtype=np.float32)  # (n, 68)
        e = np.nan_to_num(e, nan=np.inf, posinf=np.inf, neginf=np.inf)

        e_floors = np.quantile(
            np.where(np.isfinite(e), e, np.nan),
            ERR_FLOOR_QUANTILE,
            axis=0,
            method="linear",
        ).astype(np.float32)
        e_floors = np.where(
            np.isfinite(e_floors) & (e_floors > 0.0), e_floors, np.float32(0.05)
        )

        e = np.maximum(e, e_floors[None, :])
        w_pos = (w_snr[:, None] / (e * e + EPS)).astype(np.float32)

        w_cap = np.quantile(w_pos, W_CLIP_QUANTILE, axis=0, method="linear").astype(
            np.float32
        )
        w_cap = np.where(np.isfinite(w_cap) & (w_cap > 0.0), w_cap, np.float32(0.0))
        w_cap = np.maximum(w_cap, W_CLIP_MIN)
        w_pos = np.minimum(w_pos, w_cap[None, :]).astype(np.float32)
    else:
        w_pos = (w_snr[:, None] * np.ones_like(y, dtype=np.float32)).astype(np.float32)

    num = (y * w_pos).sum(axis=0)
    den = w_pos.sum(axis=0)
    den = np.where(den > 0.0, den, 1.0).astype(np.float32)
    pos_means[col] = (num / den).astype(np.float32)

clip_bounds = {}
for col in TARGET_COLS:
    vals = np.concatenate(train_hq[col].values).astype(np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        clip_bounds[col] = (-1.0, 1.0)
    else:
        lo, hi = np.quantile(vals, [0.001, 0.999])
        clip_bounds[col] = (float(lo), float(hi))

TAIL_K = 5
tail_slice = slice(max(0, SCORED_LEN - TAIL_K), SCORED_LEN)
fill_tail = {col: float(pos_means[col][tail_slice].mean()) for col in TARGET_COLS}

df = sample_sub.copy()

seqpos = df["id_seqpos"].str.rsplit("_", n=1).str[-1].astype(int).to_numpy()

for col in TARGET_COLS:
    preds = np.empty(len(df), dtype=np.float32)
    mask_scored = seqpos < SCORED_LEN
    preds[mask_scored] = pos_means[col][seqpos[mask_scored]]
    preds[~mask_scored] = fill_tail[col]
    lo, hi = clip_bounds[col]
    preds = np.clip(preds, lo, hi)
    df[col] = preds

df.head(), df.shape



## === cell 3
probe_sequence = "id_366486252"
probe_mask = df["id_seqpos"].str.startswith(probe_sequence + "_")
probe_mask.sum()



## === cell 4
required_cols = ["id_seqpos"] + TARGET_COLS
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required submission columns: {missing}")

df = df[required_cols]

if len(df) != len(sample_sub):
    raise ValueError(f"Row count mismatch: got {len(df)} expected {len(sample_sub)}")

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", df.shape)
print(df.head())
