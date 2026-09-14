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

0.48256

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I remove the dependency on a missing external “baseline submission” file and instead build a valid submission directly from the provided `sample_submission.csv`, which guarantees the required shape/columns and a `.csv` output. I also fix the cell numbering to start at 1 (your current script starts at cell 0) and make the data paths robust to the provided `/kaggle/input/...` layout. Since you currently cannot produce any submission (no score yielded), these changes are necessary to run end-to-end; the resulting submission be valid (though it be a simple baseline using the sample values). Finally, I keep the original intent of editing a specific `id`’s `reactivity`, but only if that `id` exists in the submission index (otherwise it’s a no-op).'
- What this solution (achieved 0.47889) has done: 'Your current submission is essentially the sample_submission with one id’s reactivity set to 100, which badly hurts MCRMSE; the smallest change to move toward your target is to stop injecting that extreme value and instead output a conservative constant baseline derived from training labels. I keep the same “no model” core logic (no training loop/architecture) but compute per-target means on train (only for SN_filter==1 to better match test quality) and fill all rows with those means, which is a standard minimal baseline that should substantially reduce error vs zeros/extremes. I also make sure the output rows align exactly to sample_submission (same index/order/columns) to avoid any submission-format mismatch. These changes are directly aimed at lowering the score toward 0.3519 while remaining simple and stable.'
- What this solution (achieved 0.47889) has done: 'We keep your “constant baseline from train” core logic, but compute those constants only over the scored region (first `seq_scored` positions) because Kaggle only scores those and using unscored tail positions adds noise to the means. We also compute means only for the three scored targets and set the two unscored targets to the average of the two most related scored degradation means (a conservative fill that typically reduces error versus using their own noisier means). Finally, we keep exact `sample_submission.csv` row order and add a strict sanity check that `id_seqpos` aligns to `test.json` ids/seq_length so the submission can’t silently misalign and hurt MCRMSE.'
- What this solution (achieved 0.42166) has done: 'Your current constant-mean baseline is stable but it leaves easy signal unused: the labels vary by sequence position, and MCRMSE is computed positionwise. I keep the same “no model, no training loop” core logic but switch from a single global mean per target to a per-position mean vector (length `seq_scored`) computed from `train.json` (still using `SN_filter==1`), then fill each `id_seqpos` row by its position. This is a minimal change that typically improves MCRMSE materially versus a flat constant, moving your 0.47889 down toward the 0.3519 target without altering evaluation semantics or requiring extra packages. I also add a safe clamp on `seqpos` so unscored tail positions (>=68) reuse the last scored mean, keeping the submission valid for all 107 positions.'
- What this solution (achieved 0.48258) has done: 'We keep your current “per-position mean vector” baseline (core logic) and make a minimal, score-relevant improvement by filtering out low-quality training samples more strictly using both `SN_filter==1` and a `signal_to_noise` threshold, which better matches the curated test distribution and typically lowers MCRMSE. We also compute the per-position mean with a simple weighted average using the provided per-position error arrays (inverse-variance weighting), which is still a “mean-per-position” baseline but uses label reliability to reduce noise—often a small but consistent improvement. Finally, we ensure we only use the first `seq_scored` positions for both values and errors and keep the exact `sample_submission.csv` order/shape so the submission remains valid.'
- What this solution (achieved 0.48256) has done: 'We keep your exact “per-position weighted mean baseline” core logic, but make two minimal, score-relevant corrections: (1) compute the per-position means only from the *scored region* while explicitly dropping any non-finite label values (these exist in this dataset and can silently bias weighted means), and (2) align the training filter closer to the curated test distribution by using a slightly stricter `signal_to_noise` threshold (this typically reduces MCRMSE vs 1.0 without changing the approach). We also make the weighting more robust by zeroing weights where errors are non-finite/zero and by ignoring non-finite y values rather than letting them propagate. Everything else (no model, same per-position fill, same submission format/ordering) stays the same to keep changes minimal and stable.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE_INPUT = "/kaggle/input/stanford-covid-vaccine"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "../input/stanford-covid-vaccine"

train_path = os.path.join(BASE_INPUT, "train.json")
test_path = os.path.join(BASE_INPUT, "test.json")
sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")

df_train = pd.read_json(train_path, lines=True)
df_test = pd.read_json(test_path, lines=True)
df = pd.read_csv(sample_path)

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



## === cell 2
targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
scored_targets = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]  # only these are evaluated

df_train_use = df_train.copy()

if "SN_filter" in df_train_use.columns:
    df_train_use = df_train_use[df_train_use["SN_filter"] == 1]

if "signal_to_noise" in df_train_use.columns:
    df_train_use = df_train_use[df_train_use["signal_to_noise"] >= 1.25]

df_train_use = df_train_use.reset_index(drop=True)

k_scored = 68
if "seq_scored" in df_train_use.columns and len(df_train_use):
    k_scored = int(df_train_use["seq_scored"].mode().iloc[0])

err_col = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}


def weighted_mean_vector_first_k(
    values_series: pd.Series, errors_series: pd.Series, k: int
) -> np.ndarray:
    """
    Change (score-relevant but same core logic: still per-position weighted mean):
    - Ignore non-finite y values (they exist in this dataset and can bias the mean).
    - Use safe inverse-variance weights; drop/zero weights for non-finite/zero errors.
    """
    num = np.zeros(k, dtype=np.float64)
    den = np.zeros(k, dtype=np.float64)

    eps = 1e-3

    vvals = values_series.values
    evals = (
        errors_series.values
        if errors_series is not None
        else [None] * len(values_series)
    )

    for v, e in zip(vvals, evals):
        if v is None:
            continue
        vv = np.asarray(v, dtype=np.float64)
        m = min(len(vv), k)
        if m <= 0:
            continue

        yy = vv[:m]
        ymask = np.isfinite(yy)
        if not np.any(ymask):
            continue

        if e is None:
            w = np.ones(m, dtype=np.float64)
        else:
            ee = np.asarray(e, dtype=np.float64)
            if len(ee) >= m:
                ee = ee[:m]
            else:
                ee = np.pad(ee, (0, m - len(ee)), constant_values=np.nan)

            emask = np.isfinite(ee) & (ee > 0)
            w = np.zeros(m, dtype=np.float64)
            w[emask] = 1.0 / np.maximum(ee[emask], eps) ** 2

        w = w * ymask.astype(np.float64)
        if w.sum() <= 0:
            continue

        num[:m] += w * yy
        den[:m] += w

    vec = np.divide(num, np.maximum(den, 1e-12))
    vec = vec.astype(np.float32)
    vec = np.where(np.isfinite(vec), vec, np.float32(0.0)).astype(np.float32)
    return vec


pos_means = {}
for t in scored_targets:
    ecol = err_col.get(t)
    errs = df_train_use[ecol] if (ecol in df_train_use.columns) else None
    pos_means[t] = weighted_mean_vector_first_k(df_train_use[t], errs, k_scored)

deg_avg_vec = (
    (pos_means["deg_Mg_pH10"] + pos_means["deg_Mg_50C"]) / np.float32(2.0)
).astype(np.float32)
pos_means["deg_pH10"] = deg_avg_vec
pos_means["deg_50C"] = deg_avg_vec

train_means = {t: np.float32(np.nanmean(pos_means[t])) for t in targets}
train_means



## === cell 3
for c in targets:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

seqpos = (
    df["id_seqpos"]
    .astype(str)
    .str.rsplit("_", n=1, expand=True)[1]
    .astype(int)
    .to_numpy()
)
seqpos_clamped = np.clip(seqpos, 0, k_scored - 1)

for c in targets:
    vec = pos_means.get(c, None)
    if vec is None or len(vec) != k_scored:
        df[c] = train_means[c]
    else:
        df[c] = vec[seqpos_clamped].astype(np.float32)

df[targets] = (
    df[targets].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
)



## === cell 4
expected_rows = int(df_test["seq_length"].sum()) if len(df_test) else len(df)
print("Expected rows (sum seq_length over test):", expected_rows)
print("Submission rows:", len(df))
print("Columns:", list(df.columns))

if len(df) != expected_rows:
    raise ValueError(
        f"Row count mismatch: submission has {len(df)} rows but test expects {expected_rows}"
    )

test_ids = set(df_test["id"].astype(str).tolist())


def extract_id(id_seqpos: str) -> str:
    parts = str(id_seqpos).split("_")
    return "_".join(parts[1:-1]) if len(parts) >= 3 else ""


if (
    extract_id(df["id_seqpos"].iloc[0]) not in test_ids
    or extract_id(df["id_seqpos"].iloc[-1]) not in test_ids
):
    print(
        "Warning: id_seqpos ids don't match test.json ids on boundary rows; check dataset paths."
    )



## === cell 5
out_path = "submission.csv"
df.to_csv(out_path, index=False)

print("Wrote:", out_path)
print("Submission shape:", df.shape)
print("Head:\n", df.head())
