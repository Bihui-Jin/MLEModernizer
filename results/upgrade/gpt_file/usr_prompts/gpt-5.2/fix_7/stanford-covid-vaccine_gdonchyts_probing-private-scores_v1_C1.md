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

0.48258

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42418) has done: 'I remove the dependency on a missing `../input/worst-submission/ensemble52.csv` file and instead build predictions from the provided competition data files so the notebook runs end-to-end. To keep core logic minimal and stable, I generate a valid submission by predicting constant per-target values equal to the training-set mean at each scored position (and fill unscored positions by carrying the last scored mean), which is a common safe baseline for this competition and should yield a reasonable score. I also ensure the output matches `sample_submission.csv` exactly in row order and column names, and write it to `submission.csv` in the working directory. These changes directly fix the runtime errors and guarantee a valid `.csv` submission is produced.'
- What this solution (achieved 0.42178) has done: 'Your current approach predicts per-position training means, which is a decent baseline but is hurt by low-quality/noisy training rows that the competition itself flags via `SN_filter` and `signal_to_noise`. To move your score down toward the target with minimal logic change, I compute the same per-position means but using only high-quality training examples (e.g., `SN_filter == 1`) and optionally weight by `signal_to_noise` so cleaner measurements influence the mean more. This preserves your “constant per-position mean” core logic and keeps the submission formatting identical to `sample_submission.csv`. The rest of the pipeline (paths, row order, columns, and writing `submission.csv`) remains unchanged.'
- What this solution (achieved 0.42178) has done: 'Your current score (0.42178) is worse than the target (0.35188), so we need a small, legitimate improvement while keeping the same “per-position constant mean” core logic. The biggest likely gain with minimal disruption is to compute those per-position means only over the scored region (first `seq_scored` positions) and handle missing/invalid training measurements robustly (ignore NaNs/inf), because noisy/invalid values can skew means. We also ensure we don’t accidentally let any tail padding influence the mean, and we keep the same submission row order/format by indexing into `sample_submission.csv`. These changes keep the approach identical in spirit (mean baseline) but typically reduce error on this competition.'
- What this solution (achieved 0.42183) has done: 'Your current score (0.42178) is worse than the target (0.35188), so we should make a small, legitimate improvement while keeping the same “per-position constant mean baseline” core logic. The main issue is that the baseline is currently computed using only `SN_filter==1` but still includes many noisy rows; adding a stricter, still-competition-consistent quality filter using `signal_to_noise` should reduce noise and move MCRMSE downward. To preserve semantics, we keep the same per-position mean computation and weighting approach, but compute means on a higher-quality subset (fallback to the previous subset if it becomes too small). Submission formatting, row order, and file output remain unchanged.'
- What this solution (achieved 0.48292) has done: 'We keep your “per-position constant mean” baseline intact, but reduce noise in those means by using the training-provided per-position measurement errors as inverse-variance weights (in addition to your existing `signal_to_noise` row weighting). This is a minimal, metric-aligned change for MCRMSE because it down-weights uncertain bases rather than discarding more data aggressively, which should move your score down toward the target. We also compute the mean for each target using its matching `*_error_*` column when available, with safe fallbacks to your prior behavior if anything is missing or degenerate. Submission formatting, row order, paths, and the final `submission.csv` output remain unchanged.'
- What this solution (achieved 0.48258) has done: 'Your current score is worse than the target (lower is better), so we should make a small, legitimate improvement without changing the “per-position constant mean” core logic. The biggest issue in your last change is that combining `signal_to_noise` row weights with inverse-variance per-position weights likely over-weights a small subset and can hurt generalization; we keep the same weighted-mean idea but remove the extra `signal_to_noise` weighting and rely only on the per-position measurement-error weights. We also make the weights more robust by down-weighting very noisy positions (large errors) and by normalizing weights per position to avoid extreme scaling differences, while keeping the same scored-region handling and submission alignment. Everything else (paths, shapes, columns, and writing `submission.csv`) stays the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE_DIR_CANDIDATES = [
    "/kaggle/input/stanford-covid-vaccine",
    "/kaggle/data/stanford-covid-vaccine",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_DIR = None
for d in BASE_DIR_CANDIDATES:
    if os.path.exists(os.path.join(d, "train.json")) and os.path.exists(
        os.path.join(d, "test.json")
    ):
        DATA_DIR = d
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate train.json/test.json in expected Kaggle input/data paths."
    )

train_path = os.path.join(DATA_DIR, "train.json")
test_path = os.path.join(DATA_DIR, "test.json")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_json(train_path, lines=True)
test = pd.read_json(test_path, lines=True)
sample_sub = pd.read_csv(sample_path)

train.shape, test.shape, sample_sub.shape



## === cell 2
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

ERROR_COL_MAP = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

seq_scored = int(train["seq_scored"].iloc[0])
seq_length = int(train["seq_length"].iloc[0])

train_use = train.copy()

if "SN_filter" in train_use.columns:
    train_use = train_use.loc[train_use["SN_filter"].astype(int) == 1].copy()

if len(train_use) < 50:
    train_use = train.copy()

y_means = {}
for col in TARGET_COLS:
    y = np.stack(train_use[col].values).astype(np.float32)  # (n, 68)
    y = y[:, :seq_scored]  # only scored region
    y = np.where(np.isfinite(y), y, np.nan)

    err_col = ERROR_COL_MAP.get(col, None)

    if err_col is not None and err_col in train_use.columns:
        e = np.stack(train_use[err_col].values).astype(np.float32)  # (n, 68)
        e = e[:, :seq_scored]
        e = np.where(np.isfinite(e), e, np.nan)

        e = np.clip(e, 1e-3, 5.0).astype(np.float32)
        w_pos = (1.0 / (e * e)).astype(np.float32)

        mask = np.isfinite(y).astype(np.float32)
        y_filled = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        w_eff = np.where(np.isfinite(w_pos), w_pos, 0.0).astype(np.float32) * mask

        w_sum = np.sum(w_eff, axis=0, keepdims=True) + 1e-12
        w_eff = w_eff / w_sum

        pos_mean = np.sum(y_filled * w_eff, axis=0)

        bad = ~np.isfinite(pos_mean)
        if np.any(bad):
            fallback = np.nanmean(y, axis=0)
            fallback = np.nan_to_num(fallback, nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.float32
            )
            pos_mean = pos_mean.astype(np.float32)
            pos_mean[bad] = fallback[bad]
    else:
        pos_mean = np.nanmean(y, axis=0)

    pos_mean = np.nan_to_num(pos_mean, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )

    if seq_length > seq_scored:
        tail = np.full((seq_length - seq_scored,), pos_mean[-1], dtype=np.float32)
        pos_mean_full = np.concatenate([pos_mean, tail], axis=0)  # (107,)
    else:
        pos_mean_full = pos_mean[:seq_length]

    y_means[col] = pos_mean_full

{k: v.shape for k, v in y_means.items()}



## === cell 3
id_seqpos = sample_sub["id_seqpos"].astype(str).values
seqpos = np.array([int(x.split("_")[-1]) for x in id_seqpos], dtype=np.int32)

sub = pd.DataFrame({"id_seqpos": id_seqpos})
for col in TARGET_COLS:
    pred = y_means[col][seqpos]
    sub[col] = pred.astype(np.float32)

sub = sub[["id_seqpos"] + TARGET_COLS]

assert sub.shape == sample_sub.shape, (sub.shape, sample_sub.shape)
assert list(sub.columns) == list(sample_sub.columns), (
    sub.columns.tolist(),
    sample_sub.columns.tolist(),
)

sub.head()



## === cell 4
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
