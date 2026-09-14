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

3.8

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

0.3783348710000906

# 6. Current score

0.43804

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42418) has done: 'I remove the broken dependency on external Kaggle dataset paths that aren’t present in your environment (the cause of the FileNotFoundError), and instead build a valid prediction file directly from the provided `train.json/test.json` and `sample_submission.csv`. To keep the core approach minimal and stable, I use a simple, deterministic baseline: per-position mean targets from the training set (for all 5 columns), then fill the submission rows by matching `id_seqpos` to sequence position. This runs end-to-end with only numpy/pandas, guarantees the correct row count/columns, and produces `submission.csv` with a `.csv` suffix. This should also yield a reasonable (though not SOTA) score, instead of “Not yielded”.'
- What this solution (achieved 0.42166) has done: 'We can nudge the score down (better, since lower is better) with a minimal change that stays within your baseline’s core logic: still “per-position mean targets,” but compute those means only on higher-quality training rows (`SN_filter == 1`) and fall back to all rows only if needed. This usually improves generalization for this competition because the public test is also quality-filtered, so matching the train distribution helps MCRMSE. I also add a tiny safety clamp on extreme target values (based on training quantiles) to reduce the impact of outliers without changing the modeling approach. The submission format, row alignment, and deterministic behavior remain unchanged.'
- What this solution (achieved 0.48258) has done: 'We keep your “per-position mean targets” baseline intact, but adjust it in the smallest way likely to improve MCRMSE on this competition: compute *weighted* per-position means using the provided per-target error columns (inverse-variance weighting), while still restricting to `SN_filter==1` when available. This stays within the same modeling semantics (still a mean per position), but uses the dataset’s reliability information so noisy measurements influence the mean less, typically improving generalization. We also compute clipping ranges on the same weighted-training subset for consistency, and keep the same submission alignment and tail-fill behavior to avoid any format risk. No architecture/training loops are introduced, runtime remains very small.'
- What this solution (achieved 0.44292) has done: 'We keep your per-position mean baseline exactly as-is, but fix the one change that likely pushed your score worse: the inverse-variance weighting over-emphasizes tiny error values and can overfit/noisily calibrate the mean. I replace that with a safer, still “use error columns” approach: reliability weights `w = 1/(err + eps)` with optional gentle cap at a high quantile, which preserves the same core logic (weighted per-position mean) while preventing pathological dominance. Everything else (SN_filter selection, seq_scored handling, tail fill, submission alignment, and clipping) stays the same so the effect is minimal and runtime remains tiny. This should move MCRMSE back down toward your target from 0.48258 without introducing any new modeling machinery.'
- What this solution (achieved 0.43804) has done: 'We keep your per-position mean baseline intact but make the weighting less brittle by shrinking reliability weights toward uniform weights (a convex blend), which should reduce overfitting to potentially miscalibrated error columns and move MCRMSE down from 0.44292 toward your 0.378 target. We also compute clipping bounds from the same *unweighted* training values (still on the same `use_train` subset) so clipping isn’t distorted by the weighting scheme. Finally, we fill unscored positions (seqpos ≥ seq_scored) using the mean of the last few scored positions instead of a single last position, which is still the same “tail fill” idea but less noisy and typically improves generalization with minimal semantic change. Everything else (paths, data reading, row alignment, output schema, and writing `submission.csv`) stays the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

np.random.seed(0)

DATA_DIR = "/kaggle/input/stanford-covid-vaccine"
TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
TEST_JSON = os.path.join(DATA_DIR, "test.json")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

if not os.path.exists(TRAIN_JSON):
    TRAIN_JSON = "/kaggle/input/train.json"
if not os.path.exists(TEST_JSON):
    TEST_JSON = "/kaggle/input/test.json"
if not os.path.exists(SAMPLE_SUB):
    SAMPLE_SUB = "/kaggle/input/sample_submission.csv"

print("Using:")
print("TRAIN_JSON:", TRAIN_JSON, "exists:", os.path.exists(TRAIN_JSON))
print("TEST_JSON :", TEST_JSON, "exists:", os.path.exists(TEST_JSON))
print("SAMPLE_SUB:", SAMPLE_SUB, "exists:", os.path.exists(SAMPLE_SUB))



## === cell 1
train = pd.read_json(TRAIN_JSON, lines=True)
test = pd.read_json(TEST_JSON, lines=True)
sample = pd.read_csv(SAMPLE_SUB)

print("train shape:", train.shape)
print("test shape :", test.shape)
print("sample shape:", sample.shape)
print("sample columns:", list(sample.columns))

required_sub_cols = [
    "id_seqpos",
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]
missing = [c for c in required_sub_cols if c not in sample.columns]
if missing:
    raise ValueError(f"sample_submission missing required columns: {missing}")



## === cell 2
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

ERROR_COLS = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

seq_scored = int(train["seq_scored"].iloc[0])
for c in TARGET_COLS:
    if not isinstance(train[c].iloc[0], (list, np.ndarray)):
        raise TypeError(
            f"Expected list/array in train column {c}, got {type(train[c].iloc[0])}"
        )

if "SN_filter" in train.columns:
    train_hq = train[train["SN_filter"].astype(int) == 1].reset_index(drop=True)
else:
    train_hq = train.copy()

min_rows_needed = 100  # small safety; still deterministic and minimal
use_train = train_hq if len(train_hq) >= min_rows_needed else train
print(
    f"Using rows for mean computation: {len(use_train)} (hq={len(train_hq)}, all={len(train)})"
)

y = {c: np.stack(use_train[c].values).astype(np.float64) for c in TARGET_COLS}

pos_means = {}
for c in TARGET_COLS:
    err_col = ERROR_COLS.get(c)

    if err_col in use_train.columns and isinstance(
        use_train[err_col].iloc[0], (list, np.ndarray)
    ):
        e = np.stack(use_train[err_col].values).astype(
            np.float64
        )  # (n_used, seq_scored)

        e = np.clip(e, 1e-3, None)
        w = 1.0 / (e + 1e-3)  # gentler than 1/(e^2)

        w_cap = np.quantile(w, 0.995)
        if np.isfinite(w_cap) and w_cap > 0:
            w = np.clip(w, 0.0, w_cap)

        alpha = 0.30  # 0 => unweighted mean; 1 => fully reliability-weighted mean
        w_eff = alpha * w + (1.0 - alpha) * 1.0

        pos_means[c] = (w_eff * y[c]).sum(axis=0) / np.clip(
            w_eff.sum(axis=0), 1e-12, None
        )
    else:
        pos_means[c] = y[c].mean(axis=0)

pos_means = {c: pos_means[c].astype(np.float32) for c in TARGET_COLS}

tail_k = 5
tail_fill = {c: float(pos_means[c][-tail_k:].mean()) for c in TARGET_COLS}

flat = {c: y[c].reshape(-1) for c in TARGET_COLS}
clip_lo = {c: float(np.quantile(flat[c], 0.005)) for c in TARGET_COLS}
clip_hi = {c: float(np.quantile(flat[c], 0.995)) for c in TARGET_COLS}

print("seq_scored:", seq_scored)
print(
    "Per-position mean vectors computed (reliability-weighted with shrinkage when error columns available)."
)
print(f"Tail fill uses mean of last {tail_k} scored positions.")
print("Clipping ranges (0.5%..99.5%):")
for c in TARGET_COLS:
    print(f"  {c}: [{clip_lo[c]:.4f}, {clip_hi[c]:.4f}]")




## === cell 3
def split_id_seqpos(s: str):
    base, pos = s.rsplit("_", 1)
    return base, int(pos)


test_id_to_len = dict(zip(test["id"].values, test["seq_length"].astype(int).values))

ids = []
seqpos = np.empty(len(sample), dtype=np.int32)
for i, s in enumerate(sample["id_seqpos"].astype(str).values):
    base, p = split_id_seqpos(s)
    ids.append(base)
    seqpos[i] = p

unknown = [i for i, base in enumerate(ids) if base not in test_id_to_len]
if unknown:
    raise ValueError(
        f"Found {len(unknown)} id_seqpos rows with id not in test.json. Example: {sample['id_seqpos'].iloc[unknown[0]]}"
    )

bad_pos = []
for i, base in enumerate(ids):
    L = test_id_to_len[base]
    p = int(seqpos[i])
    if p < 0 or p >= L:
        bad_pos.append(i)
if bad_pos:
    raise ValueError(
        f"Found {len(bad_pos)} id_seqpos rows with seqpos out of range. Example: {sample['id_seqpos'].iloc[bad_pos[0]]}"
    )

pred = pd.DataFrame({"id_seqpos": sample["id_seqpos"].astype(str).values})
scored_mask = seqpos < seq_scored
for c in TARGET_COLS:
    out = np.empty(len(sample), dtype=np.float32)
    out[scored_mask] = pos_means[c][seqpos[scored_mask]].astype(np.float32)
    out[~scored_mask] = np.float32(tail_fill[c])

    out = np.clip(out, np.float32(clip_lo[c]), np.float32(clip_hi[c]))
    pred[c] = out

pred = pred[required_sub_cols]

print(pred.head())
print("Pred shape:", pred.shape)



## === cell 4
if pred.isna().any().any():
    na_cols = pred.columns[pred.isna().any()].tolist()
    raise ValueError(f"Submission has NaNs in columns: {na_cols}")

if len(pred) != len(sample):
    raise ValueError(f"Row count mismatch: pred={len(pred)} sample={len(sample)}")

if not (pred["id_seqpos"].values == sample["id_seqpos"].astype(str).values).all():
    raise ValueError("id_seqpos ordering mismatch vs sample_submission")

pred.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", pred.shape)



## === cell 5
print(pred[TARGET_COLS].describe().T)
print("Done.")
