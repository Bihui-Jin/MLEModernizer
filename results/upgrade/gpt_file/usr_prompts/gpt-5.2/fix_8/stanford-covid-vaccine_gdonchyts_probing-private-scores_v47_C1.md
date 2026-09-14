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

0.42482

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'The crash is caused by trying to load a non-existent baseline submission file (`../input/worst-submission/ensemble52.csv`), which prevents `df` from ever being created and triggers downstream `NameError`s. I instead load the provided `sample_submission.csv` (always present) and use it as a safe baseline, ensuring the pipeline runs end-to-end. To keep the core intent (“make a baseline submission and tweak one id”) while avoiding invalid extreme values, I only apply the `reactivity` override if that `id` actually exists in the current test set, and I clip predictions to a reasonable numeric range to avoid potential formatting/metric issues. Finally, the script always write a valid `submission.csv` with the correct columns.'
- What this solution (achieved 0.42861) has done: 'Your current script is essentially submitting (almost) the unmodified sample submission, which explains the weak score; to move toward the much better target, we need to generate real predictions from train.json using the same minimal feature idea (sequence/structure/loop_type per position) without changing the evaluation semantics. I keep the approach very lightweight and deterministic by training simple multi-output Ridge regression models (one per target column) on one-hot encoded categorical features and position, then predict for every test id/seqpos and write the correct long-format submission. This preserves a simple “tabular model over per-base features” core logic and should substantially reduce MCRMSE from 0.638 while staying within the package constraints (numpy/pandas/sklearn). I also keep your safety checks: column order, numeric conversion, and clipping, but widen clipping to a more realistic range for this competition.'
- What this solution (achieved 0.42668) has done: 'To move your MCRMSE down toward the 0.3519 target (lower is better) with minimal changes, I keep the same per-base one-hot + Ridge core, but train only on the higher-quality subset (`SN_filter==1`) to reduce label noise that hurts generalization. I also tune Ridge regularization slightly (smaller `alpha`) and enable feature standardization (via a lightweight `StandardScaler` + `Ridge` pipeline) so the numeric `pos` feature is on a comparable scale to one-hot features, which typically improves Ridge fits without changing the modeling approach. Finally, I keep the exact submission formatting logic, but adjust clipping to a slightly wider, still safe range to avoid unnecessarily biasing predictions toward 0. These changes are small, deterministic, and aimed at reducing error without altering the overall solution structure.'
- What this solution (achieved 0.44604) has done: 'To move the MCRMSE down toward your 0.3519 target (lower is better) with minimal disruption, I keep your exact per-base one-hot + Ridge pipeline, but add two small signal-improving features that don’t change the modeling approach: an `id`-level categorical feature (lets the linear model learn per-sequence offsets) and a simple interaction feature `pos*is_paired` (captures position-dependent pairing effects). I also slightly adjust the Ridge regularization (alpha) because with the richer feature set a bit more shrinkage typically improves generalization. Finally, I keep your submission merge/formatting unchanged and retain clipping, so the script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.42489) has done: 'You’re currently worse than the target (0.44604 vs 0.35188; lower is better), so we should make a small, low-risk improvement without changing your Ridge + one-hot core. The biggest likely issue is that including `id` as a one-hot feature cannot generalize to unseen test ids and tends to overfit, so we remove `id` from the categorical design matrix (but keep everything else identical). To recover some sequence-level context in a generalizable way, we add two tiny numeric features that don’t change the modeling approach: a per-sequence GC fraction (`gc_frac`) and a per-position normalized coordinate (`pos_norm`). Everything else (SN_filter==1 training subset, StandardScaler+Ridge, long-format submission building, clipping, paths) stays the same.'
- What this solution (achieved 0.42489) has done: 'We’re currently worse than the target (0.42489 vs 0.35188; lower is better), so we should make a small, low-risk improvement without changing your Ridge + one-hot per-base core. The minimal change with the most upside here is to align training with what’s scored by optimizing the model jointly for the 3 scored targets (reactivity, deg_Mg_pH10, deg_Mg_50C) via a single multi-output Ridge, which typically helps because these targets are correlated; the two unscored targets still be predicted with the same Ridge approach. I also keep the exact feature set and data pipeline, but tune `ridge_alpha` slightly downward (less shrinkage often helps once features are standardized) to move performance toward the target band. Submission formatting/merge/clipping stays identical so you still get a valid `submission.csv`.'
- What this solution (achieved 0.42482) has done: 'To move your score down toward the 0.3519 target (lower is better) with minimal disruption, I keep the same per-base one-hot + (StandardScaler → Ridge) modeling, the SN_filter==1 subset, and the same submission formatting. The main small upgrade is to use the provided per-position experimental error arrays as sample weights during fitting (higher weight for more reliable labels), which better matches the noisy nature of this dataset without changing the model family or loss semantics in a material way. I apply weights to each target (including the multi-output scored model) using scikit-learn’s `sample_weight`, and keep everything else (features, alpha, clipping, paths) unchanged for stability and runtime. This is a low-risk change that typically improves MCRMSE on this competition because it downweights uncertain measurements.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



## === cell 2
TEST_PATH = "../input/stanford-covid-vaccine/test.json"
TRAIN_PATH = "../input/stanford-covid-vaccine/train.json"
SAMPLE_SUB_PATH = "../input/stanford-covid-vaccine/sample_submission.csv"

df_test = pd.read_json(TEST_PATH, lines=True)
df_train = pd.read_json(TRAIN_PATH, lines=True)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)



## === cell 3
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]  # scored by metric

ERROR_COL_MAP = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}


def _gc_frac(seq: str) -> float:
    s = str(seq)
    if len(s) == 0:
        return 0.0
    gc = sum(ch in ("G", "C") for ch in s)
    return float(gc) / float(len(s))


def _explode_train(df_in: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df_in.itertuples(index=False):
        Ls = int(r.seq_scored)
        L = int(r.seq_length)
        seq = r.sequence
        struct = r.structure
        loop = r.predicted_loop_type
        ys = {t: getattr(r, t) for t in TARGET_COLS}
        es = {t: getattr(r, ERROR_COL_MAP[t]) for t in TARGET_COLS}
        gc_frac = _gc_frac(seq)
        for pos in range(Ls):
            row = {
                "id": r.id,
                "pos": pos,
                "base": seq[pos],
                "struct": struct[pos],
                "loop": loop[pos],
                "pos_norm": float(pos) / float(max(L - 1, 1)),
                "gc_frac": gc_frac,
            }
            for t in TARGET_COLS:
                row[t] = float(ys[t][pos])
                row[f"{t}__err"] = float(es[t][pos])
            rows.append(row)
    return pd.DataFrame(rows)


def _explode_test(df_in: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df_in.itertuples(index=False):
        L = int(r.seq_length)
        seq = r.sequence
        struct = r.structure
        loop = r.predicted_loop_type
        gc_frac = _gc_frac(seq)
        for pos in range(L):
            rows.append(
                {
                    "id": r.id,
                    "pos": pos,
                    "base": seq[pos],
                    "struct": struct[pos],
                    "loop": loop[pos],
                    "pos_norm": float(pos) / float(max(L - 1, 1)),
                    "gc_frac": gc_frac,
                    "id_seqpos": f"{r.id}_{pos}",
                }
            )
    return pd.DataFrame(rows)


df_train_use = df_train[df_train["SN_filter"] == 1].reset_index(drop=True)

train_long = _explode_train(df_train_use)
test_long = _explode_test(df_test)

print("train_long:", train_long.shape, "test_long:", test_long.shape)
print(train_long.head())



## === cell 4
for _df in (train_long, test_long):
    _df["paired"] = (_df["struct"] != ".").astype(np.int8)
    _df["pos_paired"] = _df["pos"].astype(np.float32) * _df["paired"].astype(np.float32)


def _make_design(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols, num_cols):
    full = pd.concat(
        [train_df[cat_cols + num_cols], test_df[cat_cols + num_cols]],
        axis=0,
        ignore_index=True,
    )
    full_oh = pd.get_dummies(full, columns=cat_cols, drop_first=False)
    X_train = full_oh.iloc[: len(train_df)].to_numpy(dtype=np.float32)
    X_test = full_oh.iloc[len(train_df) :].to_numpy(dtype=np.float32)
    return X_train, X_test, full_oh.columns.tolist()


cat_cols = ["base", "struct", "loop"]
num_cols = ["pos", "pos_norm", "paired", "pos_paired", "gc_frac"]

X_train, X_test, feature_names = _make_design(train_long, test_long, cat_cols, num_cols)
print(
    "Design matrices:", X_train.shape, X_test.shape, "num_features:", len(feature_names)
)



## === cell 5
models = {}
test_preds = {}

ridge_alpha = 0.3

eps = 1e-3
w_clip_min, w_clip_max = 0.05, 20.0

weights = {}
for t in TARGET_COLS:
    err = train_long[f"{t}__err"].to_numpy(dtype=np.float32)
    w = 1.0 / (err * err + eps)
    w = np.clip(w, w_clip_min, w_clip_max).astype(np.float32)
    weights[t] = w

w_scored = np.mean(
    np.vstack([weights[t] for t in SCORED_TARGETS]).astype(np.float32), axis=0
).astype(np.float32)

Y_scored = train_long[SCORED_TARGETS].to_numpy(dtype=np.float32)

m_scored = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=ridge_alpha, random_state=0)),
    ]
)

m_scored.fit(X_train, Y_scored, ridge__sample_weight=w_scored)
P_scored = m_scored.predict(X_test).astype(np.float32)

models["__scored__"] = m_scored
for j, t in enumerate(SCORED_TARGETS):
    test_preds[t] = P_scored[:, j]

UNSCORED_TARGETS = [t for t in TARGET_COLS if t not in SCORED_TARGETS]
for t in UNSCORED_TARGETS:
    y = train_long[t].to_numpy(dtype=np.float32)
    m = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=ridge_alpha, random_state=0)),
        ]
    )
    m.fit(X_train, y, ridge__sample_weight=weights[t])
    test_preds[t] = m.predict(X_test).astype(np.float32)
    models[t] = m

print("Predictions ready for targets:", list(test_preds.keys()))



## === cell 6
pred_df = pd.DataFrame({"id_seqpos": test_long["id_seqpos"].values})
for t in TARGET_COLS:
    pred_df[t] = test_preds[t]

df = sample_sub[["id_seqpos"]].merge(pred_df, on="id_seqpos", how="left")

for t in TARGET_COLS:
    df[t] = pd.to_numeric(df[t], errors="coerce").fillna(0.0)

required_cols = [
    "id_seqpos",
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]
df = df[required_cols].copy()

pred_cols = required_cols[1:]
df[pred_cols] = df[pred_cols].clip(-3.0, 3.0)

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
print("Any NaNs:", df.isna().any().to_dict())
