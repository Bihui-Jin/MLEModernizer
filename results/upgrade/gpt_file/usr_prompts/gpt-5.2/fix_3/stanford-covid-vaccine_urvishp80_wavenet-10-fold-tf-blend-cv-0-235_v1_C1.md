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

0.3673570199225029

# 6. Current score

0.42151

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.4223) has done: 'Your notebook fails because it references two external submission files that don’t exist in this environment, so the blend DataFrames are never created. I keep the blending “core logic” intact, but make it robust by (1) searching for those files and, if missing, (2) generating two reasonable baseline prediction files from the provided `train.json` and `test.json` (so the blend can still happen). I also ensure the predictions expand to per-position rows and are aligned exactly to `sample_submission.csv`’s `id_seqpos` order, then write a valid `blend_wavenet_tf.csv` submission.'
- What this solution (achieved 0.42151) has done: 'Your current score (0.4223, lower-is-better) is worse than the target (0.36736), so we should make a small, low-risk improvement without changing the blending core logic. The biggest legitimate gain available here is to ensure the fallback baseline predictions are trained only on higher-quality training rows (SN_filter==1), matching the test set’s filtering; this typically reduces MCRMSE for simple mean baselines. I keep the same submission expansion/alignment and the same 0.85/0.15 blend, but compute per-position means from SN-filtered rows (falling back to all rows only if needed). This is minimal, fast, deterministic, and should move the score toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
SAMPLE_SUB_PATHS = [
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]
sample_sub_path = next((p for p in SAMPLE_SUB_PATHS if os.path.exists(p)), None)
if sample_sub_path is None:
    raise FileNotFoundError(
        f"Could not find sample_submission.csv in any of: {SAMPLE_SUB_PATHS}"
    )

sub = pd.read_csv(sample_sub_path)
sub.head()




## === cell 2
def find_first_existing(paths):
    return next((p for p in paths if os.path.exists(p)), None)


WAVENET_CANDIDATES = [
    "../input/wavenet-best-10-fold/GRU_LSTM1_submission1.csv",
    "/kaggle/input/wavenet-best-10-fold/GRU_LSTM1_submission1.csv",
]
TF_CANDIDATES = [
    "../input/tf-cv-2355-no-swa/GRU_LSTM1_no_swa_submission.csv",
    "/kaggle/input/tf-cv-2355-no-swa/GRU_LSTM1_no_swa_submission.csv",
]

wavenet_path = find_first_existing(WAVENET_CANDIDATES)
tf_path = find_first_existing(TF_CANDIDATES)

sub_wavenet = pd.read_csv(wavenet_path) if wavenet_path is not None else None
sub_tf = pd.read_csv(tf_path) if tf_path is not None else None

(wavenet_path, tf_path)



## === cell 3
TRAIN_JSON_PATHS = [
    "/kaggle/input/stanford-covid-vaccine/train.json",
    "/kaggle/data/stanford-covid-vaccine/train.json",
    "/kaggle/input/train.json",
    "/kaggle/data/train.json",
]
TEST_JSON_PATHS = [
    "/kaggle/input/stanford-covid-vaccine/test.json",
    "/kaggle/data/stanford-covid-vaccine/test.json",
    "/kaggle/input/test.json",
    "/kaggle/data/test.json",
]

train_path = next((p for p in TRAIN_JSON_PATHS if os.path.exists(p)), None)
test_path = next((p for p in TEST_JSON_PATHS if os.path.exists(p)), None)
if train_path is None or test_path is None:
    raise FileNotFoundError(
        f"Could not find train/test json. train: {TRAIN_JSON_PATHS}, test: {TEST_JSON_PATHS}"
    )

train_df = pd.read_json(train_path, lines=True)
test_df = pd.read_json(test_path, lines=True)

TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
SUB_COLS = ["id_seqpos"] + TARGET_COLS


def _to_2d_array(list_of_lists, L):
    """Convert series of python lists to (n, L) float array, padding/truncating safely."""
    n = len(list_of_lists)
    out = np.full((n, L), np.nan, dtype=np.float32)
    for i, v in enumerate(list_of_lists):
        if v is None:
            continue
        vv = np.asarray(v, dtype=np.float32)
        m = min(L, vv.shape[0])
        out[i, :m] = vv[:m]
    return out


def build_baseline_submission(
    train_df, test_df, sub_template, mode="global_mean", shrink=0.0, noise=0.0, seed=0
):
    """
    mode:
      - 'global_mean': per-position global mean across train
      - 'sn_weighted_mean': per-position mean weighted by signal_to_noise (clipped)
    shrink: blend toward overall mean (stabilizes tails)
    noise: small gaussian noise (kept 0 by default for determinism)
    """

    use_df = train_df
    if "SN_filter" in use_df.columns:
        filt = use_df["SN_filter"].astype(int) == 1
        if int(filt.sum()) > 0:
            use_df = use_df.loc[filt].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    L = int(test_df["seq_length"].iloc[0])  # expected 107
    L_scored = int(use_df["seq_scored"].iloc[0])  # expected 68

    per_target = {}
    for c in TARGET_COLS:
        per_target[c] = _to_2d_array(use_df[c].tolist(), L_scored)

    means = {}
    if mode == "sn_weighted_mean" and "signal_to_noise" in use_df.columns:
        w = np.asarray(use_df["signal_to_noise"].values, dtype=np.float32)
        w = np.clip(w, 0.0, 10.0)  # prevent extreme weights
        wsum = np.sum(w) + 1e-12
        for c in TARGET_COLS:
            m68 = np.nansum(per_target[c] * w[:, None], axis=0) / wsum
            means[c] = m68
    else:
        for c in TARGET_COLS:
            means[c] = np.nanmean(per_target[c], axis=0)

    for c in TARGET_COLS:
        overall = float(np.nanmean(means[c]))
        means[c] = (1.0 - shrink) * means[c] + shrink * overall

    test_ids = test_df["id"].tolist()
    n_test = len(test_ids)

    preds = {}
    for c in TARGET_COLS:
        m68 = means[c].astype(np.float32)
        m107 = np.empty((L,), dtype=np.float32)
        m107[:L_scored] = m68
        m107[L_scored:] = m68[-1]  # extend with last scored mean
        preds[c] = m107

    rows = []
    for tid in test_ids:
        for pos in range(L):
            rows.append(f"{tid}_{pos}")
    out = pd.DataFrame({"id_seqpos": rows})
    for c in TARGET_COLS:
        col = np.tile(preds[c], n_test)
        if noise != 0.0:
            col = col + rng.normal(0.0, noise, size=col.shape).astype(np.float32)
        out[c] = col

    out = out.set_index("id_seqpos").reindex(sub_template["id_seqpos"]).reset_index()
    out[TARGET_COLS] = out[TARGET_COLS].astype(np.float32).fillna(0.0)
    return out[SUB_COLS]


if sub_wavenet is None or sub_tf is None:
    if sub_wavenet is None:
        sub_wavenet = build_baseline_submission(
            train_df,
            test_df,
            sub,
            mode="sn_weighted_mean",
            shrink=0.05,
            noise=0.0,
            seed=1,
        )
    if sub_tf is None:
        sub_tf = build_baseline_submission(
            train_df, test_df, sub, mode="global_mean", shrink=0.10, noise=0.0, seed=2
        )

sub_wavenet.head()



## === cell 4
for name, df in [("sub_wavenet", sub_wavenet), ("sub_tf", sub_tf)]:
    missing = [c for c in SUB_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    if len(df) != len(sub):
        raise ValueError(f"{name} wrong number of rows: {len(df)} != {len(sub)}")

sub_wavenet = sub_wavenet.set_index("id_seqpos").reindex(sub["id_seqpos"]).reset_index()
sub_tf = sub_tf.set_index("id_seqpos").reindex(sub["id_seqpos"]).reset_index()

sub_wavenet[TARGET_COLS] = sub_wavenet[TARGET_COLS].astype(np.float32).fillna(0.0)
sub_tf[TARGET_COLS] = sub_tf[TARGET_COLS].astype(np.float32).fillna(0.0)

sub_tf.head()



## === cell 5
sub_wavenet["reactivity"] = (
    sub_wavenet["reactivity"] * 0.85 + sub_tf["reactivity"] * 0.15
)
sub_wavenet["deg_Mg_pH10"] = (
    sub_wavenet["deg_Mg_pH10"] * 0.85 + sub_tf["deg_Mg_pH10"] * 0.15
)
sub_wavenet["deg_pH10"] = sub_wavenet["deg_pH10"] * 0.85 + sub_tf["deg_pH10"] * 0.15
sub_wavenet["deg_Mg_50C"] = (
    sub_wavenet["deg_Mg_50C"] * 0.85 + sub_tf["deg_Mg_50C"] * 0.15
)
sub_wavenet["deg_50C"] = sub_wavenet["deg_50C"] * 0.85 + sub_tf["deg_50C"] * 0.15

sub_wavenet.head()



## === cell 6
out_path = "blend_wavenet_tf.csv"
sub_wavenet[SUB_COLS].to_csv(out_path, index=False)

check = pd.read_csv(out_path)
assert list(check.columns) == SUB_COLS
assert len(check) == len(sub)
check.head()
