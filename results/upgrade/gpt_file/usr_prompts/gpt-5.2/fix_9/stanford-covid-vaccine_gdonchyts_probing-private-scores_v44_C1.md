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

0.48448

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I remove the dependency on the missing `../input/worst-submission/ensemble52.csv` and instead build the submission directly from the provided `sample_submission.csv`, which guarantees correct row count and column names. I keep the same intent of the original notebook (optionally overriding a specific `id_...` reactivity to a constant) but make it safe by only applying it when that id actually exists in the test set. Finally, I add small validation checks to ensure the output matches the sample submission format and always writes a valid `submission.csv`.'
- What this solution (achieved 0.42166) has done: 'Your current script mostly outputs the sample submission (all zeros), plus an extreme reactivity overwrite for one id; that score poorly on MCRMSE. To move the score down toward the target, the smallest legitimate improvement without changing the “core approach” (still no ML training) is to replace zeros with constant baselines computed from the training labels. Since only 3 targets are scored, we compute per-position means for those 3 across the training set (restricted to SN_filter==1 for cleaner labels), and fill the other 2 unscored columns with their global means to avoid pathological outputs. Finally, we remove the 100.0 overwrite (it can only hurt) while keeping the submission-format safety checks and writing `submission.csv`.'
- What this solution (achieved 0.42297) has done: 'You’re currently using a per-position mean baseline computed on SN_filter==1; to move MCRMSE down toward the target with minimal change, we can make that baseline more representative by (a) using *all* training rows (not just SN_filter==1) and (b) computing per-position means only from rows where each position’s experimental error is finite and reasonably small, which tends to reduce noise in the averaged label. This keeps the same overall approach (no ML; still a constant/statistical baseline) while improving label quality and thus lowering RMSE. We also clamp predictions to a reasonable range to avoid occasional outliers from noisy training values hurting RMSE, while keeping the submission format logic identical and still writing `submission.csv`.'
- What this solution (achieved 0.48104) has done: 'Your current baseline is already “mean-by-position,” so the smallest likely improvement is to make the averaging closer to the leaderboard’s scored distribution without changing the overall approach (still no ML). Concretely: compute the per-position means using only higher-quality training rows (SN_filter==1 and adequate signal_to_noise), and use error-weighted averaging (inverse-variance weights) instead of a hard error cutoff; this typically reduces noise in the baseline and should lower MCRMSE toward your target. I keep the same submission construction from `sample_submission.csv`, preserve the same targets, and keep the same scored-vs-unscored handling, only changing how the baseline means are computed. I also keep clipping but widen it slightly to avoid biasing predictions when the true values fall outside the previous tight range.'
- What this solution (achieved 0.48104) has done: 'We keep your “error-weighted per-position mean baseline” core logic intact, but make three minimal changes aimed at lowering MCRMSE toward the target: (1) relax the training row filter slightly (keep SN_filter==1 but lower the signal_to_noise threshold) so the per-position averages are less variance-prone, (2) stop predicting the two unscored columns with their own noisy means and instead tie them to their corresponding scored-condition means (Mg vs non-Mg at same pH/temp), and (3) make clipping target-specific (reactivity clipped tighter; degradation clipped slightly wider) to reduce harm from rare outlier means while not over-biasing. These are small, deterministic changes and preserve the same submission construction and semantics. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.48488) has done: 'We keep your “error-weighted per-position mean baseline” approach intact, but adjust the training-row inclusion so the per-position means are computed from a distribution closer to the test set: filter to SN_filter==1 (as you already do) and also cap extremely high signal_to_noise values (which often correspond to atypically “easy/clean” sequences and can bias means away from the harder test distribution). We also add a tiny amount of shrinkage toward the global mean per target (a convex blend) to reduce per-position variance/noise without changing the overall baseline nature of the solution; this typically reduces RMSE for simple mean baselines. Finally, we keep your tying of unscored columns and clipping/submission-format safeguards unchanged so the script remains stable and produces a valid `submission.csv`.'
- What this solution (achieved 0.48942) has done: 'We keep your current “error-weighted per-position mean baseline + small shrinkage + tying unscored columns + clipping” core logic intact, and make only two targeted adjustments to move the MCRMSE down toward your target. First, we tune the shrinkage strength slightly upward (still small) to reduce per-position noise/variance that tends to hurt a pure mean-by-position baseline on this dataset. Second, we tighten the training distribution filter by excluding very low signal_to_noise examples (which are noisier and can degrade the weighted mean despite error weights), while keeping your existing cap on very high signal_to_noise. These are minimal, deterministic changes that preserve submission construction and produce the same valid `submission.csv` format.'
- What this solution (achieved 0.48448) has done: 'We keep your exact “error-weighted per-position mean baseline + shrinkage + tying unscored columns + clipping + sample_submission alignment” approach, but make two small, metric-relevant adjustments to move the MCRMSE down from 0.489 toward your 0.352 target. First, we reduce over-shrinkage by lowering `shrink_alpha` (too much shrink pulls per-position estimates toward a global mean and usually increases RMSE on this task), keeping everything else identical. Second, we slightly tighten the inverse-variance weights by using a smaller minimum variance floor (`1e-6` instead of `1e-4`), which increases the influence of genuinely low-error measurements without changing the core logic or adding any new modeling. These changes are deterministic, minimal, and preserve the same submission construction and format.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
TEST_PATH = "../input/stanford-covid-vaccine/test.json"
TRAIN_PATH = "../input/stanford-covid-vaccine/train.json"
SAMPLE_SUB_PATH = "../input/stanford-covid-vaccine/sample_submission.csv"

df_test = pd.read_json(TEST_PATH, lines=True)
df_train = pd.read_json(TRAIN_PATH, lines=True)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

df = sample_sub.copy()



## === cell 2
train_use = df_train.copy()

targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
target_to_err = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

seq_scored = int(train_use["seq_scored"].mode().iloc[0])  # expected 68

if "SN_filter" in train_use.columns:
    train_use = train_use.loc[train_use["SN_filter"].astype(int) == 1].copy()

if "signal_to_noise" in train_use.columns:
    sn = train_use["signal_to_noise"].astype(float)
    train_use = train_use.loc[np.isfinite(sn) & (sn >= 1.0) & (sn <= 10.0)].copy()

per_pos_means = {}
for t in targets:
    y = np.stack(train_use[t].values).astype(np.float32)  # (n, 68)
    e = np.stack(train_use[target_to_err[t]].values).astype(np.float32)  # (n, 68)

    good = np.isfinite(y) & np.isfinite(e) & (e > 0)

    w = np.zeros_like(y, dtype=np.float32)

    w[good] = 1.0 / np.clip(e[good] ** 2, 1e-6, 25.0)

    wsum = np.sum(w, axis=0)
    ywsum = np.sum(w * np.where(good, y, 0.0).astype(np.float32), axis=0)

    mu = np.empty(seq_scored, dtype=np.float32)
    mu[:] = np.nan
    has = wsum > 0
    mu[has] = (ywsum[has] / wsum[has]).astype(np.float32)

    global_mu = float(np.sum(w * np.where(good, y, 0.0)) / max(float(np.sum(w)), 1e-12))
    mu = np.where(np.isfinite(mu), mu, global_mu).astype(np.float32)

    shrink_alpha = 0.12  # was 0.25
    mu = ((1.0 - shrink_alpha) * mu + shrink_alpha * np.float32(global_mu)).astype(
        np.float32
    )

    per_pos_means[t] = mu

global_means = {t: float(per_pos_means[t].mean()) for t in targets}


def parse_id_seqpos(x: str):
    s = str(x)
    rid, spos = s.rsplit("_", 1)
    return rid, int(spos)


ids, pos = zip(*[parse_id_seqpos(x) for x in df["id_seqpos"].values])
ids = np.array(ids, dtype=object)
pos = np.array(pos, dtype=np.int32)

tied_baseline = {
    "deg_pH10": "deg_Mg_pH10",
    "deg_50C": "deg_Mg_50C",
}

clip_bounds = {
    "reactivity": (-0.5, 2.0),
    "deg_Mg_pH10": (-0.5, 3.0),
    "deg_pH10": (-0.5, 3.0),
    "deg_Mg_50C": (-0.5, 3.0),
    "deg_50C": (-0.5, 3.0),
}

for t in targets:
    base_t = tied_baseline.get(t, t)

    vals = np.empty(len(df), dtype=np.float32)
    in_scored = pos < seq_scored
    vals[in_scored] = per_pos_means[base_t][pos[in_scored]]
    vals[~in_scored] = global_means[base_t]

    lo, hi = clip_bounds[t]
    vals = np.clip(vals, lo, hi)

    df[t] = vals



## === cell 3
target_prefix = "id_fe2102731"
mask = df["id_seqpos"].astype(str).str.startswith(target_prefix + "_")
_ = mask  # keep variable to preserve cell intent without changing outputs



## === cell 4
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
    raise ValueError(f"Submission is missing required columns: {missing}")

for c in required_cols[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

if len(df) != len(sample_sub):
    raise ValueError(
        f"Row count mismatch: got {len(df)} rows, expected {len(sample_sub)}"
    )

if not df["id_seqpos"].equals(sample_sub["id_seqpos"]):
    df = sample_sub[["id_seqpos"]].merge(df, on="id_seqpos", how="left")
    for c in required_cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)



## === cell 5
OUT_PATH = "submission.csv"
df.to_csv(OUT_PATH, index=False)

assert OUT_PATH.endswith(".csv") and os.path.exists(OUT_PATH)
print(df.head())
print(f"Wrote {OUT_PATH} with shape {df.shape}")
