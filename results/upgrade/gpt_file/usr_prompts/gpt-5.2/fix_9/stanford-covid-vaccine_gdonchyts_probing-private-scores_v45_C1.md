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

0.42474

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I remove the dependency on the missing `../input/worst-submission/ensemble52.csv` file and instead build a valid submission directly from the provided `sample_submission.csv`, which guarantees correct row count and columns. I also fix the cell numbering (start at cell 1) and ensure `df` is always defined before it’s used, so the notebook runs end-to-end without NameErrors. The private-set ID logic is left intact but made safe (it won’t crash if the filter results are empty). The final output always be written as `submission.csv` with the required 6 columns.'
- What this solution (achieved 0.63824) has done: 'Your current submission is essentially the sample submission with one extreme outlier edit, which drives MCRMSE high; the smallest score-improving change is to remove that outlier assignment so predictions stay at the benign baseline. I keep your existing data loading and column validation intact, and only change the “private-set ID logic” cell so it never injects huge values. I also keep the numeric coercion and output exactly the same so the script still runs end-to-end and writes a valid `submission.csv`. This should move the score downward (better) toward your target with minimal risk.'
- What this solution (achieved 0.42418) has done: 'Your current code is effectively the sample submission (near-constant small values) and the “private set ID logic” is computed but not used; given your gap to the target (0.63824 vs 0.35188, lower is better), the smallest legitimate improvement is to replace the baseline constants with per-position mean targets learned from the training set. This keeps the overall approach simple (no model/architecture changes), matches the metric better than a constant, and is still fast under 600s using only pandas/numpy. I compute the mean curve for each of the 5 targets over the scored positions (0..67), then fill all test rows by seqpos (and use the last scored mean for positions >67). The submission format and paths stay the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.42166) has done: 'Your current approach is a fast per-position mean baseline; to move the MCRMSE down toward the target, the smallest safe improvement is to make those per-position means less noisy and better aligned with “high-quality” measurements. I keep the exact same prediction logic (mean curve by seqpos and constant tail) but compute the mean curves using only training rows that pass `SN_filter==1` (the competition’s intended quality filter), which typically improves generalization without changing the modeling approach. I also ensure the mean computation is robust to any NaNs by keeping the same `nanmean` behavior and not altering the submission schema or paths. Everything still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.45727) has done: 'Your current score (0.42166, lower-is-better) is still above the target (0.35188), so we should make a small, legitimate improvement without changing the overall “per-seqpos mean curve + constant tail” approach. The least invasive improvement is to compute the mean curves with weights derived from the provided per-position measurement errors, so positions with lower experimental error influence the mean more (this keeps the same prediction form but makes the estimated mean curves less noisy). We keep the SN_filter==1 restriction, and only adjust how the mean curve is estimated (weighted nan-mean instead of plain nan-mean). Submission writing, schema, and paths remain identical.'
- What this solution (achieved 0.43985) has done: 'We need to move your score down (better) from 0.45727 toward 0.35188 while keeping the same “per-seqpos mean curve + constant tail” core logic. The biggest likely issue in the current weighted-mean variant is that inverse-variance weights can become overly peaky when errors are small, hurting generalization; we keep the same weighted-mean approach but use a small weight floor + weight clipping to prevent extreme influence from a few sequences/positions. This is a minimal, metric-aligned calibration of the existing estimator (not a new model), and it should improve stability and reduce MCRMSE. Submission format/paths remain unchanged and a valid `submission.csv` is always written.'
- What this solution (achieved 0.42878) has done: 'Your current approach (per-seqpos mean curve + constant tail) is already the right “core logic” for a fast baseline, but the inverse-variance weighting can still be too peaky and hurt generalization, which likely explains the regression to 0.43985. I keep the exact same pipeline and prediction form, but make the weighting more stable by (1) increasing the regularization `eps` in the denominator and (2) clipping weights more aggressively (lower quantile) so no tiny-error measurements dominate. This should move the score down (better) toward your target without changing the model family, training loop, or submission semantics. All paths and the submission schema remain unchanged, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.42474) has done: 'We keep your exact “per-seqpos mean curve + constant tail” predictor and the same SN_filter==1 data restriction, but make the inverse-variance weighting slightly less peaky to improve generalization (lower MCRMSE) toward your target. Concretely, we (1) increase the weight-regularization `eps` a bit more and (2) clip weights more aggressively (lower clip quantile), which reduces the chance that a few tiny-error measurements dominate the mean curve. Everything else (inputs/paths, targets, submission schema, and writing `submission.csv`) stays the same to minimize risk. This is a small calibration of your current estimator, not a new model.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
df_test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
df = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")

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
    raise ValueError(f"sample_submission is missing required columns: {missing}")



## === cell 2
sequences = list(
    set(df_test[df_test.seq_length != 130].id)
)  # sequences from private set (original logic)
sequences.sort()



## === cell 3
sequences[-10:] if len(sequences) >= 10 else sequences



## === cell 4
mask = df["id_seqpos"].astype(str).str.startswith("id_fb7dfccef")



## === cell 5
train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)

target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
seq_scored = int(train["seq_scored"].iloc[0])  # expected 68
seq_length = int(train["seq_length"].iloc[0])  # expected 107

error_map = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

if "SN_filter" in train.columns:
    train_use = train.loc[train["SN_filter"].astype(int) == 1].copy()
    if len(train_use) == 0:
        train_use = train
else:
    train_use = train

means = {}

eps = 5e-2  # was 2e-2

w_clip_quantile = 0.95  # was 0.98

for col in target_cols:
    y = np.vstack(train_use[col].values).astype(np.float32)  # (n, seq_scored)
    err_col = error_map.get(col, None)

    if err_col is not None and err_col in train_use.columns:
        e = np.vstack(train_use[err_col].values).astype(np.float32)  # (n, seq_scored)

        w = 1.0 / (np.square(e) + eps)

        w_finite = w[np.isfinite(w)]
        if w_finite.size > 0:
            w_max = np.quantile(w_finite, w_clip_quantile)
            if np.isfinite(w_max) and w_max > 0:
                w = np.clip(w, 0.0, w_max)

        y_ok = np.isfinite(y)
        w_ok = np.isfinite(w)
        ok = y_ok & w_ok

        num = np.nansum(np.where(ok, y * w, 0.0), axis=0)
        den = np.nansum(np.where(ok, w, 0.0), axis=0)
        m = np.where(den > 0, num / den, np.nanmean(y, axis=0))
    else:
        m = np.nanmean(y, axis=0)

    means[col] = m.astype(np.float32)

seqpos = df["id_seqpos"].astype(str).str.rsplit("_", n=1).str[-1].astype(int).values

for col in target_cols:
    m = means[col]
    tail = float(m[-1])
    pred = np.where(
        seqpos < seq_scored, m[np.clip(seqpos, 0, seq_scored - 1)], tail
    ).astype(np.float32)
    df[col] = pred



## === cell 6
for c in required_cols[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)



## === cell 7
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print("Columns:", df.columns.tolist())
