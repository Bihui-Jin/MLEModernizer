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

0.63824

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'The current notebook fails because it tries to read three external submission files that don’t exist in your environment, so nothing downstream can run or write a CSV. I replace those reads with a robust fallback that builds a valid submission directly from the provided `sample_submission.csv` (and optionally uses any found prediction CSVs if they exist). This preserves the original “simple averaging ensemble” core logic when files are available, and otherwise guarantees a correctly formatted `submission.csv` is produced. I also align/merge by `id_seqpos` to prevent silent row-order mismatches that would hurt score.'
- What this solution (achieved 0.63824) has done: 'I fix the runtime error in `align_to_sample` by ensuring `fillna` receives a Series aligned to the merged index (instead of a NumPy ndarray), which is what pandas requires. I also make the CSV-loader and alignment a bit more robust by forcing numeric dtypes for target columns and handling any stray duplicates safely, without changing the ensemble averaging core logic. These changes are score-neutral by themselves but unblock the notebook so it runs end-to-end and reliably writes a valid `submission.csv`. The resulting submission keep the same simple 3-way averaging behavior as your current approach.'
- What this solution (achieved 0.63824) has done: 'Your current score (0.63824, lower-is-better) is far worse than the target (0.37833), so we should improve—not by changing modeling, but by ensuring the ensemble is using real prediction files rather than silently falling back to the all-zero `sample_submission.csv` (which would score very poorly). The most likely issue is that your auto-glob is scanning `/kaggle/input/**` but your environment’s files are under `/kaggle/data/**` and `/kaggle/input/stanford-covid-vaccine/**`, so you often load 0 external submissions and end up averaging zeros. I minimally expand the search to include `/kaggle/data/**`, and I filter out “degenerate” candidates that are identical to the sample submission (or near-constant), so only genuine model outputs are used. This keeps the same core logic (simple 3-way averaging after alignment) while making it much more likely to find and use actual predictions, improving the score toward your target.'
- What this solution (achieved 0.63824) has done: 'Your current score (0.63824, lower-is-better) is far from the target (0.37833), so we should improve by making sure your 3-way averaging ensemble uses the *best available* external prediction CSVs rather than arbitrarily taking the first three that pass basic checks. I keep the same core logic (load up to 3 submission-like CSVs → align to sample → mean-average), but change the selection step to rank candidates by how “non-degenerate” they are and by internal agreement: we prefer files with realistic variance and closer-to-median predictions across candidates (a robust proxy for quality without using labels). This is minimal, avoids changing architecture/training, and often significantly improves ensemble quality when many CSVs exist. The output format, alignment, and averaging semantics remain identical.'
- What this solution (achieved 0.63824) has done: 'Your current score (0.63824, lower-is-better) is far worse than the target (0.37833), so we should improve by making the ensemble selection more likely to pick genuinely good external submissions when many CSVs are present. I keep the same core logic (load submission-like CSVs → align to sample → pick 3 → mean-average), but change the ranking to prefer candidates that (a) look realistic (non-degenerate spread) and (b) agree with the *robust consensus on the scored targets only* (reactivity, deg_Mg_pH10, deg_Mg_50C), since those drive MCRMSE. This is a minimal scoring-aligned tweak that often improves the average without changing any modeling/training. The output format and 3-way averaging semantics remain identical, and it still safely falls back to `sample_submission.csv` if no candidates exist.'
- What this solution (achieved 0.63824) has done: 'Your current score (0.63824; lower-is-better) is far worse than the target (0.37833), so we should improve by making the ensemble pick better external submission CSVs when they exist. Keeping your exact core logic (load submission-like CSVs → align to sample → select 3 → simple mean-average), I adjust the candidate filtering to avoid wrongly discarding valid files just because they don’t cover the full 25680 rows (many good submissions only predict the scored 68 positions) and instead align/fill missing positions from the sample submission. Then I rank candidates using a slightly stronger “consensus agreement on scored targets” signal and select the best 3, which usually reduces MCRMSE without changing any modeling/training. The pipeline still falls back safely to `sample_submission.csv` and always writes a valid `submission.csv`.'
- What this solution (achieved 0.63824) has done: 'Your current score (0.63824, lower-is-better) is far from the target (0.37833), so we should improve by making the 3-way averaging ensemble more likely to select genuinely high-quality external submissions when many CSVs are present. Keeping your exact core logic (load submission-like CSVs → align to sample → choose 3 → mean-average), I (1) restrict auto-globbing to avoid accidentally picking the competition’s own `sample_submission.csv` or other obvious non-prediction CSVs, and (2) strengthen candidate ranking using two label-free proxies: agreement with the robust consensus on the scored columns and a mild penalty for implausibly small spread (over-smooth predictions often score worse). This should reduce the chance that the ensemble is diluted by weak/irrelevant CSVs, moving the score down toward the target without changing modeling/training. The pipeline still always writes a valid `submission.csv` with correct formatting and row alignment.'
- What this solution (achieved 0.63824) has done: 'Your current score (0.63824, lower-is-better) is far from the target (0.37833), so we should improve by making sure the 3-way average uses genuinely strong external submissions when many CSVs are present. Keeping your exact core logic (load submission-like CSVs → align to sample → select 3 → mean-average), I change only the candidate ranking to use a stronger, score-aligned proxy: agreement on the *scored* columns plus a light “non-degeneracy” spread check, and I also avoid polluting the pool with your own previously-written `submission.csv` from `/kaggle/working`. This should reduce the chance you average in weak/irrelevant CSVs, which is the most likely reason you’re stuck around 0.64. The output schema, alignment, and averaging semantics remain identical, and it still always writes a valid `submission.csv`.'
- What this solution (achieved 0.63824) has done: 'We need to move your score down (lower is better) from 0.638 to ~0.378, and the biggest likely reason you’re stuck high is that your script often ends up averaging weak/irrelevant CSVs or near-sample defaults. I keep the exact core logic (load submission-like CSVs → align to sample → pick 3 → simple mean average) but make the selection step more score-aligned by adding a lightweight “train-based plausibility” ranking: candidates whose predictions better match the distribution of the training labels on the 3 scored columns are more likely to be strong. This uses only `train.json` (no leakage) and doesn’t change the ensemble averaging itself—only which 3 files get averaged—so it’s a minimal change with a good chance to reduce MCRMSE toward your target. I also ensure we exclude any CSVs that look like our own output or contain constant/implausible values after alignment, to avoid diluting the ensemble.'
- What this solution (achieved 0.63824) has done: 'I fix the immediate crash by reading `train.json` with the correct JSON orientation (the file is a line-delimited JSON, so `pd.read_json(..., lines=True)` is required), with a safe fallback to the non-lines mode. This unblocks the train-distribution statistics that your candidate ranking depends on, which should help select better external prediction CSVs (and thus improve score) instead of erroring out. I also keep all ensemble logic the same, but add a tiny guard so the script still runs even if `train.json` can’t be parsed for some unforeseen reason (it fall back to neutral stats rather than crashing). The code still always write a valid `submission.csv` with the correct columns and row alignment.'

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd
import numpy as np



## === cell 1
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

SAMPLE_PATHS = [
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
]
sample_path = next((p for p in SAMPLE_PATHS if os.path.exists(p)), None)
if sample_path is None:
    raise FileNotFoundError(
        f"Could not find sample_submission.csv in any expected location: {SAMPLE_PATHS}"
    )

sample_sub = pd.read_csv(sample_path)
if "id_seqpos" not in sample_sub.columns:
    raise ValueError("sample_submission.csv is missing required column 'id_seqpos'")
for c in TARGET_COLS:
    if c not in sample_sub.columns:
        raise ValueError(f"sample_submission.csv is missing required column '{c}'")

for c in TARGET_COLS:
    sample_sub[c] = pd.to_numeric(sample_sub[c], errors="coerce").fillna(0.0)



## === cell 2
TRAIN_PATHS = [
    "/kaggle/input/train.json",
    "/kaggle/data/train.json",
    "/kaggle/input/stanford-covid-vaccine/train.json",
    "/kaggle/data/stanford-covid-vaccine/train.json",
]
train_path = next((p for p in TRAIN_PATHS if os.path.exists(p)), None)
if train_path is None:
    raise FileNotFoundError(
        f"Could not find train.json in any expected location: {TRAIN_PATHS}"
    )

train_df = None
train_read_err = None
try:
    train_df = pd.read_json(train_path, lines=True)
except Exception as e:
    train_read_err = e
    try:
        train_df = pd.read_json(train_path)
        train_read_err = None
    except Exception as e2:
        train_read_err = e2
        train_df = None

train_stats = {}
if train_df is None:
    for col in SCORED_COLS:
        train_stats[col] = {"med": 0.0, "iqr": 1.0}
    print("WARNING: Failed to read train.json; falling back to default train_stats.")
    print("Last error:", repr(train_read_err))
else:
    for col in SCORED_COLS:
        vals = np.concatenate(train_df[col].values).astype(np.float64)
        vals = vals[np.isfinite(vals)]
        med = float(np.median(vals))
        q25 = float(np.quantile(vals, 0.25))
        q75 = float(np.quantile(vals, 0.75))
        iqr = float(q75 - q25)
        if iqr < 1e-8:
            iqr = 1.0
        train_stats[col] = {"med": med, "iqr": iqr}

print("Using sample_submission from:", sample_path)
print("Using train.json from:", train_path)
print("Train robust stats (scored cols):", train_stats)



## === cell 3
candidate_paths = [
    "/kaggle/input/gru-lstm-mix-with-custom-loss-tunning/ensemble_final.csv",
    "/kaggle/input/mvan-covid-mrna-vaccine-analysis-notebook-268/submission.csv",
    "/kaggle/input/gru-lstm-mix-with-custom-loss/ensemble_final.csv",
]


def is_likely_prediction_csv(path: str) -> bool:
    p = path.lower()
    base = os.path.basename(p)
    if base in {"sample_submission.csv", "submission.csv", "submission_ens.csv"}:
        return False
    if "sample_submission" in p:
        return False
    if p.startswith("/kaggle/working/"):
        return False
    keep_tokens = ["submission", "sub", "ensemble", "pred", "oof"]
    return any(tok in base for tok in keep_tokens)


auto_glob_paths = []
auto_glob_paths += [
    p
    for p in glob.glob("/kaggle/input/**/*.csv", recursive=True)
    if is_likely_prediction_csv(p)
]
auto_glob_paths += [
    p
    for p in glob.glob("/kaggle/data/**/*.csv", recursive=True)
    if is_likely_prediction_csv(p)
]

all_paths = candidate_paths + auto_glob_paths


def try_load_submission_csv(path: str):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "id_seqpos" not in df.columns:
        return None
    if not all(col in df.columns for col in TARGET_COLS):
        return None

    df = df[["id_seqpos"] + TARGET_COLS].copy()
    for c in TARGET_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.drop_duplicates("id_seqpos", keep="first")
    return df


def align_to_sample(df: pd.DataFrame) -> pd.DataFrame:
    merged = sample_sub[["id_seqpos"]].merge(df, on="id_seqpos", how="left")
    for c in TARGET_COLS:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
        merged[c] = merged[c].fillna(sample_sub[c])
    return merged


def is_degenerate_like_sample_after_align(df_raw: pd.DataFrame) -> bool:
    df_a = align_to_sample(df_raw)

    if df_a[TARGET_COLS].isna().any().any():
        return True

    try:
        if (df_a[TARGET_COLS].to_numpy() == sample_sub[TARGET_COLS].to_numpy()).all():
            return True
    except Exception:
        pass

    vals_scored = df_a[SCORED_COLS].to_numpy(dtype=np.float64)
    if float(np.nanstd(vals_scored)) < 1e-8:
        return True

    if not np.isfinite(vals_scored).all():
        return True
    if float(np.nanmax(np.abs(vals_scored))) > 1e3:
        return True

    return False


loaded = []
seen = set()
for p in all_paths:
    if p in seen:
        continue
    seen.add(p)
    df = try_load_submission_csv(p)
    if df is None:
        continue
    if not is_degenerate_like_sample_after_align(df):
        loaded.append((p, df))

print(f"Auto-glob found {len(auto_glob_paths)} candidate CSV path(s).")
print(f"Loaded {len(loaded)} external submission-like CSV(s) after filtering.")




## === cell 4
def candidate_stats(df_aligned: pd.DataFrame) -> dict:
    arr_all = df_aligned[TARGET_COLS].to_numpy(dtype=np.float64)
    arr_scored = df_aligned[SCORED_COLS].to_numpy(dtype=np.float64)
    spread_all = float(np.nanmean(np.nanstd(arr_all, axis=0)))
    spread_scored = float(np.nanmean(np.nanstd(arr_scored, axis=0)))
    mean_abs_scored = float(np.nanmean(np.abs(arr_scored)))
    nan_frac_all = float(np.isnan(arr_all).mean())
    return {
        "spread_all": spread_all,
        "spread_scored": spread_scored,
        "mean_abs_scored": mean_abs_scored,
        "nan_frac_all": nan_frac_all,
    }


def dist_to_train_distribution(df_a: pd.DataFrame) -> float:
    arr = df_a[SCORED_COLS].to_numpy(dtype=np.float64)
    d = 0.0
    for j, col in enumerate(SCORED_COLS):
        v = arr[:, j]
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float("inf")
        med = float(np.median(v))
        q25 = float(np.quantile(v, 0.25))
        q75 = float(np.quantile(v, 0.75))
        iqr = float(q75 - q25)
        if iqr < 1e-8:
            iqr = 1.0

        med0 = train_stats[col]["med"]
        iqr0 = train_stats[col]["iqr"]

        d += ((med - med0) / iqr0) ** 2
        d += (np.log(iqr) - np.log(iqr0)) ** 2
    return float(d / (2.0 * len(SCORED_COLS)))


aligned_candidates = []
for p, df in loaded:
    df_a = align_to_sample(df)
    st = candidate_stats(df_a)
    aligned_candidates.append((p, df_a, st))

if len(aligned_candidates) >= 4:
    stack_scored = np.stack(
        [t[1][SCORED_COLS].to_numpy(dtype=np.float64) for t in aligned_candidates],
        axis=0,
    )
    consensus_scored = np.nanmedian(stack_scored, axis=0)

    abs_dev = np.abs(stack_scored - consensus_scored[None, :, :])
    mad = np.nanmedian(abs_dev, axis=0)
    scale = mad + 1e-6

    def dist_to_consensus_scored(df_a: pd.DataFrame) -> float:
        arr = df_a[SCORED_COLS].to_numpy(dtype=np.float64)
        z = (arr - consensus_scored) / scale
        return float(np.nanmean(z**2))

    ranked = []
    for p, df_a, st in aligned_candidates:
        d_cons = dist_to_consensus_scored(df_a)
        d_train = dist_to_train_distribution(df_a)

        low_spread_pen = max(0.0, 0.30 - st["spread_scored"])
        mag_pen = st["mean_abs_scored"]

        score = (
            d_cons
            + 0.10 * d_train
            + 0.05 * low_spread_pen
            + 1e-4 * mag_pen
            + 1e3 * st["nan_frac_all"]
        )
        ranked.append((score, p, df_a, st, d_cons, d_train))

    ranked.sort(key=lambda x: x[0])
    chosen = ranked[:3]
else:

    def rank_path(p: str) -> int:
        if p == candidate_paths[0]:
            return 0
        if p == candidate_paths[1]:
            return 1
        if p == candidate_paths[2]:
            return 2
        return 10

    ranked = []
    for p, df_a, st in aligned_candidates:
        d_train = dist_to_train_distribution(df_a)
        score = (d_train, -st["spread_scored"], rank_path(p), st["mean_abs_scored"])
        ranked.append((score, p, df_a, st, None, d_train))
    ranked.sort(key=lambda x: x[0])
    chosen = ranked[:3]

if len(chosen) == 0:
    df1 = sample_sub.copy()
    df2 = sample_sub.copy()
    df3 = sample_sub.copy()
    chosen_paths = []
else:
    dfs = [t[2] for t in chosen]
    chosen_paths = [t[1] for t in chosen]
    while len(dfs) < 3:
        dfs.append(dfs[-1].copy())
    df1, df2, df3 = dfs[0], dfs[1], dfs[2]

print(f"Selected {len(chosen_paths)} CSV(s) for 3-way averaging ensemble:")
for t in chosen:
    if len(t) >= 6:
        score, p, _, st, d_cons, d_train = t
        print(f" - {p}")
        print(
            f"   score={score:.6f} d_train={d_train:.6f}"
            + (f" d_cons={d_cons:.6f}" if d_cons is not None else "")
        )
        print(
            f"   stats: spread_scored={st['spread_scored']:.4f} mean_abs_scored={st['mean_abs_scored']:.4f}"
        )



## === cell 5
for name, df in [("df1", df1), ("df2", df2), ("df3", df3)]:
    if df.shape[0] != sample_sub.shape[0]:
        raise ValueError(
            f"{name} row count mismatch: {df.shape[0]} vs sample {sample_sub.shape[0]}"
        )
    if not df["id_seqpos"].equals(sample_sub["id_seqpos"]):
        raise ValueError(f"{name} id_seqpos order mismatch after alignment")
    if df[TARGET_COLS].isna().any().any():
        raise ValueError(f"{name} contains NaNs in target columns after alignment")



## === cell 6
df1.head()



## === cell 7
df2.head()



## === cell 8
df3.head()



## === cell 9
sub = sample_sub[["id_seqpos"]].copy()
sub["reactivity"] = (
    df1["reactivity"].values + df2["reactivity"].values + df3["reactivity"].values
) / 3.0
sub["deg_Mg_pH10"] = (
    df1["deg_Mg_pH10"].values + df2["deg_Mg_pH10"].values + df3["deg_Mg_pH10"].values
) / 3.0
sub["deg_pH10"] = (
    df1["deg_pH10"].values + df2["deg_pH10"].values + df3["deg_pH10"].values
) / 3.0
sub["deg_Mg_50C"] = (
    df1["deg_Mg_50C"].values + df2["deg_Mg_50C"].values + df3["deg_Mg_50C"].values
) / 3.0
sub["deg_50C"] = (
    df1["deg_50C"].values + df2["deg_50C"].values + df3["deg_50C"].values
) / 3.0

sub[TARGET_COLS] = sub[TARGET_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)



## === cell 10
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)



## === cell 11
sub.head()
