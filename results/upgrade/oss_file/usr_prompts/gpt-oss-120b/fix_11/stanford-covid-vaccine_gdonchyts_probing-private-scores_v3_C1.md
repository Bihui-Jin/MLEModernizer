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

0.3519866169312086

# 6. Current score

0.39112

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.47906) has done: 'The script now loads the training JSON, computes the overall mean for each target column, reads the provided sample_submission to get the correct `id_seqpos` rows, fills every prediction with these means, and writes a proper `submission.csv`. This resolves the missing‑file error, eliminates undefined variables, and guarantees a correctly‑formatted CSV ready for Kaggle submission.'
- What this solution (achieved 0.42418) has done: 'I compute per‑position mean values for each target (using the first 68 bases where measurements exist) and use those instead of a single overall mean. The submission rows are parsed to extract the base position from the `id_seqpos` field, then each target column is filled with the corresponding position‑wise mean (fallback to the overall mean for positions beyond the scored range). This modest feature‑level tweak should lower the MCRMSE toward the target without altering the core modeling logic.'
- What this solution (achieved 0.3944) has done: 'I add a simple “loop‑type” conditioning step: for each position we also compute the mean target value for each structural loop character (S, M, I, …) from the training set. When filling the submission we first try to use that conditional mean (matched by sample id and position); if it is unavailable we fall back to the original position‑wise mean, and finally to the global mean. This inexpensive refinement keeps the original logic but should lower the MCRMSE toward the target.'
- What this solution (achieved 0.39112) has done: 'I add a lightweight nucleotide + loop‑type conditioning step. For each position we now compute the mean target value for every (base, loop) pair in the training data, and during prediction we first try to use that specific mean, then fall back to the loop‑only mean, then the position mean, and finally the global mean. This extra conditioning keeps the original simple averaging logic while giving more personalized estimates, which should lower the MCRMSE toward the target.'
- What this solution (achieved 0.39046) has done: 'Implemented a lightweight count‑based fallback: predictions now use the (base, loop) conditional mean only when that combination appears at least 5 times in the training data, otherwise they fall back to the loop‑only mean (also requiring ≥5 occurrences). If those are unavailable, the original position‑wise or global means are used. This reduces noisy over‑fitting of rare combinations and is expected to lower the MCRMSE toward the target score.'
- What this solution (achieved 0.39112) has done: 'I lower the count‑threshold for using conditional (base + loop) and loop‑only means from 5 to 1, allowing those more specific averages to be applied more often. This adds useful signal without changing the overall modeling approach and is expected to reduce the MCRMSE, moving the score closer to the target.'
- What this solution (achieved 0.40566) has done: 'I replace the mean aggregations used for the conditional (loop‑only and base‑plus‑loop) statistics with medians, which are more robust to outliers, and raise the count‑threshold from 1 to 2 so that only reasonably supported patterns are used. These minimal tweaks keep the overall averaging logic intact while aiming to reduce noisy predictions and bring the MCRMSE closer to the target.'
- What this solution (achieved 0.39112) has done: 'I switch the conditional statistics back to means (instead of medians) and lower the count‑threshold to 1 so that more specific (base + loop) or loop‑only averages are used when available. These minimal tweaks restore the stronger signal from the conditional averages and should bring the MCRMSE closer to the target score.'
- What this solution (achieved 0.40565) has done: 'I replace the simple means used for position‑wise and conditional statistics with medians (more robust to outliers) and raise the count‑threshold to 2 so that only well‑supported (base,loop) or loop‑only groups are used. The fallback order stays the same, but the statistics are now median‑based, which should reduce noisy predictions and move the MCRMSE closer to the target.'
- What this solution (achieved 0.39112) has done: 'The update switches all conditional aggregations from medians back to means and lowers the count‑threshold to 1 so that more specific (base + loop) and loop‑only statistics are used whenever available. This adds useful signal without changing the overall averaging framework, moving the MCRMSE closer to the target value.'

# 9. Code solution

## === cell 0
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict



## === cell 1
train_path = Path("../input/train.json")
if not train_path.exists():
    train_path = Path("./input/train.json")  # fallback
with train_path.open("r") as f:
    train_data = [json.loads(line) for line in f]



## === cell 2
targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

global_mean_vals = {}
for t in targets:
    all_vals = []
    for row in train_data:
        vals = row.get(t, [])
        all_vals.extend(vals)
    global_mean_vals[t] = np.mean(all_vals) if all_vals else 0.0

max_scored = max(row.get("seq_scored", 0) for row in train_data)
max_scored = max(max_scored, 68)

position_means = {t: [] for t in targets}
for t in targets:
    pos_vals = [[] for _ in range(max_scored)]
    for row in train_data:
        vals = row.get(t, [])
        for i, v in enumerate(vals):
            if i < max_scored:
                pos_vals[i].append(v)
    means = [
        np.mean(pos_vals[i]) if pos_vals[i] else global_mean_vals[t]
        for i in range(max_scored)
    ]
    position_means[t] = means

cond_means = {t: [defaultdict(list) for _ in range(max_scored)] for t in targets}
for row in train_data:
    loop_str = row.get("predicted_loop_type", "")
    for t in targets:
        vals = row.get(t, [])
        for i, v in enumerate(vals):
            if i >= max_scored:
                break
            if i < len(loop_str):
                ch = loop_str[i]
                cond_means[t][i][ch].append(v)

cond_means_avg = {t: [{} for _ in range(max_scored)] for t in targets}
for t in targets:
    for i in range(max_scored):
        for ch, lst in cond_means[t][i].items():
            cond_means_avg[t][i][ch] = np.mean(lst)  # mean

cond2_means = {t: [defaultdict(list) for _ in range(max_scored)] for t in targets}
for row in train_data:
    seq = row.get("sequence", "")
    loop_str = row.get("predicted_loop_type", "")
    for t in targets:
        vals = row.get(t, [])
        for i, v in enumerate(vals):
            if i >= max_scored:
                break
            if i < len(seq) and i < len(loop_str):
                nt = seq[i]
                ch = loop_str[i]
                cond2_means[t][i][(nt, ch)].append(v)

cond2_means_avg = {t: [{} for _ in range(max_scored)] for t in targets}
for t in targets:
    for i in range(max_scored):
        for key, lst in cond2_means[t][i].items():
            cond2_means_avg[t][i][key] = np.mean(lst)  # mean



## === cell 3
sample_sub_path = Path("../input/sample_submission.csv")
if not sample_sub_path.exists():
    sample_sub_path = Path("./input/sample_submission.csv")  # fallback
submission_df = pd.read_csv(sample_sub_path)

test_path = Path("../input/test.json")
if not test_path.exists():
    test_path = Path("./input/test.json")
with test_path.open("r") as f:
    test_data = [json.loads(line) for line in f]

id_to_loop = {row["id"]: row.get("predicted_loop_type", "") for row in test_data}
id_to_seq = {row["id"]: row.get("sequence", "") for row in test_data}

submission_df["pos"] = submission_df["id_seqpos"].str.split("_").str[-1].astype(int)
submission_df["base_id"] = submission_df["id_seqpos"].apply(
    lambda x: "_".join(x.split("_")[:-1])
)

COUNT_THRESHOLD = 1

for t in targets:
    if t in submission_df.columns:
        preds = []
        for _, row in submission_df.iterrows():
            p = row["pos"]
            base_id = row["base_id"]
            val = None

            loop_str = id_to_loop.get(base_id, "")
            seq_str = id_to_seq.get(base_id, "")

            if p < max_scored and p < len(loop_str) and p < len(seq_str):
                key = (seq_str[p], loop_str[p])
                lst = cond2_means[t][p].get(key)
                if lst is not None and len(lst) >= COUNT_THRESHOLD:
                    val = cond2_means_avg[t][p].get(key)

            if val is None and p < max_scored and p < len(loop_str):
                ch = loop_str[p]
                lst = cond_means[t][p].get(ch)
                if lst is not None and len(lst) >= COUNT_THRESHOLD:
                    val = cond_means_avg[t][p].get(ch)

            if val is None:
                if p < max_scored:
                    val = position_means[t][p]
                else:
                    val = global_mean_vals[t]

            preds.append(val)
        submission_df[t] = preds

submission_df = submission_df.drop(columns=["pos", "base_id"])



## === cell 4
output_path = Path("submission.csv")
submission_df.to_csv(output_path, index=False)
