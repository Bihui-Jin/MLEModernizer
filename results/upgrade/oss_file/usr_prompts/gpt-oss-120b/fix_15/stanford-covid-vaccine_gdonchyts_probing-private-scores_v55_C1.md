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

0.41641

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.47906) has done: 'The fix replaces the missing baseline CSV with the official sample submission, computes simple global mean predictions from the training data, and writes a valid `submission.csv`. This removes the file‑not‑found and undefined‑variable errors while keeping the original workflow structure.'
- What this solution (achieved 0.42418) has done: 'I replace the simple global‑mean baseline with a per‑position mean for each target column (using the first 68 scored bases from the training set) and fall back to the overall mean for positions beyond 67. The submission rows are matched to these means via the position extracted from the `id_seqpos` field, which should lower the MCRMSE and move the score closer to the target while keeping the original workflow intact.'
- What this solution (achieved 0.42163) has done: 'Implemented a shrinkage‑adjusted per‑position mean based on high‑quality training records (SN_filter = 1 and signal_to_noise > 1). This reduces variance on sparse positions by pulling noisy per‑position estimates toward the overall mean, moving the validation MCRMSE closer to the target while preserving the original workflow. The script now filters the training data, computes counts, applies a simple James‑Stein‑style shrinkage, and uses the adjusted predictions for the submission.'
- What this solution (achieved 0.42418) has done: 'Implemented three key fixes to move the validation score closer to the target:  
1️⃣ Corrected the typo in the target column list (`deg_pth10` → `deg_pH10`).  
2️⃣ Simplified the prediction strategy by using **raw per‑position means** computed from **all training records** (no quality filtering or shrinkage), which reduces bias introduced by aggressive filtering.  
3️⃣ Updated the downstream code to use the new `pos_means` dictionary for populating the submission file.'
- What this solution (achieved 0.4242) has done: 'I add a simple shrinkage to the per‑position means: each position’s prediction is a weighted blend of its raw mean and the overall column mean, where the weight depends on how many training samples contributed to that position. This reduces variance on sparsely observed positions and should lower the MCRMSE, moving the score closer to the target while keeping the original workflow intact.'
- What this solution (achieved 0.42163) has done: 'I filter the training records to keep only high‑quality samples (SN_filter = 1 and signal_to_noise > 1) before computing per‑position statistics, and I reduce the shrinkage constant K from 10 to 5 so that per‑position means have a larger influence. This modest change keeps the overall workflow intact while giving more accurate position‑wise predictions, which should lower the MCRMSE toward the target score.'
- What this solution (achieved 0.42165) has done: 'The change reduces the shrinkage constant K from 5 to 1, giving the per‑position means computed from the high‑quality filtered records more influence while still keeping a modest regularisation toward the overall column mean. This should lower the MCRMSE (move the score closer to the target) without altering the overall workflow or model logic.'
- What this solution (achieved 0.42418) has done: 'I remove the strict quality filter and eliminate shrinkage so that per‑position predictions are based on raw means from all training records. This gives each position a stronger data‑driven estimate, which should lower the MCRMSE and move the score closer to the target while keeping the overall workflow unchanged.'
- What this solution (achieved 0.44716) has done: 'I replace the per‑position and overall statistics from means to medians, which are more robust to outliers and should lower the MCRMSE toward the target while keeping the overall workflow unchanged. The rest of the code (reading data, building the submission, and writing the CSV) remains the same.'
- What this solution (achieved 0.41641) has done: 'I replace the median‑based baseline with a mean‑based one and add a simple nucleotide‑aware adjustment: for each position we compute the average value per target column and also the average per nucleotide (A,C,G,U). During prediction the script looks up the nucleotide of the test sample at that position and uses the corresponding nucleotide‑specific mean when available, otherwise falls back to the regular per‑position mean (shrunken toward the overall column mean). This keeps the original workflow while providing more informative predictions, which should lower the MCRMSE toward the target score.'
- What this solution (achieved 0.41641) has done: 'I lower the shrinkage constant K to 0.5 so per‑position means rely more on actual data, and I make the nucleotide‑specific fallback only when enough samples (≥5) support it, otherwise falling back to the regular position mean. These changes keep the overall workflow intact while reducing noisy predictions, which should lower the MCRMSE toward the target score.'
- What this solution (achieved 0.41641) has done: 'I lower the shrinkage constant K from 0.5 to 0.2 so the per‑position means rely more on the actual observed values, and I reduce the nucleotide‑specific count threshold from 5 to 3 to allow more useful nucleotide‑aware adjustments. These minimal tweaks keep the overall workflow unchanged while expectedly tightening the predictions and moving the MCRMSE closer to the target score.'
- What this solution (achieved 0.41641) has done: 'I reduce the shrinkage (set K to 0) so per‑position means rely fully on observed data, lower the nucleotide‑count threshold to 2 to use more nucleotide‑specific information, and apply a simple three‑position moving‑average smoothing to the per‑position means. These minor tweaks keep the original workflow while expectedly lowering the MCRMSE toward the target score.'
- What this solution (achieved 0.41641) has done: 'I add a quality‑weighted aggregation so that per‑position means give more influence to high‑signal samples, and also compute loop‑type means to provide an extra fallback when nucleotide‑specific data is sparse. These modest extensions keep the original mean‑based workflow while aiming to lower the MCRMSE toward the target score.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict



## === cell 1
df_test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)

train_path = Path("../input/stanford-covid-vaccine/train.json")
with train_path.open("r") as f:
    train_records = [json.loads(line) for line in f]

sample_sub_path = Path("../input/stanford-covid-vaccine/sample_submission.csv")
df_sub = pd.read_csv(sample_sub_path)



## === cell 2
filtered_records = train_records

target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
NUM_POS = 68

pos_vals = {col: [[] for _ in range(NUM_POS)] for col in target_cols}
overall_vals = {col: [] for col in target_cols}

pos_weighted_sum = {col: np.zeros(NUM_POS) for col in target_cols}
pos_weight_sum = {col: np.zeros(NUM_POS) for col in target_cols}
overall_weighted_sum = {col: 0.0 for col in target_cols}
overall_weight_sum = {col: 0.0 for col in target_cols}

pos_nuc_vals = {col: [defaultdict(list) for _ in range(NUM_POS)] for col in target_cols}
pos_nuc_counts = {
    col: [defaultdict(int) for _ in range(NUM_POS)] for col in target_cols
}

pos_loop_vals = {
    col: [defaultdict(list) for _ in range(NUM_POS)] for col in target_cols
}
pos_loop_counts = {
    col: [defaultdict(int) for _ in range(NUM_POS)] for col in target_cols
}

for rec in filtered_records:
    seq_scored = rec.get("seq_scored", NUM_POS)
    seq = rec.get("sequence", "")
    loop_str = rec.get("predicted_loop_type", "")
    weight = float(rec.get("signal_to_noise", 1.0))  # quality weight; default 1.0

    for col in target_cols:
        vals = rec.get(col, [])
        if isinstance(vals, list):
            limit = min(seq_scored, len(vals), NUM_POS, len(seq), len(loop_str))
            for i in range(limit):
                val = vals[i]

                pos_vals[col][i].append(val)
                overall_vals[col].append(val)

                pos_weighted_sum[col][i] += val * weight
                pos_weight_sum[col][i] += weight
                overall_weighted_sum[col] += val * weight
                overall_weight_sum[col] += weight

                nuc = seq[i]
                if nuc:
                    pos_nuc_vals[col][i][nuc].append(val)
                    pos_nuc_counts[col][i][nuc] += 1

                loop_type = loop_str[i]
                if loop_type:
                    pos_loop_vals[col][i][loop_type].append(val)
                    pos_loop_counts[col][i][loop_type] += 1

overall_means = {}
for col in target_cols:
    if overall_weight_sum[col] > 0:
        overall_means[col] = overall_weighted_sum[col] / overall_weight_sum[col]
    else:
        overall_means[col] = np.mean(overall_vals[col]) if overall_vals[col] else 0.0

K = 0.0  # keep shrinkage disabled as before

pos_means = {}
for col in target_cols:
    col_means = []
    overall = overall_means[col]
    for i in range(NUM_POS):
        w_sum = pos_weight_sum[col][i]
        if w_sum > 0:
            raw_mean = pos_weighted_sum[col][i] / w_sum
        else:
            raw_mean = overall
        n_i = len(pos_vals[col][i])  # original count for possible shrinkage (K=0)
        adj_mean = (
            (n_i * raw_mean + K * overall) / (n_i + K) if (n_i + K) > 0 else overall
        )
        col_means.append(adj_mean)

    smoothed = []
    for i in range(NUM_POS):
        window_vals = col_means[max(0, i - 1) : min(NUM_POS, i + 2)]
        smoothed.append(float(np.mean(window_vals)))
    pos_means[col] = smoothed

pos_nuc_means = {}
for col in target_cols:
    nuc_means_per_pos = []
    for i in range(NUM_POS):
        nuc_dict = {}
        for nuc, lst in pos_nuc_vals[col][i].items():
            if lst:
                nuc_dict[nuc] = np.mean(lst)
        nuc_means_per_pos.append(nuc_dict)
    pos_nuc_means[col] = nuc_means_per_pos

pos_loop_means = {}
for col in target_cols:
    loop_means_per_pos = []
    for i in range(NUM_POS):
        loop_dict = {}
        for lt, lst in pos_loop_vals[col][i].items():
            if lst:
                loop_dict[lt] = np.mean(lst)
        loop_means_per_pos.append(loop_dict)
    pos_loop_means[col] = loop_means_per_pos



## === cell 3
df_sub["position"] = df_sub["id_seqpos"].str.split("_").str[-1].astype(int)
df_sub["sample_id"] = df_sub["id_seqpos"].str.rsplit("_", n=1, expand=True)[0]

id_to_seq = dict(zip(df_test["id"], df_test["sequence"]))

MIN_NUC_COUNT = 2
MIN_LOOP_COUNT = 2


def predict_value(col, pos, sample_id):
    if pos >= NUM_POS:
        return overall_means[col]
    seq = id_to_seq.get(sample_id, "")
    if pos < len(seq):
        nuc = seq[pos]
        if nuc and nuc in pos_nuc_means[col][pos]:
            if pos_nuc_counts[col][pos][nuc] >= MIN_NUC_COUNT:
                return pos_nuc_means[col][pos][nuc]
        loop_type = rec_loop_type = None
    return pos_means[col][pos]


test_loop_map = dict(zip(df_test["id"], df_test["predicted_loop_type"]))


def predict_value(col, pos, sample_id):
    if pos >= NUM_POS:
        return overall_means[col]
    seq = id_to_seq.get(sample_id, "")
    if pos < len(seq):
        nuc = seq[pos]
        if nuc and nuc in pos_nuc_means[col][pos]:
            if pos_nuc_counts[col][pos][nuc] >= MIN_NUC_COUNT:
                return pos_nuc_means[col][pos][nuc]
        loop_type = test_loop_map.get(sample_id, "")
        if pos < len(loop_type):
            lt = loop_type[pos]
            if lt and lt in pos_loop_means[col][pos]:
                if pos_loop_counts[col][pos][lt] >= MIN_LOOP_COUNT:
                    return pos_loop_means[col][pos][lt]
    return pos_means[col][pos]


for col in target_cols:
    df_sub[col] = df_sub.apply(
        lambda row: predict_value(col, row["position"], row["sample_id"]), axis=1
    )

df_sub = df_sub.drop(columns=["position", "sample_id"])



## === cell 4
output_path = Path("submission.csv")
df_sub.to_csv(output_path, index=False)
