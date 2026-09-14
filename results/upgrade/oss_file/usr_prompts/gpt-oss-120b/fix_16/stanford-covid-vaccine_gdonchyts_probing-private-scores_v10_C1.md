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

0.3520523385748121

# 6. Current score

0.39146

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.47906) has done: 'I replace the missing‑file loading and undefined‑variable code with a simple but valid submission pipeline: read the provided sample submission to get the required rows, compute column‑wise mean target values from the training data, fill every prediction with those means, and write the result to `submission.csv`. This fixes the FileNotFoundError, eliminates the NameError, and guarantees a correctly‑formatted CSV ready for Kaggle.'
- What this solution (achieved 0.3944) has done: 'The changes fix the JSON loading for the test data (using line‑oriented reading), add a fallback path for the sample submission, and ensure the `loop_char` column is correctly created before predictions are made. This resolves the runtime errors and lets the script generate a valid `submission.csv` while keeping the original prediction logic intact.'
- What this solution (achieved 0.39444) has done: 'The patch adds a lightweight fallback that uses the overall mean for each loop‑type character when a position‑specific mean is unavailable, giving the predictor a bit more information and expectedly lowering the MCRMSE toward the target. The core mean‑by‑position logic remains unchanged, only the prediction function is enhanced with this extra character‑level average.'
- What this solution (achieved 0.39146) has done: 'I fixed the typo `iterrow` → a proper loop over the training rows, using `range(len(train_df))` to keep the weight index aligned. This removes the AttributeError and ensures `sample_sub` is created before it is used, allowing the script to run end‑to‑end and generate a correctly formatted `submission.csv`.'
- What this solution (achieved 0.39146) has done: 'I replace the overall and per‑position averages with medians (which are less sensitive to outliers) while keeping the existing loop‑type weighted means unchanged. This small statistical change keeps the core logic intact but should lower the MCRMSE toward the target.'
- What this solution (achieved 0.39146) has done: 'Implemented weighted averages for overall and position‑wise statistics instead of medians. This provides more faithful estimates respecting the `signal_to_noise` weights, which should lower the MCRMSE toward the target while preserving the original loop‑type weighting logic.'
- What this solution (achieved 0.39146) has done: 'I add a light smoothing step to the position‑wise averages: each position’s mean be blended with its immediate neighbours (0.5 × center + 0.25 × left + 0.25 × right). This keeps the original weighted‑mean logic intact while reducing per‑position noise, which should lower the MCRMSE and move the score closer to the target. The smoothed values replace the original `pos_means` used in the fallback prediction.'
- What this solution (achieved 0.39146) has done: 'I slightly increase the smoothing of the position‑wise weighted means by using a 3‑point kernel (0.3 left + 0.4 center + 0.3 right) instead of the previous (0.25 + 0.5 + 0.25). This small change keeps the core averaging logic intact while reducing per‑position noise, which should lower the MCRMSE toward the target score.'
- What this solution (achieved 0.39146) has done: 'Improved smoothing of the position‑wise weighted means is added: a 5‑point uniform kernel (0.2 × each of the two neighbours on each side and the centre) replaces the previous 3‑point kernel. This extra smoothing reduces per‑position noise while keeping the original weighted‑average, loop‑type logic untouched, which is expected to lower the MCRMSE toward the target score.'
- What this solution (achieved 0.39146) has done: 'The plan adds a slightly larger smoothing kernel (7‑point) to reduce per‑position noise and adjusts the fallback prediction to blend the smoothed position mean with the overall column mean, keeping the original weighted‑average and loop‑type logic untouched while aiming to lower the MCRMSE toward the target.'
- What this solution (achieved 0.39146) has done: 'I slightly reduce the smoothing of the position‑wise weighted means (using a 3‑point kernel instead of the previous 7‑point one) and give more weight to the position‑specific mean when falling back to a generic prediction. This keeps the original weighting and loop‑type logic while expectedly lowering the MCRMSE toward the target score.'

# 9. Code solution

## === cell 0
import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict

BASE_PATH = "../input/stanford-covid-vaccine"

TRAIN_JSON = os.path.join(BASE_PATH, "train.json")
TEST_JSON = os.path.join(BASE_PATH, "test.json")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
OUTPUT_SUB = "submission.csv"

try:
    train_df = pd.read_json(TRAIN_JSON, lines=True)
except ValueError:
    with open(TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    train_df = pd.DataFrame(train_data)

TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

weights = train_df["signal_to_noise"].fillna(1.0).values
weights = np.where(np.isnan(weights), 1.0, weights)

mean_values = {}
pos_means = {}

for col in TARGET_COLS:
    stacked = np.vstack(train_df[col].values)  # shape (n_samples, 68)

    overall_w = np.repeat(weights, stacked.shape[1])
    mean_values[col] = np.average(stacked.ravel(), weights=overall_w)

    pos_weighted = []
    for p in range(68):
        vals = stacked[:, p]
        pos_weighted.append(np.average(vals, weights=weights))
    pos_means[col] = np.array(pos_weighted)

pos_means_smooth = {}
kernel = np.array([0.25, 0.5, 0.25])  # sum = 1.0
for col in TARGET_COLS:
    padded = np.pad(pos_means[col], (1, 1), mode="edge")
    smooth = np.convolve(padded, kernel, mode="valid")
    pos_means_smooth[col] = smooth
pos_means = pos_means_smooth

pos_loop_sum_weights = {
    col: [defaultdict(lambda: [0.0, 0.0]) for _ in range(68)] for col in TARGET_COLS
}
global_char_sum_weights = {col: defaultdict(lambda: [0.0, 0.0]) for col in TARGET_COLS}

for idx in range(len(train_df)):
    row = train_df.iloc[idx]
    loop_str = row["predicted_loop_type"]
    w = weights[idx]
    for col in TARGET_COLS:
        values = row[col]  # list of length 68
        for p, val in enumerate(values):
            char = loop_str[p] if p < len(loop_str) else ""
            sum_w = pos_loop_sum_weights[col][p][char]
            sum_w[0] += val * w  # weighted sum
            sum_w[1] += w  # total weight
            g_sum_w = global_char_sum_weights[col][char]
            g_sum_w[0] += val * w
            g_sum_w[1] += w

pos_loop_mean_vals = {col: [{} for _ in range(68)] for col in TARGET_COLS}
for col in TARGET_COLS:
    for p in range(68):
        for char, (s, wgt) in pos_loop_sum_weights[col][p].items():
            if wgt > 0:
                pos_loop_mean_vals[col][p][char] = s / wgt

char_global_means = {
    col: {
        char: (s / wgt) if wgt > 0 else np.nan
        for char, (s, wgt) in global_char_sum_weights[col].items()
    }
    for col in TARGET_COLS
}

if not os.path.exists(SAMPLE_SUB):
    SAMPLE_SUB = os.path.join(
        "/kaggle", "input", "stanford-covid-vaccine", "sample_submission.csv"
    )
sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["seqpos"] = sample_sub["id_seqpos"].apply(lambda x: int(x.split("_")[-1]))

try:
    test_df = pd.read_json(TEST_JSON, lines=True)
except ValueError:
    with open(TEST_JSON, "r") as f:
        test_data = [json.loads(line) for line in f if line.strip()]
    test_df = pd.DataFrame(test_data)

id_to_loop = dict(zip(test_df["id"], test_df["predicted_loop_type"]))


def get_loop_char(row):
    sample_id = row["id_seqpos"].rsplit("_", 1)[0]
    loop_str = id_to_loop.get(sample_id, "")
    pos = row["seqpos"]
    return loop_str[pos] if pos < len(loop_str) else ""


sample_sub["loop_char"] = sample_sub.apply(get_loop_char, axis=1)




## === cell 1
def predict_value(row, col):
    pos = row["seqpos"]
    if pos < 68:
        char = row["loop_char"]
        if char in pos_loop_mean_vals[col][pos]:
            return pos_loop_mean_vals[col][pos][char]
        if char in char_global_means[col]:
            return char_global_means[col][char]
        return 0.7 * pos_means[col][pos] + 0.3 * mean_values[col]
    else:
        return mean_values[col]


for col in TARGET_COLS:
    sample_sub[col] = sample_sub.apply(lambda r, c=col: predict_value(r, c), axis=1)

sample_sub.drop(columns=["seqpos", "loop_char"], inplace=True)
sample_sub.to_csv(OUTPUT_SUB, index=False)
print(f"Submission written to {OUTPUT_SUB} with shape {sample_sub.shape}")
