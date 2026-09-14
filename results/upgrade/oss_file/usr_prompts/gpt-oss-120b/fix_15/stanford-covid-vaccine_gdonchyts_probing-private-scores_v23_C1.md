# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.3677278275718549

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'The fix changes the input path to a real sample submission CSV (handling a few possible locations), keeps the original scaling step, and ensures the dataframe is written out as `submission.csv`. This resolves the file‑not‑found errors and guarantees a valid submission file is produced.'
- What this solution (achieved 0.42418) has done: 'I fix the JSON loading (the files are line‑delimited, not a single JSON array) and adjust the workflow so that the variables are defined before they are used. I also remove the unnecessary scaling factor (set to 1.0) to avoid degrading the score. These minimal edits let the script run end‑to‑end and produce a correct `submission.csv` file.'
- What this solution (achieved 0.57038) has done: 'I add a simple, per‑base calibration: compute the mean target value for each nucleotide (A, C, G, U) at every scored position and use that value instead of the overall position mean when the base is known. This keeps the same overall workflow while giving slightly more informed predictions, which should lower the MCRMSE toward the target. All other logic, file handling, and scaling remain unchanged.'
- What this solution (achieved 0.41657) has done: 'I add a lightweight internal validation split to estimate per‑target scaling factors that correct systematic bias in the simple mean‑based predictions. After computing these factors on the split, I apply them to the final predictions before writing the submission, keeping the overall averaging‑by‑position‑and‑base logic unchanged. This small calibration should lower the MCRMSE toward the target while preserving the original workflow.'
- What this solution (achieved 0.41653) has done: 'I add a simple bias‑correction step to the validation‑based calibration: instead of only a scaling factor `a`, I compute both a scale `a` and an offset `b` (linear regression on the validation split) and apply `pred = a*pred + b` to the test predictions. This keeps the original mean‑by‑position and base‑wise logic while providing a modest, targeted improvement that should lower the MCRMSE toward the target.'
- What this solution (achieved 0.41662) has done: 'The fix corrects the broadcasting error when smoothing base‑wise means, ensures the training/validation split runs without interruption, and restores the missing functions and variables so the pipeline can generate a valid `submission.csv`. No changes are made to the core modeling logic; only the tensor shapes and execution flow are repaired, which also keeps the existing calibration step that helps move the score toward the target.'
- What this solution (achieved 0.41655) has done: 'I lower the smoothing strength (`alpha`) from 5.0 to 1.0 so the base‑wise means rely more on observed data and less on the global mean. This small change keeps the overall workflow intact but should reduce over‑smoothing, bringing the MCRMSE closer to the target score while still writing a valid `submission.csv`.'
- What this solution (achieved 0.41654) has done: 'I lower the smoothing parameter `alpha` from 1.0 to 0.5 so the base‑wise means rely more on observed data and less on the global mean, which usually improves the calibration and reduces the MCRMSE toward the target. This change is minimal, keeps the core workflow intact, and be reflected wherever `alpha` is used.'

# 9. Code solution

## === cell 0
import json
import pathlib
import numpy as np
import pandas as pd

data_dir = pathlib.Path("data")
train_path = data_dir / "train.json"
test_path = data_dir / "test.json"
sample_sub_path = data_dir / "sample_submission.csv"


def load_json(path):
    with open(path, "r") as f:
        txt = f.read().strip()
    try:
        return json.loads(txt)  # single JSON array
    except json.JSONDecodeError:
        objs = []
        for line in txt.splitlines():
            if line:
                objs.append(json.loads(line))
        return objs


train_data = load_json(train_path)
test_data = load_json(test_path)

targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
base2idx = {"A": 0, "C": 1, "G": 2, "U": 3}
num_scored = max(entry["seq_scored"] for entry in train_data)  # should be 68
num_bases = len(base2idx)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2665928829.py in <cell line: 0>()
     26 
     27 
---> 28 train_data = load_json(train_path)
     29 test_data = load_json(test_path)
     30 

/tmp/ipykernel_11/2665928829.py in load_json(path)
     13 # ---------- helper to load line‑delimited or array JSON ----------
     14 def load_json(path):
---> 15     with open(path, "r") as f:
     16         txt = f.read().strip()
     17     try:

FileNotFoundError: [Errno 2] No such file or directory: 'data/train.json'

## === cell 1
means_train = {t: np.zeros(num_scored, dtype=float) for t in targets}
counts_pos = np.zeros(num_scored, dtype=int)

base_means = {t: np.zeros((num_scored, num_bases), dtype=float) for t in targets}
base_counts = np.zeros((num_scored, num_bases), dtype=int)

for entry in train_data:
    seq = entry["sequence"]
    seq_scored = entry["seq_scored"]
    for pos in range(seq_scored):
        counts_pos[pos] += 1
        base = seq[pos]
        if base not in base2idx:
            continue
        b_idx = base2idx[base]
        base_counts[pos, b_idx] += 1
        for t in targets:
            val = float(entry[t][pos])
            means_train[t][pos] += val
            base_means[t][pos, b_idx] += val

for t in targets:
    means_train[t] = np.where(counts_pos > 0, means_train[t] / counts_pos, 0.0)
    for b_idx in range(num_bases):
        cnt = base_counts[:, b_idx]
        mask = cnt > 0
        base_means[t][mask, b_idx] = base_means[t][mask, b_idx] / cnt[mask]

smoothed_base_means_train = {}
for t in targets:
    arr = np.where(base_counts > 0, base_means[t], means_train[t][:, None])
    smoothed_base_means_train[t] = arr



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2832435668.py in <cell line: 0>()
      1 # per‑position overall means
----> 2 means_train = {t: np.zeros(num_scored, dtype=float) for t in targets}
      3 counts_pos = np.zeros(num_scored, dtype=int)
      4 
      5 # per‑position, per‑base means (filled later)

NameError: name 'targets' is not defined

## === cell 2
rng = np.random.RandomState(42)
indices = np.arange(len(train_data))
rng.shuffle(indices)
split = int(0.8 * len(train_data))
train_idx, val_idx = indices[:split], indices[split:]
val_split = [train_data[i] for i in val_idx]

calibration = {}  # t -> (a_array, b_array)
for t in targets:
    a_arr = np.ones(num_scored, dtype=float)
    b_arr = np.zeros(num_scored, dtype=float)
    for pos in range(num_scored):
        true_vals = []
        pred_vals = []
        for entry in val_split:
            if pos >= entry["seq_scored"]:
                continue
            true_arr = np.asarray(entry[t], dtype=float)
            seq = entry["sequence"]
            base = seq[pos]
            if base in base2idx and base_counts[pos, base2idx[base]] > 0:
                pred = smoothed_base_means_train[t][pos, base2idx[base]]
            else:
                pred = means_train[t][pos]
            true_vals.append(true_arr[pos])
            pred_vals.append(pred)
        true_vals = np.array(true_vals, dtype=float)
        pred_vals = np.array(pred_vals, dtype=float)

        if len(true_vals) < 2 or np.sum(pred_vals**2) == 0:
            a, b = 1.0, 0.0
        else:
            a = np.sum(true_vals * pred_vals) / np.sum(pred_vals**2)
            b = true_vals.mean() - a * pred_vals.mean()
            if a <= 0:
                a, b = 1.0, 0.0
        a_arr[pos] = a
        b_arr[pos] = b
    calibration[t] = (a_arr, b_arr)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3579198456.py in <cell line: 0>()
      1 # ---------- train / validation split for calibration ----------
      2 rng = np.random.RandomState(42)
----> 3 indices = np.arange(len(train_data))
      4 rng.shuffle(indices)
      5 split = int(0.8 * len(train_data))

NameError: name 'train_data' is not defined

## === cell 3
rows = []
for entry in test_data:
    seq = entry["sequence"]
    seq_scored = entry["seq_scored"]
    seq_len = entry["seq_length"]  # should be 107
    for pos in range(seq_len):
        id_seqpos = f"{entry['id']}_{pos}"
        row = {"id_seqpos": id_seqpos}
        for t in targets:
            if pos < seq_scored:
                base = seq[pos] if pos < len(seq) else None
                if base in base2idx and base_counts[pos, base2idx[base]] > 0:
                    pred = smoothed_base_means_train[t][pos, base2idx[base]]
                else:
                    pred = means_train[t][pos]
            else:
                pred = means_train[t].mean()
            row[t] = pred
        rows.append(row)

pred_df = pd.DataFrame(rows)

pred_df["pos"] = pred_df["id_seqpos"].apply(lambda x: int(x.split("_")[-1]))
pos_idx = np.minimum(pred_df["pos"].values, num_scored - 1)

for t in targets:
    a_arr, b_arr = calibration.get(t, (np.ones(num_scored), np.zeros(num_scored)))
    pred_df[t] = pred_df[t].values * a_arr[pos_idx] + b_arr[pos_idx]

pred_df.drop(columns=["pos"], inplace=True)

submission = pred_df[["id_seqpos"] + targets]

out_path = pathlib.Path("submission.csv")
submission.to_csv(out_path, index=False)
print(f"Submission written to {out_path.resolve()}, shape={submission.shape}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1576276922.py in <cell line: 0>()
      1 # ---------- generate predictions for test ----------
      2 rows = []
----> 3 for entry in test_data:
      4     seq = entry["sequence"]
      5     seq_scored = entry["seq_scored"]

NameError: name 'test_data' is not defined
