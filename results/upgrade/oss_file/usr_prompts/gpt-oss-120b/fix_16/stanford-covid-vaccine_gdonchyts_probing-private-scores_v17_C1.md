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

0.3589839418734281

# 6. Current score

0.42214

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'The fix locates the actual sample_submission.csv instead of a missing file, loads it, applies the original scaling factor, and writes a proper submission.csv so the pipeline completes without errors.'
- What this solution (achieved 0.63824) has done: 'I replace the placeholder scaling‑only submission with a simple baseline that uses the average values of each target at every scored position from the training data. The script now loads the training JSON, computes per‑position means for the five targets, and fills the submission rows with those means (or 0 for positions beyond the scored 68 bases). This small, targeted change provides much more informed predictions and moves the MCRMSE score much closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.42166) has done: 'The fix loads the newline‑delimited `train.json` correctly (`lines=True`) and restricts the training rows to high‑quality samples (`SN_filter == 1`). This prevents the “Trailing data” error and yields more representative per‑position means, which modestly improves the MCRMSE while keeping the original baseline logic unchanged. The rest of the pipeline (reading the sample submission, filling predictions with the means, and writing `submission.csv`) remains the same.'
- What this solution (achieved 0.44716) has done: 'I remove the SN_filter restriction and replace the per‑position mean with a per‑position median, which is more robust to noisy samples. This modest change keeps the overall baseline logic intact while using the full training set, and should lower the MCRMSE toward the target value.'
- What this solution (achieved 0.4437) has done: 'I filter the training data to only high‑quality samples (`SN_filter == 1`) and keep the robust per‑position median calculation. This small change should give more reliable position‑wise targets and move the MCRMSE closer to the target score while preserving the original baseline logic.'
- What this solution (achieved 0.42178) has done: 'I replace the simple per‑position median with a quality‑aware weighted mean, using the `signal_to_noise` column as a weight. Keeping the SN_filter == 1 restriction preserves high‑quality samples, while weighting by signal‑to‑noise aligns the predictions more closely with the RMSE‑based metric, which should lower the MCRMSE toward the target. The rest of the pipeline (reading the sample submission and writing the CSV) remains unchanged.'
- What this solution (achieved 0.63824) has done: 'I keep the overall baseline structure but adjust the predictions to be a slight blend of the per‑position weighted mean (which already gave a 0.42178 score) with the global mean for each target. By mixing in a small portion of the overall mean (10 %), we reduce variance on noisy positions and move the validation metric closer to the target lower‑error value. The changes are limited to computing the global means and applying the blend when filling the submission.'
- What this solution (achieved 0.43788) has done: 'The fix corrects the target column name typo, switches the position‑wise statistic from a weighted mean to a robust median (which better matches the MCRMSE metric), and lowers the blending factor so predictions rely more on the stable global mean. These changes resolve the KeyError and should decrease the validation score toward the target while preserving the original pipeline structure.'
- What this solution (achieved 0.63824) has done: 'I replace the position‑wise median with a signal‑to‑noise weighted mean (which better matches the RMSE‑based metric) and keep the same blending logic, preserving the overall pipeline while moving the validation error closer to the target score.'
- What this solution (achieved 0.63824) has done: 'I corrected the weighted‑average calculation (providing the required `axis` argument), removed the unnecessary SN_filter restriction so all training samples contribute, and increased the blending factor to rely more on the per‑position statistics. These fixes eliminate the earlier TypeError and KeyError, ensure a valid `submission.csv` is written, and modestly improve the MCRMSE toward the target.'
- What this solution (achieved 0.43144) has done: 'I fixed the weighted‑average error when computing the overall means, added a proper flattening of the arrays, and lowered the blending factor (α) to rely a bit more on the overall weighted mean, which should reduce variance and move the MCRMSE closer to the target. The script now runs end‑to‑end and writes a correct `submission.csv`.'
- What this solution (achieved 0.42215) has done: 'I filter the training set to keep only high‑quality samples (`SN_filter == 1`) and increase the reliance on the per‑position weighted statistics by raising the blending factor α. This keeps the original baseline structure while using cleaner data and a stronger per‑position signal, which should lower the MCRMSE toward the target.'
- What this solution (achieved 0.43679) has done: 'I replace the per‑position weighted‑mean with a simple per‑position median (more robust to noisy samples) and lower the blending factor α from 0.85 to 0.65 so the overall mean has a stronger regularising effect. These minimal adjustments keep the original pipeline intact while expectedly reducing the MCRMSE toward the target score.'
- What this solution (achieved 0.4243) has done: 'I keep the overall pipeline unchanged but replace the per‑position median with a signal‑to‑noise‑weighted mean (using all training rows) and increase the blend factor so the predictions rely more on these per‑position statistics. This small, targeted change should lower the MCRMSE toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.42214) has done: 'I filter the training data to keep only high‑quality samples (`SN_filter == 1`) when computing the position‑wise statistics, while still using all samples for the overall means. This reduces noisy per‑position estimates and should lower the MCRMSE. I also raise the blending factor α slightly (to 0.85) to give a bit more weight to the now cleaner position‑wise predictions. The rest of the pipeline stays unchanged, and a valid `submission.csv` is still written.'

# 9. Code solution

## === cell 0
import numpy as np, pandas as pd, os, glob




## === cell 1
sample_paths = glob.glob(os.path.join("**", "sample_submission.csv"), recursive=True)
if not sample_paths:
    raise FileNotFoundError("sample_submission.csv not found in any subdirectory.")
df = pd.read_csv(sample_paths[0])




## === cell 2
train_paths = glob.glob(os.path.join("**", "train.json"), recursive=True)
if not train_paths:
    raise FileNotFoundError("train.json not found in any subdirectory.")
train_df = pd.read_json(train_paths[0], lines=True)

targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

weights_all = train_df["signal_to_noise"].values
overall_means = {}
for t in targets:
    stacked_all = np.vstack(train_df[t].values)
    overall_means[t] = np.average(
        stacked_all.ravel(), weights=np.repeat(weights_all, stacked_all.shape[1])
    )

hq_df = train_df[train_df["SN_filter"] == 1]
if hq_df.empty:
    hq_df = train_df
weights_hq = hq_df["signal_to_noise"].values

per_pos_means = {}
for t in targets:
    stacked_hq = np.vstack(hq_df[t].values)
    per_pos_means[t] = np.average(stacked_hq, axis=0, weights=weights_hq)




## === cell 3
alpha = 0.85  # increased reliance on the cleaner position‑wise statistic


def get_pred_vals(pos):
    """Blended prediction for all targets at a given sequence position."""
    if pos < 68:
        return [
            alpha * per_pos_means[t][pos] + (1 - alpha) * overall_means[t]
            for t in targets
        ]
    else:
        return [overall_means[t] for t in targets]


pred_positions = df["id_seqpos"].apply(lambda x: int(x.split("_")[-1]))

for i, t in enumerate(targets, start=1):
    df[t] = pred_positions.apply(lambda p: get_pred_vals(p)[i - 1])




## === cell 4
df.to_csv("submission.csv", index=False)
