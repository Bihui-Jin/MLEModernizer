# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Detect and classify harmful brain activity in electroencephalography (EEG) data: seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), or "other".

## Metric
Kullback Liebler divergence between the predicted probability and the observed target.

## Submission Format
For each `eeg_id` in the test set, you must predict a probability for each of the `vote` columns. The file should contain a header and have the following format:

```
eeg_id,seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote\
0,0.166,0.166,0.167,0.167,0.167,0.167\
1,0.166,0.166,0.167,0.167,0.167,0.167\
etc.
```

Your total predicted probabilities for each row must sum to one or your submission will fail.

## Dataset
**train.csv** Metadata for the train set. The expert annotators reviewed 50 second long EEG samples plus matched spectrograms covering 10 a minute window centered at the same time and labeled the central 10 seconds. Many of these samples overlapped and have been consolidated. `train.csv` provides the metadata that allows you to extract the original subsets that the raters annotated.

- `eeg_id` - A unique identifier for the entire EEG recording.
- `eeg_sub_id` - An ID for the specific 50 second long subsample this row's labels apply to.
- `eeg_label_offset_seconds` - The time between the beginning of the consolidated EEG and this subsample.
- `spectrogram_id` - A unique identifier for the entire EEG recording.
- `spectrogram_sub_id` - An ID for the specific 10 minute subsample this row's labels apply to.
- `spectogram_label_offset_seconds` - The time between the beginning of the consolidated spectrogram and this subsample.
- `label_id` - An ID for this set of labels.
- `patient_id` - An ID for the patient who donated the data.
- `expert_consensus` - The consensus annotator label. Provided for convenience only.
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The count of annotator votes for a given brain activity class. The full names of the activity classes are as follows: `lpd`: lateralized periodic discharges, `gpd`: generalized periodic discharges, `lrd`: lateralized rhythmic delta activity, and `grda`: generalized rhythmic delta activity . A detailed explanations of these patterns is [available here.](https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf)

**test.csv** Metadata for the test set. As there are no overlapping samples in the test set, many columns in the train metadata don't apply.

- `eeg_id`
- `spectrogram_id`
- `patient_id`

**sample_submission.csv**

- `eeg_id`
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The target columns. Your predictions must be probabilities. Note that the test samples had between 3 and 20 annotators.

**train_eegs/** EEG data from one or more overlapping samples. Use the metadata in train.csv to select specific annotated subsets. The column names are [the names of the individual electrode locations for EEG leads](https://en.wikipedia.org/wiki/10%E2%80%9320_system_%28EEG%29), with one exception. The EKG column is for an electrocardiogram lead that records data from the heart. All of the EEG data (for both train and test) was collected at a frequency of 200 samples per second.

**test_eegs/** Exactly 50 seconds of EEG data.

train_spectrograms/ Spectrograms assembled EEG data. Use the metadata in train.csv to select specific annotated subsets. The column names indicate the frequency in hertz and the recording regions of the EEG electrodes. The latter are abbreviated as LL = left lateral; RL = right lateral; LP = left parasagittal; RP = right parasagittal.

**test_spectrograms/** Spectrograms assembled using exactly 10 minutes of EEG data.

**example_figures/** Larger copies of the example case images used on the overview tab.

# 2. Python version

3.12

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        input/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        working/
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
```

-> data/hms-harmful-brain-activity-classification/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/hms-harmful-brain-activity-classification/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/hms-harmful-brain-activity-classification/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> (stopped after 10 files for performance)

# 5. Target score

0.4542823045725072

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The fix adds the missing imports, defines the target columns, loads the training data to compute a global class‑distribution, sets defaults for `NEEDTRAIN` and `PLATFORM`, reads the test metadata, creates predictions by repeating the overall probabilities for every test row, and writes a correctly‑formatted `submission.csv` whose probabilities sum to 1. This resolves the `NameError` and ensures a valid submission file is produced.'
- What this solution (achieved 1.68479) has done: 'I keep the original workflow but add a simple patient‑level prior: for each patient we compute the class distribution from the training votes and use it for test rows of the same patient (fallback to the global distribution when the patient is unseen). This modest change should lower the KL‑divergence without altering the core modelling approach.'
- What this solution (achieved 1.64506) has done: 'I replace the crude vote‑count aggregation with a per‑row normalized vote distribution, then average these uniformly per patient (and globally). This aligns the prior with the evaluation’s equal‑row weighting, keeping the same patient‑level fallback logic while improving the KL‑divergence score.'
- What this solution (achieved 1.68479) has done: 'The update replaces the equal‑row averaging with vote‑count‑weighted aggregates: patient‑level probabilities are derived from the total votes per patient, and the global fallback uses the overall vote totals. A tiny epsilon is added before normalisation to avoid division‑by‑zero, then all rows are re‑normalised to guarantee they sum to 1. This keeps the original workflow while providing more informative priors, which should lower the KL‑divergence score toward the target.'
- What this solution (achieved 1.41423) has done: 'I add a simple Bayesian smoothing step to the patient‑level prior: each patient’s vote totals are blended with the overall class votes (using a small α). This keeps the same overall workflow but makes the priors less extreme, which usually lowers the KL‑divergence and moves the score nearer the target. The rest of the code stays unchanged, and the submission file still contains rows that sum to 1.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

NEEDTRAIN = False

if os.path.isdir("/kaggle/input"):
    PLATFORM = "kaggle"
else:
    PLATFORM = "local"

train_path = (
    "./input/hms-harmful-brain-activity-classification/train.csv"
    if PLATFORM == "local"
    else "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
)
test_path = (
    "./input/hms-harmful-brain-activity-classification/test.csv"
    if PLATFORM == "local"
    else "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
)



## === cell 1
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Train shape:", train.shape)
print("Test shape :", test.shape)



## === cell 2
epsilon = 1e-9
row_sums = train[TARGETS].sum(axis=1).replace(0, np.nan) + epsilon
row_probs = train[TARGETS].div(row_sums, axis=0)

global_prob = row_probs.mean().values.astype(np.float64)
print("Global class probabilities (row‑averaged):", global_prob)

patient_counts = row_probs.groupby(train["patient_id"]).size()
patient_mean = row_probs.groupby(train["patient_id"]).mean()

alpha = 0.5
global_broadcast = pd.Series(global_prob, index=TARGETS)
patient_prob = (
    patient_mean.multiply(patient_counts, axis=0) + alpha * global_broadcast
) / (patient_counts[:, None] + alpha)

test_probs = test[["patient_id"]].merge(
    patient_prob.reset_index(),
    on="patient_id",
    how="left",
    suffixes=("", "_patient"),
)

filled_probs = test_probs[TARGETS].fillna(global_broadcast)

row_totals = filled_probs.sum(axis=1).replace(0, np.nan)
filled_probs = filled_probs.div(row_totals, axis=0)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3017987258.py in <cell line: 0>()
     18 patient_prob = (
     19     patient_mean.multiply(patient_counts, axis=0) + alpha * global_broadcast
---> 20 ) / (patient_counts[:, None] + alpha)
     21 
     22 # ---- Merge patient priors with test set ----

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __getitem__(self, key)
   1151             return self._get_rows_with_mask(key)
   1152 
-> 1153         return self._get_with(key)
   1154 
   1155     def _get_with(self, key):

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _get_with(self, key)
   1161             )
   1162         elif isinstance(key, tuple):
-> 1163             return self._get_values_tuple(key)
   1164 
   1165         elif not is_list_like(key):

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _get_values_tuple(self, key)
   1201             # the asarray is needed to avoid returning a 2D DatetimeArray
   1202             result = np.asarray(self._values[key])
-> 1203             disallow_ndim_indexing(result)
   1204             return result
   1205 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexers/utils.py in disallow_ndim_indexing(result)
    339     """
    340     if np.ndim(result) > 1:
--> 341         raise ValueError(
    342             "Multi-dimensional indexing (e.g. `obj[:, None]`) is no longer "
    343             "supported. Convert to a numpy array before indexing instead."

ValueError: Multi-dimensional indexing (e.g. `obj[:, None]`) is no longer supported. Convert to a numpy array before indexing instead.

## === cell 3
if not NEEDTRAIN:
    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
    sub[TARGETS] = filled_probs.values

    row_sums = sub[TARGETS].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), "Probabilities do not sum to 1"

    submission_path = "submission.csv"
    sub.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")
    print("Submission shape:", sub.shape)
    print(sub.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/99762496.py in <cell line: 0>()
      1 if not NEEDTRAIN:
      2     sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
----> 3     sub[TARGETS] = filled_probs.values
      4 
      5     # final sanity‑check

NameError: name 'filled_probs' is not defined
