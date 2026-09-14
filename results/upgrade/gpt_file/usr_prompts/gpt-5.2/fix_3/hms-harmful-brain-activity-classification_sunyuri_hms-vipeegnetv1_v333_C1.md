# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.13

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

0.2854737642477286

# 6. Current score

0.77501

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.01346) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory has no attribute GetPrototype`) by removing TensorFlow/Keras usage and replacing it with a lightweight, deterministic fallback that still produces a valid probability submission. Since your current score was “Not yielded”, the priority is to run end-to-end and write a correctly formatted `submission.csv` whose rows sum to 1. The fallback uses patient-level class priors from `train.csv` (and global priors for unseen patients), which is score-better than a uniform guess but keeps logic simple and stable within the 600s timeout. I also ensure column order matches `sample_submission.csv` exactly and add numerical safety (clipping + renormalization).'
- What this solution (achieved 0.77501) has done: 'Your current baseline (patient-level mean of per-row vote probabilities) is leaving performance on the table because the training metadata contains multiple rows per patient and per EEG with varying annotator counts, and KL is better matched by aggregating *counts* (Dirichlet-multinomial intuition) rather than averaging already-normalized distributions. I keep the same overall approach (priors-only, no EEG/spectrogram reading) but (1) compute patient/global priors by summing raw vote counts with a small symmetric pseudocount for stability, (2) back off unseen patients to global, and (3) for seen `eeg_id` in train, use an `eeg_id`-level prior (still a prior lookup, same semantics) which should move the score down toward your target with minimal risk. I also keep strict normalization/clipping so every row sums to 1 and submission stays valid. These changes should improve KL materially while preserving the lightweight deterministic logic and runtime.'

# 9. Code solution

## === cell 0
"""
Lightweight deterministic baseline for HMS - Harmful Brain Activity Classification.

Score-improvement intent (lower is better; current 1.01346 -> target ~0.285):
- Keep the same "priors-only" core logic (no EEG/spectrogram loading, no ML training loop).
- Improve the prior estimation to better match the KL metric:
  1) Aggregate RAW vote counts (not mean of normalized per-row probabilities).
     This better respects varying annotator totals per row.
  2) Apply a small symmetric Dirichlet pseudocount (Laplace smoothing) to avoid zeros,
     which are heavily penalized under KL.
  3) Use a more specific prior when possible:
     - if test eeg_id appears in train: use eeg_id-level prior
     - else if patient_id appears in train: use patient-level prior
     - else: use global prior
All outputs are clipped and renormalized to sum to 1 for submission validity.
"""

import os
import numpy as np
import pandas as pd

SEED = 2024
np.random.seed(SEED)

LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
sample_path = os.path.join(LOAD_DATA_FROM, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

TARGETS = [c for c in sample.columns if c != "eeg_id"]
assert len(TARGETS) == 6, f"Expected 6 target columns, got {len(TARGETS)}: {TARGETS}"

train_counts = train[TARGETS].astype(np.float64).values


def _normalize_probs(arr, eps=1e-6):
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.clip(arr, eps, 1.0)
    s = arr.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(arr, dtype=np.float64) / len(arr)
    return arr / s


def _counts_to_probs(counts_vec, alpha=0.5):
    """
    Minimal smoothing to reduce KL blow-ups from zeros while keeping priors sharp.
    alpha=0.5 (Jeffreys-like) is a conservative default.
    """
    counts_vec = np.asarray(counts_vec, dtype=np.float64)
    counts_vec = np.where(np.isfinite(counts_vec), counts_vec, 0.0)
    probs = counts_vec + alpha
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs, dtype=np.float64) / len(probs)
    else:
        probs = probs / s
    return _normalize_probs(probs)




## === cell 1

alpha = 0.5  # small pseudocount for all classes

global_counts = np.nansum(train_counts, axis=0)
global_prior_vec = _counts_to_probs(global_counts, alpha=alpha)

patient_sum = (
    pd.DataFrame(train_counts, columns=TARGETS)
    .assign(patient_id=train["patient_id"].values)
    .groupby("patient_id")[TARGETS]
    .sum()
)
patient_prior_map = {
    pid: _counts_to_probs(patient_sum.loc[pid, TARGETS].values, alpha=alpha)
    for pid in patient_sum.index.values
}

eeg_sum = (
    pd.DataFrame(train_counts, columns=TARGETS)
    .assign(eeg_id=train["eeg_id"].values)
    .groupby("eeg_id")[TARGETS]
    .sum()
)
eeg_prior_map = {
    eid: _counts_to_probs(eeg_sum.loc[eid, TARGETS].values, alpha=alpha)
    for eid in eeg_sum.index.values
}

preds = np.zeros((len(test), len(TARGETS)), dtype=np.float64)
test_eeg_ids = test["eeg_id"].values
test_patient_ids = test["patient_id"].values

for i in range(len(test)):
    eid = test_eeg_ids[i]
    pid = test_patient_ids[i]
    if eid in eeg_prior_map:
        preds[i] = eeg_prior_map[eid]
    else:
        preds[i] = patient_prior_map.get(pid, global_prior_vec)

preds = np.clip(preds, 1e-6, 1.0)
preds = preds / preds.sum(axis=1, keepdims=True)



## === cell 2
sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
for j, col in enumerate(TARGETS):
    sub[col] = preds[:, j].astype(np.float32)

sub = sub[["eeg_id"] + TARGETS]

row_sum_check = sub[TARGETS].sum(axis=1).values
if not np.all(np.isfinite(row_sum_check)):
    raise ValueError("Non-finite row sums in submission.")
max_dev = float(np.max(np.abs(row_sum_check - 1.0)))
print("Submission shape:", sub.shape)
print("Max deviation from sum=1:", max_dev)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
print(sub.head())
