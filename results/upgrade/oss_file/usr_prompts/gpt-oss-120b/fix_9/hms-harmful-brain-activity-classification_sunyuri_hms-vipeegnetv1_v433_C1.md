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

0.3435685774906769

# 6. Current score

1.68479

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'The fix forces the script to bypass TensorFlow and any heavy EEG processing, directly using the simple class‑frequency baseline to create a valid submission. By setting `TF_AVAILABLE = False` (and `NEEDTRAIN = False`) after the TensorFlow import attempt, the code follows the safe fallback path, avoiding the protobuf error triggered by later parquet handling and ensuring a correctly formatted `submission.csv` is written.'
- What this solution (achieved 1.64506) has done: 'We avoid the TensorFlow import that crashes by forcing the fallback path and set `TF_AVAILABLE=False` early. Instead of a single uniform baseline, we compute per‑patient average vote distributions from the training data and use them for test rows that share a patient ID, falling back to the global class mean otherwise. This keeps the original workflow but gives more informed probabilities, moving the KL‑divergence score closer to the target while still producing a valid `submission.csv`.'
- What this solution (achieved 1.64506) has done: 'I add a more specific fallback that first tries to use the per‑`eeg_id` average vote distribution (which is often more precise than the per‑patient average). If an `eeg_id` is not seen in the training set, the code fall back to the existing patient‑level mean, and finally to the global mean. This small enhancement keeps the same overall baseline logic while providing better calibrated probabilities, which should reduce the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.64506) has done: 'I add the missing imports and define the data folder path, then compute reliable per‑eeg and per‑patient average vote distributions (using the normalized vote vectors) instead of the previous one‑hot majority fallback. The prediction loop first use the per‑eeg mean, then the per‑patient mean, and finally the global class mean, ensuring the probabilities sum to 1 and the submission CSV is correctly written. This minimal change fixes the runtime errors and provides a better calibrated baseline that should move the KL‑divergence score toward the target while keeping the core logic unchanged.'
- What this solution (achieved 1.68479) has done: 'I improve the baseline by weighting each training sample with its total number of annotator votes when computing the global, patient‑level, and eeg‑level probability averages. This gives more influence to rows with many votes, producing better calibrated predictions and should lower the KL‑divergence toward the target. The rest of the pipeline (loading data, prediction loop, CSV output) remains unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

default_path = "/kaggle/input/hms-harmful-brain-activity-classification"
if not os.path.isdir(default_path):
    default_path = "./data/hms-harmful-brain-activity-classification"
LOAD_DATA_FROM = default_path




## === cell 1
train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
df = pd.read_csv(train_path)

TARGETS = df.columns[-6:]  # ['seizure_vote', 'lpd_vote', ... 'other_vote']
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))

vote_counts = df[TARGETS].values.astype(np.float32)  # shape (n_rows, 6)
row_totals = vote_counts.sum(axis=1, keepdims=True)
row_totals[row_totals == 0] = 1.0

norm_votes = vote_counts / row_totals

global_class_mean = vote_counts.sum(axis=0) / row_totals.sum()

patient_ids = df["patient_id"].values
patient_weighted_sums = {}
patient_weight_totals = {}
for pid, probs, weight in zip(patient_ids, norm_votes, row_totals.squeeze()):
    if pid not in patient_weighted_sums:
        patient_weighted_sums[pid] = np.zeros_like(global_class_mean)
        patient_weight_totals[pid] = 0.0
    patient_weighted_sums[pid] += probs * weight
    patient_weight_totals[pid] += weight

patient_means = {}
for pid in patient_weighted_sums:
    if patient_weight_totals[pid] > 0:
        patient_means[pid] = patient_weighted_sums[pid] / patient_weight_totals[pid]
    else:
        patient_means[pid] = global_class_mean  # fallback, should not happen

eeg_ids = df["eeg_id"].values
eeg_weighted_sums = {}
eeg_weight_totals = {}
for eid, probs, weight in zip(eeg_ids, norm_votes, row_totals.squeeze()):
    if eid not in eeg_weighted_sums:
        eeg_weighted_sums[eid] = np.zeros_like(global_class_mean)
        eeg_weight_totals[eid] = 0.0
    eeg_weighted_sums[eid] += probs * weight
    eeg_weight_totals[eid] += weight

eeg_means = {}
for eid in eeg_weighted_sums:
    if eeg_weight_totals[eid] > 0:
        eeg_means[eid] = eeg_weighted_sums[eid] / eeg_weight_totals[eid]
    else:
        eeg_means[eid] = global_class_mean  # fallback, should not happen




## === cell 2
test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
test = pd.read_csv(test_path)

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})

preds = np.zeros((len(test), len(TARGETS)), dtype=np.float32)

for i, (eid, pid) in enumerate(zip(test["eeg_id"].values, test["patient_id"].values)):
    if eid in eeg_means:
        preds[i] = eeg_means[eid]
    elif pid in patient_means:
        preds[i] = patient_means[pid]
    else:
        preds[i] = global_class_mean

row_sums = preds.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
preds = preds / row_sums

for j, col in enumerate(TARGETS):
    sub[col] = preds[:, j]

submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with shape {sub.shape}")
