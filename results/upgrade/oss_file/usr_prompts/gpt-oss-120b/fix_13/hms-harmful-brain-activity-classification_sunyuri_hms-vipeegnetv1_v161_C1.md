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

0.3465916497288482

# 6. Current score

0.72688

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'The script failed because the data directory was wrong, causing a FileNotFoundError, and consequently the `test` DataFrame was never created. I added robust path selection that picks the first existing folder among the common locations, then reloads the data using that path. The rest of the logic is unchanged; the submission now correctly references the loaded `test` DataFrame and writes a valid CSV with probabilities that sum to 1 for each row.'
- What this solution (achieved 1.64506) has done: 'I add a lightweight patient‑level prior: compute the mean normalized class distribution for each patient in the training set and use it for any test rows that share the same patient_id, falling back to the overall baseline otherwise. This keeps the original logic but gives more personalized probabilities, which should lower the KL divergence toward the target score.'
- What this solution (achieved 1.64506) has done: 'I add a per‑eeg‑id prior that overrides the patient‑level prior when the same `eeg_id` appears in the training data. This small change keeps the original baseline‑and‑patient logic but supplies a more specific probability distribution for any test rows that share an `eeg_id` with the training set, which should lower the KL divergence toward the target score.'
- What this solution (achieved 1.64506) has done: 'I keep the overall prior‑based approach but add a third level of information – the `spectrogram_id` distribution – and blend any available priors (eeg, patient, spectrogram) by averaging them instead of overriding with a single one. This small extension preserves the core logic while giving the model more relevant context, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.7224) has done: 'I keep the overall prior‑based design but replace the unweighted averaging with a simple weighted blend that gives more influence to more specific priors (eeg → patient → spectrogram) while still falling back to the overall baseline. By weighting each available distribution by the number of training rows that contributed it (and adding a small baseline weight), the predictions become better calibrated and should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.75839) has done: 'I reduce the baseline influence by lowering the `baseline_weight` from 1.0 to 0.2, letting the more specific patient/eeg/spectrogram priors dominate the prediction while preserving the overall blending logic. This small tweak should improve calibration and move the KL‑divergence closer to the target score without altering the core methodology.'
- What this solution (achieved 0.7224) has done: 'I increase the baseline weight from 0.2 to 1.0 to give the overall prior more influence and reduce over‑fitting to sparse patient/eeg/spectrogram priors. This small adjustment keeps the blending logic unchanged while providing stronger regularisation, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.72688) has done: 'I reduce the influence of the specific priors (eeg, patient, spectrogram) by scaling their counts with a logarithmic factor. This keeps the same blending logic but makes the weights smaller, giving the overall baseline more regularising effect, which should move the KL‑divergence closer to the target lower score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

possible_paths = [
    "/kaggle/input/hms-harmful-brain-activity-classification",
    "./input/hms-harmful-brain-activity-classification",
    "./data/hms-harmful-brain-activity-classification",
    "./working/hms-harmful-brain-activity-classification",
]
base_path = next((p for p in possible_paths if os.path.isdir(p)), None)
if base_path is None:
    raise FileNotFoundError("Could not locate the competition data folder.")

train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")
submission_path = "submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

TARGETS = train.columns[-6:]

train_norm = train[TARGETS].div(train[TARGETS].sum(axis=1), axis=0)

baseline_probs = train_norm.mean(axis=0).values.astype(np.float32)
baseline_probs /= baseline_probs.sum()  # ensure sum to 1


def build_dist_dict(group_series):
    dist = group_series.mean()
    weight = np.log1p(len(group_series))
    return dist.values.astype(np.float32), float(weight)


patient_group = train_norm.groupby(train["patient_id"])
patient_dist_dict = {pid: build_dist_dict(g) for pid, g in patient_group}

eeg_group = train_norm.groupby(train["eeg_id"])
eeg_dist_dict = {eid: build_dist_dict(g) for eid, g in eeg_group}

spec_group = train_norm.groupby(train["spectrogram_id"])
spectrogram_dist_dict = {sid: build_dist_dict(g) for sid, g in spec_group}




## === cell 1
baseline_weight = 1.0
probs = np.tile(baseline_probs, (len(test), 1))

test_patient_ids = test["patient_id"].values
test_eeg_ids = test["eeg_id"].values
test_spectrogram_ids = test["spectrogram_id"].values

for idx, (eid, pid, sid) in enumerate(
    zip(test_eeg_ids, test_patient_ids, test_spectrogram_ids)
):
    weighted_sum = baseline_probs * baseline_weight
    total_weight = baseline_weight

    if eid in eeg_dist_dict:
        dist, w = eeg_dist_dict[eid]
        weighted_sum += dist * w
        total_weight += w
    if pid in patient_dist_dict:
        dist, w = patient_dist_dict[pid]
        weighted_sum += dist * w
        total_weight += w
    if sid in spectrogram_dist_dict:
        dist, w = spectrogram_dist_dict[sid]
        weighted_sum += dist * w
        total_weight += w

    probs[idx] = weighted_sum / total_weight

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
for col_idx, col in enumerate(TARGETS):
    sub[col] = probs[:, col_idx]

row_sums = sub[TARGETS].sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-6), "Row probabilities do not sum to 1."

sub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with shape {sub.shape}")
