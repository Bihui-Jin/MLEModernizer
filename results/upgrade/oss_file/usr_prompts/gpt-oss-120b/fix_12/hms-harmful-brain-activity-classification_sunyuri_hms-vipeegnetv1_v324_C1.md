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

0.2883324936316291

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I adjust the data‑location helper so it correctly finds the competition folder in the Kaggle environment (checking `/kaggle/input` and `/kaggle/working`). This fixes the FileNotFoundError and allows the script to run end‑to‑end and write a valid `submission.csv` with properly normalised probabilities. No other logic is changed, preserving the original baseline model.'
- What this solution (achieved 1.41937) has done: 'I replace the constant‑global‑probability baseline with a per‑`eeg_id` probability derived from the training metadata: for each `eeg_id` the vote counts are aggregated and normalised, and these specific distributions are used for matching test rows (falling back to the overall class frequencies otherwise). This small, data‑driven tweak keeps the original workflow intact while reducing the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I blend each per‑`eeg_id` probability with the overall class distribution, weighting the blend by how many votes that `eeg_id` has in the training data (more votes → trust the per‑`eeg_id` estimate more). This simple smoothing keeps the original baseline logic but reduces over‑confident predictions, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I increase the smoothing constant `K` from 100 to 1000 so the blended predictions rely more on the stable global class distribution, which should reduce over‑confident per‑`eeg_id` estimates and lower the KL‑divergence toward the target score. The rest of the pipeline remains unchanged.'
- What this solution (achieved 1.41937) has done: 'I lower the smoothing constant `K` from 1000 to 100 so the per‑`eeg_id` vote distributions influence the predictions more strongly. This small change keeps the overall workflow unchanged while reducing over‑smoothing, which should bring the KL‑divergence closer to the target lower score.'
- What this solution (achieved 1.41937) has done: 'The change raises the smoothing constant `K` to a very large value (1 000 000) so the blend weight becomes almost zero, making the predictions rely almost entirely on the stable global class distribution. This reduces noisy per‑`eeg_id` estimates and moves the KL‑divergence much closer to the target low score while keeping the original workflow unchanged.'
- What this solution (achieved 1.41937) has done: 'I lower the smoothing constant `K` dramatically (to 1) so the per‑`eeg_id` vote distributions dominate the blended prediction instead of being almost completely overridden by the global class frequencies. This increases the influence of the training‑derived per‑ID probabilities, which should reduce the KL‑divergence and move the score closer to the lower target. I also rename the cell header to start at 1, keeping the rest of the pipeline unchanged.'
- What this solution (achieved 1.41937) has done: 'I increase the smoothing constant `K` to a very large value (1 000 000) so the blended predictions rely almost entirely on the stable global class distribution, which reduces noisy per‑`eeg_id` estimates and moves the KL‑divergence closer to the lower target score. I also rename the cell header to start at 1 to keep the notebook format consistent.'
- What this solution (achieved 1.41937) has done: 'I lower the smoothing constant `K` from the extreme value 1 000 000 to a modest 10, which lets the per‑`eeg_id` vote‑derived probabilities influence the final predictions far more while keeping the existing blending and renormalisation logic unchanged. This small adjustment is expected to move the KL‑divergence markedly closer to the target lower score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np


def locate_data_root(relative_path: str) -> str:
    """
    Search for the given relative_path in several common Kaggle locations.
    Returns the first existing directory, otherwise raises FileNotFoundError.
    """
    candidates = [
        os.path.join(".", "data", relative_path),
        os.path.join(".", "input", relative_path),
        os.path.join(".", "working", relative_path),
        os.path.join("/kaggle", "input", relative_path),
        os.path.join("/kaggle", "working", relative_path),
        os.path.join("/kaggle", "data", relative_path),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Data root not found. Checked: {candidates}")


DATA_ROOT = locate_data_root("hms-harmful-brain-activity-classification")

train_path = os.path.join(DATA_ROOT, "train.csv")
df_train = pd.read_csv(train_path)

TARGETS = df_train.columns[-6:]

global_vote_sums = df_train[TARGETS].sum()
global_total_votes = global_vote_sums.sum()
global_class_probs = global_vote_sums / global_total_votes

eeg_votes_sum = df_train.groupby("eeg_id")[list(TARGETS)].sum()
eeg_totals = eeg_votes_sum.sum(axis=1).replace(0, np.nan)  # avoid division by zero
eeg_class_probs = eeg_votes_sum.div(eeg_totals, axis=0).fillna(global_class_probs)

test_path = os.path.join(DATA_ROOT, "test.csv")
df_test = pd.read_csv(test_path)

submission = pd.DataFrame({"eeg_id": df_test["eeg_id"]})

for col in TARGETS:
    submission[col] = df_test["eeg_id"].map(eeg_class_probs[col])

submission[TARGETS] = submission[TARGETS].fillna(global_class_probs)

K = 10.0
test_votes = df_test["eeg_id"].map(eeg_totals).fillna(0.0)
weight = test_votes / (test_votes + K)  # larger weight for IDs with more training votes
weight = weight.values[:, None]  # reshape for broadcasting

global_array = global_class_probs.values.reshape(1, -1)  # shape (1, n_classes)

submission[TARGETS] = weight * submission[TARGETS].values + (1 - weight) * global_array

row_sums = submission[TARGETS].sum(axis=1)
submission[TARGETS] = submission[TARGETS].div(row_sums, axis=0)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")
