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

0.5383913144990755

# 6. Current score

0.9395

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I avoid importing TensorFlow (which raises a protobuf‑related error) and replace the heavy model inference with a simple baseline: compute the average class‑probability distribution from the training vote columns and use that constant vector for every test row. This guarantees a valid CSV with rows summing to 1, fixes the runtime crash, and provides a reasonable score without altering the core modelling logic.'
- What this solution (achieved 1.39779) has done: 'I replace the single‑class global baseline with a per‑`eeg_id` average probability derived from the training votes. For each test row, if its `eeg_id` appears in the training set we use that specific mean distribution; otherwise we fall back to the overall baseline. This keeps the original logic but adds a lightweight, data‑driven calibration that should lower the KL‑divergence (moving the score from 1.39779 toward the target 0.5383) while still guaranteeing rows sum to 1.'
- What this solution (achieved 0.92423) has done: 'I add simple Laplace smoothing to the vote counts and introduce a fallback that uses the average class distribution per `patient_id` when an unseen `eeg_id` appears in the test set. This keeps the original per‑eeg‑id baseline but gives a more informed estimate for rows without a matching `eeg_id`, which should reduce the KL‑divergence and move the score closer to the target while preserving the overall pipeline.'
- What this solution (achieved 0.97704) has done: 'I add lightweight shrinkage blending so that per‑eeg_id and per‑patient probabilities are combined with the global baseline according to how many training samples support them. This keeps the existing logic but makes noisy, low‑count estimates fall back toward the stable global distribution, which should lower the KL‑divergence and move the score from 0.924 → closer to the target 0.538. No new libraries are used and the submission format remains unchanged.'
- What this solution (achieved 0.93243) has done: 'I lower the pseudo‑count `ALPHA` used for blending the per‑eeg / per‑patient estimates with the global baseline. A smaller `ALPHA` (e.g., 1.0 instead of 10.0) lets the model rely more on the observed vote distributions when they are available, which should reduce the KL‑divergence and move the score closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 1.39879) has done: 'I simplify the fallback logic to rely only on the per‑eeg ID estimate (blended with the global baseline) and otherwise use the baseline directly, removing the noisy per‑patient fallback. I also increase the pseudo‑count `ALPHA` to 5.0 so that low‑count IDs are smoothed more toward the stable global distribution, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.9395) has done: 'I add a lightweight per‑patient fallback and use a smaller smoothing constant for blending.  
For rows with a known `eeg_id` we blend the per‑id distribution with the global baseline using `ALPHA_ID=2.0`.  
If the `eeg_id` is unseen, we fall back to a per‑patient distribution (when available) and blend it with the baseline using `ALPHA_PAT=2.0`.  
All other logic (vote smoothing, normalization, file output) is unchanged, ensuring a valid CSV while moving the KL‑divergence score closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal

PLATFORM = "kaggle"  # 'local' or 'kaggle'
if PLATFORM == "local":
    base_path = "./input/hms-harmful-brain-activity-classification"
else:
    base_path = "/kaggle/input/hms-harmful-brain-activity-classification"

train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")
submission_path = "submission.csv"

print("Loading train data...")
train_df = pd.read_csv(train_path)
print("Train shape:", train_df.shape)

print("Loading test data...")
test_df = pd.read_csv(test_path)
print("Test shape:", test_df.shape)

TARGETS = train_df.columns[-6:]  # ['seizure_vote', ..., 'other_vote']
print("Target columns:", list(TARGETS))


def row_to_prob(row):
    votes = row[TARGETS].values.astype(float) + 1.0
    s = votes.sum()
    return votes / s


train_probs = train_df.apply(row_to_prob, axis=1, result_type="expand")
train_probs.columns = TARGETS

baseline_prob = train_probs.mean().values
baseline_prob = baseline_prob / baseline_prob.sum()
print("Global baseline class probabilities:", baseline_prob)

train_probs["eeg_id"] = train_df["eeg_id"]
train_probs["patient_id"] = train_df["patient_id"]

per_id_probs = train_probs.groupby("eeg_id")[list(TARGETS)].mean()
id_counts = train_probs.groupby("eeg_id").size()
print("Per‑eeg_id probability table shape:", per_id_probs.shape)

per_patient_probs = train_probs.groupby("patient_id")[list(TARGETS)].mean()
patient_counts = train_probs.groupby("patient_id").size()
print("Per‑patient probability table shape:", per_patient_probs.shape)

test_eeg_ids = test_df["eeg_id"]
test_patient_ids = test_df["patient_id"]

aligned_id = per_id_probs.reindex(test_eeg_ids).values  # (n_test, 6)
mask_id_valid = ~np.isnan(aligned_id).any(axis=1)

final_aligned = np.tile(baseline_prob, (len(test_eeg_ids), 1))

ALPHA_ID = 2.0  # smaller smoothing → rely more on observed data
if mask_id_valid.any():
    id_counts_aligned = id_counts.reindex(test_eeg_ids).values.astype(float)
    id_counts_aligned[~mask_id_valid] = 0.0
    blended_id = (
        aligned_id * id_counts_aligned[:, None] + ALPHA_ID * baseline_prob
    ) / (id_counts_aligned[:, None] + ALPHA_ID)
    final_aligned[mask_id_valid] = blended_id[mask_id_valid]

mask_need_patient = ~mask_id_valid
if mask_need_patient.any():
    aligned_patient = per_patient_probs.reindex(test_patient_ids).values
    mask_patient_valid = ~np.isnan(aligned_patient).any(axis=1) & mask_need_patient
    if mask_patient_valid.any():
        patient_counts_aligned = patient_counts.reindex(test_patient_ids).values.astype(
            float
        )
        patient_counts_aligned[~mask_patient_valid] = 0.0
        ALPHA_PAT = 2.0
        blended_pat = (
            aligned_patient * patient_counts_aligned[:, None]
            + ALPHA_PAT * baseline_prob
        ) / (patient_counts_aligned[:, None] + ALPHA_PAT)
        final_aligned[mask_patient_valid] = blended_pat[mask_patient_valid]

row_sums = final_aligned.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
final_aligned = final_aligned / row_sums

submission = pd.DataFrame({"eeg_id": test_eeg_ids.values})
for idx, col in enumerate(TARGETS):
    submission[col] = final_aligned[:, idx]

row_sums_check = submission[TARGETS].sum(axis=1)
print(
    "Row sum stats (should be 1.0): min {:.6f}, max {:.6f}, mean {:.6f}".format(
        row_sums_check.min(), row_sums_check.max(), row_sums_check.mean()
    )
)

submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape {submission.shape}")
