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

0.2798394594864913

# 6. Current score

0.79437

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I replace the failing TensorFlow‑based pipeline with a lightweight fallback that avoids the protobuf error. The new code simply reads the training votes, computes the overall class probability distribution, and uses this same distribution for every test record, producing a valid `submission.csv` with the required columns. This fixes the runtime crash and yields a reasonable baseline score that moves toward the target.'
- What this solution (achieved 1.41937) has done: 'I replace the uniform‑distribution baseline with a per‑eeg‑id prior: for each eeg_id that appears in the training set I compute its own vote‑distribution and use it for the matching test rows, falling back to the overall class distribution when an eeg_id is unseen. This small, deterministic change keeps the original logic but should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.57587) has done: 'We keep the original per‑eeg‑id fallback but add a second‑level fallback using the patient‑level vote distribution, which is available for many test rows that lack a matching eeg_id. This extra information should bring the predicted probabilities closer to the true labels, lowering the KL‑divergence toward the target. A tiny epsilon is also added before normalisation to avoid zero probabilities.'
- What this solution (achieved 1.57587) has done: 'I replace the simple fallback hierarchy with a weighted blend of the per‑eeg‑id and per‑patient vote distributions. By combining both specific signals (when available) instead of using only one, the predictions become better calibrated to the true label distribution, which should reduce the KL‑divergence and move the score closer to the target. Missing values are still filled with the global baseline, and the final rows are re‑normalised to guarantee they sum to one.'
- What this solution (achieved 1.57587) has done: 'I remove the weighted blending and use a strict hierarchy: when a per‑eeg _id distribution is available it is taken directly, otherwise the per‑patient distribution is used, and finally the global baseline fills any remaining missing values. This simple fallback typically yields probabilities closer to the true labels, lowering the KL‑divergence toward the target score while keeping the original logic unchanged.'
- What this solution (achieved 0.78827) has done: 'I keep the overall hierarchy but improve the fallback by blending the patient‑level probabilities with a small weight of the global distribution before resorting to the pure global baseline. This smooths predictions for rows lacking an exact eeg_id match, reduces zero‑probability issues, and is expected to lower the KL‑divergence toward the target score while preserving the original logic.'
- What this solution (achieved 0.77767) has done: 'I keep the overall hierarchy but give the patient‑level distribution a higher influence when falling back to it (90 % patient + 10 % global instead of 80 % + 20 %). This small change should lower the KL‑divergence, moving the score closer to the target while preserving the existing logic and ensuring a valid submission.'
- What this solution (achieved 1.57587) has done: 'I tighten the fallback hierarchy by removing the 0.9 patient + 0.1 global blend and using the pure patient‑level distribution whenever an EEG‑id specific distribution is unavailable. This keeps the core logic unchanged while giving predictions that better reflect the observed patient patterns, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.77767) has done: 'We keep the overall hierarchy (eeg‑id → patient → global) but improve the patient fallback by blending it with a small weight of the global distribution (90 % patient + 10 % global). This smoothing reduces over‑confident patient‑only predictions, lowering the KL‑divergence and moving the score closer to the target while preserving the original logic.'
- What this solution (achieved 0.77949) has done: 'I keep the original hierarchical fallback (eeg‑id → patient → global) but add a tiny global smoothing to every row after the hierarchy is applied.  By mixing in 5 % of the overall class distribution we reduce over‑confident predictions (especially for rare classes) and keep the rows normalised, which is expected to lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.91057) has done: 'I keep the overall hierarchical fallback unchanged but increase the smoothing toward the global class distribution, which reduces over‑confident per‑eeg and patient predictions and should lower the KL‑divergence toward the target. Specifically, I lower the patient‑blend weight from 0.9 to 0.7 and increase the final global‑smoothing from 5 % to 30 %, then renormalise rows as before.'
- What this solution (achieved 0.77949) has done: 'I tighten the hierarchical fallback by giving the patient‑level distribution a higher influence (90 % patient + 10 % global) and reduce the final global smoothing to only 5 % of the prediction. This keeps the original logic while making the per‑eeg and patient priors dominate, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.79437) has done: 'I add a modest Laplace smoothing factor to the vote counts when computing the global, per‑eeg, and per‑patient probability distributions, and increase the final global‑smoothing blend from 5 % to 15 %. These tiny adjustments keep the hierarchical fallback logic unchanged while reducing over‑confident predictions, which should lower the KL‑divergence toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
else:
    PLATFORM = "local"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"

train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
df_train = pd.read_csv(train_path)

test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
df_test = pd.read_csv(test_path)

TARGETS = df_train.columns[-6:]  # seizure_vote … other_vote

smooth = 0.5

global_counts = df_train[TARGETS].sum().values.astype(np.float64) + smooth
global_probs = global_counts / global_counts.sum()
global_probs = global_probs / global_probs.sum()  # ensure sum‑1

eeg_id_sums = df_train.groupby("eeg_id")[list(TARGETS)].sum() + smooth
eeg_id_probs = eeg_id_sums.div(eeg_id_sums.sum(axis=1), axis=0)

patient_id_sums = df_train.groupby("patient_id")[list(TARGETS)].sum() + smooth
patient_id_probs = patient_id_sums.div(patient_id_sums.sum(axis=1), axis=0)

eeg_merge = df_test[["eeg_id"]].merge(
    eeg_id_probs.reset_index(), on="eeg_id", how="left"
)

patient_merge = df_test[["patient_id"]].merge(
    patient_id_probs.reset_index(), on="patient_id", how="left"
)

patient_blend = patient_merge.copy()
for i, col in enumerate(TARGETS):
    patient_blend[col] = 0.9 * patient_merge[col] + 0.1 * global_probs[i]

submission = pd.DataFrame()
submission["eeg_id"] = df_test["eeg_id"]

for i, col in enumerate(TARGETS):
    submission[col] = eeg_merge[col]

    submission[col] = submission[col].fillna(patient_blend[col])

    submission[col] = submission[col].fillna(global_probs[i])

    submission[col] = 0.85 * submission[col] + 0.15 * global_probs[i]

row_sums = submission[TARGETS].sum(axis=1)
submission[TARGETS] = submission[TARGETS].div(row_sums, axis=0)

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
print("First rows of submission:")
print(submission.head())
