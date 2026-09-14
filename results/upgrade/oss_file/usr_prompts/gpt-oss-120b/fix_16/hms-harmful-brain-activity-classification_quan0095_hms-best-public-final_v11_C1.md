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

0.2839106067584163

# 6. Current score

0.78698

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I replace the GPU‑dependent model loading and inference with a simple CPU‑only baseline: compute overall class vote proportions from the training set and assign these same probabilities to every test record, ensuring the probabilities sum to 1 and a valid `submission.csv` is written.'
- What this solution (achieved 1.57587) has done: 'I replace the uniform‑global baseline with a simple per‑patient prior: for each patient in the training data I compute the normalized vote distribution and use it for every test row belonging to that patient. Rows whose patient_id never appears in training fall back to the original global class probabilities. This adds only a lightweight aggregation step, keeps the overall structure unchanged, and should lower the KL‑divergence toward the target score while still producing a valid CSV where each row sums to 1.'
- What this solution (achieved 0.81005) has done: 'I keep the overall strategy—using per‑eeg, per‑patient and global vote distributions—but tighten the global regularisation (reduce `smooth_weight` from 0.1 to 0.02) to let the more specific probabilities influence the predictions more, which should lower the KL‑divergence toward the target. I also drop any auxiliary columns before saving so the submission matches the required format exactly.'
- What this solution (achieved 1.68479) has done: 'I lower the regularisation that blends the specific (eeg‑ or patient‑based) probabilities with the global priors, because the current 2 % smoothing pulls many predictions toward the global distribution and inflates the KL‑divergence. Setting `smooth_weight` to 0 removes this bias while keeping the original hierarchy (eeg → patient → global) and the final row‑normalisation, which should move the score nearer the target without altering the core logic.'
- What this solution (achieved 0.77767) has done: 'I increase the smoothing blend with the global class distribution (set `smooth_weight` to 0.1) so predictions are less extreme for rare EEG‑ or patient‑specific groups, which should lower the KL‑divergence toward the target score while keeping the original hierarchy and row‑normalisation unchanged. The rest of the pipeline remains identical, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.85517) has done: 'I keep the original hierarchy (global, patient, EEG priors) and only increase the blending weight toward the global class distribution. By raising `smooth_weight` from 0.1 to 0.4 each row’s probabilities are softened, reducing over‑confident predictions that cause high KL‑divergence. This small change preserves all core logic while moving the score closer to the target lower value.'
- What this solution (achieved 0.78004) has done: 'I lower the smoothing weight from 0.4 to 0.15 so predictions stay closer to the patient/eeg priors (which were already improving the score) while still benefiting from some global regularisation. Additionally, I add a tiny constant to every class probability before normalising rows; this removes near‑zero values that hurt KL‑divergence without changing the overall model logic. The rest of the pipeline and the CSV output remain unchanged.'
- What this solution (achieved 1.25479) has done: 'I increase the global‑mixing weight and add a slightly larger epsilon before normalising so predictions stay away from zero. Using a higher `smooth_weight` (0.9) moves each row much closer to the overall class distribution, which has been shown to lower KL‑divergence, and the larger `epsilon_const` (1e‑3) further prevents near‑zero probabilities that hurt the score. These tweaks keep the original hierarchy and all I/O unchanged while nudging the metric toward the target lower value.'
- What this solution (achieved 0.78698) has done: 'I lower the smoothing weight back toward the patient/eeg‑specific priors (set smooth_weight to 0.05) and reduce the added epsilon constant to a tiny value (1e‑8). This makes each row rely more on its learned per‑EEG or per‑patient distribution rather than being pulled toward the global average, which empirically moves the KL‑divergence lower and thus brings the score closer to the target. No other logic is altered, so the script still produces a valid `submission.csv` with rows summing to 1.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
DEBUG = False




## === cell 1
train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
train_df = pd.read_csv(train_path)

vote_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

epsilon = 1e-8

class_sums = train_df[vote_cols].sum().astype(float)
global_class_probs = class_sums / class_sums.sum()
if DEBUG:
    print("Global class probabilities:", global_class_probs.values)

patient_group = train_df.groupby("patient_id")[vote_cols].sum()
patient_totals = patient_group.sum(axis=1).replace(0, epsilon)
patient_probs = patient_group.div(patient_totals, axis=0)

eeg_group = train_df.groupby("eeg_id")[vote_cols].sum()
eeg_totals = eeg_group.sum(axis=1).replace(0, epsilon)
eeg_probs = eeg_group.div(eeg_totals, axis=0)

if DEBUG:
    print("Sample patient probabilities:", patient_probs.head())
    print("Sample EEG probabilities:", eeg_probs.head())




## === cell 2
test_path = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
test_df = pd.read_csv(test_path)

submission = test_df[["eeg_id", "patient_id"]].copy()

submission = submission.merge(
    eeg_probs.reset_index(),
    on="eeg_id",
    how="left",
    suffixes=("", "_eeg"),
)

submission = submission.merge(
    patient_probs.reset_index(),
    on="patient_id",
    how="left",
    suffixes=("", "_patient"),
)

for col in vote_cols:
    if col not in submission.columns:
        submission[col] = np.nan
    submission[col] = submission[col].fillna(submission.get(f"{col}_patient"))
    submission[col] = submission[col].fillna(global_class_probs[col])

smooth_weight = 0.05  # previously 0.9
if smooth_weight > 0:
    for col in vote_cols:
        submission[col] = (1 - smooth_weight) * submission[
            col
        ] + smooth_weight * global_class_probs[col]

epsilon_const = 1e-8  # previously 1e-3
submission[vote_cols] = submission[vote_cols] + epsilon_const

row_sums = submission[vote_cols].sum(axis=1).replace(0, epsilon)
submission[vote_cols] = submission[vote_cols].div(row_sums, axis=0)

submission = submission[["eeg_id"] + vote_cols]

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}, shape: {submission.shape}")




## === cell 3
print(submission.head())
print("Row sums (should be 1.0):", submission[vote_cols].sum(axis=1).head().values)
