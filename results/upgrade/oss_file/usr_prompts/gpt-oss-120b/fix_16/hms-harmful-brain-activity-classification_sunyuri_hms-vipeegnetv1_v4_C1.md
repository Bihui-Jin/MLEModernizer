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

0.4769713322007083

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I adjust the data paths so the script correctly locates the CSV files in the Kaggle environment (using /kaggle/input as the default location and falling back to the original relative path). This resolves the FileNotFoundError and allows the rest of the code to run, producing a valid submission.csv with probabilities that already sum to 1. No changes are made to the model logic or scoring, keeping the core approach intact.'
- What this solution (achieved 1.41937) has done: 'I add a per‑`eeg_id` probability estimate derived from the training votes and fall back to the global baseline when an ID isn’t present in the training set. This small calibration step keeps the original logic but gives each test record a more tailored distribution, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I smooth the per‑`eeg_id` vote‑based probabilities toward the global class baseline instead of using them raw. For each test row I compute a blending weight = votes_for_this_id / (votes_for_this_id + α) (α = 10). The final prediction is weight × per‑id prob + (1‑weight) × global prob, then re‑normalised so each row sums to 1. This modest regularisation keeps the original logic but reduces over‑confident per‑id estimates, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I smooth the per‑`eeg_id` vote counts to avoid zero probabilities, lower the blending regularisation (α = 1 instead of 10) so the model relies more on the per‑id information, and keep the same overall workflow. This modest adjustment should reduce overly‑confident or zero‑probability predictions and move the KL‑divergence closer to the target while preserving the original logic.'
- What this solution (achieved 1.41937) has done: 'I increase the regularisation strength (set ALPHA to 10.0) so the predictions rely more on the global class baseline and less on noisy per‑`eeg_id` estimates, which should lower the KL‑divergence toward the target score. I also renumber the notebook cells to start at 1 as required, keeping the rest of the workflow unchanged.'
- What this solution (achieved 1.41937) has done: 'I lower the regularisation strength (set ALPHA to 1.0) so the model relies more on the per‑`eeg_id` vote information, and I reduce the smoothing on per‑id vote counts from +1 to +0.1 to keep probabilities realistic while avoiding zeros. These minimal adjustments keep the core workflow unchanged but should bring the KL‑divergence closer to the target score.'
- What this solution (achieved 1.41937) has done: 'I lower the regularisation `ALPHA` to 0.1 so the model relies more on the per‑`eeg_id` vote information, which is usually far more predictive than the global baseline. I also reduce the smoothing constant from 0.1 to 0.01 when computing per‑id probabilities to keep the estimates realistic while preserving the original blending logic. These small adjustments keep the core workflow unchanged but should move the KL‑divergence closer to the target lower score.'
- What this solution (achieved 1.41937) has done: 'I increase the regularisation strength by setting `ALPHA = 10.0` so the blended predictions rely more on the stable global class baseline and less on noisy per‑`eeg_id` vote estimates. This small change keeps the overall workflow unchanged while tempering over‑confident per‑id probabilities, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I lower the blending regularisation constant `ALPHA` from 10.0 to 1.0 so the predictions rely more on the per‑`eeg_id` vote information, which is far more predictive than the global baseline. This single change keeps the overall workflow unchanged while expectedly reducing the KL‑divergence toward the target lower score.'
- What this solution (achieved 1.41937) has done: 'The changes increase regularisation and smooth the per‑`eeg_id` vote counts: `ALPHA` is raised from 1.0 to 10.0 so predictions rely more on the stable global class baseline, and the smoothing constant is increased from 0.01 to 0.1 to avoid zero probabilities. These minimal adjustments keep the original workflow intact while making the output distribution less noisy, which should lower the KL‑divergence and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

TF_AVAILABLE = False
print("TensorFlow disabled; using simple baseline predictions.")



## === cell 1
DEFAULT_ROOT = os.path.join("data", "hms-harmful-brain-activity-classification")
KAGGLE_ROOT = os.path.join(
    "/kaggle", "input", "hms-harmful-brain-activity-classification"
)
DATA_ROOT = KAGGLE_ROOT if os.path.isdir(KAGGLE_ROOT) else DEFAULT_ROOT

TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

train_df = pd.read_csv(TRAIN_PATH)

VOTE_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

total_votes_per_class = train_df[VOTE_COLS].sum(axis=0).astype(float)
global_total_votes = total_votes_per_class.sum()
class_probs = total_votes_per_class / global_total_votes
print("Global class probabilities (baseline):")
print(class_probs)

per_id_votes = train_df.groupby("eeg_id")[VOTE_COLS].sum()
per_id_votes_smooth = per_id_votes + 0.1  # raised from 0.01 to 0.1
row_sums_smooth = per_id_votes_smooth.sum(axis=1)
per_id_probs = per_id_votes_smooth.div(row_sums_smooth, axis=0)

per_id_totals = per_id_votes.sum(axis=1).rename("total_votes")
per_id_info = per_id_probs.join(per_id_totals)



## === cell 2
test_df = pd.read_csv(TEST_PATH)

submission = test_df[["eeg_id"]].copy()

submission = submission.merge(
    per_id_info,
    left_on="eeg_id",
    right_index=True,
    how="left",
    suffixes=("", "_per_id"),
)

ALPHA = 10.0  # raised from 1.0 to 10.0
total_votes_series = submission["total_votes"].fillna(0.0)
blend_weight = total_votes_series / (total_votes_series + ALPHA)

for col in VOTE_COLS:
    per_id_vals = submission[col].fillna(0.0)  # per‑id probability (or 0 if unseen)
    blended = blend_weight * per_id_vals + (1.0 - blend_weight) * class_probs[col]
    submission[col] = blended

submission = submission.drop(columns=["total_votes"] + VOTE_COLS, errors="ignore")
for col in VOTE_COLS:
    if col not in submission.columns:
        submission[col] = class_probs[col]

row_sums = submission[VOTE_COLS].sum(axis=1)
submission[VOTE_COLS] = submission[VOTE_COLS].div(row_sums, axis=0)



## === cell 3
OUTPUT_PATH = "submission.csv"
submission.to_csv(OUTPUT_PATH, index=False)
print(f"Submission written to {OUTPUT_PATH}")
