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

0.633040988619371

# 6. Current score

1.04747

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The script failed because the hard‑coded relative paths (`./input/...`) do not exist in the current environment. I added a small helper that searches common data locations (`./data/...`, `./input/...`, `/kaggle/input/...`) and picks the first existing file. The training data is then loaded, global class probabilities are computed, and the test set is read using the same robust path logic. Finally, a submission file is created where every row receives the global class probabilities and is re‑normalized so each row sums to 1, guaranteeing a valid Kaggle submission.'
- What this solution (achieved 1.68479) has done: 'I keep the existing robust file‑location helper and global class baseline, but add a lightweight personalization: compute per‑patient vote distributions from the training set and use them for any test rows whose patient appears in training. This small change gives more informative probabilities than the pure global average, which should lower the KL‑divergence toward the target without altering the overall pipeline or requiring heavy data loading.'
- What this solution (achieved 1.05318) has done: 'The changes add per‑eeg‑id vote distributions (more specific than per‑patient) and a tiny smoothing epsilon before the final row‑wise normalization. This gives each test record a closer‑matching probability when its eeg_id appears in the training data, while still falling back to patient‑level or global averages, and the smoothing helps avoid zero‑probability KL penalties, moving the score toward the target.'
- What this solution (achieved 0.77767) has done: 'I keep the existing robust file‑location logic and hierarchical probability fallback, but add a light blending of the final per‑row probabilities with the global class distribution (90 % learned, 10 % global). This reduces over‑confident predictions for rare classes and should lower the KL‑divergence, moving the score closer to the target while preserving the overall pipeline.'
- What this solution (achieved 0.86229) has done: 'I slightly reduce the weight of the learned (per‑eeg / per‑patient) probabilities and apply a mild temperature scaling (power 0.8) before the final row‑wise normalization. This flattens over‑confident predictions, which typically lowers KL‑divergence when the current score is higher than the target, while keeping the overall hierarchical fallback logic intact.'
- What this solution (achieved 1.04717) has done: 'I make the predictions a bit less specific by lowering the weight given to the learned per‑eeg / per‑patient probabilities and flattening them further with a stronger temperature scaling. This moves the output closer to the global (more uniform) distribution, which should reduce the KL‑divergence and bring the score nearer the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.78096) has done: 'I increase the influence of the learned per‑eeg / per‑patient probabilities (which are more specific than the global average) and apply a milder temperature scaling. This makes the predictions less flattened and closer to the true label distribution, which should lower the KL‑divergence toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.83301) has done: 'I lower the temperature exponent from 0.8 to 0.6, which makes the per‑row probability vectors more uniform (adds stronger flattening). This simple tweak keeps the overall hierarchical fallback, blending, and post‑processing unchanged while reducing over‑confidence, which should lower the KL‑divergence and move the score closer to the target 0.633.'
- What this solution (achieved 1.04747) has done: 'I lower the influence of the learned per‑eeg / per‑patient probabilities and increase the flattening effect so predictions become less over‑confident. Specifically, I reduce BLEND_LEARNED from 0.9 to 0.6 (and raise the global weight to 0.4) and decrease the temperature exponent from 0.6 to 0.5. These minimal adjustments keep the original pipeline intact while moving the KL‑divergence score closer to the target.'

# 9. Code solution

## === cell 0
import os
import pathlib
import pandas as pd


def locate_file(relative_path: str) -> str:
    """Return the first existing path for `relative_path` searched in common locations."""
    candidates = [
        pathlib.Path(relative_path),  # as‑is (e.g., ./input/...)
        pathlib.Path("./data") / relative_path,  # ./data/...
        pathlib.Path("./input") / relative_path,  # ./input/...
        pathlib.Path("/kaggle/input") / relative_path,  # Kaggle default
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    raise FileNotFoundError(f"Could not find {relative_path} in any known location.")


TRAIN_PATH = locate_file("hms-harmful-brain-activity-classification/train.csv")
TEST_PATH = locate_file("hms-harmful-brain-activity-classification/test.csv")

train_df = pd.read_csv(TRAIN_PATH)
TARGETS = train_df.columns[-6:]  # last six columns are the vote targets
print("Train shape:", train_df.shape)
print("Targets:", list(TARGETS))

class_counts = train_df[TARGETS].sum()
class_probs = class_counts / class_counts.sum()
print("Global class probabilities:", class_probs.values)

patient_votes = train_df.groupby("patient_id")[list(TARGETS)].sum()
patient_probs = patient_votes.div(patient_votes.sum(axis=1), axis=0)
print("Computed per‑patient probabilities for", patient_probs.shape[0], "patients")

eeg_votes = train_df.groupby("eeg_id")[list(TARGETS)].sum()
eeg_probs = eeg_votes.div(eeg_votes.sum(axis=1), axis=0)
print("Computed per‑eeg_id probabilities for", eeg_probs.shape[0], "eeg_ids")



## === cell 1
test_df = pd.read_csv(TEST_PATH)
print("Test shape:", test_df.shape)

submission = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})

merged = test_df[["eeg_id", "patient_id"]].merge(
    eeg_probs.reset_index(), how="left", on="eeg_id"
)

patient_probs_reset = patient_probs.reset_index()
for col in TARGETS:
    merged[col] = merged[col].fillna(merged["patient_id"].map(patient_probs[col]))

for col in TARGETS:
    merged[col] = merged[col].fillna(class_probs[col])

BLEND_LEARNED = 0.6  # less specific influence
BLEND_GLOBAL = 1.0 - BLEND_LEARNED
merged[TARGETS] = merged[TARGETS] * BLEND_LEARNED + BLEND_GLOBAL * class_probs.values

TEMPERATURE = 0.5
merged[TARGETS] = merged[TARGETS] ** TEMPERATURE

merged[TARGETS] = merged[TARGETS].div(merged[TARGETS].sum(axis=1), axis=0)

for col in TARGETS:
    submission[col] = merged[col].values

epsilon = 1e-6
submission[TARGETS] = submission[TARGETS].clip(lower=0) + epsilon
submission[TARGETS] = submission[TARGETS].div(submission[TARGETS].sum(axis=1), axis=0)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Submission written to", submission_path)
print("Submission shape:", submission.shape)
print(submission.head())
