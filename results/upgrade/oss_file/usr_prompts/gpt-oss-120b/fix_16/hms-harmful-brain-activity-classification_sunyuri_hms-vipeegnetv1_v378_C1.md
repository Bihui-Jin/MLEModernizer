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

0.3323973539334477

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'We set the pipeline to skip all heavy parquet loading and model training, and directly create a valid submission by assigning uniform probabilities that sum to 1 for every test row. This eliminates the protobuf MessageFactory error while still producing a correctly‑formatted `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'The fix removes the failing TensorFlow import, replaces the uniform baseline with class‑frequency probabilities derived from the training votes (a lightweight improvement that lowers the KL score), and keeps the rest of the pipeline unchanged so a valid `submission.csv` is written.'
- What this solution (achieved 1.68479) has done: 'Implemented safe TensorFlow imports and lightweight placeholder classes to eliminate the NameError. Added per‑patient vote‑based probability generation: compute class‑frequency distributions per patient from the training metadata and use them for test rows when the patient appears in the training set, falling back to global class frequencies otherwise. Rows are re‑normalized to ensure each probability vector sums to 1 before writing the required `submission.csv`. This fixes the runtime crash and provides a more informed baseline, moving the KL score closer to the target.'
- What this solution (achieved 1.68479) has done: 'Implemented robust TensorFlow import handling by removing inheritance from TensorFlow‑specific classes, avoiding the protobuf‑related crash. Added per‑eeg_id probability distributions (computed from training votes) and enhanced the prediction logic: test rows first try to use their own `eeg_id` distribution, then fall back to per‑patient probabilities, and finally to global class frequencies. All probabilities are re‑normalized to sum to 1 per row before writing the required `submission.csv`. This fixes the runtime error and provides a more informed baseline to move the KL score toward the target.'
- What this solution (achieved 1.05318) has done: 'Implemented a tiny epsilon smoothing step for the predicted class probabilities to eliminate zero values before normalization. This change prevents infinite KL penalties for unseen classes, stabilizes the score, and keeps the original logic intact. Added comments explaining the fix and performed the smoothing right before the final normalization step.'
- What this solution (achieved 1.05318) has done: 'Implemented robust data path handling so the script reliably finds the training and test CSV files in typical Kaggle environments. Added fallback directory checks and a clear error if none are found. This fixes the FileNotFoundError, allows the probability tables to be built, and ensures a valid `submission.csv` is written with properly normalized probabilities.'
- What this solution (achieved 1.07509) has done: 'We keep the existing data loading and fallback logic, but add a mild “temperature” smoothing step that flattens overly‑confident probability vectors before the tiny epsilon addition and final renormalisation. Raising the probabilities to a power < 1 (here 0.5) moves predictions toward a more moderate distribution, which is expected to lower the KL divergence and bring the score closer to the target without altering the core modelling approach.'
- What this solution (achieved 0.98646) has done: 'I slightly increase the smoothing (lower temperature α) to make predictions less confident and then blend each row with the overall class‑frequency distribution. This modest adjustment keeps the original per‑eeg / per‑patient logic, adds only a tiny averaging step, and re‑normalises so the rows still sum to 1, which should reduce the KL divergence toward the target score.'
- What this solution (achieved 1.26991) has done: 'The changes lower the temperature (making per‑eeg/patient probabilities flatter) and reduce the weight of these specialized predictions, blending more heavily toward the global class‑frequency distribution, which should move the KL score closer to the target lower value while keeping the same pipeline and valid CSV output.'
- What this solution (achieved 1.22091) has done: 'Implemented robust path resolution to locate training and test CSV files in typical Kaggle input directories and added a fallback to the filename itself. Adjusted the temperature scaling and blending weight to give a stronger bias toward the global class‑frequency distribution, which helps lower the KL divergence while preserving the original prediction pipeline.'
- What this solution (achieved 1.41937) has done: 'I lower the temperature scaling to flatten the probability vectors more aggressively and set the blending weight to 0 so the submission relies entirely on the global class‑frequency distribution. This reduces over‑confident per‑eeg or per‑patient predictions, which should bring the KL score closer to the target while preserving the original pipeline logic.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np


def resolve_path(*parts):
    """
    Try several common locations for Kaggle datasets:
    1) The exact relative path given.
    2) Inside /kaggle/input (the standard mount point).
    3) Directly under the current directory.
    Returns the first existing path, otherwise just the basename.
    """
    candidate_paths = [os.path.join(*parts)]
    if parts and parts[0] == "data":
        candidate_paths.append(os.path.join("/kaggle/input", *parts[1:]))
    else:
        candidate_paths.append(os.path.join("/kaggle/input", *parts))
    candidate_paths.append(os.path.basename(parts[-1]))

    for p in candidate_paths:
        if os.path.exists(p):
            return p
    return candidate_paths[-1]


TRAIN_CSV = resolve_path(
    "data", "hms-harmful-brain-activity-classification", "train.csv"
)
TEST_CSV = resolve_path("data", "hms-harmful-brain-activity-classification", "test.csv")

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)

global_votes = train_df[TARGETS].sum()
global_prob = (global_votes / global_votes.sum()).to_dict()

eeg_votes = train_df.groupby("eeg_id")[TARGETS].sum()
eeg_prob = (eeg_votes.T / eeg_votes.sum(axis=1)).T
eeg_prob = eeg_prob.fillna(0)  # handle any eeg_id with no votes

patient_votes = train_df.groupby("patient_id")[TARGETS].sum()
patient_prob = (patient_votes.T / patient_votes.sum(axis=1)).T
patient_prob = patient_prob.fillna(0)



## === cell 2
if __name__ == "__main__":
    test = pd.read_csv(TEST_CSV)

    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})

    merged = test.merge(eeg_prob, on="eeg_id", how="left", suffixes=("", "_eeg"))
    merged = merged.merge(
        patient_prob, on="patient_id", how="left", suffixes=("", "_pat")
    )

    for col in TARGETS:
        sub[col] = merged[col]  # per‑eeg (may be NaN)
        sub[col] = sub[col].fillna(merged[f"{col}_pat"])  # per‑patient fallback
        sub[col] = sub[col].fillna(global_prob[col])  # global fallback

    temperature_alpha = 0.1  # lower than before to make predictions less confident
    sub[TARGETS] = np.power(sub[TARGETS].values, temperature_alpha)

    epsilon = 1e-6
    sub[TARGETS] = sub[TARGETS].fillna(0) + epsilon
    prob_sum = sub[TARGETS].sum(axis=1)
    sub[TARGETS] = sub[TARGETS].div(prob_sum, axis=0)

    blend_weight = 0.0  # 0 = pure global, 1 = pure derived probs
    global_arr = np.array([global_prob[col] for col in TARGETS])
    sub[TARGETS] = blend_weight * sub[TARGETS].values + (1 - blend_weight) * global_arr

    prob_sum = sub[TARGETS].sum(axis=1)
    sub[TARGETS] = sub[TARGETS].div(prob_sum, axis=0)

    sub.to_csv("submission.csv", index=False)
    print("Submission shape:", sub.shape)
    print("Saved to submission.csv")
