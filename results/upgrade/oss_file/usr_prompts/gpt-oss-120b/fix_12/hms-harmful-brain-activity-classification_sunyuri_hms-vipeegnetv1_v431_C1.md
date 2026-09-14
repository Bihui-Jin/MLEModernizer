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

0.2746703339616783

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I guard the TensorFlow import with a try‑except and, if TensorFlow cannot be loaded, skip all model‑related code. In that fallback path I generate a simple uniform‑probability prediction for each test row, which guarantees a valid `submission.csv` and avoids the protobuf‑related crash. The core logic and data handling remain unchanged; only the failing import and model usage are conditionally bypassed.'
- What this solution (achieved 1.41937) has done: 'Implemented robust path handling and removed the stray markdown cell that caused a `NameError`.  
The script now checks both the original relative `data/...` location and the typical Kaggle `/kaggle/input/...` directory, loading the training and test metadata correctly.  
Fallback logic for missing per‑EEG statistics is kept, and NaNs are safely filled.  
Finally, the submission CSV is written with proper column order, guaranteeing a valid file for Kaggle evaluation.'
- What this solution (achieved 1.41937) has done: 'The fix replaces the invalid `fillna` call with a Series‑based fill, ensuring per‑EEG probabilities are correctly completed with the global class distribution. A small blending of per‑EEG and global probabilities is added before normalisation to make predictions more robust and lower the KL‑divergence score. All other logic and file handling remain unchanged, and the script now always writes a valid `submission.csv` with correctly summed probabilities.'
- What this solution (achieved 1.41937) has done: 'The update adds an adaptive blending weight that depends on how many annotator votes each `eeg_id` has: EEGs with many votes keep the strong per‑EEG signal, while those with few votes rely more on the stable global class distribution. This simple calibration reduces noisy per‑EEG predictions and is expected to lower the KL‑divergence score, moving it closer to the target while keeping the original workflow unchanged.'
- What this solution (achieved 1.41937) has done: 'The update reduces reliance on noisy per‑EEG statistics by limiting their influence to at most 30 % of the final prediction. This smoother blending pushes the model toward the stable global class distribution, which is expected to lower the KL‑divergence and move the score closer to the target while keeping the original workflow intact.'
- What this solution (achieved 1.41937) has done: 'I smooth the per‑EEG vote counts to avoid zero probabilities, increase the influence of these per‑EEG statistics, and make the blending weight reach its maximum faster (with a lower vote‑count threshold). These minimal adjustments keep the original workflow intact while providing stronger, better‑calibrated predictions that should move the KL‑divergence score closer to the target.'
- What this solution (achieved 1.41937) has done: 'I lower the influence of the noisy per‑EEG statistics by setting the blending weight to 0, so the prediction relies solely on the stable global class distribution computed from the training votes. This single change keeps the overall workflow intact, guarantees the probabilities sum to 1, and is expected to reduce the KL‑divergence, moving the score closer to the target (lower is better).'
- What this solution (achieved 1.41937) has done: 'I enable a modest per‑EEG contribution by setting a non‑zero blend weight (0.4) and keeping the same vote‑based scaling logic. This uses the already‑computed per‑EEG probabilities when enough votes are present, which generally yields predictions closer to the true distribution and should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I simplify the prediction step to rely solely on the stable global class distribution, removing the per‑EEG blending that was worsening the KL‑divergence. By setting the blend weight to 0 the model outputs the same globally‑computed probabilities for every test record, which is a more reliable baseline and should move the score down toward the target while keeping the overall workflow unchanged.'

# 9. Code solution

## === cell 0
import os, pandas as pd, numpy as np, gc, time, signal

possible_paths = [
    os.path.join("data", "hms-harmful-brain-activity-classification"),
    os.path.join("/kaggle", "input", "hms-harmful-brain-activity-classification"),
]
for p in possible_paths:
    if os.path.isdir(p):
        LOAD_DATA_FROM = p
        break
else:
    raise FileNotFoundError(
        "Cannot locate competition data. Checked paths: " + ", ".join(possible_paths)
    )

TF_AVAILABLE = False  # TensorFlow is not available in this environment
NEEDTRAIN = False  # No training will be performed
READ_EEG_FILES = False  # Skip heavy EEG file loading
filter_range = None  # No filtering applied

train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
df = pd.read_csv(train_path)
TARGETS = df.columns[-6:]  # ['seizure_vote', ..., 'other_vote']
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

_class_counts = df[TARGETS].sum()
GLOBAL_PROBS = (_class_counts / _class_counts.sum()).values.astype(np.float32)

_per_eeg_counts = df.groupby("eeg_id")[TARGETS].sum()
_per_eeg_counts_smooth = _per_eeg_counts + 1.0
PER_EEG_PROBS = _per_eeg_counts_smooth.div(_per_eeg_counts_smooth.sum(axis=1), axis=0)

_global_series = pd.Series(GLOBAL_PROBS, index=TARGETS)
PER_EEG_PROBS = PER_EEG_PROBS.fillna(_global_series)

PER_EEG_TOTALS = _per_eeg_counts.sum(axis=1)



## === cell 1
if __name__ == "__main__":
    test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
    test = pd.read_csv(test_path)
    test["sign_id"] = test.index.values
    print("Test shape", test.shape)

    pred_array = np.tile(GLOBAL_PROBS, (len(test), 1))

    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
    sub[TARGETS] = pred_array
    sub[TARGETS] = sub[TARGETS].div(sub[TARGETS].sum(axis=1), axis=0)

    submission_path = "submission.csv"
    sub.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")
    print("Submission shape", sub.shape)
    print(sub.head())
