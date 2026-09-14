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

0.468439133027326

# 6. Current score

1.20739

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I wrap the EfficientNet import in a safe fallback to avoid the protobuf error, and replace the inference section with a simple uniform‑probability prediction that guarantees a valid `.csv` submission (rows sum to 1). This fixes the runtime crash and ensures the script finishes, while keeping the overall pipeline structure unchanged.'
- What this solution (achieved 1.41937) has done: 'The fix removes the unnecessary EfficientNet import that triggers a protobuf error and replaces the uniform prediction with a simple class‑frequency baseline derived from the training vote counts. Using these empirical class probabilities provides a more realistic distribution, reducing the KL‑divergence toward the target score while keeping the overall pipeline unchanged. The script now safely reads the data, computes the baseline probabilities, creates the submission, and writes a valid CSV.'
- What this solution (achieved 1.40995) has done: 'The fix removes the problematic TensorFlow import (which caused the protobuf error) and replaces the class‑frequency baseline with a simple uniform probability vector, which yields a lower KL‑divergence on the validation split. This keeps the original pipeline intact while guaranteeing a valid CSV submission whose rows sum to 1.'
- What this solution (achieved 1.41937) has done: 'The fix corrects the faulty import statement (`import pandas as pd, np as np`) which caused a `ModuleNotFoundError` and prevented all subsequent variables (like `NEEDTRAIN`, `train_prob`, etc.) from being defined. Changing it to `import pandas as pd, numpy as np` restores the proper environment, allowing the script to load data, compute baseline class probabilities, generate normalized predictions for the test set, and write a valid `submission.csv` file.'
- What this solution (achieved 1.68479) has done: 'The fix removes the problematic TensorFlow import that caused a protobuf error and adds a fallback that uses patient‑level class probabilities when an `eeg_id` is not present in the training data. This keeps the original baseline logic but provides more informative predictions for unseen `eeg_id`s, helping lower the KL‑divergence toward the target while ensuring a valid `.csv` submission is written.'
- What this solution (achieved 1.68479) has done: 'I keep the overall baseline logic but improve the probability estimates:  
1. Apply Laplace smoothing when computing the global class distribution.  
2. When a test `eeg_id` is known, blend its per‑eeg probabilities with the corresponding patient‑level probabilities (60 % / 40 %).  
3. For unseen `eeg_id`s fall back to the patient‑level estimate, and finally to the smoothed global distribution.  
These small, targeted tweaks should make the predictions more realistic and move the KL‑divergence score closer to the target while preserving the original pipeline.'
- What this solution (achieved 1.1548) has done: 'I keep the overall pipeline but improve the baseline probabilities: add a tiny Laplace‑style smoothing to the per‑eeg and per‑patient distributions and increase the weight of the reliable per‑eeg estimate (80 % / 20 %). This reduces overly‑confident zero‑probabilities that inflate KL‑divergence and should move the score closer to the target while preserving the original logic.'
- What this solution (achieved 1.1548) has done: 'I simplify the prediction logic to rely on the more stable patient‑level class probabilities instead of the per‑eeg blend, which tends to be over‑confident and inflates the KL‑divergence. By using only the patient distribution (with the existing Laplace smoothing) and falling back to the global class probabilities when a patient has no data, we expect the score to move closer to the target lower‑KL value while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.80052) has done: 'I add a lightweight per‑eeg probability component and blend it with the existing patient‑level and global probabilities (weights 0.6 / 0.3 / 0.1). This uses the already computed `train_prob` distribution, adds just a few lines, and keeps the overall pipeline unchanged while giving more specific predictions that should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.1548) has done: 'I replace the per‑eeg + patient + global blending with a simpler patient‑level probability that falls back to the globally smoothed distribution when a patient has no training data. This keeps the same data sources but removes the noisy per‑eeg component, which in earlier experiments produced a lower KL‑divergence and should move the score closer to the target (lower is better). The rest of the pipeline and file‑writing logic remain unchanged.'
- What this solution (achieved 0.98616) has done: 'I add a lightweight probability smoothing step after the patient‑level blending: raising each predicted probability to a power < 1 (α = 0.7) and renormalising the rows. This makes the distributions slightly less confident, which typically reduces KL‑divergence when the model is over‑confident, moving the score closer to the target while keeping the original pipeline unchanged.'
- What this solution (achieved 1.20739) has done: 'I lower the confidence of the predictions by flattening the blended probabilities more strongly (using a smaller exponent) and by mixing the patient‑level estimate with the global class distribution before normalisation. This should bring the KL‑divergence closer to the target (lower is better) while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import os, sys
import pandas as pd, numpy as np

PLATFORM = "kaggle"  # or 'local'
NEEDTRAIN = False
LOAD_MODELS_FROM = "models20240220"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # seconds
SFREQ = 100
HIGH = 64
LENGTH = 256
filter_range = [0.5, 40]

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
else:
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
TARGETS = df.columns[-6:]  # last six columns are the vote counts
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

n_classes = len(TARGETS)

global_counts = df[TARGETS].sum() + 1
global_prob = global_counts / global_counts.sum()
print("Global class probabilities (smoothed):", global_prob.values)

train_prob = df.groupby("eeg_id")[TARGETS].sum()
train_prob += 1e-6  # Laplace‑style smoothing
row_sums = train_prob.sum(axis=1)
train_prob = train_prob.div(row_sums.replace(0, np.finfo(float).eps), axis=0)

patient_prob = df.groupby("patient_id")[TARGETS].sum()
patient_prob += 1e-6
patient_row_sums = patient_prob.sum(axis=1)
patient_prob = patient_prob.div(
    patient_row_sums.replace(0, np.finfo(float).eps), axis=0
)



## === cell 1
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
    print("Test shape", test.shape)

    patient_merge = test[["patient_id"]].merge(
        patient_prob, left_on="patient_id", right_index=True, how="left"
    )
    patient_vals = patient_merge[TARGETS].values.astype(float)
    patient_vals_clean = np.where(np.isnan(patient_vals), 0.0, patient_vals)

    row_is_zero = patient_vals_clean.sum(axis=1) == 0
    patient_vals_clean[row_is_zero] = global_prob.values

    blended = 0.5 * patient_vals_clean + 0.5 * global_prob.values

    blended = blended / blended.sum(axis=1, keepdims=True)

    alpha = 0.3
    blended = np.power(blended, alpha)
    blended = blended / blended.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
    for i, col in enumerate(TARGETS):
        sub[col] = blended[:, i]

    submission_path = "submission.csv"
    sub.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")
    print("First few rows of submission:")
    print(sub.head())
