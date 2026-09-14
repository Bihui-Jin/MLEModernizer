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

0.5298586086507827

# 6. Current score

0.81597

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.48867) has done: 'The fix removes TensorFlow imports and any related code when training is not required, preventing the protobuf `GetPrototype` error. All TensorFlow‑dependent sections (GPU setup, data generator, model building) are now wrapped in a `if NEEDTRAIN:` block, so they are skipped when `NEEDTRAIN=False`. The inference part simply uses the class‑probability means, normalizes them to guarantee each row sums to 1, and writes a correct `submission.csv`.'
- What this solution (achieved 1.67064) has done: 'I replace the uniform class‑mean prediction with a patient‑level conditional prediction: for each test row I look up the mean vote distribution of the same patient in the training data (if present) and use it; otherwise I fall back to the overall class means. This uses only existing aggregates, keeps the same model‑free pipeline, and should lower the KL divergence toward the target score while still guaranteeing each row sums to 1.'
- What this solution (achieved 1.69138) has done: 'I replace the simple un‑weighted patient‑level mean with a **vote‑weighted patient distribution** (summing raw annotator votes per patient and normalising). This uses more information from the training data, keeping the same overall pipeline, and is expected to lower the KL‑divergence toward the target while still guaranteeing each row sums to 1.'
- What this solution (achieved 1.69138) has done: 'We keep the overall pipeline identical but improve the prediction hierarchy: first try to use the exact eeg_id distribution from the training data (when the same eeg_id appears in test), then fall back to the patient‑level vote‑weighted distribution, and finally to the overall class means. This adds more specific information without altering any model logic and should lower the KL‑divergence, moving the score toward the target.'
- What this solution (achieved 1.68479) has done: 'I replace the uniform class‑mean fallback with a fallback based on the overall vote distribution across the whole training set (i.e., the true class priors weighted by annotator votes). This keeps the same hierarchy – exact eeg_id distribution first, then patient‑level weighted votes – but uses a more informative prior when no specific information is available, which should lower the KL‑divergence and move the score toward the target.'
- What this solution (achieved 1.68479) has done: 'I add a spectrogram‑level vote distribution as an extra fallback tier (eeg_id → patient → spectrogram → overall). This uses only existing aggregated statistics, keeps the same inference‑only pipeline, and improves calibration without altering any core modeling logic, helping to lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.68479) has done: 'I replace the simple hierarchical fill‑na with a slight averaging of patient‑ and spectrogram‑level distributions when the specific eeg_id distribution is missing. This uses the same aggregated statistics but combines two informative priors instead of falling back to just one, which should bring the KL‑divergence closer to the target without altering the core pipeline.'
- What this solution (achieved 1.68479) has done: 'I keep the original hierarchical logic but improve the combined patient‑spectrogram fallback by weighting each distribution with the total number of annotator votes it represents. This simple weighting uses only existing aggregate statistics, leaves the overall pipeline unchanged, and is expected to bring the KL‑divergence down toward the target value.'
- What this solution (achieved 0.77767) has done: 'I keep the overall hierarchical logic but add a light smoothing step that blends each row’s predicted class distribution with the overall class prior (10 % prior, 90 % hierarchical estimate). This reduces overly confident predictions on sparse rows, which should lower the KL‑divergence and move the score closer to the target while preserving the existing pipeline.'
- What this solution (achieved 0.81597) has done: 'I keep the whole pipeline unchanged and only adjust the smoothing that blends the hierarchical predictions with the overall class prior. By increasing the weight of the overall prior (from 10 % to 30 %), the predictions become less over‑confident on sparse rows, which should lower the KL‑divergence and move the score closer to the target while preserving all core logic.'
- What this solution (achieved 0.96626) has done: 'I keep the original inference‑only pipeline intact and only adjust the blending of the hierarchical predictions with the overall class prior. By increasing the weight of the overall prior from 30 % to 60 % (i.e., using 0.4 * hierarchical + 0.6 * overall), the predictions become less over‑confident on sparse rows, which should lower the KL‑divergence and move the score closer to the target while preserving all core logic.'
- What this solution (achieved 0.81597) has done: 'I reduce the weight of the overall class‑prior when blending with the hierarchical predictions (currently 0.4 × hierarchical + 0.6 × overall). Using more of the EEG‑/patient‑/spectrogram‑specific estimates (e.g., 0.7 × hierarchical + 0.3 × overall) should lower the KL‑divergence and move the score closer to the target while keeping the core logic unchanged. The only change is the blending coefficients and an explanatory comment.'

# 9. Code solution

## === cell 0
PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False  # No training/inference with TF needed
LOAD_MODELS_FROM = "models202402211"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # s
SFREQ = 200

HIGH = 128  # 128
LENGTH = 256  # 256

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

import os
import pandas as pd, numpy as np
import matplotlib
import matplotlib.pyplot as plt

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
    test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
else:  # kaggle
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    )

TARGETS = df.columns[-6:]  # seizure, lpd, gpd, lrda, grda, other
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))

train_agg = df.groupby("eeg_id")[
    ["spectrogram_id", "spectrogram_label_offset_seconds", "eeg_label_offset_seconds"]
].agg(
    {
        "spectrogram_id": "first",
        "spectrogram_label_offset_seconds": "min",
        "eeg_label_offset_seconds": "median",
    }
)
train_agg.columns = ["spec_id", "min", "eeg_median"]

max_offsets = (
    df.groupby("eeg_id")["spectrogram_label_offset_seconds"].max().rename("max")
)
train_agg = train_agg.join(max_offsets)

train_agg["patient_id"] = df.groupby("eeg_id")["patient_id"].first()

vote_sums = df.groupby("eeg_id")[TARGETS].sum()
y_vals = vote_sums.values
y_vals = y_vals / y_vals.sum(axis=1, keepdims=True)
train_agg[TARGETS] = y_vals

overall_votes = df[TARGETS].sum()
overall_probs = (overall_votes / overall_votes.sum()).values
print(
    "Overall class prior (vote‑weighted):",
    overall_probs,
    "Sum =",
    overall_probs.sum(),
)

patient_raw_votes = df.groupby("patient_id")[TARGETS].sum()
patient_probs = patient_raw_votes.div(
    patient_raw_votes.sum(axis=1), axis=0
).reset_index()  # columns: patient_id + TARGETS
patient_total_votes = patient_raw_votes.sum(axis=1).rename("patient_total")  # Series

spectro_raw_votes = df.groupby("spectrogram_id")[TARGETS].sum()
spectro_probs = spectro_raw_votes.div(
    spectro_raw_votes.sum(axis=1), axis=0
).reset_index()  # columns: spectrogram_id + TARGETS
spectro_total_votes = spectro_raw_votes.sum(axis=1).rename("spectro_total")  # Series

test_pred = test[["eeg_id", "patient_id", "spectrogram_id"]].copy()

eeg_probs = train_agg[TARGETS].reset_index()  # columns: eeg_id + TARGETS
test_pred = test_pred.merge(eeg_probs, on="eeg_id", how="left", suffixes=("", "_eeg"))

test_pred = test_pred.merge(
    patient_probs, on="patient_id", how="left", suffixes=("", "_patient")
)
test_pred = test_pred.merge(
    patient_total_votes, left_on="patient_id", right_index=True, how="left"
)

test_pred = test_pred.merge(
    spectro_probs, on="spectrogram_id", how="left", suffixes=("", "_spectro")
)
test_pred = test_pred.merge(
    spectro_total_votes, left_on="spectrogram_id", right_index=True, how="left"
)

for i, col in enumerate(TARGETS):
    eeg_arr = test_pred[col].values
    patient_arr = test_pred[col + "_patient"].values
    spectro_arr = test_pred[col + "_spectro"].values

    patient_total_arr = test_pred["patient_total"].values
    spectro_total_arr = test_pred["spectro_total"].values

    have_eeg = ~np.isnan(eeg_arr)

    have_both = (~np.isnan(patient_arr)) & (~np.isnan(spectro_arr))

    have_patient_only = (~np.isnan(patient_arr)) & np.isnan(spectro_arr)

    have_spectro_only = np.isnan(patient_arr) & (~np.isnan(spectro_arr))

    combined = np.full_like(eeg_arr, overall_probs[i], dtype=float)

    combined[have_patient_only] = patient_arr[have_patient_only]
    combined[have_spectro_only] = spectro_arr[have_spectro_only]

    denom = patient_total_arr + spectro_total_arr
    weighted_vals = (
        patient_arr * patient_total_arr + spectro_arr * spectro_total_arr
    ) / denom
    combined[have_both] = np.where(
        denom[have_both] == 0, overall_probs[i], weighted_vals[have_both]
    )

    combined[have_eeg] = eeg_arr[have_eeg]

    test_pred[col] = combined

drop_cols = [
    c for c in test_pred.columns if c.endswith("_patient") or c.endswith("_spectro")
]
test_pred = test_pred.drop(columns=drop_cols)

pred_vals = test_pred[TARGETS].values
pred_vals = 0.7 * pred_vals + 0.3 * overall_probs.reshape(1, -1)

row_sums = pred_vals.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
pred_vals = pred_vals / row_sums

submission = pd.DataFrame({"eeg_id": test["eeg_id"].values})
submission[TARGETS] = pred_vals
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission written to {submission_path}")
print("Submission shape:", submission.shape)
print("Row sums (first 5 rows, should be 1.0):")
print(submission.iloc[:5, -6:].sum(axis=1))
