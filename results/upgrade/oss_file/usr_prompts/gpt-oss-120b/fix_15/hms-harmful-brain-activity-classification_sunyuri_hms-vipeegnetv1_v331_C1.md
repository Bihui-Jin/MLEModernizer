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

0.2853635641079756

# 6. Current score

0.91113

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.68479) has done: 'The fix forces the script to skip TensorFlow‑dependent training and directly use the fallback baseline that creates a valid CSV submission, eliminating the protobuf import error and guaranteeing an output file.'
- What this solution (achieved 1.68479) has done: 'I remove the TensorFlow import (which raises a protobuf error) and replace it with a pure‑pandas fallback that also leverages per‑eeg‑id vote fractions in addition to the existing per‑patient fractions. The script now picks a prediction hierarchy : use the exact eeg_id if seen in training, otherwise the patient‑level distribution, otherwise the overall global distribution. This fixes the runtime crash and gives a more informed baseline, moving the KL score closer to the target while still writing a valid `submission.csv`.'
- What this solution (achieved 1.68479) has done: 'Implemented a lightweight improvement to the baseline hierarchy: added spectrogram‑level vote fractions and use them when neither the exact `eeg_id` nor the patient appears in the training data. This provides a more informed prior than the global distribution, helping lower the KL divergence while keeping the original logic intact.'
- What this solution (achieved 1.68479) has done: 'The fix removes the forced `sys.exit(0)` (which caused a runtime error), adds a small epsilon to avoid zero‑probability issues, and improves the baseline hierarchy by averaging patient‑ and spectrogram‑level distributions when both are available. This keeps the original logic while making the script run cleanly and produce a valid `submission.csv` that is better calibrated, moving the KL score toward the target.'
- What this solution (achieved 1.68479) has done: 'I replace the simple un‑weighted averaging of patient‑ and spectrogram‑level vote fractions with a confidence‑weighted blend that uses the total number of votes each source contributes. By weighting more‑populated groups higher, the predictions become better calibrated, which should lower the KL‑divergence and move the score closer to the target while keeping the original hierarchy and logic intact.'
- What this solution (achieved 0.77767) has done: 'I adjust the blending logic so that the global distribution is used only as a small regularising prior instead of dominating the weight sum. This keeps the original hierarchy but gives more influence to patient‑ and spectrogram‑level fractions, which are more informative and should lower the KL divergence toward the target. The rest of the script remains unchanged, and the submission file is still written correctly.'
- What this solution (achieved 0.78698) has done: 'I lower the weight of the global prior (from 0.1 to 0.05) so that predictions rely more on the patient‑, spectrogram‑, or exact‑eeg distributions, which are more informative and should reduce the KL‑divergence toward the target. The blending logic is updated accordingly while keeping the original hierarchy and normalization unchanged.'
- What this solution (achieved 0.7981) has done: 'I add a tiny Laplace‑smoothing when building the vote fractions so that no class gets a zero probability, and I lower the global prior weight from 0.05 to 0.03 to let the patient‑ and spectrogram‑level information dominate the prediction. These small adjustments keep the original hierarchy intact while improving calibration, which should reduce the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.78065) has done: 'I slightly increase the global prior weight (from 0.03 to 0.07) and recompute the complementary specific weight. This provides a modest regularising influence from the overall class distribution, which should improve calibration and reduce the KL‑divergence, moving the score closer to the target while keeping the original hierarchy unchanged.'
- What this solution (achieved 0.94238) has done: 'I replace the fixed GLOBAL_WEIGHT with an adaptive blend that gives more global regularisation when the available patient / spectrogram vote counts are low, and lets the specific distribution dominate only for well‑supported groups. This keeps the original hierarchy while improving calibration, which should lower the KL‑divergence toward the target.'
- What this solution (achieved 0.82922) has done: 'The script keep the same hierarchical prediction logic but lower the `GLOBAL_REG` constant from 500 to 100, giving more weight to patient‑ and spectrogram‑specific vote distributions. This modest adjustment is expected to produce predictions that are better calibrated to the training data, thereby reducing the KL‑divergence score and moving it closer to the target while preserving all core functionality and the required CSV output.'
- What this solution (achieved 0.81714) has done: 'The changes lower the smoothing to 1e‑6 and greatly reduce `GLOBAL_REG` to 0.5, which makes the predictions rely far more on patient‑ and spectrogram‑level vote fractions (the most informative signals) while still keeping a tiny global prior for stability. This adjustment is expected to move the KL‑divergence score much closer to the target 0.285 without altering the overall hierarchy or core logic.'
- What this solution (achieved 0.91113) has done: 'The adjustment reduces the global regularisation weight (`GLOBAL_REG`) from 0.5 to 0.02, letting patient‑ and spectrogram‑specific vote fractions dominate the prediction when available. This stronger reliance on more informative local distributions should lower the KL‑divergence toward the target while preserving the original hierarchy and output format.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri
email: syuri@tju.edu.cn
"""

NEEDTRAIN = False  # Disable training to avoid TensorFlow errors
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

import os
import warnings
import pandas as pd
import numpy as np

os.environ["KERAS_BACKEND"] = "tensorflow"
if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"  # *** local kaggle *** local training or online testing
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
    NEEDTRAIN = False
    for dir_name in os.listdir("/kaggle/input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print("DATATYPE:", DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
else:
    LOAD_DATA_FROM = "./"

warnings.filterwarnings("ignore")

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]  # last six columns are the vote counts
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))

SMOOTH = 1e-6

global_votes = df[TARGETS].sum() + SMOOTH
global_frac = global_votes / global_votes.sum()
global_count = global_votes.sum()  # total number of votes globally (unused later)

patient_votes = df.groupby("patient_id")[TARGETS].sum() + SMOOTH
patient_frac = patient_votes.div(patient_votes.sum(axis=1), axis=0)
patient_counts = patient_votes.sum(axis=1)  # total (smoothed) votes per patient

eeg_votes = df.groupby("eeg_id")[TARGETS].sum() + SMOOTH
eeg_frac = eeg_votes.div(eeg_votes.sum(axis=1), axis=0)

spectro_votes = df.groupby("spectrogram_id")[TARGETS].sum() + SMOOTH
spectro_frac = spectro_votes.div(spectro_votes.sum(axis=1), axis=0)
spectro_counts = spectro_votes.sum(axis=1)  # total (smoothed) votes per spectrogram

test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))

GLOBAL_REG = 0.02  # smaller => more reliance on specific distributions

preds = []
for _, row in test.iterrows():
    eid = row["eeg_id"]
    pid = row["patient_id"]
    sid = row["spectrogram_id"]

    if eid in eeg_frac.index:
        probs = eeg_frac.loc[eid].values
    else:
        patient_present = pid in patient_frac.index
        spectro_present = sid in spectro_frac.index

        specific_parts = []
        specific_weights = []

        if patient_present:
            specific_parts.append(patient_frac.loc[pid].values)
            specific_weights.append(patient_counts.loc[pid])
        if spectro_present:
            specific_parts.append(spectro_frac.loc[sid].values)
            specific_weights.append(spectro_counts.loc[sid])

        if specific_parts:
            w = np.array(specific_weights, dtype=float)
            parts = np.stack(specific_parts, axis=0)
            probs_specific = np.average(parts, axis=0, weights=w)

            total_specific = w.sum()
            specific_weight = total_specific / (total_specific + GLOBAL_REG)
            probs = (
                specific_weight * probs_specific
                + (1.0 - specific_weight) * global_frac.values
            )
        else:
            probs = global_frac.values

    preds.append(probs)

pred_arr = np.stack(preds)

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
for i, col in enumerate(TARGETS):
    sub[col] = pred_arr[:, i]

epsilon = 1e-9
row_sums = sub[TARGETS].sum(axis=1) + epsilon
sub[TARGETS] = sub[TARGETS].div(row_sums, axis=0)

submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print(f"Baseline submission written to {submission_path} with shape {sub.shape}")
