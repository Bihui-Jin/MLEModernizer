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

0.312444673135533

# 6. Current score

1.14521

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I wrap the EfficientNet import in a safe try/except block and replace the heavy model‑inference section (which fails due to protobuf incompatibility) with a lightweight fallback: compute the overall class distribution from the training data and use it as a constant prediction for every test sample. This guarantees a valid .csv submission while keeping the original data‑loading logic intact and avoids the error that prevented the script from completing.'
- What this solution (achieved 1.68479) has done: 'I keep the overall fallback approach but add a cheap patient‑level calibration: compute the average vote distribution for each patient in the training data and use it for test rows when the patient appears in training, otherwise fall back to the global class probabilities. This simple personalization should lower the KL‑divergence (bring the score down toward the target) while preserving the original logic and without adding heavy model inference.'
- What this solution (achieved 1.68479) has done: 'I add a simple per‑eeg_id probability fallback (which is more specific than the patient‑level fallback) and keep the global fallback as a last resort. This minor hierarchy should make predictions better calibrated and move the KL‑divergence score closer to the target without changing the core model‑free logic.'
- What this solution (achieved 1.68479) has done: 'I add a lightweight spectrogram‑level fallback (using the `spectrogram_id` grouped vote distribution) before the existing eeg‑ and patient‑level lookups. This more specific prior is cheap, respects the original hierarchy, and should move the KL‑divergence lower toward the target without altering any core modeling logic.'
- What this solution (achieved 0.81597) has done: 'I keep the same hierarchical fallback logic but add a small smoothing step and blend each specific prediction with the global class distribution. This avoids zero probabilities (which heavily penalise KL‑divergence) and nudges overly specific priors toward the overall average, likely moving the score closer to the target while preserving the core approach.'
- What this solution (achieved 0.9785) has done: 'I lower the reliance on the specific hierarchical priors by reducing `SPECIFIC_WEIGHT` from 0.7 to 0.4 and add a tiny Laplace smoothing (+1 count) when building all group‑level probability tables. These minimal tweaks keep the overall fallback hierarchy intact while nudging predictions toward the more stable global distribution, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.19964) has done: 'I lower the reliance on the highly specific priors (spectrogram/eeg) by removing the spectrogram fallback and reducing the blending weight to 0.15, so predictions are dominated by the stable global class distribution while still keeping a modest personalization via patient‑level priors. This small adjustment should markedly cut the KL‑divergence, moving the score closer to the target without altering the overall pipeline.'
- What this solution (achieved 1.09682) has done: 'I add a lightweight spectrogram‑level prior (the most specific grouping) and increase the blending weight slightly so predictions benefit from this extra information while still being anchored by the stable global distribution. This small hierarchy extension and modest weight raise are expected to lower the KL‑divergence toward the target without altering the overall fallback logic.'
- What this solution (achieved 1.33362) has done: 'I lower the influence of the highly specific priors by reducing `SPECIFIC_WEIGHT` from 0.25 to 0.05. This makes each prediction rely more on the stable global class distribution, which empirically reduces KL‑divergence and moves the score closer to the target while keeping the original hierarchy and all other logic unchanged.'
- What this solution (achieved 1.26166) has done: 'I keep the existing data‑loading and probability‑building logic, but replace the single‑weight blending with a small hierarchical blending: start from the global class distribution, then successively blend in patient, EEG‑id and spectrogram priors using modest weights (0.1, 0.2, 0.5). This adds useful specific information while keeping the model‑free core unchanged, and should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I reduce the influence of all hierarchical priors by setting the blending weights to 0.0, so every test row uses only the globally‑computed class distribution. This removes noisy specific adjustments that were inflating the KL‑divergence, moving the score lower toward the target while keeping the overall pipeline unchanged and still producing a valid submission.csv.'
- What this solution (achieved 1.14521) has done: 'I activate a modest hierarchical blending of the patient, EEG‑id and spectrogram priors (instead of the all‑zero weights) so that predictions use specific information when available while still being anchored by the stable global distribution. Small positive weights (patient 0.2, EEG 0.3, spectrogram 0.5) are chosen to lower the KL‑divergence toward the target without altering the core pipeline.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np


def find_file(rel_path):
    candidates = [
        os.path.join("data", rel_path),  # repository root
        os.path.join("/kaggle/input", rel_path),  # Kaggle default input dir
        os.path.join("working", rel_path),  # alternate workspace dir
        rel_path,  # direct relative path
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Unable to locate file {rel_path} in any known location.")


LOAD_DATA_FROM = os.path.join("data", "hms-harmful-brain-activity-classification")
train_path = find_file(
    os.path.join("hms-harmful-brain-activity-classification", "train.csv")
)
df = pd.read_csv(train_path)
TARGETS = df.columns[-6:]  # ['seizure_vote', ... 'other_vote']
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))

class_counts = df[TARGETS].sum() + 1
class_probs = class_counts / class_counts.sum()
print("Overall class probabilities (fallback):")
print(class_probs)

patient_group = df.groupby("patient_id")[TARGETS].sum() + 1
patient_probs = patient_group.div(patient_group.sum(axis=1), axis=0)
patient_probs = patient_probs.fillna(class_probs)

eeg_group = df.groupby("eeg_id")[TARGETS].sum() + 1
eeg_probs = eeg_group.div(eeg_group.sum(axis=1), axis=0)
eeg_probs = eeg_probs.fillna(patient_probs)  # fallback to patient

spectrogram_group = df.groupby("spectrogram_id")[TARGETS].sum() + 1
spectrogram_probs = spectrogram_group.div(spectrogram_group.sum(axis=1), axis=0)
spectrogram_probs = spectrogram_probs.fillna(eeg_probs)  # fallback chain

test_path = find_file(
    os.path.join("hms-harmful-brain-activity-classification", "test.csv")
)
test = pd.read_csv(test_path)
test["sign_id"] = test.index.values
print("Test shape:", test.shape)




## === cell 1
PATIENT_WEIGHT = 0.2  # contribution of patient‑level prior
EEG_WEIGHT = 0.3  # contribution of EEG‑id prior
SPECIFIC_WEIGHT = 0.5  # contribution of spectrogram prior
EPS = 1e-9  # avoid exact zeros

preds_list = []
for _, row in test.iterrows():
    eid = row["eeg_id"]
    pid = row["patient_id"]
    sid = row["spectrogram_id"]

    prob = class_probs.values.copy()  # start from global prior

    if PATIENT_WEIGHT > 0 and pid in patient_probs.index:
        prob = (1 - PATIENT_WEIGHT) * prob + PATIENT_WEIGHT * patient_probs.loc[
            pid
        ].values

    if EEG_WEIGHT > 0 and eid in eeg_probs.index:
        prob = (1 - EEG_WEIGHT) * prob + EEG_WEIGHT * eeg_probs.loc[eid].values

    if SPECIFIC_WEIGHT > 0 and sid in spectrogram_probs.index:
        prob = (1 - SPECIFIC_WEIGHT) * prob + SPECIFIC_WEIGHT * spectrogram_probs.loc[
            sid
        ].values

    prob = prob + EPS  # ensure no zero entries
    prob = prob / prob.sum()  # renormalise
    preds_list.append(prob)

preds_all = np.vstack(preds_list)  # (n_test, n_classes)

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
sub[TARGETS] = preds_all
submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape {sub.shape}")
print(sub.head())
