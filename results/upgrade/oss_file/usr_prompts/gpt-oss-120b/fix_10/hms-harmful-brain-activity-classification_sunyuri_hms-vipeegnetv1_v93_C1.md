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

0.343317868053277

# 6. Current score

0.81032

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'We guard the EfficientNet import (only needed when training) and replace the inference section with a simple baseline that predicts the overall class distribution from the training data. This avoids the protobuf error and still outputs a correctly‑formatted `submission.csv` containing probabilities that sum to 1 for each row.'
- What this solution (achieved 0.78827) has done: 'The fix adds a robust file‑search helper that looks for the CSV files in all typical Kaggle locations ( `/kaggle/input`, `./data`, `./input` ) and falls back to the current directory. This prevents the `FileNotFoundError` and lets the script run end‑to‑end, producing a correctly‑formatted `submission.csv` with probabilities that sum to 1. No core modeling logic is changed, keeping the original baseline behavior intact.'
- What this solution (achieved 0.8694) has done: 'I replace the fixed blending weight with a dynamic one that depends on how many training samples a patient has: patients with many records keep most of their own distribution, while rare patients fall back toward the global class distribution. This modest change preserves the overall baseline logic but should produce probabilities that better match the true label distribution, thereby lowering the KL‑divergence score toward the target.'
- What this solution (achieved 0.82635) has done: 'I add per‑eeg_id probability blending in addition to the existing patient‑based blending. For each test row we first try to use the specific eeg_id distribution (weighted by how many training samples that eeg_id has); if it is missing we fall back to the patient‑specific blend, and finally to the global prior. This extra specificity should make the predicted probabilities closer to the true distribution and thus lower the KL‑divergence toward the target score, while keeping the overall baseline logic unchanged.'
- What this solution (achieved 0.83976) has done: 'I add a small Laplace smoothing term when computing the per‑patient and per‑eeg probability tables and make the blending weights less aggressive (scaled by 0.5). These tweaks keep the overall baseline logic but pull extreme entity‑specific distributions toward the global prior, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.96311) has done: 'I lower the entity‑specific blending strength by introducing a `BLEND_SCALE` constant (set to 0.25) and using it when computing the weight for patient‑ and EEG‑based probabilities. This keeps the overall blending logic unchanged but pulls the predictions closer to the global class distribution, which should reduce the KL‑divergence and move the score toward the target.'
- What this solution (achieved 0.81032) has done: 'I increase the blending strength (raise `BLEND_SCALE`) and modify the probability lookup so that when both an eeg_id and a patient_id have training data we blend their entity‑specific distributions together before mixing with the global prior. This uses the same blending mechanism but gives a richer, more specific prediction, which should lower the KL‑divergence and move the score toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from pathlib import Path


def locate_file(relative_path: str) -> Path:
    """
    Search for a file in common Kaggle directories.
    Returns the first matching Path, or raises FileNotFoundError.
    """
    candidates = [
        Path("/kaggle/input") / relative_path,
        Path("./data") / relative_path,
        Path("./input") / relative_path,
        Path(".") / relative_path,
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Could not find {relative_path} in any known location.")


train_path = locate_file("hms-harmful-brain-activity-classification/train.csv")
test_path = locate_file("hms-harmful-brain-activity-classification/test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

TARGETS = train_df.columns[-6:]  # seizure_vote … other_vote

class_sums = train_df[TARGETS].sum()
global_probs = class_sums / class_sums.sum()

patient_sums = (
    train_df.groupby("patient_id")[list(TARGETS)].sum() + 1
)  # Laplace smoothing
patient_probs = patient_sums.div(patient_sums.sum(axis=1), axis=0)

patient_counts = train_df.groupby("patient_id").size()
MAX_PATIENT_COUNT = patient_counts.quantile(0.75)  # 75‑th percentile as “many”

eeg_sums = train_df.groupby("eeg_id")[list(TARGETS)].sum() + 1
eeg_probs = eeg_sums.div(eeg_sums.sum(axis=1), axis=0)

eeg_counts = train_df.groupby("eeg_id").size()
MAX_EEG_COUNT = eeg_counts.quantile(0.75)  # 75‑th percentile for EEG blending

BLEND_SCALE = 0.70


def blend_probs(
    base_probs: np.ndarray, specific_probs: np.ndarray, weight: float
) -> np.ndarray:
    """Blend specific probabilities toward a base (global) distribution."""
    blended = weight * specific_probs + (1.0 - weight) * base_probs
    blended = np.clip(blended, 1e-8, None)
    return blended


def get_row_probs(row) -> np.ndarray:
    """
    Return a probability vector for a test row.
    If both eeg_id and patient_id have training data we combine them;
    otherwise we fall back to the available entity or the global prior.
    The blend weight grows with the amount of training data (scaled by BLEND_SCALE).
    """
    pid = row["patient_id"]
    eid = row["eeg_id"]

    have_eeg = eid in eeg_probs.index
    have_pat = pid in patient_probs.index

    weight_eeg = 0.0
    weight_pat = 0.0
    if have_eeg:
        count_eeg = eeg_counts.loc[eid]
        weight_eeg = min(1.0, BLEND_SCALE * np.sqrt(count_eeg / MAX_EEG_COUNT))
    if have_pat:
        count_pat = patient_counts.loc[pid]
        weight_pat = min(1.0, BLEND_SCALE * np.sqrt(count_pat / MAX_PATIENT_COUNT))

    if have_eeg and have_pat:
        specific = (
            weight_eeg * eeg_probs.loc[eid].values
            + weight_pat * patient_probs.loc[pid].values
        ) / (weight_eeg + weight_pat)
        overall_weight = max(weight_eeg, weight_pat)
        return blend_probs(global_probs.values, specific, overall_weight)

    if have_eeg:
        prob = eeg_probs.loc[eid].values
        return blend_probs(global_probs.values, prob, weight_eeg)

    if have_pat:
        prob = patient_probs.loc[pid].values
        return blend_probs(global_probs.values, prob, weight_pat)

    return global_probs.values.copy()


prob_matrix = np.vstack(test_df.apply(get_row_probs, axis=1).values)
prob_df = pd.DataFrame(prob_matrix, columns=TARGETS)

row_sums = prob_df[TARGETS].sum(axis=1).replace(0, 1)
prob_df[TARGETS] = prob_df[TARGETS].div(row_sums, axis=0)

sub = pd.DataFrame({"eeg_id": test_df["eeg_id"]})
sub = pd.concat([sub, prob_df[TARGETS]], axis=1)

submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print("First few rows:")
print(sub.head())
