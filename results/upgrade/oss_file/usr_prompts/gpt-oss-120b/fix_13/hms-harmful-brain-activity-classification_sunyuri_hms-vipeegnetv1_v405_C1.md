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

0.283964982625568

# 6. Current score

0.90905

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I remove the TensorFlow import that causes the protobuf `MessageFactory` error and replace the training pipeline with a lightweight baseline: compute the overall class vote distribution from the training data and use this same normalized distribution for every test entry. This avoids the runtime crash, guarantees a valid CSV with the correct columns, and yields a reasonable KL‑divergence score (moving toward the target without altering any core model logic).'
- What this solution (achieved 1.39779) has done: 'The fix adds the missing CSV helper, correct imports, and a safe fallback that uses patient‑level vote distributions before resorting to the global distribution. This resolves the NameError, guarantees a proper submission.csv with rows summing to 1, and modestly improves the KL‑divergence by providing more specific priors for unseen eeg_id entries.'
- What this solution (achieved 1.40995) has done: 'I keep the overall blending approach but tighten it up: compute the global baseline from total vote counts (more representative), add a tiny smoothing term before normalising blended probabilities so no class gets a zero prediction (which heavily penalises KL), and search the weight grid a bit more finely (0.02 steps). These small adjustments stay within the original logic while expected to lower the validation KL and move the score closer to the target.'
- What this solution (achieved 1.40995) has done: 'I add a small temperature‑scaling step after the blended probabilities.  
The code now searches for the best temperature α (0.5‑2.0) on the validation split, picks the α that yields the lowest KL, and then applies the same α to the test predictions. This keeps the original blending logic intact while giving a modest KL improvement and still writes a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I add a tiny Dirichlet‑style smoothing to the per‑eeg and per‑patient probability tables and increase the post‑blend smoothing from 1e‑6 to 1e‑3. This prevents zero predictions for classes that appear in the true labels, which heavily penalises KL‑divergence, and should move the validation score closer to the target while keeping the original blending logic unchanged.'
- What this solution (achieved 0.90905) has done: 'I replace the grid‑search blending with a simple hierarchical fallback: use the per‑EEG distribution when available, otherwise the per‑patient distribution, otherwise the global distribution. This keeps the original smoothing idea, adds a tiny smoothing term, and still applies the temperature‑scaling search that was already present, but removes the unnecessary weight‑grid loop. The change is minimal to the core logic while expected to lower the KL score toward the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os
from pathlib import Path


def read_csv_safe(path: str) -> pd.DataFrame:
    """
    Safely read a CSV file, trying several common base directories.
    Raises a clear error if the file cannot be found or read.
    """
    possible_bases = [
        Path("."),  # current directory
        Path("data"),  # typical data folder
        Path("kaggle/data"),  # Kaggle kernel layout
        Path("kaggle/working/hms-harmful-brain-activity-classification"),
        Path("hms-harmful-brain-activity-classification"),
        Path("working/hms-harmful-brain-activity-classification"),
    ]
    for base in possible_bases:
        candidate = base / path
        if candidate.is_file():
            try:
                return pd.read_csv(candidate)
            except Exception as exc:
                raise RuntimeError(f"Unable to read CSV at {candidate}: {exc}")
    raise FileNotFoundError(
        f"Could not locate CSV file '{path}' in any of the expected locations: {possible_bases}"
    )




## === cell 1
train_df = read_csv_safe("train.csv")

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

assert all(col in train_df.columns for col in TARGET_COLS), "Missing target columns"

vote_counts = train_df[TARGET_COLS].values.astype(float)
row_sums = vote_counts.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
row_probs = vote_counts / row_sums

global_counts = train_df[TARGET_COLS].sum()
global_probs = global_counts / global_counts.sum()
global_prob_vec = global_probs.values  # numpy array for broadcasting

smooth_eps = 1e-4  # a tiny Dirichlet‑style smoothing

eeg_group_counts = train_df.groupby("eeg_id")[TARGET_COLS].sum() + smooth_eps
eeg_group_sums = eeg_group_counts.sum(axis=1).replace(0, np.nan)
eeg_group_probs = eeg_group_counts.div(eeg_group_sums, axis=0).fillna(0)

patient_group_counts = train_df.groupby("patient_id")[TARGET_COLS].sum() + smooth_eps
patient_group_sums = patient_group_counts.sum(axis=1).replace(0, np.nan)
patient_group_probs = patient_group_counts.div(patient_group_sums, axis=0).fillna(0)


def get_fallback_probs(eeg_ids, patient_ids, eps_smooth=1e-4):
    """
    For each row, use the per‑EEG distribution if available,
    else per‑patient, else the global distribution.
    A small epsilon is added to avoid zero probabilities.
    """
    eeg_probs = eeg_group_probs.reindex(eeg_ids).fillna(0).values
    pat_probs = patient_group_probs.reindex(patient_ids).fillna(0).values

    has_eeg = eeg_probs.sum(axis=1) > 0
    has_pat = pat_probs.sum(axis=1) > 0

    probs = np.empty((len(eeg_ids), len(TARGET_COLS)), dtype=float)

    probs[has_eeg] = eeg_probs[has_eeg]
    only_pat = ~has_eeg & has_pat
    probs[only_pat] = pat_probs[only_pat]
    neither = ~has_eeg & ~has_pat
    probs[neither] = global_prob_vec  # broadcast automatically

    probs += eps_smooth
    row_sum = probs.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return probs / row_sum


def apply_temperature(probs, alpha):
    """Sharpen or smooth a probability matrix with exponentiation and re‑normalisation."""
    if alpha == 1.0:
        return probs
    scaled = np.power(probs, alpha)
    row_sum = scaled.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return scaled / row_sum


rng = np.random.default_rng(42)
perm = rng.permutation(len(train_df))
val_size = int(0.1 * len(train_df))
val_idx = perm[:val_size]
val_df = train_df.iloc[val_idx].reset_index(drop=True)

val_eeg_ids = val_df["eeg_id"]
val_patient_ids = val_df["patient_id"]
val_true = row_probs[val_idx]  # true distributions for validation rows

best_alpha = 1.0
best_kl = np.inf
eps = 1e-12

base_pred = get_fallback_probs(val_eeg_ids, val_patient_ids, eps_smooth=smooth_eps)

for alpha in np.arange(0.5, 2.01, 0.1):
    pred_temp = apply_temperature(base_pred, alpha)
    kl = np.where(
        val_true > 0, val_true * np.log((val_true + eps) / (pred_temp + eps)), 0.0
    ).sum(axis=1)
    avg_kl = kl.mean()
    if avg_kl < best_kl:
        best_kl = avg_kl
        best_alpha = alpha

print(f"Chosen temperature α = {best_alpha:.2f}, Validation KL = {best_kl:.6f}")

test_df = read_csv_safe("test.csv")
assert "eeg_id" in test_df.columns, "test.csv must contain eeg_id"

test_probs_matrix = get_fallback_probs(
    test_df["eeg_id"], test_df["patient_id"], eps_smooth=smooth_eps
)

test_probs_matrix = apply_temperature(test_probs_matrix, best_alpha)

submission = pd.DataFrame()
submission["eeg_id"] = test_df["eeg_id"]
for idx, col in enumerate(TARGET_COLS):
    submission[col] = test_probs_matrix[:, idx]

row_sum = submission[TARGET_COLS].sum(axis=1)
submission[TARGET_COLS] = submission[TARGET_COLS].div(row_sum.replace(0, 1), axis=0)

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
print(submission.head())
