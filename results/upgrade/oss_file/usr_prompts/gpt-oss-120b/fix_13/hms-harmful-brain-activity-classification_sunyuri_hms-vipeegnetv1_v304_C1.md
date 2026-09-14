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

0.3151513005943226

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I avoid the protobuf import error by wrapping the EfficientNet import in a safe try/except and, since training is disabled, replace the whole inference pipeline with a simple baseline that uses the class‑frequency averages from the training set to generate constant predictions for every test sample. This fixes the runtime crash and guarantees a valid `submission.csv` while keeping the core logic untouched. The change is minimal and score‑neutral (it only produces a valid submission).'
- What this solution (achieved 1.64506) has done: 'We wrap TensorFlow imports in a safe try/except and skip all TF‑related setup when the import fails, because the baseline solution does not need TF.  
Then we improve the baseline from a single global class‑frequency vector to a **patient‑wise** probability vector: for each patient we compute the mean normalized vote distribution from the training set and use it for test rows belonging to that patient (fallback to the global average otherwise). This keeps the core logic unchanged while providing a modest, score‑reducing calibration.'
- What this solution (achieved 1.68479) has done: 'The fix prevents TensorFlow‑related crashes by disabling any TF usage if it cannot be safely imported, and improves the baseline predictions by using raw vote counts → patient‑wise normalized vote distributions (fallback to the global distribution). This keeps the original workflow but yields probabilities that better match the training label statistics, moving the KL divergence toward the target.'
- What this solution (achieved 1.68479) has done: 'We wrap all TensorFlow‑related configuration in a safe try/except so that any protobuf incompatibility is caught and the script continues without TF.  
Then we enrich the baseline by also computing per‑eeg‑id vote distributions (fallback to patient‑wise, then global) and use these probabilities for the test rows, keeping the overall workflow unchanged while improving calibration and lowering the KL‑divergence score.'
- What this solution (achieved 0.78827) has done: 'The fix disables TensorFlow entirely (avoiding the protobuf `MessageFactory` crash) and keeps the original baseline logic while adding a small smoothing blend between patient‑wise probabilities and the global class distribution to improve calibration and lower the KL‑divergence score.'
- What this solution (achieved 1.13446) has done: 'Implemented a lighter patient‑wise blending and added tiny Laplace smoothing to avoid zero‑probability penalties, which should move the KL‑divergence closer to the target (lower is better). The core logic and data handling remain unchanged; only the blend weight and probability handling were tweaked.'
- What this solution (achieved 0.90499) has done: 'I raise the influence of the patient‑wise vote distribution by increasing the blending weight (`ALPHA_PATIENT`) from 0.2 to 0.5 (and adjust `ALPHA_GLOBAL` accordingly). This gives more personalized probabilities while still falling back to the global class distribution, which should lower the KL‑divergence and move the score closer to the target without altering any core logic.'
- What this solution (achieved 1.05318) has done: 'I fix the incorrect import (`np` should be imported from NumPy) and adjust the blending weights so that patient‑wise and EEG‑wise distributions are used directly (no global mixing). This removes the import error, ensures a valid submission CSV, and modestly improves calibration to move the KL‑divergence score closer to the target while preserving the original logic.'
- What this solution (achieved 0.80306) has done: 'I keep the overall workflow but replace the hard‑selection of a single distribution with a small blend of EEG‑wise, patient‑wise and global class probabilities. By giving a modest weight to the global prior (e.g., 0.1) and mixing EEG‑wise 0.6 with patient‑wise 0.3, the predictions become better calibrated and should lower the KL‑divergence toward the target while preserving the original logic. I also increase the Laplace smoothing slightly to avoid zero‑probability spikes.'
- What this solution (achieved 1.41937) has done: 'I add a lightweight validation step that searches a small grid of blending weights (EEG‑wise, patient‑wise, global) to find the combination that minimizes KL‑divergence on a hold‑out split of the training data. The best weights are then used for the final test predictions. This keeps the original baseline logic while moving the score closer to the target (lower KL).'
- What this solution (achieved 1.41937) has done: 'The update keeps the original baseline logic (global, patient‑wise and EEG‑wise probability blending) but adds a two‑stage grid search with a finer resolution around the best coarse‑grid weights. This finds a better combination of the three blending coefficients, which lowers the KL‑divergence on the validation split and moves the score closer to the target. A fixed random seed ensures reproducible validation splits.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri
email: syuri@tju.edu.cn
"""

import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

PLATFORM = "kaggle"  # "local" or "kaggle"
if PLATFORM == "local":
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
else:
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]  # last six columns are the vote counts
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))

total_votes = df[TARGETS].sum()
global_class_probs = (total_votes / total_votes.sum()).values  # (6,)

patient_sum = df.groupby("patient_id")[list(TARGETS)].sum()
patient_prob_dict = {}
for pid, row in patient_sum.iterrows():
    row_sum = row.sum()
    patient_prob_dict[pid] = (
        (row / row_sum).values if row_sum > 0 else global_class_probs
    )

eeg_sum = df.groupby("eeg_id")[list(TARGETS)].sum()
eeg_prob_dict = {}
for eid, row in eeg_sum.iterrows():
    row_sum = row.sum()
    eeg_prob_dict[eid] = (row / row_sum).values if row_sum > 0 else global_class_probs

np.random.seed(42)
val_frac = 0.2
val_mask = np.random.rand(len(df)) < val_frac
val_df = df[val_mask].reset_index(drop=True)


def predict_probs(eeg_ids, patient_ids, al_eeg, al_pat, al_glob, eps=1e-6):
    """Return probability matrix for given IDs and blend weights."""
    probs = np.empty((len(eeg_ids), len(TARGETS)), dtype=np.float64)
    for i, (eid, pid) in enumerate(zip(eeg_ids, patient_ids)):
        prob = np.zeros_like(global_class_probs)
        if eid in eeg_prob_dict:
            prob += al_eeg * eeg_prob_dict[eid]
        if pid in patient_prob_dict:
            prob += al_pat * patient_prob_dict[pid]
        prob += al_glob * global_class_probs
        if prob.sum() == 0:
            prob = global_class_probs.copy()
        prob = prob + eps
        prob = prob / prob.sum()
        probs[i] = prob
    return probs


def kl_divergence(true_counts, pred_probs, eps=1e-12):
    true_probs = true_counts / true_counts.sum(axis=1, keepdims=True)
    return np.mean(
        np.sum(true_probs * np.log((true_probs + eps) / (pred_probs + eps)), axis=1)
    )


best_score = np.inf
best_weights = (0.6, 0.3, 0.1)  # fallback
coarse_steps = np.arange(0.0, 1.01, 0.1)

for al_eeg in coarse_steps:
    for al_pat in coarse_steps:
        if al_eeg + al_pat > 1.0:
            continue
        al_glob = 1.0 - (al_eeg + al_pat)
        preds = predict_probs(
            val_df["eeg_id"].values,
            val_df["patient_id"].values,
            al_eeg,
            al_pat,
            al_glob,
        )
        score = kl_divergence(val_df[list(TARGETS)].values, preds)
        if score < best_score:
            best_score = score
            best_weights = (al_eeg, al_pat, al_glob)

fine_range = 0.1
fine_steps = np.arange(-fine_range, fine_range + 1e-9, 0.01)

base_eeg, base_pat, base_glob = best_weights
for delta_eeg in fine_steps:
    for delta_pat in fine_steps:
        al_eeg = base_eeg + delta_eeg
        al_pat = base_pat + delta_pat
        if al_eeg < 0 or al_pat < 0:
            continue
        if al_eeg + al_pat > 1.0:
            continue
        al_glob = 1.0 - (al_eeg + al_pat)
        if al_glob < 0:
            continue
        preds = predict_probs(
            val_df["eeg_id"].values,
            val_df["patient_id"].values,
            al_eeg,
            al_pat,
            al_glob,
        )
        score = kl_divergence(val_df[list(TARGETS)].values, preds)
        if score < best_score:
            best_score = score
            best_weights = (al_eeg, al_pat, al_glob)

ALPHA_EEG, ALPHA_PATIENT, ALPHA_GLOBAL = best_weights
print(
    f"Chosen weights → EEG: {ALPHA_EEG:.3f}, PATIENT: {ALPHA_PATIENT:.3f}, GLOBAL: {ALPHA_GLOBAL:.3f}"
)
print(f"Validation KL‑divergence (lower is better): {best_score:.5f}")

test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
print("Test shape:", test.shape)

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})

for idx, col in enumerate(TARGETS):
    sub[col] = global_class_probs[idx]

EPS = 1e-12

for i, (eeg_id, patient_id) in enumerate(
    zip(test["eeg_id"].values, test["patient_id"].values)
):
    prob = np.zeros_like(global_class_probs)

    if eeg_id in eeg_prob_dict:
        prob += ALPHA_EEG * eeg_prob_dict[eeg_id]

    if patient_id in patient_prob_dict:
        prob += ALPHA_PATIENT * patient_prob_dict[patient_id]

    prob += ALPHA_GLOBAL * global_class_probs

    if prob.sum() == 0:
        prob = global_class_probs.copy()

    prob = prob + EPS
    prob = prob / prob.sum()

    sub.iloc[i, sub.columns.get_indexer(TARGETS)] = prob

prob_sum = sub[TARGETS].sum(axis=1)
if not np.allclose(prob_sum, 1.0):
    sub[TARGETS] = sub[TARGETS].div(prob_sum, axis=0)

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print(f"Submission written to {sub_path}")
print("Submission shape:", sub.shape)
sub.head()
