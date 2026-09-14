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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pyarrow==19.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

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

0.7981938799409479

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'Implemented a minimal, robust pipeline that reads the provided training and test metadata, computes overall class vote fractions from the training set, and generates a valid `submission.csv` where each row’s probabilities sum to 1. All heavy processing, clustering, and model steps have been replaced with this concise baseline, ensuring the script runs end‑to‑end without file‑not‑found errors and produces a correctly formatted submission file.'
- What this solution (achieved 1.41937) has done: 'I add a per‑eeg ID probability lookup: for every `eeg_id` that appears in the training metadata I compute the class vote fractions from its own rows and use those as predictions for the same `eeg_id` in the test set. If a test `eeg_id` is unseen, the overall mean probabilities are used as before. This keeps the original baseline logic but adds useful conditioning, which should lower the KL‑divergence (move the score from 1.42 closer to the target 0.80). The rest of the pipeline – loading data, checking that rows sum to 1, and writing `submission.csv` – stays unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

above_dir = "../input/hms-harmful-brain-activity-classification/"

HBA_votes = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

SMOOTH_K = 5.0




## === cell 1
train_path = os.path.join(above_dir, "train.csv")
test_path = os.path.join(above_dir, "test.csv")

train_meta = pd.read_csv(train_path)
test_meta = pd.read_csv(test_path)

print("Train rows:", train_meta.shape[0])
print("Test rows :", test_meta.shape[0])

total_votes_per_class = train_meta[HBA_votes].sum()
overall_votes = total_votes_per_class.sum()
mean_probs = total_votes_per_class / overall_votes
print("Mean class probabilities (global prior):")
print(mean_probs.round(6))

eeg_votes_full = train_meta.groupby("eeg_id")[HBA_votes].sum()
eeg_counts_full = eeg_votes_full.sum(axis=1)
eeg_probs_full = (eeg_votes_full + SMOOTH_K * mean_probs) / (
    eeg_counts_full.values[:, None] + SMOOTH_K
)
eeg_probs_full = eeg_probs_full.fillna(mean_probs)  # safety for any zero‑count IDs

patient_votes_full = train_meta.groupby("patient_id")[HBA_votes].sum()
patient_counts_full = patient_votes_full.sum(axis=1)
patient_probs_full = (patient_votes_full + SMOOTH_K * mean_probs) / (
    patient_counts_full.values[:, None] + SMOOTH_K
)
patient_probs_full = patient_probs_full.fillna(mean_probs)




## === cell 2
train_split, val_split = train_test_split(
    train_meta, test_size=0.2, random_state=42, shuffle=True
)

eeg_votes_tr = train_split.groupby("eeg_id")[HBA_votes].sum()
eeg_counts_tr = eeg_votes_tr.sum(axis=1)
eeg_probs_tr = (eeg_votes_tr + SMOOTH_K * mean_probs) / (
    eeg_counts_tr.values[:, None] + SMOOTH_K
)
eeg_probs_tr = eeg_probs_tr.fillna(mean_probs)

patient_votes_tr = train_split.groupby("patient_id")[HBA_votes].sum()
patient_counts_tr = patient_votes_tr.sum(axis=1)
patient_probs_tr = (patient_votes_tr + SMOOTH_K * mean_probs) / (
    patient_counts_tr.values[:, None] + SMOOTH_K
)
patient_probs_tr = patient_probs_tr.fillna(mean_probs)


def build_predictions(df, eeg_probs, patient_probs, mean_probs, alpha):
    """Baseline prediction used for alpha tuning (no beta)."""
    pred = df[["eeg_id", "patient_id"]].copy()
    pred = pred.merge(eeg_probs, left_on="eeg_id", right_index=True, how="left")
    for col in HBA_votes:
        miss = pred[col].isna()
        if miss.any():
            pred.loc[miss, col] = pred.loc[miss, "patient_id"].map(patient_probs[col])
    for col in HBA_votes:
        pred[col] = pred[col].fillna(mean_probs[col])
    for col in HBA_votes:
        pred[col] = pred[col] * alpha + mean_probs[col] * (1 - alpha)
    row_sums = pred[HBA_votes].sum(axis=1)
    pred[HBA_votes] = pred[HBA_votes].div(row_sums, axis=0)
    return pred


def build_predictions_with_beta(df, eeg_probs, patient_probs, mean_probs, alpha, beta):
    """Prediction that incorporates patient info with weight beta."""
    pred = df[["eeg_id", "patient_id"]].copy()
    pred = pred.merge(eeg_probs, left_on="eeg_id", right_index=True, how="left")
    patient_aligned = patient_probs.reindex(pred["patient_id"]).reset_index(drop=True)
    for col in HBA_votes:
        pred[col] = (1 - beta) * pred[col] + beta * patient_aligned[col]
    for col in HBA_votes:
        miss = pred[col].isna()
        if miss.any():
            pred.loc[miss, col] = pred.loc[miss, "patient_id"].map(patient_probs[col])
    for col in HBA_votes:
        pred[col] = pred[col].fillna(mean_probs[col])
    for col in HBA_votes:
        pred[col] = pred[col] * alpha + mean_probs[col] * (1 - alpha)
    row_sums = pred[HBA_votes].sum(axis=1)
    pred[HBA_votes] = pred[HBA_votes].div(row_sums, axis=0)
    return pred


def kl_divergence(true_counts, pred_probs):
    """Mean per‑row KL(true || pred)."""
    true_probs = true_counts / true_counts.sum(axis=1, keepdims=True)
    eps = 1e-12
    pred_probs = np.clip(pred_probs, eps, 1.0)
    kl = np.sum(true_probs * np.log(true_probs / pred_probs), axis=1)
    return np.mean(kl)


candidate_alphas = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
candidate_betas = np.linspace(0, 0.5, 6)  # 0.0, 0.1, ..., 0.5

best_alpha = None
best_beta = None
best_score = np.inf

for a in candidate_alphas:
    for b in candidate_betas:
        pred_val = build_predictions_with_beta(
            val_split, eeg_probs_tr, patient_probs_tr, mean_probs, a, b
        )
        score = kl_divergence(val_split[HBA_votes].values, pred_val[HBA_votes].values)
        print(f"alpha={a:.2f}, beta={b:.2f} -> KL={score:.5f}")
        if score < best_score:
            best_score = score
            best_alpha = a
            best_beta = b

if best_alpha is not None and best_beta is not None:
    refined_alphas = np.linspace(
        max(0.0, best_alpha - 0.1), min(1.0, best_alpha + 0.1), 11
    )
    refined_betas = np.linspace(
        max(0.0, best_beta - 0.05), min(0.5, best_beta + 0.05), 11
    )
    for a in refined_alphas:
        for b in refined_betas:
            pred_val = build_predictions_with_beta(
                val_split, eeg_probs_tr, patient_probs_tr, mean_probs, a, b
            )
            score = kl_divergence(
                val_split[HBA_votes].values, pred_val[HBA_votes].values
            )
            if score < best_score:
                best_score = score
                best_alpha = a
                best_beta = b
                print(f"Refined: alpha={a:.3f}, beta={b:.3f} -> KL={score:.5f}")

if best_alpha is None or best_beta is None:
    best_alpha, best_beta = 0.5, 0.2
    print("Fallback to default alpha=0.5, beta=0.2")

print(
    f"Chosen alpha = {best_alpha:.3f}, beta = {best_beta:.3f} (best KL={best_score:.5f})"
)




## === cell 3
alpha = best_alpha
beta = best_beta

submission = test_meta[["eeg_id", "patient_id"]].copy()
submission = submission.merge(
    eeg_probs_full, left_on="eeg_id", right_index=True, how="left"
)

patient_aligned_test = patient_probs_full.reindex(submission["patient_id"]).reset_index(
    drop=True
)

for vote_col in HBA_votes:
    submission[vote_col] = (1 - beta) * submission[
        vote_col
    ] + beta * patient_aligned_test[vote_col]

for vote_col in HBA_votes:
    missing_mask = submission[vote_col].isna()
    if missing_mask.any():
        submission.loc[missing_mask, vote_col] = submission.loc[
            missing_mask, "patient_id"
        ].map(patient_probs_full[vote_col])

for vote_col in HBA_votes:
    submission[vote_col] = submission[vote_col].fillna(mean_probs[vote_col])

for vote_col in HBA_votes:
    submission[vote_col] = submission[vote_col] * alpha + mean_probs[vote_col] * (
        1 - alpha
    )

row_sums = submission[HBA_votes].sum(axis=1)
submission[HBA_votes] = submission[HBA_votes].div(row_sums, axis=0)

assert np.allclose(
    submission[HBA_votes].sum(axis=1), 1.0, atol=1e-6
), "Probabilities do not sum to 1!"

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False, float_format="%.6f")
print(f"Submission written to {submission_path}")
